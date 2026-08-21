"""Statistical analysis for repeated-runs experiments.

Provides the effect-size and significance machinery required for a rigorous
empirical section: Mann-Whitney U (non-parametric, no normality assumption),
bootstrap confidence intervals (percentile method), and Cliff's delta
(non-parametric effect size with the standard magnitude thresholds).

These back the 30-run comparisons in exp04: for each metric we report
mean +/- 95% bootstrap CI per method, and pairwise Reverse-N-Wise-vs-baseline
p-values with effect sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class BootstrapCI:
    mean: float
    lo: float
    hi: float
    level: float = 0.95

    def as_tuple(self) -> tuple[float, float, float]:
        return self.mean, self.lo, self.hi


def bootstrap_ci(
    samples: Sequence[float],
    level: float = 0.95,
    n_boot: int = 10000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``samples``."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return BootstrapCI(mean=float("nan"), lo=float("nan"), hi=float("nan"), level=level)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return BootstrapCI(mean=float(arr.mean()), lo=float(lo), hi=float(hi), level=level)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> tuple[float, str]:
    """Cliff's delta effect size between ``a`` and ``b`` with magnitude label.

    delta = P(a > b) - P(a < b) in [-1, 1]. Thresholds (Romano et al. 2006):
      |d|<0.147 negligible, <0.33 small, <0.474 medium, else large.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0, "undefined"
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    delta = (gt - lt) / (x.size * y.size)
    ad = abs(delta)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return float(delta), mag


@dataclass
class ComparisonResult:
    """One Reverse-N-Wise-vs-baseline comparison on a single metric."""

    metric: str
    method_a: str
    method_b: str
    mean_a: float
    mean_b: float
    u_statistic: float
    p_value: float
    cliffs_delta: float
    effect_magnitude: str
    significant: bool

    def row(self) -> dict:
        return {
            "metric": self.metric,
            "a": self.method_a,
            "b": self.method_b,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "U": round(self.u_statistic, 1),
            "p_value": self.p_value,
            "cliffs_delta": round(self.cliffs_delta, 3),
            "effect": self.effect_magnitude,
            "significant": self.significant,
        }


def mann_whitney(
    a: Sequence[float],
    b: Sequence[float],
    alternative: str = "greater",
) -> tuple[float, float]:
    """Mann-Whitney U test. Returns (U, p). ``alternative='greater'`` tests
    whether ``a`` stochastically dominates ``b``. Handles the degenerate
    all-identical case (returns p=1.0)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
        return 0.0, 1.0
    try:
        u, p = stats.mannwhitneyu(x, y, alternative=alternative)
    except ValueError:
        return 0.0, 1.0
    return float(u), float(p)


def compare(
    metric: str,
    method_a: str,
    a: Sequence[float],
    method_b: str,
    b: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "greater",
) -> ComparisonResult:
    """Full comparison: MWU p-value + Cliff's delta + significance flag."""
    u, p = mann_whitney(a, b, alternative=alternative)
    delta, mag = cliffs_delta(a, b)
    return ComparisonResult(
        metric=metric,
        method_a=method_a,
        method_b=method_b,
        mean_a=float(np.mean(a)) if len(a) else float("nan"),
        mean_b=float(np.mean(b)) if len(b) else float("nan"),
        u_statistic=u,
        p_value=p,
        cliffs_delta=delta,
        effect_magnitude=mag,
        significant=p < alpha,
    )
