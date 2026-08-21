"""Generate figures from experiment results (matplotlib, optional dependency).

Produces the coverage-comparison bar chart, the strength-scaling curve, and the
prioritisation coverage curve. matplotlib is an optional extra
(``pip install rnwise[analysis]``); import is deferred.

Usage:
    python -m analysis.plots --exp04 results/exp04_30runs.json --exp03 results/exp03_seed0.json --out results/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("plots require matplotlib; install with `pip install rnwise[analysis]`") from exc


def plot_coverage_bars(exp04: dict, out: Path) -> None:
    plt = _mpl()
    agg = exp04["aggregate"]
    methods = list(agg.keys())
    ocov = [agg[m]["OCov_s"]["mean"] * 100 for m in methods]
    fdr = [agg[m]["FDR_pct"]["mean"] * 100 for m in methods]

    import numpy as np

    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, ocov, w, label="OCov$_s$")
    ax.bar(x + w / 2, fdr, w, label="FDR")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.set_title("Output coverage and fault detection (30-run means)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "coverage_bars.png", dpi=150)
    plt.close(fig)


def plot_strength_curve(exp03: dict, out: Path) -> None:
    plt = _mpl()
    rows = exp03["by_strength"]
    s = [r["s"] for r in rows]
    universe = [r["full_universe"] for r in rows]
    tests = [r["prioritized_tests"] for r in rows]
    lb = [r["corollary1_lower_bound"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(s, universe, "o-", label="full universe")
    ax.plot(s, tests, "s-", label="RNWise tests")
    ax.plot(s, lb, "^--", label="$v^s$ lower bound")
    ax.set_yscale("log")
    ax.set_xlabel("coverage strength $s$")
    ax.set_ylabel("count (log)")
    ax.set_title("Strength scaling (Adult + XGBoost)")
    ax.set_xticks(s)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "strength_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="generate result figures")
    ap.add_argument("--exp04", type=str, default="")
    ap.add_argument("--exp03", type=str, default="")
    ap.add_argument("--out", type=str, default="results/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.exp04:
        plot_coverage_bars(json.loads(Path(args.exp04).read_text()), out)
        print(f"wrote {out}/coverage_bars.png")
    if args.exp03:
        plot_strength_curve(json.loads(Path(args.exp03).read_text()), out)
        print(f"wrote {out}/strength_curve.png")


if __name__ == "__main__":
    main()
