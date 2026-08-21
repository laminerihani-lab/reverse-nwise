"""Unit tests for analysis.stats: bootstrap CI, Cliff's delta, Mann-Whitney."""

from __future__ import annotations

import numpy as np

from analysis.stats import bootstrap_ci, cliffs_delta, compare, mann_whitney


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(10.0, 1.0, size=200)
    ci = bootstrap_ci(data, seed=1)
    assert ci.lo <= ci.mean <= ci.hi
    assert abs(ci.mean - 10.0) < 0.5


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([])
    assert np.isnan(ci.mean)


def test_bootstrap_ci_constant():
    ci = bootstrap_ci([5.0] * 20, seed=0)
    assert ci.lo == ci.hi == ci.mean == 5.0


def test_cliffs_delta_full_dominance():
    d, mag = cliffs_delta([3, 4, 5], [0, 1, 2])
    assert d == 1.0
    assert mag == "large"


def test_cliffs_delta_reverse():
    d, mag = cliffs_delta([0, 1, 2], [3, 4, 5])
    assert d == -1.0
    assert mag == "large"


def test_cliffs_delta_negligible():
    d, mag = cliffs_delta([1, 2, 3, 4], [1, 2, 3, 4])
    assert d == 0.0
    assert mag == "negligible"


def test_mann_whitney_greater():
    u, p = mann_whitney([5, 6, 7, 8], [1, 2, 3, 4], alternative="greater")
    assert p < 0.05


def test_mann_whitney_identical_constants():
    u, p = mann_whitney([1, 1, 1], [1, 1, 1])
    assert p == 1.0


def test_compare_significant():
    a = [1.0] * 10  # RNWise: perfect
    b = [0.6] * 10  # baseline: worse
    res = compare("OCov_s", "RNWise", a, "Baseline", b)
    assert res.mean_a > res.mean_b
    assert res.significant
    assert res.cliffs_delta == 1.0
    assert res.effect_magnitude == "large"


def test_compare_row_keys():
    res = compare("FDR", "RNWise", [1.0, 1.0], "B", [0.5, 0.5])
    row = res.row()
    for k in ("metric", "a", "b", "mean_a", "mean_b", "U", "p_value", "cliffs_delta", "effect", "significant"):
        assert k in row
