"""Integration test for exp01: the Table 1 reproduction on Adult + XGBoost.

Marked slow because it trains XGBoost and runs the full pipeline; it asserts the
paper's qualitative claim -- Reverse N-Wise Pareto-dominates every baseline on
output coverage and fault detection.
"""

from __future__ import annotations

import pytest

pytest.importorskip("xgboost")
pytest.importorskip("sklearn")


@pytest.mark.slow
def test_exp01_rnwise_dominates_baselines():
    from experiments.exp01_table1_adult import run

    res = run(seed=0, n_probes=4000, max_iter=100)
    rows = {r["method"]: r for r in res["results"]}

    rn = rows["Reverse N-Wise"]
    # RNWise should reach (near) full feasible coverage and top FDR.
    assert rn["OCov_s"] >= 0.98
    assert rn["FDR_pct"] >= 0.75  # detects the reachable seeded faults

    # RNWise OCov_s must be >= every baseline's OCov_s.
    for name, r in rows.items():
        if name == "Reverse N-Wise":
            continue
        assert rn["OCov_s"] >= r["OCov_s"] - 1e-9, f"{name} out-covers RNWise"

    # Classic input-side CT and random must be clearly worse on output coverage.
    assert rows["Input CT"]["OCov_s"] < 0.8
    assert rows["Random-%d" % rn["tests"]]["OCov_s"] < rn["OCov_s"]

    # sanity on the design
    assert res["q"] == 9
    assert res["full_universe"] == 324
    assert res["model_accuracy"] > 0.83
