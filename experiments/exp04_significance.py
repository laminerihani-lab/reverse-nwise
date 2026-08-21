"""exp04 -- repeated-runs significance study (30 seeds) on Adult + XGBoost.

Runs exp01 across 30 fixed seeds, aggregates per-method OCov_s / eta_s / FDR with
means and 95% bootstrap CIs, and reports pairwise Reverse-N-Wise-vs-baseline
Mann-Whitney U tests with Cliff's delta effect sizes. This produces the genuine,
statistically-defensible replacement for the paper's single-shot Table 1.

Usage:
    python -m experiments.exp04_significance --runs 30 --probes 8000 --json results/exp04.json
    python -m experiments.exp04_significance --runs 5 --probes 4000   # quick check
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.stats import bootstrap_ci, compare
from experiments.exp01_table1_adult import run as run_exp01

METRICS = ["OCov_s", "eta_s", "FDR_pct"]
RNWISE = "Reverse N-Wise"


def _canonical_method(name: str) -> str:
    """Collapse budget-suffixed names (Random-20 / Random-21) to a stable key."""
    if name.startswith("Random-"):
        return "Random"
    return name


def run(n_runs: int = 30, n_probes: int = 8000, max_iter: int = 200) -> dict:
    t0 = time.time()
    # metric -> method -> list of per-seed values
    collected: dict[str, dict[str, list[float]]] = {
        m: defaultdict(list) for m in METRICS
    }
    per_seed: list[dict] = []
    tests_by_method: dict[str, list[int]] = defaultdict(list)

    for seed in range(n_runs):
        res = run_exp01(seed=seed, n_probes=n_probes, max_iter=max_iter)
        per_seed.append(
            {"seed": seed, "model_accuracy": res["model_accuracy"], "results": res["results"]}
        )
        for r in res["results"]:
            method = _canonical_method(r["method"])
            tests_by_method[method].append(r["tests"])
            for m in METRICS:
                collected[m][method].append(float(r[m]))

    methods = list(collected["OCov_s"].keys())

    # Per-method aggregate with bootstrap CIs.
    aggregate: dict[str, dict] = {}
    for method in methods:
        agg = {"tests_mean": round(float(np.mean(tests_by_method[method])), 1)}
        for m in METRICS:
            ci = bootstrap_ci(collected[m][method], seed=0)
            agg[m] = {
                "mean": round(ci.mean, 4),
                "ci95_lo": round(ci.lo, 4),
                "ci95_hi": round(ci.hi, 4),
            }
        aggregate[method] = agg

    # Pairwise RNWise vs each baseline, per metric.
    comparisons: list[dict] = []
    for m in METRICS:
        a = collected[m][RNWISE]
        for method in methods:
            if method == RNWISE:
                continue
            cmp = compare(m, RNWISE, a, method, collected[m][method])
            comparisons.append(cmp.row())

    return {
        "n_runs": n_runs,
        "n_probes": n_probes,
        "max_iter": max_iter,
        "metrics": METRICS,
        "methods": methods,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "total_time_s": round(time.time() - t0, 1),
        "per_seed": per_seed,
    }


def _print_report(res: dict) -> None:
    print(f"\n=== exp04: {res['n_runs']}-run significance (Adult + XGBoost) ===")
    print(f"probes={res['n_probes']} max_iter={res['max_iter']} total={res['total_time_s']}s\n")

    print(f"{'Method':<16}{'Tests':>7}{'OCov_s (95% CI)':>26}{'FDR (95% CI)':>24}")
    print("-" * 73)
    for method, agg in res["aggregate"].items():
        o = agg["OCov_s"]
        f = agg["FDR_pct"]
        ocov = f"{o['mean']*100:5.1f}% [{o['ci95_lo']*100:.1f},{o['ci95_hi']*100:.1f}]"
        fdr = f"{f['mean']*100:5.1f}% [{f['ci95_lo']*100:.1f},{f['ci95_hi']*100:.1f}]"
        print(f"{method:<16}{agg['tests_mean']:>7}{ocov:>26}{fdr:>24}")

    print(f"\nPairwise Mann-Whitney U (RNWise > baseline) + Cliff's delta:")
    print(f"{'Metric':<9}{'vs Baseline':<16}{'p-value':>12}{'delta':>9}{'effect':>12}{'sig':>5}")
    print("-" * 63)
    for c in res["comparisons"]:
        star = "***" if c["significant"] else ""
        print(
            f"{c['metric']:<9}{c['b']:<16}{c['p_value']:>12.2e}"
            f"{c['cliffs_delta']:>9.2f}{c['effect']:>12}{star:>5}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="exp04 significance study")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--probes", type=int, default=8000)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    res = run(n_runs=args.runs, n_probes=args.probes, max_iter=args.max_iter)
    _print_report(res)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
