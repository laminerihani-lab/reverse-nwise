"""exp06 -- cost profile of the Reverse N-Wise pipeline (paper Table 3 analogue).

Measures, on Adult + XGBoost, the wall-clock time of each Algorithm 1 step, peak
memory, the number of SUT evaluations, and the search-budget ratio versus an
exhaustive input sweep. Produces genuine, machine-annotated cost numbers to
replace the paper's hardware-specific Table 3.

Usage:
    python -m experiments.exp06_cost --seed 0 --probes 8000 --json results/exp06.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np

from benchmarks.adult.adult_sut import (
    NUMERIC_FEATURES,
    AdultSUT,
    adult_partition_scheme,
    adult_search_space,
    load_adult,
    train_xgb,
)
from rnwise.covering_array import ExhaustiveFeasibleGenerator
from rnwise.feasibility import EmpiricalFeasibility
from rnwise.inverse_map import InverseMapper
from rnwise.metrics import output_coverage_from_rows, tuple_efficiency
from rnwise.oca import count_all_tuples
from rnwise.optimizers.jaya import Jaya
from rnwise.prioritize import prioritize


def run(seed: int = 0, n_probes: int = 8000, max_iter: int = 200) -> dict:
    tracemalloc.start()
    rng = np.random.default_rng(seed)
    s = 2

    t = time.time()
    data = load_adult(seed=seed)
    model = train_xgb(data, seed=seed)
    t_train = time.time() - t

    sut = AdultSUT(model=model, data=data)
    scheme = adult_partition_scheme()
    space = adult_search_space(data)
    idx = [data.feature_names.index(c) for c in NUMERIC_FEATURES]
    probes = data.X_train.to_numpy()[:n_probes][:, idx]

    # Step 1: feasibility
    t = time.time()
    feas = EmpiricalFeasibility(scheme, s).analyze(sut, probes)
    t_feas = time.time() - t

    # Step 2: covering array (one row per feasible output)
    t = time.time()
    oca = ExhaustiveFeasibleGenerator(scheme, s).generate(feas, rng)
    t_ca = time.time() - t

    # Step 3: inverse map (warm-started)
    t = time.time()
    mapper = InverseMapper(
        sut=sut,
        scheme=scheme,
        space=space,
        optimizer=Jaya(max_iter=max_iter),
        warm_start=feas.exemplars,
    )
    inv = mapper.run(oca, rng)
    t_inv = time.time() - t

    # Step 4: prioritize + truncate
    t = time.time()
    prio = prioritize(inv.test_cases, scheme.q, s, feas.feasible_tuples)
    curve = prio.coverage_curve
    mx = max(curve) if curve else 0.0
    plateau = (next((i for i, c in enumerate(curve) if c >= mx), len(curve) - 1) + 1)
    t_prio = time.time() - t

    realised = inv.realised_abstract_outputs()
    ocov = output_coverage_from_rows(realised, scheme.symbol_space, s, feas.feasible_tuples)
    eta = tuple_efficiency(realised, scheme.symbol_space, s)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Exhaustive-input baseline budget: product of per-feature discretisation.
    # A naive exhaustive sweep at 10 levels per numeric feature.
    exhaustive_budget = 10 ** len(NUMERIC_FEATURES)
    rnwise_budget = inv.total_evals + n_probes  # SUT calls: probes + inverse-map

    return {
        "seed": seed,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor() or platform.machine(),
        },
        "n_probes": n_probes,
        "max_iter": max_iter,
        "timings_s": {
            "train_model": round(t_train, 3),
            "step1_feasibility": round(t_feas, 3),
            "step2_covering_array": round(t_ca, 3),
            "step3_inverse_map": round(t_inv, 3),
            "step4_prioritize": round(t_prio, 3),
            "pipeline_total": round(t_feas + t_ca + t_inv + t_prio, 3),
        },
        "peak_memory_mb": round(peak / 1e6, 2),
        "sut_evaluations": {
            "feasibility_probes": n_probes,
            "inverse_map_evals": inv.total_evals,
            "rnwise_total": rnwise_budget,
        },
        "budget_ratio": {
            "exhaustive_input_sweep": exhaustive_budget,
            "rnwise_calls": rnwise_budget,
            "fraction_of_exhaustive": rnwise_budget / exhaustive_budget,
        },
        "suite": {
            "feasible_outputs": feas.n_feasible_outputs,
            "feasible_tuples": feas.n_feasible_tuples,
            "full_universe": count_all_tuples(scheme.symbol_space, s),
            "full_suite_tests": len(inv.test_cases),
            "prioritized_tests": plateau,
            "OCov_s": round(ocov, 4),
            "eta_s": round(eta, 2),
            "realisation_rate": round(inv.realisation_rate, 4),
        },
    }


def _print(res: dict) -> None:
    print(f"\n=== exp06: cost profile (Adult + XGBoost, seed {res['seed']}) ===")
    print(f"machine: {res['machine']['platform']} | py{res['machine']['python']}")
    print(f"\nAlgorithm 1 step timings (s):")
    for k, v in res["timings_s"].items():
        print(f"  {k:<24}{v:>8.3f}")
    print(f"\npeak memory: {res['peak_memory_mb']} MB")
    su = res["suite"]
    print(
        f"suite: full={su['full_suite_tests']} tests -> prioritized={su['prioritized_tests']} tests"
        f"  OCov_s={su['OCov_s']*100:.1f}%  eta_s={su['eta_s']}  realise={su['realisation_rate']}"
    )
    br = res["budget_ratio"]
    print(
        f"budget: {br['rnwise_calls']:,} SUT calls vs {br['exhaustive_input_sweep']:,} exhaustive"
        f"  ({br['fraction_of_exhaustive']*100:.4f}% of exhaustive)\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="exp06 cost profile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=8000)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    res = run(seed=args.seed, n_probes=args.probes, max_iter=args.max_iter)
    _print(res)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
