"""exp01 -- Table 1 reproduction on UCI Adult + XGBoost (pairwise, s=2).

Runs the full Reverse N-Wise pipeline (Algorithm 1) and all five baselines on the
same trained XGBoost model, scoring each by the OUTPUT coverage it achieves on a
shared feasible pairwise universe, plus fault-detection rate over 8 seeded
behavioural faults.

Usage:
    python -m experiments.exp01_table1_adult --seed 0
    python -m experiments.exp01_table1_adult --seed 0 --probes 8000 --json results/exp01.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analysis.harness import SuiteScore, score_suite
from baselines.deepct import DeepCT
from baselines.input_ct import InputCT
from baselines.metamorphic import Metamorphic
from baselines.property_based import PropertyBased
from baselines.random_test import RandomTest
from benchmarks.adult.adult_sut import (
    NUMERIC_FEATURES,
    AdultSUT,
    adult_partition_scheme,
    adult_search_space,
    load_adult,
    train_xgb,
)
from benchmarks.adult.faults import SEEDED_FAULTS
from rnwise.covering_array import ExhaustiveFeasibleGenerator, GreedyGenerator
from rnwise.feasibility import EmpiricalFeasibility
from rnwise.optimizers.jaya import Jaya
from rnwise.pipeline import ReverseNWisePipeline


def run(seed: int = 0, n_probes: int = 8000, max_iter: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()

    data = load_adult(seed=seed)
    model = train_xgb(data, seed=seed)
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(data.y_test, model.predict(data.X_test.to_numpy()))

    sut = AdultSUT(model=model, data=data)
    scheme = adult_partition_scheme()
    space = adult_search_space(data)
    s = 2

    # Shared feasible universe (empirical) used to score EVERY method identically.
    idx = [data.feature_names.index(c) for c in NUMERIC_FEATURES]
    probes = data.X_train.to_numpy()[:n_probes][:, idx]
    feas = EmpiricalFeasibility(scheme, strength=s).analyze(sut, probes)
    feasible = feas.feasible_tuples

    scores: list[SuiteScore] = []

    # --- Reverse N-Wise (our method) --------------------------------------
    t_rn = time.time()
    pipe = ReverseNWisePipeline(
        sut=sut,
        scheme=scheme,
        space=space,
        strength=s,
        optimizer=Jaya(max_iter=max_iter),
        generator=ExhaustiveFeasibleGenerator(scheme, s),
    )
    rn = pipe.run(probe_inputs=probes, seed=seed)
    # Prioritized suite: greedy OCov_s ordering. Truncate at the coverage plateau
    # (the minimal prefix that reaches maximal feasible coverage) -- extra tests
    # beyond the plateau add no new pairwise output interactions.
    curve = rn.prioritization.coverage_curve
    max_cov = max(curve) if curve else 0.0
    plateau = next((i for i, c in enumerate(curve) if c >= max_cov), len(curve) - 1) + 1
    rn_inputs = rn.prioritization.ordered[:plateau]
    rn_realised_inputs = [tc.decoded for tc in rn_inputs]
    rn_time = time.time() - t_rn
    scores.append(
        score_suite(
            "Reverse N-Wise",
            sut,
            scheme,
            rn_realised_inputs,
            s,
            feasible,
            SEEDED_FAULTS,
        )
    )
    budget = len(rn_realised_inputs)  # match baselines to RNWise test count

    # --- Input CT (classical pairwise over inputs) ------------------------
    levels = []
    for c in NUMERIC_FEATURES:
        lo, hi = space.lower[NUMERIC_FEATURES.index(c)], space.upper[NUMERIC_FEATURES.index(c)]
        levels.append(np.array([lo, (lo + hi) / 2, hi]))
    ct = InputCT(feature_levels=levels, strength=2).generate(np.random.default_rng(seed))
    scores.append(
        score_suite("Input CT", sut, scheme, ct.inputs, s, feasible, SEEDED_FAULTS)
    )

    # --- Random (matched budget) ------------------------------------------
    rand = RandomTest(space=space, n_tests=budget).generate(np.random.default_rng(seed + 1))
    scores.append(
        score_suite(f"Random-{budget}", sut, scheme, rand.inputs, s, feasible, SEEDED_FAULTS)
    )

    # --- Property-based ----------------------------------------------------
    pb = PropertyBased(space=space, n_tests=budget).generate(np.random.default_rng(seed + 2))
    scores.append(
        score_suite("Property-Based", sut, scheme, pb.inputs, s, feasible, SEEDED_FAULTS)
    )

    # --- Metamorphic -------------------------------------------------------
    n_sources = max(1, budget // 4)
    mm = Metamorphic(space=space, n_sources=n_sources).generate(np.random.default_rng(seed + 3))
    scores.append(
        score_suite("Metamorphic", sut, scheme, mm.inputs, s, feasible, SEEDED_FAULTS)
    )

    # --- DeepCT ------------------------------------------------------------
    dct = DeepCT(sut=sut, scheme=scheme, space=space, n_tests=budget).generate(
        np.random.default_rng(seed + 4)
    )
    scores.append(
        score_suite("DeepCT", sut, scheme, dct.inputs, s, feasible, SEEDED_FAULTS)
    )

    return {
        "seed": seed,
        "model_accuracy": round(float(acc), 4),
        "strength": s,
        "q": scheme.q,
        "levels": list(scheme.levels),
        "feasible_outputs": feas.n_feasible_outputs,
        "feasible_tuples": feas.n_feasible_tuples,
        "full_universe": __import__("rnwise.oca", fromlist=["count_all_tuples"]).count_all_tuples(
            scheme.symbol_space, s
        ),
        "rnwise_time_s": round(rn_time, 2),
        "total_time_s": round(time.time() - t0, 2),
        "rnwise_realisation_rate": round(rn.realisation_rate, 4),
        "results": [sc.row() for sc in scores],
    }


def _print_table(res: dict) -> None:
    print(f"\n=== exp01: Table 1 reproduction (Adult + XGBoost, s={res['strength']}) ===")
    print(
        f"model acc={res['model_accuracy']}  q={res['q']}  levels={res['levels']}  "
        f"feasible_tuples={res['feasible_tuples']}/{res['full_universe']}  "
        f"feasible_outputs={res['feasible_outputs']}"
    )
    print(
        f"RNWise realisation_rate={res['rnwise_realisation_rate']}  "
        f"rnwise_time={res['rnwise_time_s']}s  total={res['total_time_s']}s\n"
    )
    hdr = f"{'Method':<16}{'Tests':>7}{'OCov_s':>9}{'OCov_full':>11}{'eta_s':>8}{'FDR':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in res["results"]:
        print(
            f"{r['method']:<16}{r['tests']:>7}{r['OCov_s']*100:>8.1f}%"
            f"{r['OCov_s_full']*100:>10.1f}%{r['eta_s']:>8.1f}{r['FDR']:>8}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="exp01 Table 1 reproduction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=8000)
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    res = run(seed=args.seed, n_probes=args.probes, max_iter=args.max_iter)
    _print_table(res)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
