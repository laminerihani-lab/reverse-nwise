"""exp03 -- coverage strength study (s = 2, 3) on Adult + XGBoost.

Higher strength demands covering every s-way OUTPUT interaction, which grows the
feasible universe (and the lower bound M >= v^s, Corollary 1). This experiment
measures, per strength, the feasible universe size, the RNWise prioritized suite
size, achieved OCov_s, and verifies the Corollary 1 bound holds. It answers the
reviewer question "does the method scale beyond pairwise?".

Usage:
    python -m experiments.exp03_strength --seed 0 --strengths 2 3 --json results/exp03.json
"""

from __future__ import annotations

import argparse
import json
import time
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


def run(seed: int = 0, n_probes: int = 8000, strengths=(2, 3), max_iter: int = 200) -> dict:
    rng_seed = seed
    data = load_adult(seed=seed)
    model = train_xgb(data, seed=seed)
    sut = AdultSUT(model=model, data=data)
    scheme = adult_partition_scheme()
    space = adult_search_space(data)
    idx = [data.feature_names.index(c) for c in NUMERIC_FEATURES]
    probes = data.X_train.to_numpy()[:n_probes][:, idx]

    rows = []
    for s in strengths:
        t = time.time()
        feas = EmpiricalFeasibility(scheme, s).analyze(sut, probes)
        oca = ExhaustiveFeasibleGenerator(scheme, s).generate(
            feas, np.random.default_rng(rng_seed)
        )
        mapper = InverseMapper(
            sut=sut,
            scheme=scheme,
            space=space,
            optimizer=Jaya(max_iter=max_iter),
            warm_start=feas.exemplars,
        )
        inv = mapper.run(oca, np.random.default_rng(rng_seed))
        prio = prioritize(inv.test_cases, scheme.q, s, feas.feasible_tuples)
        curve = prio.coverage_curve
        mx = max(curve) if curve else 0.0
        plateau = (next((i for i, c in enumerate(curve) if c >= mx), len(curve) - 1) + 1)
        realised = inv.realised_abstract_outputs()
        ocov = output_coverage_from_rows(realised, scheme.symbol_space, s, feas.feasible_tuples)
        eta = tuple_efficiency(realised, scheme.symbol_space, s)
        v = max(scheme.levels)
        rows.append(
            {
                "s": s,
                "full_universe": count_all_tuples(scheme.symbol_space, s),
                "feasible_tuples": feas.n_feasible_tuples,
                "feasible_outputs": feas.n_feasible_outputs,
                "prioritized_tests": plateau,
                "full_suite_tests": len(inv.test_cases),
                "OCov_s": round(ocov, 4),
                "eta_s": round(eta, 2),
                "corollary1_lower_bound": v**s,
                "bound_satisfied": len(inv.test_cases) >= v**s,
                "time_s": round(time.time() - t, 2),
            }
        )

    return {"seed": seed, "n_probes": n_probes, "q": scheme.q, "v": max(scheme.levels), "by_strength": rows}


def _print(res: dict) -> None:
    print(f"\n=== exp03: strength study (Adult + XGBoost, seed {res['seed']}, q={res['q']}, v={res['v']}) ===\n")
    hdr = f"{'s':>3}{'full_uni':>10}{'feasible':>10}{'tests':>8}{'OCov_s':>9}{'eta_s':>8}{'v^s(LB)':>9}{'LB_ok':>7}{'time':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in res["by_strength"]:
        print(
            f"{r['s']:>3}{r['full_universe']:>10}{r['feasible_tuples']:>10}"
            f"{r['prioritized_tests']:>8}{r['OCov_s']*100:>8.1f}%{r['eta_s']:>8.1f}"
            f"{r['corollary1_lower_bound']:>9}{str(r['bound_satisfied']):>7}{r['time_s']:>8}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="exp03 strength study")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=8000)
    ap.add_argument("--strengths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    res = run(seed=args.seed, n_probes=args.probes, strengths=tuple(args.strengths), max_iter=args.max_iter)
    _print(res)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
