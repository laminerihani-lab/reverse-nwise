"""exp02 -- multi-domain generality: Reverse N-Wise on a quantum NISQ SUT.

Demonstrates the method is domain-agnostic by running the SAME Algorithm 1
pipeline on a parameterised noisy quantum circuit (statevector simulator), using
the VQE inverse-map optimizer. Compares RNWise against random parameter testing
on quantum output coverage and fault detection, mirroring exp01's structure in a
completely different domain.

Seeded quantum faults are rare-but-reachable pairwise output-behaviour signatures
(e.g. high-entropy state paired with high fidelity -- a noise-model inconsistency).

Usage:
    python -m experiments.exp02_multidomain --seed 0 --json results/exp02.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from analysis.harness import score_suite
from benchmarks.quantum.nisq_sut import (
    NISQCircuitSUT,
    quantum_partition_scheme,
    quantum_probe_thetas,
    quantum_search_space,
)
from rnwise.covering_array import ExhaustiveFeasibleGenerator
from rnwise.feasibility import EmpiricalFeasibility
from rnwise.inverse_map import InverseMapper
from rnwise.oca import OutputTuple, count_all_tuples, project
from rnwise.optimizers.vqe import VQE
from rnwise.prioritize import prioritize


def _seed_quantum_faults(feas, n: int = 5) -> list[OutputTuple]:
    """Pick the rarest reachable pairwise tuples as seeded quantum faults."""
    q = 4
    freq: Counter = Counter()
    for ao in feas.abstract_outputs:
        for f in combinations(range(q), 2):
            freq[project(ao, f)] += 1
    rare = sorted(feas.feasible_tuples, key=lambda t: freq[t])
    return list(rare[:n])


def run(seed: int = 0, n_probes: int = 2000, noise: float = 0.08, max_iter: int = 150) -> dict:
    t0 = time.time()
    sut = NISQCircuitSUT(noise=noise)
    scheme = quantum_partition_scheme()
    space = quantum_search_space(4)
    probes = quantum_probe_thetas(n_probes, seed=seed)
    s = 2

    feas = EmpiricalFeasibility(scheme, s).analyze(sut, probes)
    faults = _seed_quantum_faults(feas, n=5)

    # -- Reverse N-Wise with VQE inverse map --------------------------------
    t = time.time()
    oca = ExhaustiveFeasibleGenerator(scheme, s).generate(feas, np.random.default_rng(seed))
    mapper = InverseMapper(
        sut=sut,
        scheme=scheme,
        space=space,
        optimizer=VQE(max_iter=max_iter, n_restarts=3),
        warm_start=feas.exemplars,
    )
    inv = mapper.run(oca, np.random.default_rng(seed))
    prio = prioritize(inv.test_cases, scheme.q, s, feas.feasible_tuples)
    curve = prio.coverage_curve
    mx = max(curve) if curve else 0.0
    plateau = (next((i for i, c in enumerate(curve) if c >= mx), len(curve) - 1) + 1)
    rn_inputs = [tc.decoded for tc in prio.ordered[:plateau]]
    rn_time = time.time() - t

    rn_score = score_suite("Reverse N-Wise (VQE)", sut, scheme, rn_inputs, s, feas.feasible_tuples, faults)

    # -- Random quantum testing (matched budget) ----------------------------
    rng = np.random.default_rng(seed + 100)
    rand_inputs = [x for x in space.sample(rng, len(rn_inputs))]
    rand_score = score_suite("Random-Q", sut, scheme, rand_inputs, s, feas.feasible_tuples, faults)

    # -- Larger random pool (grid probes) as a strong baseline --------------
    pool_score = score_suite("Random-Pool", sut, scheme, probes[: len(rn_inputs) * 10], s, feas.feasible_tuples, faults)

    return {
        "domain": "quantum-nisq",
        "seed": seed,
        "noise": noise,
        "q": scheme.q,
        "full_universe": count_all_tuples(scheme.symbol_space, s),
        "feasible_outputs": feas.n_feasible_outputs,
        "feasible_tuples": feas.n_feasible_tuples,
        "n_faults": len(faults),
        "rnwise_time_s": round(rn_time, 2),
        "total_time_s": round(time.time() - t0, 2),
        "realisation_rate": round(inv.realisation_rate, 4),
        "results": [rn_score.row(), rand_score.row(), pool_score.row()],
    }


def _print(res: dict) -> None:
    print(f"\n=== exp02: multi-domain (QUANTUM NISQ, seed {res['seed']}, noise={res['noise']}) ===")
    print(
        f"q={res['q']} feasible_tuples={res['feasible_tuples']}/{res['full_universe']} "
        f"feasible_outputs={res['feasible_outputs']} faults={res['n_faults']} "
        f"realise={res['realisation_rate']} rnwise_time={res['rnwise_time_s']}s\n"
    )
    print(f"{'Method':<22}{'Tests':>7}{'OCov_s':>9}{'eta_s':>8}{'FDR':>8}")
    print("-" * 54)
    for r in res["results"]:
        print(f"{r['method']:<22}{r['tests']:>7}{r['OCov_s']*100:>8.1f}%{r['eta_s']:>8.1f}{r['FDR']:>8}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="exp02 multi-domain (quantum)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--max-iter", type=int, default=150)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    res = run(seed=args.seed, n_probes=args.probes, noise=args.noise, max_iter=args.max_iter)
    _print(res)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
