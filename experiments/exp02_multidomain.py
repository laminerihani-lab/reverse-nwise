"""exp02 -- multi-domain generality of Reverse N-Wise.

Runs the SAME Algorithm 1 pipeline across three structurally different domains,
each with its own SUT, output-behaviour partition, search space, and inverse-map
optimizer, then compares RNWise against a matched-budget random baseline on output
coverage and fault detection:

  * quantum : parameterised NISQ circuit (statevector sim), VQE inverse map
  * vision  : MLP on sklearn digits over a PCA latent space, Jaya inverse map
  * nlp     : logistic-regression text classifier over a TF-IDF+SVD embedding
              space (20 newsgroups), Jaya inverse map

Seeded faults per domain are the rarest reachable pairwise output-behaviour
signatures. This is the empirical backbone of the "domain-agnostic" claim.

Usage:
    python -m experiments.exp02_multidomain --domains quantum vision nlp --json results/exp02.json
    python -m experiments.exp02_multidomain --domains vision --seed 0
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
from rnwise.covering_array import ExhaustiveFeasibleGenerator
from rnwise.feasibility import EmpiricalFeasibility
from rnwise.inverse_map import InverseMapper
from rnwise.oca import OutputTuple, count_all_tuples, project
from rnwise.optimizers.jaya import Jaya
from rnwise.optimizers.vqe import VQE
from rnwise.prioritize import prioritize


def _rarest_faults(feas, q: int, n: int) -> list[OutputTuple]:
    """The n rarest reachable pairwise tuples (subtle, hard-to-hit behaviours)."""
    freq: Counter = Counter()
    for ao in feas.abstract_outputs:
        for f in combinations(range(q), 2):
            freq[project(ao, f)] += 1
    return sorted(feas.feasible_tuples, key=lambda t: freq[t])[:n]


def _build_domain(name: str, seed: int, n_probes: int, max_iter: int):
    """Return (sut, scheme, space, probes, optimizer) for a named domain."""
    if name == "quantum":
        from benchmarks.quantum.nisq_sut import (
            NISQCircuitSUT,
            quantum_partition_scheme,
            quantum_probe_thetas,
            quantum_search_space,
        )

        sut = NISQCircuitSUT(noise=0.08)
        scheme = quantum_partition_scheme()
        space = quantum_search_space(4)
        probes = quantum_probe_thetas(n_probes, seed=seed)
        opt = VQE(max_iter=max_iter, n_restarts=3)
        return sut, scheme, space, probes, opt

    if name == "vision":
        from benchmarks.vision.vision_sut import (
            VisionSUT,
            load_vision,
            train_mlp,
            vision_partition_scheme,
            vision_probe_latents,
            vision_search_space,
        )

        data = load_vision(seed=seed)
        model = train_mlp(data, seed=seed)
        sut = VisionSUT(model=model, data=data)
        scheme = vision_partition_scheme()
        space = vision_search_space(data)
        probes = vision_probe_latents(data, n_probes)
        opt = Jaya(max_iter=max_iter)
        return sut, scheme, space, probes, opt

    if name == "nlp":
        from benchmarks.nlp.nlp_sut import (
            NLPSUT,
            load_nlp,
            nlp_partition_scheme,
            nlp_probe_embeddings,
            nlp_search_space,
            train_text_clf,
        )

        data = load_nlp(seed=seed)
        model = train_text_clf(data, seed=seed)
        sut = NLPSUT(model=model, data=data)
        scheme = nlp_partition_scheme()
        space = nlp_search_space(data)
        probes = nlp_probe_embeddings(data, n_probes)
        opt = Jaya(max_iter=max_iter)
        return sut, scheme, space, probes, opt

    raise ValueError(f"unknown domain {name!r}")


def run_domain(name: str, seed: int, n_probes: int, max_iter: int, n_faults: int = 5) -> dict:
    t0 = time.time()
    sut, scheme, space, probes, opt = _build_domain(name, seed, n_probes, max_iter)
    s = 2
    q = scheme.q

    feas = EmpiricalFeasibility(scheme, s).analyze(sut, probes)
    faults = _rarest_faults(feas, q, n_faults)

    t = time.time()
    oca = ExhaustiveFeasibleGenerator(scheme, s).generate(feas, np.random.default_rng(seed))
    mapper = InverseMapper(sut=sut, scheme=scheme, space=space, optimizer=opt, warm_start=feas.exemplars)
    inv = mapper.run(oca, np.random.default_rng(seed))
    prio = prioritize(inv.test_cases, q, s, feas.feasible_tuples)
    curve = prio.coverage_curve
    mx = max(curve) if curve else 0.0
    plateau = (next((i for i, c in enumerate(curve) if c >= mx), len(curve) - 1) + 1)
    rn_inputs = [tc.decoded for tc in prio.ordered[:plateau]]
    rn_time = time.time() - t

    rn = score_suite(f"Reverse N-Wise [{name}]", sut, scheme, rn_inputs, s, feas.feasible_tuples, faults)
    rng = np.random.default_rng(seed + 100)
    rand_inputs = [space.to_input(x) for x in space.sample(rng, len(rn_inputs))]
    rand = score_suite("Random (matched)", sut, scheme, rand_inputs, s, feas.feasible_tuples, faults)

    return {
        "domain": name,
        "seed": seed,
        "optimizer": type(opt).__name__,
        "q": q,
        "full_universe": count_all_tuples(scheme.symbol_space, s),
        "feasible_outputs": feas.n_feasible_outputs,
        "feasible_tuples": feas.n_feasible_tuples,
        "n_faults": len(faults),
        "realisation_rate": round(inv.realisation_rate, 4),
        "rnwise_time_s": round(rn_time, 2),
        "total_time_s": round(time.time() - t0, 2),
        "results": [rn.row(), rand.row()],
    }


def run(domains, seed: int = 0, n_probes: int = 2000, max_iter: int = 120) -> dict:
    out = {"seed": seed, "domains": {}}
    for d in domains:
        out["domains"][d] = run_domain(d, seed, n_probes, max_iter)
    return out


def _print(res: dict) -> None:
    print(f"\n=== exp02: multi-domain generality (seed {res['seed']}) ===\n")
    hdr = f"{'Domain':<9}{'Opt':<7}{'q':>3}{'feas/uni':>10}{'Method':<22}{'Tests':>6}{'OCov_s':>9}{'FDR':>7}"
    print(hdr)
    print("-" * len(hdr))
    for d, r in res["domains"].items():
        fu = f"{r['feasible_tuples']}/{r['full_universe']}"
        for i, row in enumerate(r["results"]):
            dom = d if i == 0 else ""
            opt = r["optimizer"] if i == 0 else ""
            qv = str(r["q"]) if i == 0 else ""
            fuv = fu if i == 0 else ""
            print(
                f"{dom:<9}{opt:<7}{qv:>3}{fuv:>10}{row['method']:<22}"
                f"{row['tests']:>6}{row['OCov_s']*100:>8.1f}%{row['FDR']:>7}"
            )
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="exp02 multi-domain generality")
    ap.add_argument("--domains", nargs="+", default=["quantum", "vision", "nlp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=2000)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--json", type=str, default="")
    args = ap.parse_args()

    res = run(args.domains, seed=args.seed, n_probes=args.probes, max_iter=args.max_iter)
    _print(res)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
