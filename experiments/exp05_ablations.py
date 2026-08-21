"""exp05 -- ablation studies on Adult + XGBoost.

Three ablations answering design-justification reviewer questions:

  A. Optimizer choice (Step 3): Jaya vs Whale vs Firefly vs Bayesian Optimization.
     With warm-start the metaheuristic rarely fires (exemplars solve most rows),
     so we ablate WITHOUT warm-start to expose the optimizer's raw inverse-map
     power: realisation rate, achieved OCov_s, and SUT-evaluation cost.

  B. Partition granularity: 2-level vs 3-level (default) vs 4-level output factors.
     Coarser partitions shrink the universe (cheaper, less discriminating);
     finer partitions grow it (more faults expressible, harder to cover).

  C. Regularisation lambda sweep: effect of the R(x) penalty weight on realisation
     (feasibility of found inputs) vs coverage. Uses a box-distance regulariser.

Usage:
    python -m experiments.exp05_ablations --seed 0 --probes 6000 --json results/exp05.json
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
    adult_search_space,
    load_adult,
    train_xgb,
)
from rnwise.covering_array import ExhaustiveFeasibleGenerator
from rnwise.feasibility import EmpiricalFeasibility
from rnwise.inverse_map import InverseMapper
from rnwise.metrics import output_coverage_from_rows
from rnwise.oca import count_all_tuples
from rnwise.optimizers.firefly import Firefly
from rnwise.optimizers.jaya import Jaya
from rnwise.optimizers.whale import Whale
from rnwise.partition import CategoricalPassthrough, PartitionScheme, ThresholdBands


# ---- partition schemes at 3 granularities ---------------------------------


def scheme_granularity(levels: int) -> PartitionScheme:
    """Vary the granularity of the two CONTINUOUS output factors (confidence and
    margin) across 2 / 3 / 4 bins, holding the 7 categorical factors at their raw
    ternary alphabets.

    Only the threshold-band factors can be freely re-binned; the categorical
    factors are emitted verbatim by the SUT and cannot be relabelled. This
    isolates the effect of output-abstraction granularity on the universe size and
    achievable coverage.
    """
    cat = (
        CategoricalPassthrough(alphabet=("neg", "boundary", "pos")),
        CategoricalPassthrough(alphabet=("consistent", "weak", "inconsistent")),
        CategoricalPassthrough(alphabet=("young", "mid", "senior")),
        CategoricalPassthrough(alphabet=("private", "gov", "other")),
        CategoricalPassthrough(alphabet=("drop", "stable", "rise")),
        CategoricalPassthrough(alphabet=("drop", "stable", "rise")),
        CategoricalPassthrough(alphabet=("none", "moderate", "high")),
    )
    if levels == 2:
        conf = ThresholdBands(edges=(0.75,), labels=("low", "high"))
        marg = ThresholdBands(edges=(0.0,), labels=("neg_m", "pos_m"))
    elif levels == 4:
        conf = ThresholdBands(edges=(0.55, 0.7, 0.9), labels=("l", "ml", "mh", "h"))
        marg = ThresholdBands(edges=(-0.2, 0.0, 0.2), labels=("fn", "n", "p", "fp"))
    else:  # 3 (default)
        conf = ThresholdBands(edges=(0.6, 0.85), labels=("low", "med", "high"))
        marg = ThresholdBands(edges=(-0.15, 0.15), labels=("far_neg", "near", "far_pos"))
    # factor order matches AdultSUT.predict: decision, confidence, margin, ...
    return PartitionScheme((cat[0], conf, marg, cat[1], cat[2], cat[3], cat[4], cat[5], cat[6]))


def box_regularizer(space):
    """R(x) = normalised squared distance from the box centre (keeps inputs
    realistic / near the data centroid). Zero at centre, ~1 at corners."""
    centre = (space.lower + space.upper) / 2.0
    half = (space.upper - space.lower) / 2.0
    half = np.where(half == 0, 1.0, half)

    def R(x: np.ndarray) -> float:
        z = (np.asarray(x, dtype=float) - centre) / half
        return float(np.mean(z**2))

    return R


def _rnwise_once(sut, scheme, space, probes, s, optimizer, seed, warm, lam=0.0, reg=None, target_cap=0):
    feas = EmpiricalFeasibility(scheme, s).analyze(sut, probes)
    oca = ExhaustiveFeasibleGenerator(scheme, s).generate(feas, np.random.default_rng(seed))
    # Optional target cap: invert only a representative random subset of rows.
    # Used for the (expensive) no-warm-start optimizer ablation so it stays
    # tractable while still fairly comparing raw inverse-map power.
    if target_cap and len(oca.rows) > target_cap:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(oca.rows), size=target_cap, replace=False)
        oca.rows = [oca.rows[i] for i in keep]
    mapper = InverseMapper(
        sut=sut,
        scheme=scheme,
        space=space,
        optimizer=optimizer,
        lam=lam,
        regularizer=reg,
        warm_start=feas.exemplars if warm else None,
    )
    inv = mapper.run(oca, np.random.default_rng(seed))
    realised = inv.realised_abstract_outputs()
    ocov = output_coverage_from_rows(realised, scheme.symbol_space, s, feas.feasible_tuples)
    return {
        "realisation_rate": round(inv.realisation_rate, 4),
        "OCov_s": round(ocov, 4),
        "sut_evals": inv.total_evals,
        "feasible_tuples": feas.n_feasible_tuples,
        "targets_inverted": len(oca.rows),
    }


def run(seed: int = 0, n_probes: int = 6000, max_iter: int = 120) -> dict:
    from benchmarks.adult.adult_sut import adult_partition_scheme

    data = load_adult(seed=seed)
    model = train_xgb(data, seed=seed)
    sut = AdultSUT(model=model, data=data)
    space = adult_search_space(data)
    idx = [data.feature_names.index(c) for c in NUMERIC_FEATURES]
    probes = data.X_train.to_numpy()[:n_probes][:, idx]
    scheme = adult_partition_scheme()
    s = 2

    # -- A. optimizer ablation (NO warm-start, exposes raw inverse-map power) --
    optimizers = {
        "Jaya": Jaya(pop_size=25, max_iter=max_iter),
        "Whale": Whale(pop_size=25, max_iter=max_iter),
        "Firefly": Firefly(pop_size=20, max_iter=max_iter),
    }
    try:
        from rnwise.optimizers.bayesopt import BayesOpt

        import skopt  # noqa: F401

        optimizers["BayesOpt"] = BayesOpt(n_calls=max_iter, n_initial_points=15)
    except Exception:
        pass  # scikit-optimize not installed; skip BO

    opt_rows = []
    for name, opt in optimizers.items():
        t = time.time()
        r = _rnwise_once(sut, scheme, space, probes, s, opt, seed, warm=False, target_cap=40)
        r.update({"optimizer": name, "time_s": round(time.time() - t, 2)})
        opt_rows.append(r)

    # -- B. partition-granularity ablation (warm-start on) --------------------
    gran_rows = []
    for lv in (2, 3, 4):
        sc = scheme_granularity(lv)
        t = time.time()
        r = _rnwise_once(sut, sc, space, probes, s, Jaya(max_iter=max_iter), seed, warm=True)
        r.update(
            {
                "levels": lv,
                "full_universe": count_all_tuples(sc.symbol_space, s),
                "time_s": round(time.time() - t, 2),
            }
        )
        gran_rows.append(r)

    # -- C. lambda sweep (regularised inverse loss, NO warm-start) ------------
    reg = box_regularizer(space)
    lam_rows = []
    for lam in (0.0, 0.1, 0.5, 1.0):
        t = time.time()
        r = _rnwise_once(
            sut, scheme, space, probes, s, Jaya(max_iter=max_iter), seed, warm=False, lam=lam, reg=reg, target_cap=40
        )
        r.update({"lambda": lam, "time_s": round(time.time() - t, 2)})
        lam_rows.append(r)

    return {
        "seed": seed,
        "n_probes": n_probes,
        "max_iter": max_iter,
        "optimizer_ablation": opt_rows,
        "granularity_ablation": gran_rows,
        "lambda_ablation": lam_rows,
    }


def _print(res: dict) -> None:
    print(f"\n=== exp05: ablations (Adult + XGBoost, seed {res['seed']}) ===")

    print("\n[A] Optimizer (Step 3), NO warm-start, 40 sampled targets -- raw inverse-map power:")
    print(f"{'optimizer':<12}{'realise':>9}{'OCov_s':>9}{'sut_evals':>11}{'time':>8}")
    print("-" * 49)
    for r in res["optimizer_ablation"]:
        print(f"{r['optimizer']:<12}{r['realisation_rate']*100:>8.1f}%{r['OCov_s']*100:>8.1f}%{r['sut_evals']:>11}{r['time_s']:>8}")

    print("\n[B] Output-abstraction granularity (confidence+margin bins; warm-start on):")
    print(f"{'bins':<8}{'universe':>10}{'feasible':>10}{'OCov_s':>9}{'time':>8}")
    print("-" * 45)
    for r in res["granularity_ablation"]:
        print(f"{r['levels']:<8}{r['full_universe']:>10}{r['feasible_tuples']:>10}{r['OCov_s']*100:>8.1f}%{r['time_s']:>8}")

    print("\n[C] Regularisation lambda, NO warm-start, 40 sampled targets:")
    print(f"{'lambda':<8}{'realise':>9}{'OCov_s':>9}{'sut_evals':>11}{'time':>8}")
    print("-" * 45)
    for r in res["lambda_ablation"]:
        print(f"{r['lambda']:<8}{r['realisation_rate']*100:>8.1f}%{r['OCov_s']*100:>8.1f}%{r['sut_evals']:>11}{r['time_s']:>8}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="exp05 ablations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probes", type=int, default=6000)
    ap.add_argument("--max-iter", type=int, default=120)
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
