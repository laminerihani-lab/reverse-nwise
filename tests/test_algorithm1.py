"""Tests for feasibility (Step 1), covering-array generation (Step 2),
prioritization (Step 4), and the full pipeline on a synthetic SUT."""

from __future__ import annotations

import numpy as np

from rnwise.covering_array import GreedyGenerator
from rnwise.feasibility import (
    ConstraintFeasibility,
    EmpiricalFeasibility,
    all_feasible,
)
from rnwise.inverse_map import TestCase
from rnwise.metrics import output_coverage
from rnwise.model import CallableSUT
from rnwise.optimizers.base import SearchSpace
from rnwise.optimizers.jaya import Jaya
from rnwise.partition import CategoricalPassthrough, PartitionScheme, ThresholdBands
from rnwise.pipeline import ReverseNWisePipeline
from rnwise.prioritize import prioritize


def make_sut_scheme_space():
    def fn(x):
        x = np.asarray(x, dtype=float)
        cls = "pos" if x[0] >= 0 else "neg"
        return (cls, float(x[1]))

    sut = CallableSUT(fn=fn, arity=2)
    scheme = PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("neg", "pos")),
            ThresholdBands(edges=(0.5,), labels=("lo", "hi")),
        )
    )
    space = SearchSpace(lower=np.array([-1.0, 0.0]), upper=np.array([1.0, 1.0]))
    return sut, scheme, space


def test_empirical_feasibility_observes_all_four_outputs():
    sut, scheme, _ = make_sut_scheme_space()
    probes = [
        np.array([-1.0, 0.0]),  # neg, lo
        np.array([-1.0, 1.0]),  # neg, hi
        np.array([1.0, 0.0]),  # pos, lo
        np.array([1.0, 1.0]),  # pos, hi
    ]
    feas = EmpiricalFeasibility(scheme, strength=2).analyze(sut, probes)
    assert feas.n_feasible_outputs == 4
    # pairwise universe for two binary factors = 4 tuples, all feasible here
    assert feas.n_feasible_tuples == 4
    assert feas.probes_used == 4


def test_constraint_feasibility_prunes():
    scheme = PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("neg", "pos")),
            ThresholdBands(edges=(0.5,), labels=("lo", "hi")),
        )
    )
    # Only allow matching "polarity": (neg,lo) and (pos,hi).
    def constraint(out):
        return out in {("neg", "lo"), ("pos", "hi")}

    feas = ConstraintFeasibility(scheme, constraint, strength=2).analyze()
    assert feas.n_feasible_outputs == 2
    assert feas.n_feasible_tuples == 2


def test_all_feasible_matches_universe():
    scheme = PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("a", "b", "c")),
            CategoricalPassthrough(alphabet=("x", "y")),
        )
    )
    feas = all_feasible(scheme, s=2)
    assert feas.n_feasible_outputs == 6  # 3*2
    assert feas.n_feasible_tuples == 6  # single factor-pair, 3*2 combos


def test_greedy_generator_full_coverage():
    sut, scheme, _ = make_sut_scheme_space()
    probes = [
        np.array([-1.0, 0.0]),
        np.array([-1.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
    ]
    feas = EmpiricalFeasibility(scheme, 2).analyze(sut, probes)
    oca = GreedyGenerator(scheme, 2).generate(feas, np.random.default_rng(0))
    assert output_coverage(oca) == 1.0
    assert oca.is_complete()


def test_prioritize_monotone_and_complete():
    scheme = PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("neg", "pos")),
            ThresholdBands(edges=(0.5,), labels=("lo", "hi")),
        )
    )
    feas = all_feasible(scheme, 2)
    outs = [("neg", "lo"), ("neg", "hi"), ("pos", "lo"), ("pos", "hi")]
    cases = [
        TestCase(
            x=np.zeros(2),
            decoded=None,
            target=o,
            realised=o,
            loss=0.0,
            success=True,
            n_evals=1,
            n_iters=1,
        )
        for o in outs
    ]
    res = prioritize(cases, q=2, s=2, feasible_tuples=feas.feasible_tuples)
    # coverage curve must be non-decreasing
    assert all(b >= a for a, b in zip(res.coverage_curve, res.coverage_curve[1:]))
    assert res.final_coverage == 1.0


def test_pipeline_end_to_end_reaches_full_coverage():
    sut, scheme, space = make_sut_scheme_space()
    probes = [
        np.array([-1.0, 0.0]),
        np.array([-1.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 1.0]),
    ]
    pipe = ReverseNWisePipeline(
        sut=sut,
        scheme=scheme,
        space=space,
        strength=2,
        optimizer=Jaya(max_iter=100),
    )
    result = pipe.run(probe_inputs=probes, seed=0)
    assert result.ocov_s == 1.0
    assert result.realisation_rate == 1.0
    assert result.n_tests >= 1
    summary = result.summary()
    assert summary["OCov_s"] == 1.0
    assert summary["s"] == 2
