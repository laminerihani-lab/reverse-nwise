"""Tests for the inverse-mapping optimizers on synthetic objectives.

We construct a SUT whose abstract output is a deterministic function of a
continuous input, so each optimizer should be able to drive the symbol-mismatch
loss to 0 for reachable targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from rnwise.model import CallableSUT, StochasticOutput
from rnwise.optimizers.base import InverseObjective, SearchSpace, hamming_mismatch
from rnwise.optimizers.firefly import Firefly
from rnwise.optimizers.jaya import Jaya
from rnwise.optimizers.vqe import VQE, wasserstein_1d
from rnwise.optimizers.whale import Whale
from rnwise.partition import CategoricalPassthrough, PartitionScheme, ThresholdBands


def make_objective(target):
    """SUT: output = (sign of x0, band of x1). 2 factors."""

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
    return InverseObjective(sut=sut, scheme=scheme, target=target, space=space)


def test_hamming_mismatch():
    assert hamming_mismatch(("a", "b"), ("a", "b")) == 0
    assert hamming_mismatch(("a", "b"), ("a", "c")) == 1
    assert hamming_mismatch(("x", "b"), ("a", "c")) == 2


def test_search_space_validation():
    with pytest.raises(ValueError):
        SearchSpace(lower=np.array([1.0]), upper=np.array([0.0]))
    with pytest.raises(ValueError):
        SearchSpace(lower=np.array([0.0, 0.0]), upper=np.array([1.0]))


def test_objective_arity_check():
    with pytest.raises(ValueError):
        make_objective(("pos",))  # arity 1 vs scheme arity 2


@pytest.mark.parametrize("opt", [Jaya(max_iter=100), Whale(max_iter=100), Firefly(max_iter=60)])
def test_metaheuristics_hit_reachable_target(opt):
    rng = np.random.default_rng(42)
    obj = make_objective(("pos", "hi"))  # x0>=0, x1>=0.5
    res = opt.optimize(obj, rng)
    assert res.success
    assert res.loss == 0.0
    assert obj.matches(res.x)


def test_jaya_other_corner():
    rng = np.random.default_rng(7)
    obj = make_objective(("neg", "lo"))  # x0<0, x1<0.5
    res = Jaya(max_iter=100).optimize(obj, rng)
    assert res.success


def test_vqe_reachable_target():
    rng = np.random.default_rng(1)
    obj = make_objective(("pos", "hi"))
    res = VQE(max_iter=100, n_restarts=4).optimize(obj, rng)
    assert res.success


def test_wasserstein_1d():
    assert wasserstein_1d([1, 0, 0], [1, 0, 0]) == 0.0
    # all mass moves one bin over: EMD = 1
    assert wasserstein_1d([1, 0, 0], [0, 1, 0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        wasserstein_1d([1, 0], [1, 0, 0])


def test_optimizer_counts_evals():
    rng = np.random.default_rng(3)
    obj = make_objective(("pos", "hi"))
    res = Jaya(pop_size=10, max_iter=50).optimize(obj, rng)
    assert res.n_evals > 0
    assert obj.n_evals == res.n_evals
