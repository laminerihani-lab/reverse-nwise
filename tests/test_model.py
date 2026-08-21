"""Unit tests for rnwise.model: SUT protocol and StochasticOutput."""

from __future__ import annotations

import numpy as np
import pytest

from rnwise.model import (
    CallableSUT,
    StochasticOutput,
    SystemUnderTest,
    as_output_vector,
    batch_predict,
)


def test_stochastic_output_deterministic():
    so = StochasticOutput.deterministic((">50K", 0.9))
    assert so.q == 2
    assert so.mode() == (">50K", 0.9)
    np.testing.assert_allclose(so.probabilities(), [1.0])


def test_stochastic_output_uniform_weights():
    so = StochasticOutput(samples=(("a",), ("b",), ("c",)))
    np.testing.assert_allclose(so.probabilities(), [1 / 3, 1 / 3, 1 / 3])


def test_stochastic_output_weighted_mode():
    so = StochasticOutput(samples=(("a",), ("b",)), weights=(0.2, 0.8))
    assert so.mode() == ("b",)
    np.testing.assert_allclose(so.probabilities(), [0.2, 0.8])


def test_stochastic_output_weight_length_mismatch():
    with pytest.raises(ValueError):
        StochasticOutput(samples=(("a",),), weights=(0.5, 0.5))


def test_stochastic_output_nonpositive_weights():
    so = StochasticOutput(samples=(("a",),), weights=(0.0,))
    with pytest.raises(ValueError):
        so.probabilities()


def test_callable_sut_deterministic_scalar():
    sut = CallableSUT(fn=lambda x: x + 1, arity=1)
    assert isinstance(sut, SystemUnderTest)
    out = sut.predict(41)
    assert out.mode() == (42,)
    assert sut.q == 1


def test_callable_sut_tuple_output():
    sut = CallableSUT(fn=lambda x: (">50K", 0.7), arity=2)
    out = sut.predict("row")
    assert out.mode() == (">50K", 0.7)


def test_callable_sut_passthrough_stochastic():
    so = StochasticOutput(samples=(("a",), ("b",)), weights=(0.3, 0.7))
    sut = CallableSUT(fn=lambda x: so, arity=1, stochastic=True)
    assert sut.predict(0) is so


def test_as_output_vector():
    assert as_output_vector(5, 1) == (5,)
    assert as_output_vector([1, 2], 2) == (1, 2)
    assert as_output_vector((1, 2, 3), 3) == (1, 2, 3)
    assert as_output_vector(np.array([1.0, 2.0]), 2) == (1.0, 2.0)
    with pytest.raises(ValueError):
        as_output_vector((1, 2), 3)


def test_batch_predict():
    sut = CallableSUT(fn=lambda x: (x,), arity=1)
    outs = batch_predict(sut, [1, 2, 3])
    assert [o.mode() for o in outs] == [(1,), (2,), (3,)]
