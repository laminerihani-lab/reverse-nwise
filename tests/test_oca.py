"""Unit tests for rnwise.oca: OCA datatypes and coverage arithmetic."""

from __future__ import annotations

from itertools import product

import pytest

from rnwise.oca import (
    OutputCoveringArray,
    OutputTuple,
    count_all_tuples,
    enumerate_all_tuples,
    project,
)


def test_output_tuple_validation():
    with pytest.raises(ValueError):
        OutputTuple(factors=(0, 1), symbols=("a",))  # length mismatch
    with pytest.raises(ValueError):
        OutputTuple(factors=(1, 0), symbols=("a", "b"))  # not ascending


def test_output_tuple_matches():
    row = ("hi", 2, "x")
    t = OutputTuple(factors=(0, 2), symbols=("hi", "x"))
    assert t.matches(row)
    assert not OutputTuple(factors=(0, 2), symbols=("hi", "y")).matches(row)
    assert t.strength == 2


def test_project():
    row = (0, 1, 2, 3)
    t = project(row, (1, 3))
    assert t == OutputTuple(factors=(1, 3), symbols=(1, 3))


def test_count_matches_enumeration():
    space = [("a", "b"), (0, 1, 2), ("x", "y")]
    for s in (1, 2, 3):
        assert count_all_tuples(space, s) == len(list(enumerate_all_tuples(space, s)))


def test_count_all_tuples_pairwise_homogeneous():
    # q=3, each level 2, s=2 -> C(3,2) subsets * 2*2 = 3*4 = 12
    space = [("a", "b")] * 3
    assert count_all_tuples(space, 2) == 12


def test_full_factorial_is_complete():
    # An exhaustive enumeration of the abstract-output universe must cover 100%.
    space = [("a", "b"), (0, 1, 2)]
    rows = [tuple(r) for r in product(*space)]
    arr = OutputCoveringArray(rows=rows, symbol_space=tuple(space), strength=2)
    assert arr.is_complete()
    assert arr.covered_count() == arr.target_count()
    assert not arr.uncovered()


def test_lower_bound_corollary1():
    # v = max level = 3, s = 2 -> M >= 9
    space = [("a", "b"), (0, 1, 2)]
    arr = OutputCoveringArray(rows=[("a", 0)], symbol_space=tuple(space), strength=2)
    assert arr.v == 3
    assert arr.lower_bound() == 9
    assert not arr.satisfies_lower_bound()
    arr.extend([("a", 0)] * 8)
    assert arr.M == 9
    assert arr.satisfies_lower_bound()


def test_feasible_restriction():
    space = [("a", "b"), (0, 1)]
    # Suppose only these pairwise tuples are feasible.
    feasible = frozenset(
        {
            OutputTuple(factors=(0, 1), symbols=("a", 0)),
            OutputTuple(factors=(0, 1), symbols=("b", 1)),
        }
    )
    arr = OutputCoveringArray(
        rows=[("a", 0)], symbol_space=tuple(space), strength=2, feasible=feasible
    )
    assert arr.target_count() == 2
    assert arr.covered_count() == 1
    arr.add_row(("b", 1))
    assert arr.covered_count() == 2
    assert arr.is_complete()


def test_add_row_invalidates_cache():
    space = [("a", "b"), (0, 1)]
    arr = OutputCoveringArray(rows=[("a", 0)], symbol_space=tuple(space), strength=2)
    first = arr.covered_count()
    arr.add_row(("b", 1))
    assert arr.covered_count() > first


def test_row_arity_validation():
    with pytest.raises(ValueError):
        OutputCoveringArray(rows=[("a",)], symbol_space=(("a",), (0,)), strength=1)
