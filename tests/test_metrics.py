"""Unit tests for rnwise.metrics: OCov_s, eta_s, FDR."""

from __future__ import annotations

from itertools import product

from rnwise.metrics import (
    coverage_report,
    fault_detection_rate,
    fdr_by_tuple_coverage,
    output_coverage,
    output_coverage_from_rows,
    tuple_efficiency,
)
from rnwise.oca import OutputCoveringArray, OutputTuple


def test_ocov_full_coverage():
    space = [("a", "b"), (0, 1)]
    rows = [tuple(r) for r in product(*space)]
    arr = OutputCoveringArray(rows=rows, symbol_space=tuple(space), strength=2)
    assert output_coverage(arr) == 1.0


def test_ocov_partial():
    space = [("a", "b"), (0, 1)]
    # one row covers exactly 1 of the 4 pairwise tuples
    arr = OutputCoveringArray(rows=[("a", 0)], symbol_space=tuple(space), strength=2)
    assert output_coverage(arr) == 0.25


def test_ocov_from_rows_matches():
    space = [("a", "b"), (0, 1)]
    rows = [("a", 0), ("b", 1)]
    assert output_coverage_from_rows(rows, space, 2) == 0.5


def test_tuple_efficiency_single_row():
    # one row over q=3 factors, s=2 -> C(3,2)=3 distinct tuples / 1 test = 3.0
    space = [("a", "b"), (0, 1), ("x", "y")]
    assert tuple_efficiency([("a", 0, "x")], space, 2) == 3.0


def test_tuple_efficiency_dedup_across_rows():
    # two identical rows -> distinct tuples stay 3, /2 tests = 1.5
    space = [("a", "b"), (0, 1), ("x", "y")]
    rows = [("a", 0, "x"), ("a", 0, "x")]
    assert tuple_efficiency(rows, space, 2) == 1.5


def test_tuple_efficiency_empty():
    assert tuple_efficiency([], [("a",)], 1) == 0.0


def test_fault_detection_rate_all_detected():
    rows = [("a", 0), ("b", 1)]
    oracles = [
        lambda r: r[0] == "a",
        lambda r: r[1] == 1,
    ]
    detected, total, rate = fault_detection_rate(rows, oracles)
    assert (detected, total, rate) == (2, 2, 1.0)


def test_fault_detection_rate_partial():
    rows = [("a", 0)]
    oracles = [
        lambda r: r[0] == "a",  # detected
        lambda r: r[1] == 1,  # not detected
    ]
    detected, total, rate = fault_detection_rate(rows, oracles)
    assert detected == 1 and total == 2 and rate == 0.5


def test_fdr_by_tuple_coverage():
    covered = frozenset({OutputTuple(factors=(0, 1), symbols=("a", 0))})
    faults = [
        OutputTuple(factors=(0, 1), symbols=("a", 0)),  # covered
        OutputTuple(factors=(0, 1), symbols=("b", 1)),  # not
    ]
    detected, total, rate = fdr_by_tuple_coverage(covered, faults)
    assert detected == 1 and total == 2 and rate == 0.5


def test_coverage_report_keys():
    space = [("a", "b"), (0, 1)]
    arr = OutputCoveringArray(rows=[("a", 0)], symbol_space=tuple(space), strength=2)
    rep = coverage_report(arr)
    for k in (
        "M",
        "q",
        "s",
        "target_tuples",
        "covered_tuples",
        "OCov_s",
        "eta_s",
        "lower_bound_v_pow_s",
        "satisfies_bound",
    ):
        assert k in rep
