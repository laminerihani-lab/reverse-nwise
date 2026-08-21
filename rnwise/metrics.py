"""Evaluation metrics: output coverage (OCov_s), tuple efficiency (eta_s), and
fault detection rate (FDR).

These are the headline numbers reported in the paper's Table 1. Definitions
follow the paper:

  * OCov_s -- fraction of feasible ``s``-way output tuples covered by a test set.
  * eta_s  -- output-tuple efficiency: total ``s``-way tuples realised per test
              (a density measure; higher means each test exercises more
              interactions).
  * FDR    -- fault detection rate: fraction of a seeded fault set detected by a
              test suite.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

from rnwise.oca import (
    AbstractOutput,
    OutputCoveringArray,
    OutputTuple,
    count_all_tuples,
    project,
)
from rnwise.partition import PartitionScheme


def output_coverage(oca: OutputCoveringArray) -> float:
    """OCov_s: covered feasible tuples / total feasible tuples, in ``[0, 1]``.

    When the array carries an explicit feasible set the denominator is that set;
    otherwise it is the full combinatorial universe.
    """
    total = oca.target_count()
    if total == 0:
        return 1.0
    return oca.covered_count() / total


def output_coverage_from_rows(
    rows: Sequence[AbstractOutput],
    symbol_space: Sequence[Sequence[object]],
    s: int,
    feasible: frozenset[OutputTuple] | None = None,
) -> float:
    """Convenience: build a transient OCA from raw abstract-output rows and score
    its OCov_s. Used by baselines that produce outputs directly."""
    arr = OutputCoveringArray(
        rows=[tuple(r) for r in rows],
        symbol_space=tuple(tuple(w) for w in symbol_space),
        strength=s,
        feasible=feasible,
    )
    return output_coverage(arr)


def tuple_efficiency(
    rows: Sequence[AbstractOutput],
    symbol_space: Sequence[Sequence[object]],
    s: int,
) -> float:
    """eta_s: total number of (not necessarily distinct) ``s``-way tuples realised
    across the suite, divided by the number of tests.

    Each row contributes ``C(q, s)`` tuples, so for a suite of ``M`` rows over
    ``q`` factors ``eta_s = M * C(q, s) / M = C(q, s)``. To capture *effective*
    density we instead report the mean number of *distinct-per-row* tuples, which
    for a single row equals ``C(q, s)`` but for the whole suite reflects how many
    interactions each test carries on average. We report the paper's convention:
    distinct covered tuples per test.
    """
    n = len(rows)
    if n == 0:
        return 0.0
    q = len(symbol_space)
    distinct: set[OutputTuple] = set()
    for row in rows:
        for factors in combinations(range(q), s):
            distinct.add(project(tuple(row), factors))
    return len(distinct) / n


def gross_tuple_efficiency(rows: Sequence[AbstractOutput], q: int, s: int) -> float:
    """Gross variant: total ``s``-way tuples emitted per test = ``C(q, s)`` (each
    test always realises exactly one tuple per factor subset). Provided for
    comparison / sanity checking."""
    from math import comb

    return float(comb(q, s))


# ---------------------------------------------------------------------------
# Fault detection
# ---------------------------------------------------------------------------

# A fault oracle takes a realised abstract output (or a test input) and returns
# True iff the fault is triggered/observed.
FaultOracle = Callable[[AbstractOutput], bool]


def fault_detection_rate(
    rows: Sequence[AbstractOutput],
    fault_oracles: Sequence[FaultOracle],
) -> tuple[int, int, float]:
    """FDR over a set of seeded faults.

    A fault is *detected* if at least one row triggers its oracle. Returns
    ``(detected, total, rate)`` so callers can render "8/8 (100%)" as in Table 1.
    """
    total = len(fault_oracles)
    if total == 0:
        return 0, 0, 1.0
    detected = 0
    for oracle in fault_oracles:
        if any(oracle(tuple(r)) for r in rows):
            detected += 1
    return detected, total, detected / total


def fdr_by_tuple_coverage(
    covered: frozenset[OutputTuple],
    fault_tuples: Sequence[OutputTuple],
) -> tuple[int, int, float]:
    """FDR under the paper's assumption that each seeded fault manifests as a
    specific ``s``-way output tuple: a fault is detected iff its signature tuple
    is covered. Returns ``(detected, total, rate)``."""
    total = len(fault_tuples)
    if total == 0:
        return 0, 0, 1.0
    detected = sum(1 for ft in fault_tuples if ft in covered)
    return detected, total, detected / total


def coverage_report(oca: OutputCoveringArray) -> dict[str, float | int]:
    """Bundle the core metrics for one array into a dict (for tables/logging)."""
    return {
        "M": oca.M,
        "q": oca.q,
        "s": oca.strength,
        "target_tuples": oca.target_count(),
        "covered_tuples": oca.covered_count(),
        "OCov_s": output_coverage(oca),
        "eta_s": tuple_efficiency(oca.rows, oca.symbol_space, oca.strength),
        "lower_bound_v_pow_s": oca.lower_bound(),
        "satisfies_bound": oca.satisfies_lower_bound(),
    }
