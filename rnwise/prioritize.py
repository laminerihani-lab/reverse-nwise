"""Algorithm 1 - Step 4: Prioritize.

Order the realised test cases so that output coverage grows as fast as possible:
a greedy incremental ordering that, at each step, picks the remaining test whose
*realised* abstract output adds the most not-yet-covered feasible ``s``-way
tuples. This yields a monotone OCov_s curve, so a truncated suite (budget cap)
still maximises early coverage - the standard test-case-prioritisation objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from rnwise.inverse_map import TestCase
from rnwise.oca import OutputTuple, project


@dataclass
class PrioritizationResult:
    """Ordered test cases plus the cumulative coverage curve."""

    ordered: list[TestCase]
    coverage_curve: list[float]  # OCov_s after 1, 2, ..., k tests
    feasible_total: int

    @property
    def final_coverage(self) -> float:
        return self.coverage_curve[-1] if self.coverage_curve else 0.0


def _row_tuples(realised: tuple, q: int, s: int) -> set[OutputTuple]:
    return {project(realised, f) for f in combinations(range(q), s)}


def prioritize(
    test_cases: list[TestCase],
    q: int,
    s: int,
    feasible_tuples: frozenset[OutputTuple],
) -> PrioritizationResult:
    """Greedy incremental OCov_s ordering of ``test_cases``.

    Ties are broken by original order (stable). Coverage is measured against the
    feasible tuple set so the curve tops out at the achievable maximum.
    """
    remaining = list(test_cases)
    ordered: list[TestCase] = []
    covered: set[OutputTuple] = set()
    curve: list[float] = []
    total = len(feasible_tuples) or 1

    # Precompute each case's feasible tuple contribution once.
    contrib: dict[int, set[OutputTuple]] = {
        i: _row_tuples(tc.realised, q, s) & feasible_tuples
        for i, tc in enumerate(remaining)
    }
    used: set[int] = set()

    while len(used) < len(remaining):
        best_i = -1
        best_gain = -1
        for i, tc in enumerate(remaining):
            if i in used:
                continue
            gain = len(contrib[i] - covered)
            if gain > best_gain:
                best_gain, best_i = gain, i
        if best_i < 0:
            break
        used.add(best_i)
        ordered.append(remaining[best_i])
        covered |= contrib[best_i]
        curve.append(len(covered) / total)

    return PrioritizationResult(
        ordered=ordered, coverage_curve=curve, feasible_total=len(feasible_tuples)
    )
