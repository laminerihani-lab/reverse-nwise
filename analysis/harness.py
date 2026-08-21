"""Shared scoring harness: score any set of inputs by the output coverage they
achieve, on a fixed feasible universe. Used to compare Reverse N-Wise against all
baselines apples-to-apples (paper Table 1)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

from rnwise.metrics import fdr_by_tuple_coverage
from rnwise.model import SystemUnderTest
from rnwise.oca import OutputTuple, count_all_tuples, project
from rnwise.partition import PartitionScheme


@dataclass
class SuiteScore:
    """Scored metrics for one test suite against a shared feasible universe."""

    name: str
    n_tests: int
    ocov_s: float  # vs feasible universe
    ocov_s_full: float  # vs full (unpruned) universe
    eta_s: float  # distinct covered tuples / test
    covered_tuples: int
    fdr_detected: int
    fdr_total: int

    @property
    def fdr(self) -> float:
        return self.fdr_detected / self.fdr_total if self.fdr_total else 1.0

    def row(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "tests": self.n_tests,
            "OCov_s": round(self.ocov_s, 4),
            "OCov_s_full": round(self.ocov_s_full, 4),
            "eta_s": round(self.eta_s, 2),
            "covered_tuples": self.covered_tuples,
            "FDR": f"{self.fdr_detected}/{self.fdr_total}",
            "FDR_pct": round(self.fdr, 4),
        }


def realised_outputs(
    sut: SystemUnderTest, scheme: PartitionScheme, inputs: Sequence[Any]
) -> list[tuple]:
    """Map inputs through the SUT + partition scheme to abstract outputs."""
    return [scheme.apply(sut.predict(x).mode()) for x in inputs]


def covered_pairwise_tuples(rows: Sequence[tuple], q: int, s: int) -> set[OutputTuple]:
    covered: set[OutputTuple] = set()
    for row in rows:
        for factors in combinations(range(q), s):
            covered.add(project(row, factors))
    return covered


def score_suite(
    name: str,
    sut: SystemUnderTest,
    scheme: PartitionScheme,
    inputs: Sequence[Any],
    s: int,
    feasible_tuples: frozenset[OutputTuple],
    fault_tuples: Sequence[OutputTuple],
) -> SuiteScore:
    """Score a test suite's output coverage, efficiency, and fault detection."""
    q = scheme.q
    rows = realised_outputs(sut, scheme, inputs)
    covered = covered_pairwise_tuples(rows, q, s)

    covered_feasible = covered & feasible_tuples
    n_feasible = len(feasible_tuples) or 1
    n_full = count_all_tuples(scheme.symbol_space, s)

    n_tests = len(inputs) or 1
    detected, total, _ = fdr_by_tuple_coverage(frozenset(covered), fault_tuples)

    return SuiteScore(
        name=name,
        n_tests=len(inputs),
        ocov_s=len(covered_feasible) / n_feasible,
        ocov_s_full=len(covered_feasible) / n_full,
        eta_s=len(covered_feasible) / n_tests,
        covered_tuples=len(covered_feasible),
        fdr_detected=detected,
        fdr_total=total,
    )
