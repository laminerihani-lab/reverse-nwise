"""Algorithm 1 - Step 1: FeasibilitySAT.

Not every abstract-output symbol combination is realisable by the SUT. Before
building a covering array we must determine the set of *feasible* abstract
outputs (and hence feasible ``s``-way output tuples), otherwise the array would
demand coverage of impossible interactions and OCov_s could never reach 100%.

Two feasibility strategies are provided:

  * ``EmpiricalFeasibility`` -- sample the SUT on a probe set of inputs and record
    which abstract outputs actually occur. Model-agnostic; the default for AI/ML
    SUTs where no symbolic model of ``f`` exists.
  * ``ConstraintFeasibility`` -- when explicit logical constraints among output
    symbols are known (e.g. domain rules, mutually exclusive labels), prune the
    universe with a SAT/CP solver (python-sat / OR-Tools). Falls back gracefully
    if the optional solver is unavailable.

Both yield the feasible abstract-output set and the induced feasible ``s``-way
tuple set consumed by the covering-array generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np

from rnwise.model import SystemUnderTest
from rnwise.oca import OutputTuple, enumerate_all_tuples, project
from rnwise.partition import PartitionScheme, Symbol

AbstractOutput = tuple


@dataclass
class FeasibilityResult:
    """Feasible abstract outputs and the induced feasible ``s``-way tuples."""

    abstract_outputs: frozenset[AbstractOutput]
    feasible_tuples: frozenset[OutputTuple]
    strength: int
    probes_used: int = 0
    exemplars: dict[AbstractOutput, object] = field(default_factory=dict)

    @property
    def n_feasible_outputs(self) -> int:
        return len(self.abstract_outputs)

    @property
    def n_feasible_tuples(self) -> int:
        return len(self.feasible_tuples)


def _induce_tuples(
    abstract_outputs: Iterable[AbstractOutput], q: int, s: int
) -> frozenset[OutputTuple]:
    """Project a set of feasible abstract outputs onto every ``s``-subset to get
    the set of feasible ``s``-way tuples."""
    tuples: set[OutputTuple] = set()
    for row in abstract_outputs:
        for factors in combinations(range(q), s):
            tuples.add(project(row, factors))
    return frozenset(tuples)


@dataclass
class EmpiricalFeasibility:
    """Estimate feasibility by sampling the SUT over a probe input set.

    ``probe_inputs`` should broadly exercise the input space (e.g. the training/
    validation set, or Latin-hypercube samples). The observed abstract outputs are
    taken as the feasible set. Optionally augment with a random-search phase.
    """

    scheme: PartitionScheme
    strength: int = 2

    def analyze(
        self,
        sut: SystemUnderTest,
        probe_inputs: Sequence,
    ) -> FeasibilityResult:
        observed: set[AbstractOutput] = set()
        exemplars: dict[AbstractOutput, object] = {}
        for x in probe_inputs:
            raw = sut.predict(x).mode()
            ao = self.scheme.apply(raw)
            observed.add(ao)
            if ao not in exemplars:  # keep first input that realises this output
                exemplars[ao] = x
        q = self.scheme.q
        return FeasibilityResult(
            abstract_outputs=frozenset(observed),
            feasible_tuples=_induce_tuples(observed, q, self.strength),
            strength=self.strength,
            probes_used=len(probe_inputs),
            exemplars=exemplars,
        )


@dataclass
class ConstraintFeasibility:
    """Prune the abstract-output universe with explicit symbol constraints.

    ``constraint`` is a predicate over a candidate abstract output returning True
    iff that combination is logically admissible. This mirrors a SAT/CP feasibility
    oracle without hard-coding a particular solver; when a python-sat/OR-Tools model
    is available the predicate can wrap a solver call.
    """

    scheme: PartitionScheme
    constraint: Callable[[AbstractOutput], bool]
    strength: int = 2

    def analyze(self) -> FeasibilityResult:
        from itertools import product

        q = self.scheme.q
        space = self.scheme.symbol_space
        feasible_outputs: set[AbstractOutput] = set()
        for combo in product(*space):
            if self.constraint(combo):
                feasible_outputs.add(combo)
        return FeasibilityResult(
            abstract_outputs=frozenset(feasible_outputs),
            feasible_tuples=_induce_tuples(feasible_outputs, q, self.strength),
            strength=self.strength,
        )


def all_feasible(scheme: PartitionScheme, s: int) -> FeasibilityResult:
    """Baseline: treat the entire combinatorial universe as feasible (no pruning).

    Useful for synthetic checks and as an upper bound on required tuples.
    """
    from itertools import product

    q = scheme.q
    outputs = frozenset(product(*scheme.symbol_space))
    tuples = frozenset(enumerate_all_tuples(scheme.symbol_space, s))
    return FeasibilityResult(
        abstract_outputs=outputs, feasible_tuples=tuples, strength=s
    )
