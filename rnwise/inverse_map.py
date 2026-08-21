"""Algorithm 1 - Step 3: InverseMap.

Given an ``OCA`` of target abstract outputs, recover a concrete input ``x`` for
each row by minimising the inverse-mapping loss ``L(x; w*)`` with a gradient-free
optimizer (``max_iter = 200`` per the paper). Rows for which no input hits the
target within budget are recorded as *unrealised* and reported (they bound the
achievable OCov_s from below).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rnwise.model import SystemUnderTest
from rnwise.oca import AbstractOutput, OutputCoveringArray
from rnwise.optimizers.base import (
    InverseObjective,
    Optimizer,
    Regularizer,
    SearchSpace,
)
from rnwise.partition import PartitionScheme, Symbol


@dataclass
class TestCase:
    """A concrete test input produced by inverting one OCA row."""

    __test__ = False  # not a pytest test class

    x: np.ndarray
    decoded: Any  # x passed through SearchSpace.decode (model-native input)
    target: tuple[Symbol, ...]  # requested abstract output w*
    realised: tuple[Symbol, ...]  # actual f_hat(x)
    loss: float
    success: bool
    n_evals: int
    n_iters: int


@dataclass
class InverseMapResult:
    """All inverted test cases plus realisation statistics."""

    test_cases: list[TestCase]
    n_targets: int
    n_realised: int

    @property
    def realisation_rate(self) -> float:
        return self.n_realised / self.n_targets if self.n_targets else 1.0

    @property
    def total_evals(self) -> int:
        return sum(tc.n_evals for tc in self.test_cases)

    def realised_abstract_outputs(self) -> list[AbstractOutput]:
        """The abstract outputs actually achieved (feeds coverage scoring)."""
        return [tc.realised for tc in self.test_cases]

    def inputs(self) -> list[Any]:
        return [tc.decoded for tc in self.test_cases]


@dataclass
class InverseMapper:
    """Runs Step 3 over every row of an OCA using a chosen optimizer.

    Attributes:
        sut: The system under test.
        scheme: Partition scheme defining ``f_hat``.
        space: Continuous search space over inputs (with optional decode).
        optimizer: Any :class:`Optimizer` (Jaya / Whale / Firefly / BO / VQE).
        lam: Regularisation weight lambda in ``L(x; w*)``.
        regularizer: Optional ``R(x)`` (manifold/box realism term).
        keep_unrealised: If True, still emit a TestCase for missed targets (its
            ``realised`` output is whatever the best-found ``x`` produced), so the
            downstream suite reflects true achieved coverage rather than the ideal.
    """

    sut: SystemUnderTest
    scheme: PartitionScheme
    space: SearchSpace
    optimizer: Optimizer
    lam: float = 0.0
    regularizer: Regularizer | None = None
    keep_unrealised: bool = True
    warm_start: dict[tuple, np.ndarray] | None = None

    def invert_row(
        self, target: AbstractOutput, rng: np.random.Generator
    ) -> TestCase | None:
        # Warm start: if feasibility handed us an input already realising this
        # target, accept it directly (Step 3 is trivially solved for that row).
        if self.warm_start and tuple(target) in self.warm_start:
            x0 = np.asarray(self.warm_start[tuple(target)], dtype=float)
            objective0 = InverseObjective(
                sut=self.sut,
                scheme=self.scheme,
                target=tuple(target),
                space=self.space,
                lam=self.lam,
                regularizer=self.regularizer,
            )
            if objective0.matches(x0):
                return TestCase(
                    x=x0,
                    decoded=self.space.to_input(x0),
                    target=tuple(target),
                    realised=objective0.abstract_output(x0),
                    loss=objective0(x0),
                    success=True,
                    n_evals=objective0.n_evals,
                    n_iters=0,
                )

        objective = InverseObjective(
            sut=self.sut,
            scheme=self.scheme,
            target=tuple(target),
            space=self.space,
            lam=self.lam,
            regularizer=self.regularizer,
        )
        res = self.optimizer.optimize(objective, rng)
        realised = objective.abstract_output(res.x)
        if not res.success and not self.keep_unrealised:
            return None
        return TestCase(
            x=res.x,
            decoded=self.space.to_input(res.x),
            target=tuple(target),
            realised=realised,
            loss=res.loss,
            success=res.success,
            n_evals=res.n_evals,
            n_iters=res.n_iters,
        )

    def run(self, oca: OutputCoveringArray, rng: np.random.Generator) -> InverseMapResult:
        cases: list[TestCase] = []
        realised = 0
        for row in oca.rows:
            tc = self.invert_row(row, rng)
            if tc is None:
                continue
            cases.append(tc)
            if tc.success:
                realised += 1
        return InverseMapResult(
            test_cases=cases, n_targets=oca.M, n_realised=realised
        )
