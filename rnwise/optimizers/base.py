"""Optimizer protocol and the inverse-mapping objective ``L(x; w*)``.

Step 3 of Algorithm 1 seeks an input ``x`` whose abstract output matches a target
abstract output ``w*``. We frame it as minimising

    L(x; w*) = sum_j delta(pi_j(f_j(x)), w*_j) + lambda * R(x)

where ``delta`` is 0/1 symbol mismatch, ``R`` is an optional input regulariser
(e.g. keep ``x`` near a data manifold / within box bounds), and ``lambda`` trades
off feasibility against realism. A loss of 0 means ``f_hat(x) == w*`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from rnwise.model import SystemUnderTest
from rnwise.partition import PartitionScheme, Symbol

# A regulariser R(x) -> non-negative float. Default: none.
Regularizer = Callable[[np.ndarray], float]


def hamming_mismatch(abstract: Sequence[Symbol], target: Sequence[Symbol]) -> int:
    """sum_j delta(abstract_j, target_j) -- number of mismatched output symbols."""
    return sum(1 for a, b in zip(abstract, target) if a != b)


@dataclass
class SearchSpace:
    """A bounded continuous input search space (box constraints).

    Discrete/categorical inputs are handled by rounding/decoding inside the
    ``decode`` hook; by default we pass the raw continuous vector through.
    """

    lower: np.ndarray
    upper: np.ndarray
    decode: Callable[[np.ndarray], Any] | None = None

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper bounds must have identical shape")
        if np.any(self.lower > self.upper):
            raise ValueError("each lower bound must be <= its upper bound")

    @property
    def dim(self) -> int:
        return int(self.lower.size)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def sample(self, rng: np.random.Generator, n: int = 1) -> np.ndarray:
        u = rng.random((n, self.dim))
        return self.lower + u * (self.upper - self.lower)

    def to_input(self, x: np.ndarray) -> Any:
        return self.decode(x) if self.decode is not None else x


@dataclass
class InverseObjective:
    """Callable objective ``L(x; w*)`` bound to a SUT, a partition scheme, a target
    abstract output, and an optional regulariser.

    Instances are cheap to construct per target; they cache nothing across calls
    so they are safe to reuse across optimizer restarts.
    """

    sut: SystemUnderTest
    scheme: PartitionScheme
    target: tuple[Symbol, ...]
    space: SearchSpace
    lam: float = 0.0
    regularizer: Regularizer | None = None
    n_evals: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if len(self.target) != self.scheme.q:
            raise ValueError(
                f"target arity {len(self.target)} != scheme arity {self.scheme.q}"
            )

    def abstract_output(self, x: np.ndarray) -> tuple[Symbol, ...]:
        """Compute ``f_hat(x)`` = partitioned mode of the SUT's output at ``x``."""
        raw = self.sut.predict(self.space.to_input(x)).mode()
        return self.scheme.apply(raw)

    def __call__(self, x: np.ndarray) -> float:
        self.n_evals += 1
        x = self.space.clip(np.asarray(x, dtype=float))
        mismatch = hamming_mismatch(self.abstract_output(x), self.target)
        reg = 0.0
        if self.lam and self.regularizer is not None:
            reg = self.lam * float(self.regularizer(x))
        return float(mismatch) + reg

    def matches(self, x: np.ndarray) -> bool:
        """True iff ``f_hat(x)`` equals the target (loss's mismatch term is 0)."""
        return hamming_mismatch(self.abstract_output(x), self.target) == 0


@dataclass
class OptimizeResult:
    """Outcome of one inverse-mapping search."""

    x: np.ndarray
    loss: float
    success: bool  # loss's mismatch term reached 0 (abstract output == target)
    n_evals: int
    n_iters: int
    history: list[float] = field(default_factory=list)


@runtime_checkable
class Optimizer(Protocol):
    """Minimise an :class:`InverseObjective` over its :class:`SearchSpace`.

    Implementations are gradient-free population/metaheuristic methods. They stop
    early on an exact hit (loss==0) or after ``max_iter`` iterations.
    """

    def optimize(
        self, objective: InverseObjective, rng: np.random.Generator
    ) -> OptimizeResult:
        ...
