"""System-under-test model: ``f: X -> P(Y)``.

The paper models a system under test as a (possibly stochastic) map from an
input space ``X`` to a distribution over a structured output space
``Y = Y_1 x ... x Y_q``. Deterministic systems are the degenerate case where the
distribution places all mass on a single output vector.

This module defines the light-weight protocols the rest of the tool programs
against, so that a scikit-learn classifier, an XGBoost model, a neural network,
or a quantum circuit can all be wrapped uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

# An input point. We keep this deliberately generic: a feature vector, a token
# sequence, an image tensor, or a circuit parameterisation are all valid.
Input = Any

# A single realised output vector y = (y_1, ..., y_q). Each component y_j lives
# in its own (possibly heterogeneous) sub-space Y_j.
Output = tuple[Any, ...]


@dataclass(frozen=True)
class StochasticOutput:
    """An empirical distribution over output vectors for one input.

    Attributes:
        samples: Realised output vectors, shape ``(n_samples, q)`` conceptually
            (a sequence of length-``q`` tuples).
        weights: Optional probability mass per sample. If ``None`` the samples
            are treated as equally weighted (Monte-Carlo draws).
    """

    samples: tuple[Output, ...]
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.weights is not None and len(self.weights) != len(self.samples):
            raise ValueError("weights and samples must have equal length")

    @property
    def q(self) -> int:
        """Number of output components (dimensionality of Y)."""
        if not self.samples:
            raise ValueError("empty StochasticOutput has no defined arity")
        return len(self.samples[0])

    def probabilities(self) -> np.ndarray:
        """Return a normalised probability vector over the samples."""
        n = len(self.samples)
        if n == 0:
            return np.array([], dtype=float)
        if self.weights is None:
            return np.full(n, 1.0 / n, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        total = w.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        return w / total

    def mode(self) -> Output:
        """Most probable output vector (MAP point)."""
        p = self.probabilities()
        return self.samples[int(np.argmax(p))]

    @classmethod
    def deterministic(cls, y: Output) -> "StochasticOutput":
        """Wrap a single deterministic output as a point-mass distribution."""
        return cls(samples=(y,), weights=(1.0,))


@runtime_checkable
class SystemUnderTest(Protocol):
    """Protocol for any system the tool can test.

    Concrete adapters (XGBoost, sklearn, torch, quantum circuits) implement this
    so the pipeline stays model-agnostic. ``q`` is the number of structured
    output components; ``predict`` returns one ``StochasticOutput`` per input.
    """

    @property
    def q(self) -> int:
        """Arity of the structured output space Y = Y_1 x ... x Y_q."""
        ...

    def predict(self, x: Input) -> StochasticOutput:
        """Evaluate f at a single input, returning a distribution over outputs."""
        ...


@dataclass
class CallableSUT:
    """Adapt a plain callable ``x -> Output`` (or ``-> StochasticOutput``) to the
    :class:`SystemUnderTest` protocol.

    This is the common case for deterministic classifiers where a single forward
    pass yields one output vector.
    """

    fn: Any
    arity: int
    stochastic: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def q(self) -> int:
        return self.arity

    def predict(self, x: Input) -> StochasticOutput:
        out = self.fn(x)
        if isinstance(out, StochasticOutput):
            return out
        if not isinstance(out, tuple):
            out = (out,)
        return StochasticOutput.deterministic(out)


def as_output_vector(y: Any, q: int) -> Output:
    """Coerce a raw model return value into a length-``q`` output tuple."""
    if isinstance(y, tuple):
        vec = y
    elif isinstance(y, (list, np.ndarray)):
        vec = tuple(np.asarray(y).tolist())
    else:
        vec = (y,)
    if len(vec) != q:
        raise ValueError(f"expected output arity {q}, got {len(vec)}: {vec!r}")
    return vec


def batch_predict(sut: SystemUnderTest, inputs: Sequence[Input]) -> list[StochasticOutput]:
    """Evaluate ``sut`` over a batch of inputs (reference loop; adapters may
    override with a vectorised path for speed)."""
    return [sut.predict(x) for x in inputs]
