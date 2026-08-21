"""Semantic output partitioners ``pi_j: Y_j -> W_j``.

The abstraction map ``f_hat(x) = (pi_1(y_1), ..., pi_q(y_q))`` turns a raw output
vector into a discrete *abstract output* over symbol alphabets ``W_j``. Covering
arrays are then built over the product ``W_1 x ... x W_q``.

Each partitioner maps a raw component value to a discrete symbol (label). A named
registry lets configs refer to partitioners as strings (e.g. in YAML), and lets
benchmarks contribute domain-specific schemes (fairness bins, confidence bands,
decision-boundary regions, embedding clusters, quantum syndromes, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

Symbol = Any  # a discrete abstract-output label (usually int or str)


@runtime_checkable
class Partitioner(Protocol):
    """Maps a single raw output component ``y_j`` to a discrete symbol in ``W_j``."""

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """The finite alphabet ``W_j`` this partitioner emits."""
        ...

    def __call__(self, value: Any) -> Symbol:
        """Assign ``value`` to its abstract symbol."""
        ...


# ---------------------------------------------------------------------------
# Concrete partitioner families
# ---------------------------------------------------------------------------


@dataclass
class ThresholdBands:
    """Partition a scalar into ordered bands via ascending cut points.

    ``edges = [a, b]`` yields 3 bands, left-closed on each edge:
    ``(-inf, a)``, ``[a, b)``, ``[b, +inf)``. Useful for confidence calibration
    (e.g. low/medium/high) and any monotone scalar behavioural property.
    """

    edges: tuple[float, ...]
    labels: tuple[Symbol, ...] | None = None

    def __post_init__(self) -> None:
        if list(self.edges) != sorted(self.edges):
            raise ValueError("edges must be ascending")
        n = len(self.edges) + 1
        if self.labels is None:
            self.labels = tuple(range(n))
        elif len(self.labels) != n:
            raise ValueError(f"expected {n} labels for {len(self.edges)} edges")

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        assert self.labels is not None
        return self.labels

    def __call__(self, value: Any) -> Symbol:
        idx = int(np.searchsorted(self.edges, float(value), side="right"))
        assert self.labels is not None
        return self.labels[idx]


@dataclass
class CategoricalPassthrough:
    """Treat a discrete label as its own symbol (identity), over a fixed alphabet.

    Used for classifier decisions, fairness group membership, ranking-stability
    classes, quantum measurement outcomes, etc.
    """

    alphabet: tuple[Symbol, ...]

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return self.alphabet

    def __call__(self, value: Any) -> Symbol:
        if value not in self.alphabet:
            raise ValueError(f"value {value!r} not in alphabet {self.alphabet}")
        return value


@dataclass
class RegionPartitioner:
    """Assign a vector-valued component to a region label via a user function.

    Covers decision-boundary regions and embedding clusters where the raw
    component is itself a vector and the symbol is the region/cluster id.
    """

    region_fn: Callable[[Any], Symbol]
    alphabet: tuple[Symbol, ...]

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        return self.alphabet

    def __call__(self, value: Any) -> Symbol:
        sym = self.region_fn(value)
        if sym not in self.alphabet:
            raise ValueError(f"region_fn produced {sym!r} outside alphabet {self.alphabet}")
        return sym


@dataclass
class PartitionScheme:
    """A tuple of per-component partitioners defining ``f_hat`` for the whole SUT.

    ``apply`` maps a raw output vector to its abstract output; ``symbol_space``
    exposes each ``W_j`` so covering-array generators know the factor levels.
    """

    partitioners: tuple[Partitioner, ...]

    @property
    def q(self) -> int:
        return len(self.partitioners)

    @property
    def symbol_space(self) -> tuple[tuple[Symbol, ...], ...]:
        """The list of alphabets ``(W_1, ..., W_q)``."""
        return tuple(p.symbols for p in self.partitioners)

    @property
    def levels(self) -> tuple[int, ...]:
        """Number of symbols per component ``(|W_1|, ..., |W_q|)``."""
        return tuple(len(w) for w in self.symbol_space)

    def apply(self, output: Sequence[Any]) -> tuple[Symbol, ...]:
        """Compute ``f_hat(x)`` from a raw output vector ``y``."""
        if len(output) != self.q:
            raise ValueError(f"output arity {len(output)} != scheme arity {self.q}")
        return tuple(p(v) for p, v in zip(self.partitioners, output))


# ---------------------------------------------------------------------------
# Named registry (string -> factory), so configs/benchmarks stay declarative
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., Partitioner]] = {}


def register_partitioner(name: str, factory: Callable[..., Partitioner]) -> None:
    """Register a partitioner factory under a name (idempotent overwrite warns)."""
    _REGISTRY[name] = factory


def get_partitioner(name: str, /, **kwargs: Any) -> Partitioner:
    """Instantiate a registered partitioner by name with keyword args."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown partitioner {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def registered_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def scheme_from_spec(spec: Iterable[Mapping[str, Any]]) -> PartitionScheme:
    """Build a :class:`PartitionScheme` from a declarative list of specs.

    Each spec is ``{"kind": <name>, ...kwargs}``. Used to parse YAML configs.
    """
    parts: list[Partitioner] = []
    for entry in spec:
        d = dict(entry)
        kind = d.pop("kind")
        parts.append(get_partitioner(kind, **d))
    return PartitionScheme(tuple(parts))


# Built-in registrations
register_partitioner("threshold_bands", ThresholdBands)
register_partitioner("categorical", CategoricalPassthrough)
register_partitioner("region", RegionPartitioner)
