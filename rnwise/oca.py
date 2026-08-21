"""Output Covering Array ``OCA(M; s, q, w)`` datatypes and coverage math.

Definition (paper, Def. 2): an Output Covering Array of strength ``s`` over ``q``
output factors, each with alphabet of size ``w`` (heterogeneous ``w_j`` allowed),
is a multiset of ``M`` abstract-output rows such that every combination of
symbols on every ``s``-subset of factors that is *feasible* appears in at least
one row.

Key objects:
  * ``AbstractOutput`` -- one row ``w* = (w*_1, ..., w*_q)`` in ``W_1 x ... x W_q``.
  * ``OutputTuple``    -- an ``s``-way projection: which factors + which symbols.
  * ``OutputCoveringArray`` -- the array plus the strength/level metadata and the
    coverage arithmetic (total feasible tuples, covered tuples, ``M >= v^s`` check
    from Corollary 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterable, Iterator, Sequence

Symbol = object
AbstractOutput = tuple  # (w*_1, ..., w*_q); alias kept simple for typing ergonomics


@dataclass(frozen=True)
class OutputTuple:
    """An ``s``-way output interaction: a set of factor indices and the symbols
    assigned to them.

    ``factors`` is an ascending tuple of factor indices (length ``s``); ``symbols``
    are the corresponding assigned symbols (same length/order). Hashable so it can
    key coverage sets.
    """

    factors: tuple[int, ...]
    symbols: tuple[Symbol, ...]

    def __post_init__(self) -> None:
        if len(self.factors) != len(self.symbols):
            raise ValueError("factors and symbols must have equal length")
        if list(self.factors) != sorted(self.factors):
            raise ValueError("factors must be ascending")

    @property
    def strength(self) -> int:
        return len(self.factors)

    def matches(self, row: AbstractOutput) -> bool:
        """True if ``row`` exhibits this interaction on its factors."""
        return all(row[f] == s for f, s in zip(self.factors, self.symbols))


def project(row: AbstractOutput, factors: tuple[int, ...]) -> OutputTuple:
    """Project an abstract-output row onto the given factor indices."""
    return OutputTuple(factors=factors, symbols=tuple(row[f] for f in factors))


def enumerate_all_tuples(
    symbol_space: Sequence[Sequence[Symbol]], s: int
) -> Iterator[OutputTuple]:
    """Yield every possible ``s``-way :class:`OutputTuple` over the symbol space.

    This is the *combinatorial universe* before feasibility pruning: for each
    ``s``-subset of factors, the Cartesian product of their alphabets.
    """
    q = len(symbol_space)
    if not 1 <= s <= q:
        raise ValueError(f"strength s={s} must satisfy 1<=s<=q={q}")
    for factors in combinations(range(q), s):
        alphabets = [symbol_space[f] for f in factors]
        for combo in product(*alphabets):
            yield OutputTuple(factors=factors, symbols=tuple(combo))


def count_all_tuples(symbol_space: Sequence[Sequence[Symbol]], s: int) -> int:
    """Number of ``s``-way tuples in the universe (sum over factor subsets of the
    product of the involved alphabet sizes). Closed-form; no enumeration."""
    q = len(symbol_space)
    if not 1 <= s <= q:
        raise ValueError(f"strength s={s} must satisfy 1<=s<=q={q}")
    total = 0
    for factors in combinations(range(q), s):
        prod = 1
        for f in factors:
            prod *= len(symbol_space[f])
        total += prod
    return total


@dataclass
class OutputCoveringArray:
    """An ``OCA(M; s, q, w)`` over a heterogeneous symbol space.

    Attributes:
        rows: The ``M`` abstract-output rows (each length ``q``).
        symbol_space: The alphabets ``(W_1, ..., W_q)`` (defines factor levels).
        strength: Target coverage strength ``s`` (pairwise = 2).
        feasible: Optional set of feasible ``s``-way tuples the array is required
            to cover. If ``None``, every tuple in the universe is deemed feasible.
    """

    rows: list[AbstractOutput]
    symbol_space: tuple[tuple[Symbol, ...], ...]
    strength: int
    feasible: frozenset[OutputTuple] | None = None
    _covered_cache: frozenset[OutputTuple] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        q = len(self.symbol_space)
        for r in self.rows:
            if len(r) != q:
                raise ValueError(f"row arity {len(r)} != q={q}")
        if not 1 <= self.strength <= q:
            raise ValueError(f"strength s={self.strength} must satisfy 1<=s<=q={q}")

    # -- basic dimensions ---------------------------------------------------

    @property
    def M(self) -> int:
        """Number of rows (test count on the output side)."""
        return len(self.rows)

    @property
    def q(self) -> int:
        return len(self.symbol_space)

    @property
    def levels(self) -> tuple[int, ...]:
        return tuple(len(w) for w in self.symbol_space)

    @property
    def v(self) -> int:
        """Max alphabet size ``v = max_j |W_j|`` (used in the ``M >= v^s`` bound)."""
        return max(self.levels) if self.levels else 0

    # -- coverage arithmetic -----------------------------------------------

    def covered_tuples(self) -> frozenset[OutputTuple]:
        """Set of ``s``-way tuples actually realised by the rows (cached)."""
        if self._covered_cache is None:
            covered: set[OutputTuple] = set()
            q = self.q
            for row in self.rows:
                for factors in combinations(range(q), self.strength):
                    covered.add(project(row, factors))
            object.__setattr__(self, "_covered_cache", frozenset(covered))
        assert self._covered_cache is not None
        return self._covered_cache

    def universe(self) -> frozenset[OutputTuple]:
        """All ``s``-way tuples (feasibility-restricted if ``feasible`` given)."""
        if self.feasible is not None:
            return self.feasible
        return frozenset(enumerate_all_tuples(self.symbol_space, self.strength))

    def target_count(self) -> int:
        """Number of tuples the array is required to cover (denominator of OCov_s)."""
        if self.feasible is not None:
            return len(self.feasible)
        return count_all_tuples(self.symbol_space, self.strength)

    def covered_count(self) -> int:
        """Number of *required* tuples that are covered (numerator of OCov_s)."""
        covered = self.covered_tuples()
        if self.feasible is not None:
            return len(covered & self.feasible)
        return len(covered)

    def uncovered(self) -> frozenset[OutputTuple]:
        """Required tuples not yet covered (drives greedy prioritisation/gap fill)."""
        return self.universe() - self.covered_tuples()

    def is_complete(self) -> bool:
        """True iff every required (feasible) ``s``-way tuple is covered."""
        return not self.uncovered()

    # -- theoretical lower bound (Corollary 1) ------------------------------

    def lower_bound(self) -> int:
        """Corollary 1 lower bound ``M >= v^s`` on the number of rows."""
        return self.v**self.strength

    def satisfies_lower_bound(self) -> bool:
        return self.M >= self.lower_bound()

    # -- mutation helpers ---------------------------------------------------

    def add_row(self, row: AbstractOutput) -> None:
        """Append a row and invalidate the coverage cache."""
        if len(row) != self.q:
            raise ValueError(f"row arity {len(row)} != q={self.q}")
        self.rows.append(tuple(row))
        object.__setattr__(self, "_covered_cache", None)

    def extend(self, rows: Iterable[AbstractOutput]) -> None:
        for r in rows:
            self.add_row(r)
