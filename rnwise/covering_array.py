"""Algorithm 1 - Step 2: CoveringArrayGenerator.

Build an ``OCA(M; s, q, w)`` over the *feasible* abstract-output universe. We ship
a self-contained greedy generator (one-row-per-step, maximise newly covered
feasible tuples) and optional wrappers around external combinatorial tools:

  * ACTS (NIST) -- via a bundled jar, invoked as a subprocess (IPOG algorithm).
  * PICT (Microsoft) -- via the ``pict`` binary.

External tools generate arrays over abstract *factors*; we then keep only rows
whose full abstract output is feasible, and top up any residual gap greedily so
the final array provably covers every feasible ``s``-way tuple (OCov_s = 1.0 on
the feasible universe) while respecting the ``M >= v^s`` lower bound (Corollary 1).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

from rnwise.feasibility import FeasibilityResult
from rnwise.oca import (
    AbstractOutput,
    OutputCoveringArray,
    OutputTuple,
    project,
)
from rnwise.partition import PartitionScheme, Symbol


def _row_new_coverage(
    row: AbstractOutput, factors_list: list[tuple[int, ...]], covered: set[OutputTuple]
) -> set[OutputTuple]:
    """Feasible tuples a candidate row would newly cover."""
    gained: set[OutputTuple] = set()
    for factors in factors_list:
        t = project(row, factors)
        if t not in covered:
            gained.add(t)
    return gained


@dataclass
class ExhaustiveFeasibleGenerator:
    """One OCA row per feasible abstract output (output-oriented suite).

    Rather than compressing to a minimal covering array, emit every feasible
    abstract output as a distinct target. This realises the paper's
    "one test per feasible output behaviour" philosophy: after inverse mapping,
    the suite size approaches the number of feasible outputs (minus inverse-map
    misses). It trivially covers 100% of feasible pairwise tuples on the target
    side; achieved coverage then depends on inverse-map realisation.
    """

    scheme: PartitionScheme
    strength: int = 2

    def generate(
        self, feasibility: FeasibilityResult, rng: np.random.Generator | None = None
    ) -> OutputCoveringArray:
        rows = list(feasibility.abstract_outputs)
        if rng is not None:
            rng.shuffle(rows)
        return OutputCoveringArray(
            rows=rows,
            symbol_space=self.scheme.symbol_space,
            strength=self.strength,
            feasible=feasibility.feasible_tuples,
        )


@dataclass
class GreedyGenerator:
    """Self-contained greedy OCA builder (no external tool dependency).

    Repeatedly appends the feasible abstract output that covers the most
    not-yet-covered feasible ``s``-way tuples, until every feasible tuple is
    covered. Deterministic given the feasible-output ordering and RNG.
    """

    scheme: PartitionScheme
    strength: int = 2

    def generate(
        self, feasibility: FeasibilityResult, rng: np.random.Generator | None = None
    ) -> OutputCoveringArray:
        rng = rng or np.random.default_rng(0)
        q = self.scheme.q
        s = self.strength
        factors_list = list(combinations(range(q), s))
        candidates = list(feasibility.abstract_outputs)
        rng.shuffle(candidates)

        target = set(feasibility.feasible_tuples)
        covered: set[OutputTuple] = set()
        rows: list[AbstractOutput] = []

        while covered != target:
            best_row = None
            best_gain: set[OutputTuple] = set()
            for row in candidates:
                gain = _row_new_coverage(row, factors_list, covered)
                gain &= target  # only count feasible tuples
                if len(gain) > len(best_gain):
                    best_row, best_gain = row, gain
            if best_row is None or not best_gain:
                break  # remaining target tuples unreachable from feasible outputs
            rows.append(best_row)
            covered |= best_gain

        return OutputCoveringArray(
            rows=rows,
            symbol_space=self.scheme.symbol_space,
            strength=s,
            feasible=feasibility.feasible_tuples,
        )


@dataclass
class ACTSGenerator:
    """Wrapper around the NIST ACTS jar (IPOG). Optional; requires Java + the jar.

    Produces an initial array over abstract factors; rows are filtered to feasible
    abstract outputs and any residual gap is closed with :class:`GreedyGenerator`.
    """

    scheme: PartitionScheme
    strength: int = 2
    acts_jar: Path | None = None
    java_bin: str = "java"

    def _available(self) -> bool:
        return (
            self.acts_jar is not None
            and Path(self.acts_jar).exists()
            and shutil.which(self.java_bin) is not None
        )

    def generate(
        self, feasibility: FeasibilityResult, rng: np.random.Generator | None = None
    ) -> OutputCoveringArray:
        if not self._available():
            # Graceful fallback keeps the pipeline runnable without the jar.
            return GreedyGenerator(self.scheme, self.strength).generate(feasibility, rng)
        rows = self._run_acts(feasibility)
        arr = OutputCoveringArray(
            rows=rows,
            symbol_space=self.scheme.symbol_space,
            strength=self.strength,
            feasible=feasibility.feasible_tuples,
        )
        _close_gap(arr, feasibility)
        return arr

    def _run_acts(self, feasibility: FeasibilityResult) -> list[AbstractOutput]:
        # ACTS consumes a system model file and emits a CSV covering array.
        space = self.scheme.symbol_space
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model.txt"
            out = Path(td) / "array.csv"
            model.write_text(_acts_model_text(space, self.strength))
            subprocess.run(
                [
                    self.java_bin,
                    "-Ddoi=%d" % self.strength,
                    "-jar",
                    str(self.acts_jar),
                    "cmd",
                    str(model),
                    str(out),
                ],
                check=True,
                capture_output=True,
            )
            return _parse_acts_csv(out, space, feasibility)


@dataclass
class PICTGenerator:
    """Wrapper around Microsoft PICT (``pict`` binary). Optional."""

    scheme: PartitionScheme
    strength: int = 2
    pict_bin: str = "pict"

    def _available(self) -> bool:
        return shutil.which(self.pict_bin) is not None

    def generate(
        self, feasibility: FeasibilityResult, rng: np.random.Generator | None = None
    ) -> OutputCoveringArray:
        if not self._available():
            return GreedyGenerator(self.scheme, self.strength).generate(feasibility, rng)
        space = self.scheme.symbol_space
        with tempfile.TemporaryDirectory() as td:
            model = Path(td) / "model.txt"
            model.write_text(_pict_model_text(space))
            proc = subprocess.run(
                [self.pict_bin, str(model), "/o:%d" % self.strength],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = _parse_pict_output(proc.stdout, space, feasibility)
        arr = OutputCoveringArray(
            rows=rows,
            symbol_space=space,
            strength=self.strength,
            feasible=feasibility.feasible_tuples,
        )
        _close_gap(arr, feasibility)
        return arr


# --- helpers ---------------------------------------------------------------


def _close_gap(arr: OutputCoveringArray, feasibility: FeasibilityResult) -> None:
    """Greedily append feasible outputs until all feasible tuples are covered."""
    q = arr.q
    s = arr.strength
    factors_list = list(combinations(range(q), s))
    target = set(feasibility.feasible_tuples)
    candidates = list(feasibility.abstract_outputs)
    covered = set(arr.covered_tuples()) & target
    while covered != target:
        best_row = None
        best_gain: set[OutputTuple] = set()
        for row in candidates:
            gain = (_row_new_coverage(row, factors_list, covered)) & target
            if len(gain) > len(best_gain):
                best_row, best_gain = row, gain
        if best_row is None or not best_gain:
            break
        arr.add_row(best_row)
        covered |= best_gain


def _acts_model_text(space: Sequence[Sequence[Symbol]], strength: int) -> str:
    lines = ["[System]", "Name: rnwise_oca", "", "[Parameter]"]
    for j, alphabet in enumerate(space):
        vals = ",".join(str(v) for v in alphabet)
        lines.append(f"F{j} (enum): {vals}")
    return "\n".join(lines) + "\n"


def _pict_model_text(space: Sequence[Sequence[Symbol]]) -> str:
    lines = []
    for j, alphabet in enumerate(space):
        vals = ", ".join(str(v) for v in alphabet)
        lines.append(f"F{j}: {vals}")
    return "\n".join(lines) + "\n"


def _symbol_index(space: Sequence[Sequence[Symbol]]) -> list[dict[str, Symbol]]:
    """Map the string form of each symbol back to the original symbol object."""
    return [{str(v): v for v in alphabet} for alphabet in space]


def _parse_acts_csv(
    path: Path, space: Sequence[Sequence[Symbol]], feasibility: FeasibilityResult
) -> list[AbstractOutput]:
    idx = _symbol_index(space)
    rows: list[AbstractOutput] = []
    text = path.read_text().strip().splitlines()
    if not text:
        return rows
    for line in text[1:]:  # skip header
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(space):
            continue
        row = tuple(idx[j].get(cells[j], cells[j]) for j in range(len(space)))
        if row in feasibility.abstract_outputs:
            rows.append(row)
    return rows


def _parse_pict_output(
    stdout: str, space: Sequence[Sequence[Symbol]], feasibility: FeasibilityResult
) -> list[AbstractOutput]:
    idx = _symbol_index(space)
    rows: list[AbstractOutput] = []
    lines = stdout.strip().splitlines()
    for line in lines[1:]:  # skip header row
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) != len(space):
            continue
        row = tuple(idx[j].get(cells[j], cells[j]) for j in range(len(space)))
        if row in feasibility.abstract_outputs:
            rows.append(row)
    return rows
