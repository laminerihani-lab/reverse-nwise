"""Baseline test-generation strategies compared against Reverse N-Wise in Table 1.

Each baseline produces a set of concrete inputs; the harness runs them through the
SUT + partition scheme to obtain realised abstract outputs, then scores OCov_s,
eta_s and FDR on the same feasible universe as Reverse N-Wise. This keeps the
comparison apples-to-apples: every method is judged by the *output* coverage it
achieves, regardless of how it selects inputs.

Baselines (paper Table 1):
  * input_ct     -- classical INPUT-side pairwise combinatorial testing.
  * random_test  -- uniform random inputs (matched test budget).
  * property_based -- Hypothesis-style property sampling around input strata.
  * metamorphic  -- metamorphic relations (perturb-and-compare) seed inputs.
  * deepct       -- neuron/output-activation-guided combinatorial sampling.

All baselines share the ``BaselineResult`` container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaselineResult:
    """Inputs proposed by a baseline plus its name (for reporting)."""

    name: str
    inputs: list[Any]

    @property
    def n_tests(self) -> int:
        return len(self.inputs)
