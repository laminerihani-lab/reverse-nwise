"""Property-based testing baseline (Hypothesis-style).

Samples inputs from stratified regions of the search space (mimicking a
property-based tester's shrinking/strategy that emphasises boundaries and typical
values) at a matched budget. Output-agnostic like random, but with boundary bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines import BaselineResult
from rnwise.optimizers.base import SearchSpace


@dataclass
class PropertyBased:
    space: SearchSpace
    n_tests: int
    boundary_frac: float = 0.4  # fraction of tests placed at box boundaries

    def generate(self, rng: np.random.Generator | None = None) -> BaselineResult:
        rng = rng or np.random.default_rng(0)
        dim = self.space.dim
        n_boundary = int(self.n_tests * self.boundary_frac)
        inputs: list[np.ndarray] = []

        # boundary-biased samples: each dim snapped to lower/upper/mid at random
        for _ in range(n_boundary):
            choice = rng.integers(0, 3, size=dim)
            x = np.where(
                choice == 0,
                self.space.lower,
                np.where(choice == 1, self.space.upper, (self.space.lower + self.space.upper) / 2),
            )
            inputs.append(x.astype(float))

        # remaining: uniform interior samples
        rest = self.space.sample(rng, self.n_tests - n_boundary)
        inputs.extend(x for x in rest)
        return BaselineResult(name="Property-Based", inputs=inputs)
