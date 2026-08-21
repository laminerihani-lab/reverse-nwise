"""Random testing baseline: uniform random inputs at a matched test budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines import BaselineResult
from rnwise.optimizers.base import SearchSpace


@dataclass
class RandomTest:
    space: SearchSpace
    n_tests: int

    def generate(self, rng: np.random.Generator | None = None) -> BaselineResult:
        rng = rng or np.random.default_rng(0)
        xs = self.space.sample(rng, self.n_tests)
        return BaselineResult(name=f"Random-{self.n_tests}", inputs=[x for x in xs])
