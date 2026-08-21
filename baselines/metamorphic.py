"""Metamorphic testing baseline.

Seeds a set of source inputs, then generates follow-up inputs via metamorphic
relations (feature perturbations that a metamorphic tester would apply, e.g.
scale hours, shift age). The source+follow-up inputs form the test set. Coverage
of the *output* space is incidental, which is why metamorphic testing under-covers
in Table 1 despite being effective at finding specific relational violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from baselines import BaselineResult
from rnwise.optimizers.base import SearchSpace


@dataclass
class Metamorphic:
    space: SearchSpace
    n_sources: int
    # each relation maps a source input to a perturbed follow-up input
    relations: list[Callable[[np.ndarray], np.ndarray]] = field(default_factory=list)

    def _default_relations(self) -> list[Callable[[np.ndarray], np.ndarray]]:
        span = self.space.upper - self.space.lower

        def scale_up(x):
            return self.space.clip(x + 0.15 * span)

        def scale_down(x):
            return self.space.clip(x - 0.15 * span)

        def perturb_first(x):
            y = x.copy()
            y[0] = min(y[0] + 0.25 * span[0], self.space.upper[0])
            return self.space.clip(y)

        return [scale_up, scale_down, perturb_first]

    def generate(self, rng: np.random.Generator | None = None) -> BaselineResult:
        rng = rng or np.random.default_rng(0)
        rels = self.relations or self._default_relations()
        sources = self.space.sample(rng, self.n_sources)
        inputs: list[np.ndarray] = []
        for src in sources:
            inputs.append(src)
            for rel in rels:
                inputs.append(rel(src))
        return BaselineResult(name="Metamorphic", inputs=inputs)
