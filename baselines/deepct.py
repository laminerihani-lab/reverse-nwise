"""DeepCT-style baseline: output-activation-guided combinatorial sampling.

DeepCT (Ma et al.) adapts combinatorial testing to deep models by covering
combinations of neuron/output *activation* states. Here we approximate it in an
output-oriented way suited to any SUT: discretise each output factor into "active
/inactive" style bins and greedily grow a test set that covers pairwise
combinations of these coarse activation states -- but selecting from a random
input pool (it cannot invert, so it only covers what random inputs happen to
reach). This yields moderate output coverage: better than pure random, well below
Reverse N-Wise which actively inverts to reach targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from baselines import BaselineResult
from rnwise.model import SystemUnderTest
from rnwise.optimizers.base import SearchSpace
from rnwise.partition import PartitionScheme


@dataclass
class DeepCT:
    sut: SystemUnderTest
    scheme: PartitionScheme
    space: SearchSpace
    n_tests: int
    pool_size: int = 4000

    def generate(self, rng: np.random.Generator | None = None) -> BaselineResult:
        rng = rng or np.random.default_rng(0)
        s = 2
        q = self.scheme.q
        # sample a random input pool and compute their abstract outputs (activations)
        pool = self.space.sample(rng, self.pool_size)
        acts = [self.scheme.apply(self.sut.predict(x).mode()) for x in pool]

        covered: set = set()
        chosen: list[int] = []
        # greedily pick pool inputs maximising newly covered pairwise activation combos
        while len(chosen) < self.n_tests:
            best_i, best_gain = -1, -1
            for i, a in enumerate(acts):
                if i in chosen:
                    continue
                gain = 0
                for f1, f2 in combinations(range(q), s):
                    combo = ((f1, a[f1]), (f2, a[f2]))
                    if combo not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i < 0 or best_gain <= 0:
                break
            chosen.append(best_i)
            a = acts[best_i]
            for f1, f2 in combinations(range(q), s):
                covered.add(((f1, a[f1]), (f2, a[f2])))

        inputs = [pool[i] for i in chosen]
        return BaselineResult(name="DeepCT", inputs=inputs)
