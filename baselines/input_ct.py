"""Input-side classical combinatorial testing (pairwise over INPUT features).

The traditional CT baseline: discretise each input feature into a few levels and
build a pairwise (t=2) covering array over the *inputs*. This is exactly the
method Reverse N-Wise inverts -- it covers input interactions but is blind to
whether the resulting *outputs* are diverse, which is why its output coverage
(OCov_s) is low in Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

import numpy as np

from baselines import BaselineResult


@dataclass
class InputCT:
    """Pairwise input covering array over discretised numeric features.

    Attributes:
        feature_levels: For each input dimension, the discrete candidate values
            (e.g. quantile representatives) to combine.
        strength: Input interaction strength (2 = pairwise).
    """

    feature_levels: list[np.ndarray]
    strength: int = 2

    def generate(self, rng: np.random.Generator | None = None) -> BaselineResult:
        rng = rng or np.random.default_rng(0)
        dim = len(self.feature_levels)
        s = self.strength
        # Greedy pairwise input covering array (IPO-like, output-agnostic).
        target: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for a, b in combinations(range(dim), s):
            for va in range(len(self.feature_levels[a])):
                for vb in range(len(self.feature_levels[b])):
                    target.add(((a, va), (b, vb)))

        rows: list[np.ndarray] = []
        covered: set = set()
        # candidate pool: random full assignments, greedily selected
        pool_size = 2000
        pool = np.stack(
            [
                np.array(
                    [rng.integers(len(self.feature_levels[d])) for d in range(dim)]
                )
                for _ in range(pool_size)
            ]
        )
        while covered != target:
            best_row = None
            best_gain = -1
            for assign in pool:
                gain = 0
                for a, b in combinations(range(dim), s):
                    pair = ((a, int(assign[a])), (b, int(assign[b])))
                    if pair not in covered:
                        gain += 1
                if gain > best_gain:
                    best_gain, best_row = gain, assign
            if best_row is None or best_gain <= 0:
                break
            rows.append(best_row)
            for a, b in combinations(range(dim), s):
                covered.add(((a, int(best_row[a])), (b, int(best_row[b]))))

        inputs = [
            np.array([self.feature_levels[d][idx] for d, idx in enumerate(row)])
            for row in rows
        ]
        return BaselineResult(name="Input CT", inputs=inputs)
