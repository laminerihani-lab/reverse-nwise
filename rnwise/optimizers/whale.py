"""Whale Optimization Algorithm (Mirjalili & Lewis, 2016).

Models humpback bubble-net hunting with three moves per agent:
  * encircling prey (shrink toward best),
  * spiral update (log-spiral around best),
  * exploration (move toward a random agent when |A| >= 1).

The coefficient ``a`` decreases linearly from 2 to 0 over the run, shifting the
search from exploration to exploitation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rnwise.optimizers.base import InverseObjective, OptimizeResult, SearchSpace


@dataclass
class Whale:
    pop_size: int = 30
    max_iter: int = 200
    b: float = 1.0  # spiral shape constant

    def optimize(
        self, objective: InverseObjective, rng: np.random.Generator
    ) -> OptimizeResult:
        space: SearchSpace = objective.space
        pop = space.sample(rng, self.pop_size)
        fit = np.array([objective(ind) for ind in pop])
        best_idx = int(np.argmin(fit))
        best = pop[best_idx].copy()
        best_fit = float(fit[best_idx])
        history: list[float] = [best_fit]

        for it in range(self.max_iter):
            a = 2.0 - 2.0 * it / self.max_iter  # linearly 2 -> 0
            for i in range(self.pop_size):
                r = rng.random(space.dim)
                A = 2.0 * a * r - a
                C = 2.0 * rng.random(space.dim)
                p = rng.random()
                if p < 0.5:
                    if np.linalg.norm(A) < 1.0:
                        D = np.abs(C * best - pop[i])
                        cand = best - A * D
                    else:
                        rand = pop[rng.integers(self.pop_size)]
                        D = np.abs(C * rand - pop[i])
                        cand = rand - A * D
                else:
                    l = rng.uniform(-1.0, 1.0, space.dim)
                    D = np.abs(best - pop[i])
                    cand = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best
                cand = space.clip(cand)
                cf = objective(cand)
                if cf < fit[i]:
                    pop[i], fit[i] = cand, cf
                    if cf < best_fit:
                        best, best_fit = cand.copy(), cf
            history.append(best_fit)
            if best_fit == 0.0:
                break

        return OptimizeResult(
            x=best,
            loss=best_fit,
            success=objective.matches(best),
            n_evals=objective.n_evals,
            n_iters=it + 1,
            history=history,
        )
