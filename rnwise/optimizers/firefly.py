"""Firefly Algorithm (Yang, 2009).

Fireflies are attracted to brighter (lower-loss) fireflies; attractiveness decays
with distance as ``beta0 * exp(-gamma * r^2)``. A random walk term (scaled by
``alpha``, annealed each iteration) maintains exploration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rnwise.optimizers.base import InverseObjective, OptimizeResult, SearchSpace


@dataclass
class Firefly:
    pop_size: int = 25
    max_iter: int = 200
    alpha: float = 0.5  # randomisation strength (annealed)
    beta0: float = 1.0  # attractiveness at r=0
    gamma: float = 1.0  # light absorption
    alpha_decay: float = 0.97

    def optimize(
        self, objective: InverseObjective, rng: np.random.Generator
    ) -> OptimizeResult:
        space: SearchSpace = objective.space
        scale = space.upper - space.lower
        pop = space.sample(rng, self.pop_size)
        fit = np.array([objective(ind) for ind in pop])
        history: list[float] = [float(fit.min())]
        alpha = self.alpha

        for it in range(self.max_iter):
            for i in range(self.pop_size):
                for k in range(self.pop_size):
                    if fit[k] < fit[i]:  # k is brighter -> i moves toward k
                        r2 = float(np.sum((pop[i] - pop[k]) ** 2))
                        beta = self.beta0 * np.exp(-self.gamma * r2)
                        rand = alpha * (rng.random(space.dim) - 0.5) * scale
                        cand = space.clip(pop[i] + beta * (pop[k] - pop[i]) + rand)
                        cf = objective(cand)
                        if cf < fit[i]:
                            pop[i], fit[i] = cand, cf
            alpha *= self.alpha_decay
            history.append(float(fit.min()))
            if fit.min() == 0.0:
                break

        j = int(np.argmin(fit))
        return OptimizeResult(
            x=pop[j],
            loss=float(fit[j]),
            success=objective.matches(pop[j]),
            n_evals=objective.n_evals,
            n_iters=it + 1,
            history=history,
        )
