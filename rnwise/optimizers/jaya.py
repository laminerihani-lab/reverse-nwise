"""Jaya optimizer (Rao, 2016): parameter-free population metaheuristic.

Each candidate moves toward the current best and away from the current worst:

    x' = x + r1 * (best - |x|) - r2 * (worst - |x|)

No algorithm-specific control parameters beyond population size and iterations,
which makes it a robust default for the InverseMap step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rnwise.optimizers.base import InverseObjective, OptimizeResult, SearchSpace


@dataclass
class Jaya:
    pop_size: int = 30
    max_iter: int = 200

    def optimize(
        self, objective: InverseObjective, rng: np.random.Generator
    ) -> OptimizeResult:
        space: SearchSpace = objective.space
        pop = space.sample(rng, self.pop_size)
        fit = np.array([objective(ind) for ind in pop])
        history: list[float] = [float(fit.min())]

        for it in range(self.max_iter):
            best = pop[int(np.argmin(fit))]
            worst = pop[int(np.argmax(fit))]
            for i in range(self.pop_size):
                r1, r2 = rng.random(space.dim), rng.random(space.dim)
                cand = pop[i] + r1 * (best - np.abs(pop[i])) - r2 * (worst - np.abs(pop[i]))
                cand = space.clip(cand)
                cf = objective(cand)
                if cf < fit[i]:
                    pop[i], fit[i] = cand, cf
            history.append(float(fit.min()))
            if fit.min() == 0.0:  # exact abstract-output hit
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
