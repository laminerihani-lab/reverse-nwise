"""Bayesian optimization via a Gaussian-process surrogate with Expected
Improvement.

Uses ``scikit-optimize`` (``skopt``) when available; the objective is the same
inverse-mapping loss ``L(x; w*)``. BO is sample-efficient, which matters when
each SUT evaluation is expensive (large models, quantum backends).

scikit-optimize is an optional dependency (``pip install rnwise[bayesopt]``); the
import is deferred so the core package does not require it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rnwise.optimizers.base import InverseObjective, OptimizeResult, SearchSpace


@dataclass
class BayesOpt:
    n_calls: int = 100
    n_initial_points: int = 15
    acq_func: str = "EI"  # Expected Improvement

    def optimize(
        self, objective: InverseObjective, rng: np.random.Generator
    ) -> OptimizeResult:
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "BayesOpt requires scikit-optimize; install with `pip install rnwise[bayesopt]`"
            ) from exc

        space: SearchSpace = objective.space
        dims = [Real(float(lo), float(hi)) for lo, hi in zip(space.lower, space.upper)]
        history: list[float] = []

        def f(params: list[float]) -> float:
            val = objective(np.asarray(params, dtype=float))
            history.append(min(history[-1], val) if history else val)
            return val

        seed = int(rng.integers(0, 2**31 - 1))
        res = gp_minimize(
            f,
            dims,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acq_func,
            random_state=seed,
        )
        x = np.asarray(res.x, dtype=float)
        return OptimizeResult(
            x=x,
            loss=float(res.fun),
            success=objective.matches(x),
            n_evals=objective.n_evals,
            n_iters=self.n_calls,
            history=history,
        )
