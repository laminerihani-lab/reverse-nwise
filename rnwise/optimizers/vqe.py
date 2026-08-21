"""VQE-style inverse mapping for quantum systems under test.

For quantum SUTs the "input" is a vector of circuit parameters ``theta`` and the
"output" is a measurement-outcome distribution. The InverseMap step searches for
``theta`` whose *abstracted* measurement distribution matches a target abstract
output (e.g. |0>-dominant / |1>-dominant / superposition, or a syndrome class).

We minimise the same symbol-mismatch loss ``L(theta; w*)`` with the classical
COBYLA optimizer (the standard VQE outer loop). When the target is expressed as a
reference *distribution* rather than a symbol, a Wasserstein distance term can be
substituted for the 0/1 mismatch (see ``wasserstein_target``).

COBYLA comes from SciPy (a core dep); the quantum backend (qiskit) is only needed
by the benchmark that supplies the SUT, not by this optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from rnwise.optimizers.base import InverseObjective, OptimizeResult, SearchSpace


def wasserstein_1d(p: np.ndarray, q: np.ndarray) -> float:
    """1-Wasserstein (earth-mover) distance between two 1-D histograms over the
    same ordered support. Used as a soft output-matching loss for quantum
    measurement distributions."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("distributions must share support")
    ps = p / p.sum() if p.sum() > 0 else p
    qs = q / q.sum() if q.sum() > 0 else q
    return float(np.sum(np.abs(np.cumsum(ps) - np.cumsum(qs))))


@dataclass
class VQE:
    """COBYLA-driven inverse mapping for quantum SUTs.

    Attributes:
        max_iter: COBYLA iteration budget (paper uses 200 for InverseMap).
        rhobeg: Initial COBYLA trust-region radius.
        n_restarts: Random restarts to escape local minima on the NISQ landscape.
        dist_target: Optional reference output distribution; when supplied the loss
            adds a Wasserstein term ``wasserstein_1d(f_dist(theta), dist_target)``
            on top of the symbol mismatch.
        dist_fn: Callable ``theta -> np.ndarray`` returning the SUT's raw output
            distribution (required iff ``dist_target`` is set).
    """

    max_iter: int = 200
    rhobeg: float = 0.5
    n_restarts: int = 3
    dist_target: np.ndarray | None = None
    dist_fn: Callable[[np.ndarray], np.ndarray] | None = None
    history: list[float] = field(default_factory=list, init=False)

    def _loss(self, objective: InverseObjective, theta: np.ndarray) -> float:
        theta = objective.space.clip(theta)
        base = objective(theta)
        if self.dist_target is not None:
            if self.dist_fn is None:
                raise ValueError("dist_fn must be provided when dist_target is set")
            base = base + wasserstein_1d(self.dist_fn(theta), self.dist_target)
        self.history.append(min(self.history[-1], base) if self.history else base)
        return base

    def optimize(
        self, objective: InverseObjective, rng: np.random.Generator
    ) -> OptimizeResult:
        space: SearchSpace = objective.space
        best_x: np.ndarray | None = None
        best_loss = np.inf
        total_iters = 0

        for _ in range(self.n_restarts):
            x0 = space.sample(rng, 1)[0]
            res = minimize(
                lambda t: self._loss(objective, np.asarray(t, dtype=float)),
                x0,
                method="COBYLA",
                options={"maxiter": self.max_iter, "rhobeg": self.rhobeg},
            )
            total_iters += int(getattr(res, "nfev", self.max_iter))
            xr = space.clip(np.asarray(res.x, dtype=float))
            lr = float(res.fun)
            if lr < best_loss:
                best_loss, best_x = lr, xr
            if best_loss == 0.0:
                break

        assert best_x is not None
        return OptimizeResult(
            x=best_x,
            loss=best_loss,
            success=objective.matches(best_x),
            n_evals=objective.n_evals,
            n_iters=total_iters,
            history=list(self.history),
        )
