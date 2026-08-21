"""Gradient-free optimizers for Step 3 (InverseMap) of Algorithm 1.

Each optimizer minimises the inverse-mapping loss

    L(x; w*) = sum_j delta(pi_j(f_j(x)), w*_j) + lambda * R(x)

over an input search space, trying to find an input ``x`` whose *abstract output*
``f_hat(x)`` equals a requested target abstract output ``w*``. The paper uses
metaheuristics (Jaya, Whale, Firefly), Bayesian optimisation, and (for quantum)
VQE with a Wasserstein loss.
"""

from rnwise.optimizers.base import (
    InverseObjective,
    Optimizer,
    OptimizeResult,
    SearchSpace,
    hamming_mismatch,
)

__all__ = [
    "InverseObjective",
    "Optimizer",
    "OptimizeResult",
    "SearchSpace",
    "hamming_mismatch",
]
