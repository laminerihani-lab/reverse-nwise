"""Self-contained NISQ quantum benchmark (no external qiskit dependency).

Implements a small parameterised quantum circuit as a statevector simulator with
a depolarising-noise channel, exposed as a :class:`SystemUnderTest`. This lets the
Reverse N-Wise pipeline (with the VQE inverse-map optimizer) build output covering
arrays over *quantum measurement-outcome behaviours* -- the paper's quantum
integration claim -- while staying fully reproducible and artifact-friendly.

Input:  circuit parameters theta (rotation angles).
Output (structured, q factors):
  0. dominant_state  -- which basis state dominates measurement (|0..0>, |1..1>,
                        or a superposition/other class)
  1. entropy_band    -- Shannon entropy of the outcome distribution
                        (low / mid / high = pure / partially-mixed / maximally-mixed)
  2. parity_class    -- expected parity of measured bitstrings (even / odd / balanced)
  3. fidelity_band   -- fidelity to the noiseless ideal state (low / mid / high),
                        capturing NISQ noise degradation

The abstract-output partition turns these into discrete symbols; covering arrays
then demand that every pairwise combination of quantum output behaviours be
exercised by some parameter setting -- exactly output-oriented testing for a
quantum SUT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from rnwise.model import StochasticOutput, SystemUnderTest
from rnwise.optimizers.base import SearchSpace
from rnwise.partition import CategoricalPassthrough, PartitionScheme, ThresholdBands


def _rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


@dataclass
class NISQCircuitSUT:
    """A 2-qubit parameterised ansatz with depolarising noise.

    Parameters (theta, length 4): [Ry(0) q0, Ry(1) q1, Rx(2) q0, Rx(3) q1] with a
    CNOT entangler between the layers. Depolarising noise mixes the ideal outcome
    distribution toward uniform with strength ``noise``.
    """

    n_qubits: int = 2
    noise: float = 0.08  # depolarising strength in [0,1]
    _q: int = field(default=4, init=False)

    @property
    def q(self) -> int:
        return self._q

    def _ideal_probs(self, theta: np.ndarray) -> np.ndarray:
        # |psi> = CNOT (Rx2 x Rx3) (Ry0 x Ry1) |00>
        psi = np.zeros(4, dtype=complex)
        psi[0] = 1.0
        layer1 = np.kron(_ry(theta[0]), _ry(theta[1]))
        layer2 = np.kron(_rx(theta[2]), _rx(theta[3]))
        psi = layer1 @ psi
        psi = _CNOT @ psi
        psi = layer2 @ psi
        return np.abs(psi) ** 2

    def outcome_distribution(self, theta: np.ndarray) -> np.ndarray:
        """Noisy measurement-outcome distribution over the 4 basis states."""
        p = self._ideal_probs(np.asarray(theta, dtype=float))
        uniform = np.full_like(p, 1.0 / p.size)
        return (1.0 - self.noise) * p + self.noise * uniform

    def predict(self, theta) -> StochasticOutput:
        probs = self.outcome_distribution(theta)
        ideal = self._ideal_probs(np.asarray(theta, dtype=float))

        # factor 0: dominant basis state class
        top = int(np.argmax(probs))
        if probs[top] < 0.4:
            dominant = "superpos"
        elif top == 0:
            dominant = "zero"  # |00>
        elif top == 3:
            dominant = "one"  # |11>
        else:
            dominant = "mixed"  # |01> or |10> dominant

        # factor 1: entropy of the distribution
        ent = float(-np.sum(probs * np.log2(probs + 1e-12)))  # in [0, 2] for 4 states

        # factor 2: parity (even = states 00,11 ; odd = 01,10)
        p_even = probs[0] + probs[3]
        parity = p_even - (1.0 - p_even)  # in [-1,1]; >0 even-leaning

        # factor 3: fidelity to the noiseless ideal (classical fidelity of dists)
        fid = float(np.sum(np.sqrt(probs * ideal)) ** 2)

        raw = (dominant, ent, parity, fid)
        return StochasticOutput.deterministic(raw)


def quantum_partition_scheme() -> PartitionScheme:
    """Partition the quantum SUT's raw output into a 4-factor abstract output."""
    return PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("zero", "one", "mixed", "superpos")),
            ThresholdBands(edges=(0.8, 1.6), labels=("low", "mid", "high")),  # entropy
            ThresholdBands(edges=(-0.25, 0.25), labels=("odd", "balanced", "even")),
            ThresholdBands(edges=(0.6, 0.9), labels=("low", "mid", "high")),  # fidelity
        )
    )


def quantum_search_space(n_params: int = 4) -> SearchSpace:
    """Rotation angles in [0, 2*pi) per parameter."""
    return SearchSpace(
        lower=np.zeros(n_params), upper=np.full(n_params, 2 * np.pi)
    )


def quantum_probe_thetas(n: int = 4000, seed: int = 0) -> list[np.ndarray]:
    """A probe set of parameter vectors (grid + random) for feasibility scanning."""
    rng = np.random.default_rng(seed)
    grid_pts = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    probes = [np.array(c, dtype=float) for c in product(grid_pts, repeat=4)]
    if len(probes) > n:
        idx = rng.choice(len(probes), size=n, replace=False)
        probes = [probes[i] for i in idx]
    else:
        extra = rng.uniform(0, 2 * np.pi, size=(n - len(probes), 4))
        probes.extend(x for x in extra)
    return probes
