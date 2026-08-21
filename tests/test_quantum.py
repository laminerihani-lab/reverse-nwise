"""Tests for the NISQ quantum benchmark and multi-domain pipeline (exp02)."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.quantum.nisq_sut import (
    NISQCircuitSUT,
    quantum_partition_scheme,
    quantum_probe_thetas,
    quantum_search_space,
)
from rnwise.feasibility import EmpiricalFeasibility
from rnwise.oca import count_all_tuples


def test_outcome_distribution_normalised():
    sut = NISQCircuitSUT(noise=0.08)
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta = rng.uniform(0, 2 * np.pi, size=4)
        p = sut.outcome_distribution(theta)
        assert p.shape == (4,)
        assert np.isclose(p.sum(), 1.0)
        assert np.all(p >= 0)


def test_noise_increases_entropy():
    theta = np.zeros(4)  # ideal |00>, near-deterministic
    low = NISQCircuitSUT(noise=0.0).outcome_distribution(theta)
    high = NISQCircuitSUT(noise=0.5).outcome_distribution(theta)
    ent = lambda p: -np.sum(p * np.log2(p + 1e-12))
    assert ent(high) > ent(low)


def test_predict_arity_and_symbols():
    sut = NISQCircuitSUT()
    scheme = quantum_partition_scheme()
    out = sut.predict(np.zeros(4)).mode()
    assert len(out) == sut.q == 4
    sym = scheme.apply(out)
    assert len(sym) == 4


def test_quantum_universe_size():
    scheme = quantum_partition_scheme()
    assert scheme.levels == (4, 3, 3, 3)
    assert count_all_tuples(scheme.symbol_space, 2) == 63


def test_quantum_feasibility_nonempty():
    sut = NISQCircuitSUT(noise=0.08)
    scheme = quantum_partition_scheme()
    probes = quantum_probe_thetas(1000, seed=0)
    feas = EmpiricalFeasibility(scheme, 2).analyze(sut, probes)
    assert feas.n_feasible_outputs > 5
    assert feas.n_feasible_tuples > 10
    # exemplars recorded for warm start
    assert len(feas.exemplars) == feas.n_feasible_outputs


def test_search_space_bounds():
    space = quantum_search_space(4)
    assert space.dim == 4
    assert np.allclose(space.lower, 0.0)
    assert np.allclose(space.upper, 2 * np.pi)


@pytest.mark.slow
def test_exp02_rnwise_beats_random_quantum():
    from experiments.exp02_multidomain import run_domain

    r = run_domain("quantum", seed=0, n_probes=1500, max_iter=80)
    rows = {row["method"].split(" [")[0].strip(): row for row in r["results"]}
    rn = next(v for k, v in rows.items() if k.startswith("Reverse"))
    assert rn["OCov_s"] >= 0.98
    assert rn["OCov_s"] > rows["Random (matched)"]["OCov_s"]
    assert r["realisation_rate"] == 1.0
