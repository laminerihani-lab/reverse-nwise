"""Tests for the vision and NLP benchmarks and the multi-domain pipeline (exp02)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")


# --- vision ----------------------------------------------------------------


def test_vision_scheme_and_sizing():
    from benchmarks.vision.vision_sut import vision_partition_scheme
    from rnwise.oca import count_all_tuples

    scheme = vision_partition_scheme()
    assert scheme.q == 5
    assert scheme.levels == (3, 3, 3, 3, 3)
    assert count_all_tuples(scheme.symbol_space, 2) == 90


@pytest.mark.slow
def test_vision_pipeline_full_coverage():
    from benchmarks.vision.vision_sut import (
        VisionSUT,
        load_vision,
        train_mlp,
        vision_partition_scheme,
        vision_probe_latents,
        vision_search_space,
    )
    from rnwise.covering_array import ExhaustiveFeasibleGenerator
    from rnwise.feasibility import EmpiricalFeasibility
    from rnwise.inverse_map import InverseMapper
    from rnwise.metrics import output_coverage_from_rows
    from rnwise.optimizers.jaya import Jaya

    data = load_vision(seed=0)
    sut = VisionSUT(model=train_mlp(data, seed=0), data=data)
    scheme = vision_partition_scheme()
    space = vision_search_space(data)
    probes = vision_probe_latents(data, 1500)
    feas = EmpiricalFeasibility(scheme, 2).analyze(sut, probes)
    assert feas.n_feasible_outputs > 10
    oca = ExhaustiveFeasibleGenerator(scheme, 2).generate(feas, np.random.default_rng(0))
    inv = InverseMapper(
        sut=sut, scheme=scheme, space=space, optimizer=Jaya(max_iter=80), warm_start=feas.exemplars
    ).run(oca, np.random.default_rng(0))
    ocov = output_coverage_from_rows(
        inv.realised_abstract_outputs(), scheme.symbol_space, 2, feas.feasible_tuples
    )
    assert inv.realisation_rate == 1.0
    assert ocov >= 0.99


# --- nlp -------------------------------------------------------------------


def test_nlp_scheme_and_sizing():
    from benchmarks.nlp.nlp_sut import nlp_partition_scheme
    from rnwise.oca import count_all_tuples

    scheme = nlp_partition_scheme()
    assert scheme.q == 5
    assert scheme.levels == (3, 3, 3, 3, 3)
    assert count_all_tuples(scheme.symbol_space, 2) == 90


@pytest.mark.slow
def test_nlp_pipeline_full_coverage():
    from benchmarks.nlp.nlp_sut import (
        NLPSUT,
        load_nlp,
        nlp_partition_scheme,
        nlp_probe_embeddings,
        nlp_search_space,
        train_text_clf,
    )
    from rnwise.covering_array import ExhaustiveFeasibleGenerator
    from rnwise.feasibility import EmpiricalFeasibility
    from rnwise.inverse_map import InverseMapper
    from rnwise.metrics import output_coverage_from_rows
    from rnwise.optimizers.jaya import Jaya

    data = load_nlp(seed=0)
    sut = NLPSUT(model=train_text_clf(data, seed=0), data=data)
    scheme = nlp_partition_scheme()
    space = nlp_search_space(data)
    probes = nlp_probe_embeddings(data, 1500)
    feas = EmpiricalFeasibility(scheme, 2).analyze(sut, probes)
    assert feas.n_feasible_outputs > 5
    oca = ExhaustiveFeasibleGenerator(scheme, 2).generate(feas, np.random.default_rng(0))
    inv = InverseMapper(
        sut=sut, scheme=scheme, space=space, optimizer=Jaya(max_iter=80), warm_start=feas.exemplars
    ).run(oca, np.random.default_rng(0))
    ocov = output_coverage_from_rows(
        inv.realised_abstract_outputs(), scheme.symbol_space, 2, feas.feasible_tuples
    )
    assert inv.realisation_rate == 1.0
    assert ocov >= 0.99


# --- multi-domain driver ----------------------------------------------------


@pytest.mark.slow
def test_exp02_all_domains_dominate_random():
    from experiments.exp02_multidomain import run

    res = run(["quantum", "vision", "nlp"], seed=0, n_probes=1200, max_iter=60)
    for dom, r in res["domains"].items():
        rows = {row["method"].split(" [")[0].strip(): row for row in r["results"]}
        rn = next(v for k, v in rows.items() if k.startswith("Reverse"))
        rand = rows["Random (matched)"]
        assert rn["OCov_s"] >= 0.98, f"{dom}: RNWise OCov_s too low"
        assert rn["OCov_s"] > rand["OCov_s"], f"{dom}: RNWise did not beat random"
