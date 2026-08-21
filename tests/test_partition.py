"""Unit tests for rnwise.partition: partitioners, scheme, registry."""

from __future__ import annotations

import pytest

from rnwise.partition import (
    CategoricalPassthrough,
    PartitionScheme,
    RegionPartitioner,
    ThresholdBands,
    get_partitioner,
    registered_names,
    scheme_from_spec,
)


def test_threshold_bands_basic():
    p = ThresholdBands(edges=(0.3, 0.7), labels=("low", "med", "high"))
    # Bands are left-closed on the edge: (-inf,0.3), [0.3,0.7), [0.7,+inf).
    assert p(0.1) == "low"
    assert p(0.3) == "med"  # value equal to an edge starts the new band
    assert p(0.5) == "med"
    assert p(0.7) == "high"
    assert p(0.9) == "high"
    assert p.symbols == ("low", "med", "high")


def test_threshold_bands_default_labels():
    p = ThresholdBands(edges=(1.0,))
    assert p.symbols == (0, 1)
    assert p(0.0) == 0
    assert p(2.0) == 1


def test_threshold_bands_bad_order():
    with pytest.raises(ValueError):
        ThresholdBands(edges=(0.7, 0.3))
    with pytest.raises(ValueError):
        ThresholdBands(edges=(0.3,), labels=("a", "b", "c"))


def test_categorical_passthrough():
    p = CategoricalPassthrough(alphabet=("<=50K", ">50K"))
    assert p(">50K") == ">50K"
    assert p.symbols == ("<=50K", ">50K")
    with pytest.raises(ValueError):
        p("unknown")


def test_region_partitioner():
    p = RegionPartitioner(region_fn=lambda v: "pos" if v[0] > 0 else "neg", alphabet=("pos", "neg"))
    assert p((1.0, 2.0)) == "pos"
    assert p((-1.0, 0.0)) == "neg"


def test_region_partitioner_out_of_alphabet():
    p = RegionPartitioner(region_fn=lambda v: "other", alphabet=("pos", "neg"))
    with pytest.raises(ValueError):
        p((0.0,))


def test_partition_scheme_apply():
    scheme = PartitionScheme(
        (
            CategoricalPassthrough(alphabet=("<=50K", ">50K")),
            ThresholdBands(edges=(0.5,), labels=("uncertain", "confident")),
        )
    )
    assert scheme.q == 2
    assert scheme.levels == (2, 2)
    assert scheme.apply((">50K", 0.9)) == (">50K", "confident")
    assert scheme.symbol_space == (("<=50K", ">50K"), ("uncertain", "confident"))


def test_partition_scheme_arity_mismatch():
    scheme = PartitionScheme((CategoricalPassthrough(alphabet=("a",)),))
    with pytest.raises(ValueError):
        scheme.apply(("a", "b"))


def test_registry_builtin_names():
    names = registered_names()
    assert "threshold_bands" in names
    assert "categorical" in names
    assert "region" in names


def test_get_partitioner_by_name():
    p = get_partitioner("categorical", alphabet=("a", "b"))
    assert isinstance(p, CategoricalPassthrough)
    assert p("a") == "a"


def test_scheme_from_spec():
    spec = [
        {"kind": "categorical", "alphabet": ("<=50K", ">50K")},
        {"kind": "threshold_bands", "edges": (0.5,), "labels": ("lo", "hi")},
    ]
    scheme = scheme_from_spec(spec)
    assert scheme.q == 2
    assert scheme.apply((">50K", 0.8)) == (">50K", "hi")


def test_get_partitioner_unknown():
    with pytest.raises(KeyError):
        get_partitioner("does_not_exist")
