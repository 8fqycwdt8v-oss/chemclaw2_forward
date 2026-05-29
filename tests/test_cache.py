"""Tests for the disk-backed prediction cache."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")
diskcache = pytest.importorskip("diskcache")

from chemclaw2_forward.cache import PredictionCache  # noqa: E402


def test_forward_cache_round_trip(tmp_path: Path):
    cache = PredictionCache(tmp_path / "fwd", enabled=True, ttl_seconds=60)
    assert cache.enabled

    payload = [{"product_smiles": "CCO", "score": 0.9, "rank": 1, "source_model": "m"}]
    cache.set_forward("m", "OCC", 3, payload)
    # Different SMILES spelling, same canonical form -> cache hit
    out = cache.get_forward("m", "CCO", 3)
    assert out == payload


def test_conditions_cache_keyed_on_both_sides(tmp_path: Path):
    cache = PredictionCache(tmp_path / "cond", enabled=True, ttl_seconds=60)
    payload = [
        {
            "catalysts": [], "solvents": ["O"], "reagents": [],
            "temperature_c": 25.0, "score": 0.7, "rank": 1, "source_model": "m",
        }
    ]
    cache.set_conditions("m", "CCO.CC(=O)O", "CCOC(C)=O", 3, payload)
    hit = cache.get_conditions("m", "CCO.CC(=O)O", "CCOC(C)=O", 3)
    assert hit == payload
    # Different product -> miss
    miss = cache.get_conditions("m", "CCO.CC(=O)O", "CC(=O)Nc1ccccc1", 3)
    assert miss is None


def test_reagent_context_distinguishes_cache_keys(tmp_path: Path):
    """A forward input that carries a reagent after '>' must NOT collide with the
    bare-reactants form — they produce different predictions (regression for the
    cache-key reagent-stripping bug)."""
    cache = PredictionCache(tmp_path / "rgt", enabled=True, ttl_seconds=60)
    with_reagent = "CC(=O)Cl.Nc1ccccc1>CCN(CC)CC>"
    no_reagent = "CC(=O)Cl.Nc1ccccc1"
    cache.set_forward("m", with_reagent, 3, [{"tag": "with_reagent"}])
    # The bare-reactants form must miss (different chemistry → different key).
    assert cache.get_forward("m", no_reagent, 3) is None
    assert cache.get_forward("m", with_reagent, 3) == [{"tag": "with_reagent"}]


def test_disabled_cache_is_noop(tmp_path: Path):
    cache = PredictionCache(tmp_path / "off", enabled=False, ttl_seconds=60)
    cache.set_forward("m", "CCO", 3, [{"x": 1}])
    assert cache.get_forward("m", "CCO", 3) is None


def test_top_k_differentiates_cache_keys(tmp_path: Path):
    cache = PredictionCache(tmp_path / "k", enabled=True, ttl_seconds=60)
    cache.set_forward("m", "CCO", 3, [{"rank": 3}])
    cache.set_forward("m", "CCO", 5, [{"rank": 5}])
    assert cache.get_forward("m", "CCO", 3) == [{"rank": 3}]
    assert cache.get_forward("m", "CCO", 5) == [{"rank": 5}]


def test_clear(tmp_path: Path):
    cache = PredictionCache(tmp_path / "c", enabled=True, ttl_seconds=60)
    cache.set_forward("m", "CCO", 3, [{"x": 1}])
    n = cache.clear()
    assert n >= 1
    assert cache.get_forward("m", "CCO", 3) is None
