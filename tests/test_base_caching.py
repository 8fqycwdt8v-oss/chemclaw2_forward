"""Tests for caching behaviour in the predictor base class.

These use a fake in-memory predictor so no model dependencies are required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("diskcache")

from chemclaw2_forward import cache as cache_mod  # noqa: E402
from chemclaw2_forward import config as config_mod  # noqa: E402
from chemclaw2_forward.predictors.base import BaseForwardPredictor  # noqa: E402
from chemclaw2_forward.schemas import ForwardPrediction  # noqa: E402


class _FakeForward(BaseForwardPredictor):
    name = "fake_forward"
    description = "fake"

    def __init__(self, outputs: list[list[ForwardPrediction]]) -> None:
        super().__init__()
        self._outputs = outputs
        self.call_count = 0

    def load(self) -> None:  # no-op
        pass

    def predict_sync(self, reactants: str, top_k: int) -> list[ForwardPrediction]:
        out = self._outputs[min(self.call_count, len(self._outputs) - 1)]
        self.call_count += 1
        return out


@pytest.fixture()
def fresh_cache(tmp_path: Path, monkeypatch):
    """Point the cache + settings singletons at a temp dir with caching on."""
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CACHE_ENABLED", "true")
    config_mod.reset_settings_for_tests()
    cache_mod.reset_cache_for_tests()
    yield
    config_mod.reset_settings_for_tests()
    cache_mod.reset_cache_for_tests()


async def test_empty_result_is_not_cached(fresh_cache):
    """An empty prediction list must not be cached: a transient soft-failure
    should not silently disable the predictor for the whole TTL."""
    empty: list[ForwardPrediction] = []
    good = [ForwardPrediction(product_smiles="CCO", score=0.9, rank=1, source_model="fake_forward")]
    predictor = _FakeForward(outputs=[empty, good])

    first = await predictor.predict("CCO", 3)
    assert first == []
    # Second call must re-invoke predict_sync (not serve a cached empty list).
    second = await predictor.predict("CCO", 3)
    assert [p.product_smiles for p in second] == ["CCO"]
    assert predictor.call_count == 2


async def test_nonempty_result_is_cached(fresh_cache):
    """A non-empty result IS cached: the second call is served without
    re-invoking predict_sync."""
    good = [ForwardPrediction(product_smiles="CCO", score=0.9, rank=1, source_model="fake_forward")]
    predictor = _FakeForward(outputs=[good, []])  # 2nd output would be empty if recomputed

    first = await predictor.predict("CCO", 3)
    assert [p.product_smiles for p in first] == ["CCO"]
    second = await predictor.predict("CCO", 3)
    assert [p.product_smiles for p in second] == ["CCO"]  # served from cache
    assert predictor.call_count == 1
