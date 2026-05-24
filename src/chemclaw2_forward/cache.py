"""Disk-backed prediction cache keyed on canonical reaction SMILES.

Wraps `diskcache.Cache` so per-predictor inference can be memoised across
server restarts. Each cache entry stores a list of Prediction dicts; the
predictor base class checks the cache before calling `predict_sync` and
writes the result on a hit-miss.

The cache is **opt-in** via `Settings.cache_enabled`. When disabled, the
class still exposes a no-op interface so the call sites don't need null
checks.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from .preprocessing import canonical_multi_smiles, canonical_smiles

logger = logging.getLogger(__name__)


def _hash_key(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:32]


def _safe_canon_reactants(s: str) -> str:
    try:
        left = s.split(">")[0]
        return canonical_multi_smiles(left)
    except ValueError:
        return s


def _safe_canon_product(s: str) -> str:
    try:
        return canonical_smiles(s)
    except ValueError:
        return s


class PredictionCache:
    """Thin wrapper over diskcache. No-ops cleanly when disabled or when the
    `diskcache` package is missing.
    """

    def __init__(self, cache_dir: Path, *, enabled: bool, ttl_seconds: int) -> None:
        self.enabled = enabled
        self.ttl = ttl_seconds
        self._impl: Any = None
        if not enabled:
            return
        try:
            import diskcache  # noqa: PLC0415

            cache_dir.mkdir(parents=True, exist_ok=True)
            self._impl = diskcache.Cache(str(cache_dir))
            logger.info("Prediction cache enabled at %s (ttl=%ds)", cache_dir, ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise diskcache (%r); cache disabled.", exc)
            self.enabled = False

    def _key_forward(self, model_name: str, reactants: str, top_k: int) -> str:
        return _hash_key(["fwd", model_name, _safe_canon_reactants(reactants), str(top_k)])

    def _key_conditions(
        self, model_name: str, reactants: str, product: str, top_k: int
    ) -> str:
        return _hash_key(
            [
                "cond",
                model_name,
                _safe_canon_reactants(reactants),
                _safe_canon_product(product),
                str(top_k),
            ]
        )

    def get_forward(
        self, model_name: str, reactants: str, top_k: int
    ) -> list[dict] | None:
        if not self.enabled or self._impl is None:
            return None
        return self._impl.get(self._key_forward(model_name, reactants, top_k))

    def set_forward(
        self, model_name: str, reactants: str, top_k: int, predictions: list[dict]
    ) -> None:
        if not self.enabled or self._impl is None:
            return
        self._impl.set(
            self._key_forward(model_name, reactants, top_k),
            predictions,
            expire=self.ttl,
        )

    def get_conditions(
        self, model_name: str, reactants: str, product: str, top_k: int
    ) -> list[dict] | None:
        if not self.enabled or self._impl is None:
            return None
        return self._impl.get(self._key_conditions(model_name, reactants, product, top_k))

    def set_conditions(
        self,
        model_name: str,
        reactants: str,
        product: str,
        top_k: int,
        predictions: list[dict],
    ) -> None:
        if not self.enabled or self._impl is None:
            return
        self._impl.set(
            self._key_conditions(model_name, reactants, product, top_k),
            predictions,
            expire=self.ttl,
        )

    def clear(self) -> int:
        if not self.enabled or self._impl is None:
            return 0
        return self._impl.clear()


_cache_singleton: PredictionCache | None = None


def get_cache() -> PredictionCache:
    global _cache_singleton
    if _cache_singleton is None:
        from .config import get_settings

        s = get_settings()
        _cache_singleton = PredictionCache(
            cache_dir=s.model_cache_dir / "predictions",
            enabled=s.cache_enabled,
            ttl_seconds=s.cache_ttl_seconds,
        )
    return _cache_singleton


def reset_cache_for_tests() -> None:
    """Force a fresh PredictionCache on next get_cache() call."""
    global _cache_singleton
    _cache_singleton = None
