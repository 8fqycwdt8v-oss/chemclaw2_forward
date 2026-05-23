"""Predictor abstract base classes."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

from ..schemas import ConditionsPrediction, ForwardPrediction

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Shared metadata for forward and conditions predictors."""

    name: str
    description: str
    citation: str | None = None
    extras_install: str | None = None

    def __init__(self) -> None:
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights / open files. Called lazily on first predict()."""

    def is_loaded(self) -> bool:
        return self._loaded


class BaseForwardPredictor(BasePredictor):
    """Given reactants (and optional agents) SMILES, predict product SMILES."""

    @abstractmethod
    def predict_sync(self, reactants: str, top_k: int) -> list[ForwardPrediction]:
        """Synchronous prediction. Override this in subclasses."""

    async def predict(self, reactants: str, top_k: int) -> list[ForwardPrediction]:
        """Async wrapper; offloads sync inference to a worker thread."""
        if not self._loaded:
            await asyncio.to_thread(self.load)
            self._loaded = True
        return await asyncio.to_thread(self.predict_sync, reactants, top_k)


class BaseConditionsPredictor(BasePredictor):
    """Given reactants + product SMILES, predict reaction conditions."""

    @abstractmethod
    def predict_sync(
        self, reactants: str, product: str, top_k: int
    ) -> list[ConditionsPrediction]:
        ...

    async def predict(
        self, reactants: str, product: str, top_k: int
    ) -> list[ConditionsPrediction]:
        if not self._loaded:
            await asyncio.to_thread(self.load)
            self._loaded = True
        return await asyncio.to_thread(self.predict_sync, reactants, product, top_k)
