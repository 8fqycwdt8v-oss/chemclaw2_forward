"""MEGAN forward predictor.

Sacha et al. 2021 — `molecule-one/megan`. Graph-edit attention network that
models a reaction as a sequence of graph edits. Best known for retrosynthesis
but also runs in the forward direction with the same architecture.

This wrapper assumes the molecule-one/megan repo is installed (it's not on
PyPI; install via `pip install git+https://github.com/molecule-one/megan.git`)
and a pretrained checkpoint sits at $MEGAN_MODEL_PATH.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ...preprocessing import canonical_smiles
from ...schemas import ForwardPrediction
from .. import mark_unavailable, register_forward
from ..base import BaseForwardPredictor

logger = logging.getLogger(__name__)


class MeganForward(BaseForwardPredictor):
    name = "megan"
    description = "MEGAN — Molecule Edit Graph Attention Network (Sacha 2021)."
    citation = "Sacha et al., J. Chem. Inf. Model. 2021, 61, 3273 (molecule-one/megan)"
    extras_install = "megan"

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._predictor: Any = None
        self._model_path: str | None = None

    def load(self) -> None:
        from ...config import get_settings

        settings = get_settings()
        self._model_path = os.environ.get(
            "MEGAN_MODEL_PATH",
            str(settings.model_cache_dir / "megan" / "uspto_50k_forward"),
        )
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"MEGAN model directory not found at {self._model_path}. "
                "Download from https://github.com/molecule-one/megan/releases."
            )

        # MEGAN's public predict API:
        from src.model.megan import Megan  # type: ignore  # noqa: PLC0415
        from src.predict.beam_search import BeamSearch  # type: ignore  # noqa: PLC0415

        self._model = Megan.load_from_dir(self._model_path)
        self._predictor = BeamSearch(self._model)

    def predict_sync(self, reactants: str, top_k: int) -> list[ForwardPrediction]:
        results = self._predictor.beam_search(
            input_smiles=reactants,
            beam_size=max(top_k, 10),
            max_steps=16,
            direction="forward",
        )

        preds: list[ForwardPrediction] = []
        for i, hit in enumerate(results[:top_k]):
            smi = hit.get("smiles") if isinstance(hit, dict) else getattr(hit, "smiles", None)
            prob = hit.get("prob") if isinstance(hit, dict) else getattr(hit, "prob", None)
            if not smi:
                continue
            try:
                product = canonical_smiles(smi)
            except ValueError:
                continue
            score = float(prob) if prob is not None else max(0.01, 1.0 - 0.1 * i)
            preds.append(
                ForwardPrediction(
                    product_smiles=product,
                    score=min(1.0, max(0.0, score)),
                    rank=i + 1,
                    source_model=self.name,
                )
            )
        return preds


try:
    import dgl  # noqa: F401, PLC0415
    import torch  # noqa: F401, PLC0415

    register_forward(MeganForward())
except Exception as exc:  # noqa: BLE001
    mark_unavailable(
        MeganForward.name,
        "forward",
        f"missing optional deps (install `chemclaw2_forward[megan]` and the "
        f"molecule-one/megan repo + checkpoint): {exc!r}",
    )
