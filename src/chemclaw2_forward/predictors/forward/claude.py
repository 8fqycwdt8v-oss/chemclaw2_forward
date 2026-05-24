"""Anthropic Claude forward predictor.

Prompts a Claude model (Sonnet 4.6 by default) with the reactants and a few
similarity-retrieved reference reactions, parses a JSON reply into
ForwardPrediction objects. Acts as one heterogeneous voter in the meta-model
alongside the transformer / graph-based predictors.

Disabled silently if ANTHROPIC_API_KEY is not set or the `anthropic` SDK
isn't installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ...preprocessing import canonical_smiles
from ...schemas import ForwardPrediction
from .. import mark_unavailable, register_forward
from ..base import BaseForwardPredictor
from ..llm_prompts import build_forward_prompt, parse_json_payload

logger = logging.getLogger(__name__)


class ClaudeForward(BaseForwardPredictor):
    name = "claude"
    description = "Anthropic Claude (via API) prompted for forward reaction prediction."
    citation = "Anthropic Claude API, chemistry few-shot prompting"
    extras_install = "claude"

    def __init__(self) -> None:
        super().__init__()
        self._client: Any = None
        self._model_id: str = ""
        self._max_tokens: int = 0
        self._n_examples: int = 0

    def load(self) -> None:
        import anthropic  # noqa: PLC0415

        from ...config import get_settings

        settings = get_settings()
        api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not configured; ClaudeForward disabled. "
                "Set the env var or `ANTHROPIC_API_KEY` in .env."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = settings.claude_model_id
        self._max_tokens = settings.claude_max_tokens
        self._n_examples = settings.claude_n_examples
        logger.info("ClaudeForward configured with model=%s", self._model_id)

    def predict_sync(self, reactants: str, top_k: int) -> list[ForwardPrediction]:
        system, user = build_forward_prompt(reactants, top_k=top_k, n_examples=self._n_examples)
        msg = self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = _collect_text(msg)
        try:
            payload = parse_json_payload(text)
        except ValueError as exc:
            logger.warning("Claude returned unparseable JSON: %s", exc)
            return []

        items = payload.get("predictions") or payload.get("products") or []
        preds: list[ForwardPrediction] = []
        for i, item in enumerate(items[:top_k]):
            # SMILES strings can be "" if the LLM whiffs; `or`-chain handles
            # that as desired (skip), but we still want explicit truthiness for
            # legibility. The 0-vs-falsy issue doesn't apply to strings here.
            smi = item.get("product_smiles") or item.get("smiles") or item.get("product")
            if not smi:
                continue
            try:
                product = canonical_smiles(str(smi))
            except ValueError:
                continue
            score = _clamp_score(item.get("score"), fallback_rank=i)
            preds.append(
                ForwardPrediction(
                    product_smiles=product,
                    score=score,
                    rank=i + 1,
                    source_model=self.name,
                )
            )
        return preds


def _collect_text(message: Any) -> str:
    """Concatenate text blocks from an Anthropic Message object."""
    chunks: list[str] = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks)


def _clamp_score(raw: Any, fallback_rank: int) -> float:
    try:
        s = float(raw)
    except (TypeError, ValueError):
        s = max(0.01, 1.0 - 0.1 * fallback_rank)
    return min(1.0, max(0.0, s))


try:
    import anthropic  # noqa: F401, PLC0415

    # We register the predictor even without an API key — load() will raise on
    # first call, which the server reports as a per-predictor failure rather
    # than crashing the meta-model. This lets users add the key after startup.
    register_forward(ClaudeForward())
except Exception as exc:  # noqa: BLE001
    mark_unavailable(
        ClaudeForward.name,
        "forward",
        f"missing optional dep `anthropic` (install `chemclaw2_forward[claude]`): {exc!r}",
    )
