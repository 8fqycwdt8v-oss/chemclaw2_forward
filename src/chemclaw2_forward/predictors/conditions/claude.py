"""Anthropic Claude condition predictor.

Counterpart to ClaudeForward: given reactants + product, prompts Claude for
catalyst/solvent/reagent/temperature suggestions and parses the JSON reply.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ...schemas import ConditionsPrediction
from .. import mark_unavailable, register_conditions
from ..base import BaseConditionsPredictor
from ..llm_prompts import build_conditions_prompt, parse_json_payload

logger = logging.getLogger(__name__)


class ClaudeConditions(BaseConditionsPredictor):
    name = "claude"
    description = "Anthropic Claude (via API) prompted for reaction condition prediction."
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
                "ANTHROPIC_API_KEY not configured; ClaudeConditions disabled. "
                "Set the env var or `ANTHROPIC_API_KEY` in .env."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = settings.claude_model_id
        self._max_tokens = settings.claude_max_tokens
        self._n_examples = settings.claude_n_examples
        logger.info("ClaudeConditions configured with model=%s", self._model_id)

    def predict_sync(
        self, reactants: str, product: str, top_k: int
    ) -> list[ConditionsPrediction]:
        system, user = build_conditions_prompt(
            reactants, product, top_k=top_k, n_examples=self._n_examples
        )
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

        items = payload.get("predictions") or payload.get("conditions") or []
        preds: list[ConditionsPrediction] = []
        for i, item in enumerate(items[:top_k]):
            # Use _first_present so a literal 0 °C doesn't get clobbered by `or`.
            preds.append(
                ConditionsPrediction(
                    catalysts=_as_list(_first_present(item, "catalysts", "catalyst")),
                    solvents=_as_list(_first_present(item, "solvents", "solvent")),
                    reagents=_as_list(_first_present(item, "reagents", "reagent")),
                    temperature_c=_as_float(_first_present(item, "temperature_c", "temperature")),
                    score=_clamp_score(item.get("score"), fallback_rank=i),
                    rank=i + 1,
                    source_model=self.name,
                )
            )
        return preds


def _first_present(d: dict, *keys: str):
    """Return the value of the first key in `keys` that's present in `d` (even if it's 0 / [] / None)."""
    for k in keys:
        if k in d:
            return d[k]
    return None


def _collect_text(message: Any) -> str:
    chunks: list[str] = []
    for block in getattr(message, "content", []):
        text = getattr(block, "text", None)
        if text:
            chunks.append(text)
    return "".join(chunks)


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    return [str(x) for x in v if x is not None and str(x).strip()]


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _clamp_score(raw: Any, fallback_rank: int) -> float:
    try:
        s = float(raw)
    except (TypeError, ValueError):
        s = max(0.01, 1.0 - 0.1 * fallback_rank)
    return min(1.0, max(0.0, s))


try:
    import anthropic  # noqa: F401, PLC0415

    register_conditions(ClaudeConditions())
except Exception as exc:  # noqa: BLE001
    mark_unavailable(
        ClaudeConditions.name,
        "conditions",
        f"missing optional dep `anthropic` (install `chemclaw2_forward[claude]`): {exc!r}",
    )
