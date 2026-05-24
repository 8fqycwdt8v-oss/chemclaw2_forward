"""Load and save per-class model trust priors.

A trust prior is a float weight applied to each predictor's votes in the
aggregator. Higher = more trusted. Priors can be global (one value per model)
or per reaction class (one value per (class, model)).

The default global priors live in `Settings.model_trust_priors`. Per-class
priors are stored as a JSON file (default: $MODEL_CACHE_DIR/trust_priors.json)
written by `scripts/calibrate_priors.py` after running a benchmark.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .classifier import CLASS_OTHER

logger = logging.getLogger(__name__)


def load_priors_file(path: Path) -> dict[str, dict[str, float]]:
    """Load a per-class priors JSON file.

    Schema:
        {
          "amide_formation": {"reaction_t5_v2": 1.0, "molecular_transformer": 0.85, ...},
          "suzuki_coupling": {...},
          ...
        }

    Missing file returns an empty dict (no per-class overrides).
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            logger.warning("trust_priors file at %s is not a JSON object; ignoring.", path)
            return {}
        return {str(k): {str(m): float(v) for m, v in (vals or {}).items()} for k, vals in data.items()}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Could not parse %s: %s", path, exc)
        return {}


def save_priors_file(path: Path, priors: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(priors, indent=2, sort_keys=True))


def effective_prior(
    model_name: str,
    reaction_class: str | None,
    global_priors: dict[str, float],
    per_class_priors: dict[str, dict[str, float]],
    *,
    default: float = 0.5,
) -> float:
    """Pick the most specific available prior for (model, class)."""
    if reaction_class and reaction_class != CLASS_OTHER:
        class_map = per_class_priors.get(reaction_class)
        if class_map and model_name in class_map:
            return class_map[model_name]
    return global_priors.get(model_name, default)
