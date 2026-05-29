"""Benchmark-driven per-class trust prior calibration.

Loads a labelled reaction dataset (JSON or JSONL with records
{reactants, expected_product, [conditions]}), runs every registered forward
predictor on each reaction, partitions hits by reaction class
(`meta.classifier.classify_reaction`), and writes per-class trust priors to
`trust_priors_path` (default `~/.cache/chemclaw2_forward/trust_priors.json`).

Per-class prior for model M on class C is the smoothed top-1 accuracy of M on
the subset of records that classify to C. Used by the aggregator's MoE gating
when `Settings.use_class_priors=True`.

Run:
    python scripts/calibrate_priors.py --dataset tests/fixtures/sample_reactions.json
    python scripts/calibrate_priors.py --dataset uspto_mit_test.jsonl --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

from chemclaw2_forward.config import get_settings
from chemclaw2_forward.meta.classifier import CLASS_OTHER, classify_reaction
from chemclaw2_forward.meta.trust_priors import save_priors_file
from chemclaw2_forward.predictors import discover_predictors, list_forward
from chemclaw2_forward.preprocessing import canonical_smiles

logger = logging.getLogger(__name__)

# Laplace smoothing: avoids a class with one sample dominating with a 1.0 prior
# when it's just lucky.
SMOOTH_ALPHA = 1.0
SMOOTH_BETA = 1.0


def _load_records(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


async def _evaluate(records: list[dict], top_k: int) -> dict[str, dict[str, float]]:
    discover_predictors()
    predictors = list_forward()
    if not predictors:
        logger.error("No forward predictors registered; nothing to calibrate.")
        return {}

    # hits[class][model] = (correct, total)
    hits: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )

    for rec in records:
        try:
            expected = canonical_smiles(rec["expected_product"])
        except (KeyError, ValueError):
            continue
        klass = classify_reaction(rec["reactants"], product=rec["expected_product"])
        for p in predictors:
            try:
                preds = await p.predict(rec["reactants"], top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] failed on '%s': %r", p.name, rec["reactants"], exc)
                continue
            top1 = preds[0].product_smiles if preds else None
            hits[klass][p.name][1] += 1
            if top1 == expected:
                hits[klass][p.name][0] += 1

    # Convert (correct, total) -> smoothed accuracy
    priors: dict[str, dict[str, float]] = {}
    for klass, model_hits in hits.items():
        if klass == CLASS_OTHER:
            continue  # don't pollute "other" — it falls back to global priors anyway
        priors[klass] = {}
        for model, (correct, total) in model_hits.items():
            acc = (correct + SMOOTH_ALPHA) / (total + SMOOTH_ALPHA + SMOOTH_BETA)
            priors[klass][model] = round(acc, 4)
    return priors


def main() -> int:
    # Calibration must reflect each model's *current* behaviour, so disable the
    # prediction cache before any Settings/cache singleton is constructed —
    # otherwise stale cached predictions (from a prior model version) would skew
    # the per-class priors. Set before the first get_settings() call below.
    os.environ["CACHE_ENABLED"] = "false"

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "tests" / "fixtures" / "sample_reactions.json",
        help="JSON list or JSONL file with {reactants, expected_product} records.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to Settings.trust_priors_path.",
    )
    args = parser.parse_args()

    records = _load_records(args.dataset)
    if not records:
        logger.error("Dataset %s is empty.", args.dataset)
        return 1

    settings = get_settings()
    output_path = args.output or settings.trust_priors_path

    priors = asyncio.run(_evaluate(records, args.top_k))
    if not priors:
        logger.error("Calibration produced no priors; aborting.")
        return 1

    save_priors_file(output_path, priors)
    logger.info("Wrote per-class priors for %d classes -> %s", len(priors), output_path)
    print(json.dumps(priors, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
