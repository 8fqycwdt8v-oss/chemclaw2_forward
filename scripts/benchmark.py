"""Per-model + meta-model accuracy benchmark on a small reaction sample.

Reads tests/fixtures/sample_reactions.json (or a user-supplied JSONL with
{reactants, expected_product} records) and reports top-1 / top-5 accuracy
per registered forward predictor and for the meta-model consensus.

This is a tiny smoke benchmark to validate the meta-model hypothesis (consensus
≥ best single model). For real evaluation, point --dataset at a USPTO-MIT
test split or ORD subset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from chemclaw2_forward.config import get_settings
from chemclaw2_forward.meta.aggregator import aggregate_forward
from chemclaw2_forward.predictors import discover_predictors, list_forward
from chemclaw2_forward.preprocessing import canonical_smiles


async def evaluate(records: list[dict], top_k: int) -> dict:
    discover_predictors()
    settings = get_settings()
    predictors = list_forward()
    if not predictors:
        print("No forward predictors registered.", file=sys.stderr)
        return {}

    per_model_hits = {p.name: {1: 0, 5: 0} for p in predictors}
    meta_hits = {1: 0, 5: 0}
    total = 0

    for rec in records:
        try:
            expected = canonical_smiles(rec["expected_product"])
        except ValueError:
            continue
        total += 1

        per_model_preds = {}
        for p in predictors:
            try:
                preds = await p.predict(rec["reactants"], top_k)
                per_model_preds[p.name] = preds
                tops = [pp.product_smiles for pp in preds]
                if tops and tops[0] == expected:
                    per_model_hits[p.name][1] += 1
                if expected in tops[:5]:
                    per_model_hits[p.name][5] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [{p.name}] failed on '{rec['reactants']}': {exc!r}", file=sys.stderr)

        consensus = aggregate_forward(per_model_preds, settings, top_k)
        consensus_tops = [c.product_smiles for c in consensus]
        if consensus_tops and consensus_tops[0] == expected:
            meta_hits[1] += 1
        if expected in consensus_tops[:5]:
            meta_hits[5] += 1

    return {
        "total": total,
        "per_model": {
            name: {k: v / total for k, v in hits.items()}
            for name, hits in per_model_hits.items()
        },
        "meta": {k: v / total for k, v in meta_hits.items()},
    }


def _load_records(path: Path) -> list[dict]:
    """Load a JSON list (`.json`) or JSON-lines (`.jsonl`) dataset."""
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_reactions.json",
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    records = _load_records(args.dataset)

    results = asyncio.run(evaluate(records, args.top_k))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
