"""Meta-model aggregators for forward and conditions predictions.

Strategy: Borda-style weighted rank voting.
  - For each candidate (canonical product SMILES or canonical condition tuple),
    sum a contribution from every model that ranked it: weight = prior * score * 1/rank.
  - Sort candidates by total weight, descending.
  - Ties broken by higher vote count.
  - Returned consensus_score is renormalised so the top candidate scores ~1.0.

This requires no training data and degrades gracefully when models are missing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable

from ..config import Settings
from ..preprocessing import canonical_smiles
from ..schemas import (
    AggregatedConditionsPrediction,
    AggregatedForwardPrediction,
    ConditionsPrediction,
    ForwardPrediction,
)

logger = logging.getLogger(__name__)


def _normalise_product(smiles: str) -> str:
    try:
        return canonical_smiles(smiles)
    except ValueError:
        return smiles  # leave malformed strings as-is; they'll get bottom rank by score


def aggregate_forward(
    per_model: dict[str, list[ForwardPrediction]],
    settings: Settings,
    top_k: int,
) -> list[AggregatedForwardPrediction]:
    """Borda-weighted voting across forward predictors.

    `per_model[model_name]` is that model's top-K predictions (rank 1 = best).
    """
    weights: dict[str, float] = defaultdict(float)
    voters: dict[str, set[str]] = defaultdict(set)

    for model_name, preds in per_model.items():
        prior = settings.model_trust_priors.get(model_name, 0.5)
        for p in preds:
            canon = _normalise_product(p.product_smiles)
            contribution = prior * p.score / p.rank
            weights[canon] += contribution
            voters[canon].add(model_name)

    if not weights:
        return []

    sorted_candidates = sorted(
        weights.items(),
        key=lambda kv: (-kv[1], -len(voters[kv[0]])),
    )

    max_weight = sorted_candidates[0][1] or 1.0
    aggregated: list[AggregatedForwardPrediction] = []
    for rank, (smiles, weight) in enumerate(sorted_candidates[:top_k], start=1):
        aggregated.append(
            AggregatedForwardPrediction(
                product_smiles=smiles,
                consensus_score=min(1.0, weight / max_weight),
                rank=rank,
                vote_count=len(voters[smiles]),
                contributing_models=sorted(voters[smiles]),
            )
        )
    return aggregated


def _canon_set(items: Iterable[str]) -> tuple[str, ...]:
    """Canonicalise & sort a set of SMILES strings into a hashable tuple."""
    out: list[str] = []
    for x in items:
        x = x.strip()
        if not x:
            continue
        try:
            out.append(canonical_smiles(x))
        except ValueError:
            out.append(x)
    return tuple(sorted(set(out)))


def _temperature_bucket(t: float | None) -> int | None:
    """Bucket temperature into 10 °C bins using floor division.

    floor() avoids round-half-to-even surprises near bin boundaries (where
    e.g. 25 and 28 would otherwise land in different bins). Negative
    temperatures stay correctly bucketed (-15 → -20, not -10).
    """
    import math

    if t is None:
        return None
    return int(math.floor(t / 10.0)) * 10


def aggregate_conditions(
    per_model: dict[str, list[ConditionsPrediction]],
    settings: Settings,
    top_k: int,
) -> list[AggregatedConditionsPrediction]:
    """Borda-weighted voting across condition predictors.

    Conditions are higher-dimensional than products. We treat the whole
    (catalysts, solvents, reagents, temperature_bucket) tuple as the voting unit
    so that "the same recipe" gets reinforced, and bucket temperature into 10 °C
    bins to avoid trivial mismatches drowning out agreement.
    """
    weights: dict[tuple, float] = defaultdict(float)
    voters: dict[tuple, set[str]] = defaultdict(set)
    temps_for_key: dict[tuple, list[float]] = defaultdict(list)

    for model_name, preds in per_model.items():
        prior = settings.model_trust_priors.get(model_name, 0.5)
        for p in preds:
            cats = _canon_set(p.catalysts)
            sols = _canon_set(p.solvents)
            rgs = _canon_set(p.reagents)
            tbucket = _temperature_bucket(p.temperature_c)
            key = (cats, sols, rgs, tbucket)

            contribution = prior * p.score / p.rank
            weights[key] += contribution
            voters[key].add(model_name)
            if p.temperature_c is not None:
                temps_for_key[key].append(p.temperature_c)

    if not weights:
        return []

    sorted_candidates = sorted(
        weights.items(),
        key=lambda kv: (-kv[1], -len(voters[kv[0]])),
    )

    max_weight = sorted_candidates[0][1] or 1.0
    aggregated: list[AggregatedConditionsPrediction] = []
    for rank, (key, weight) in enumerate(sorted_candidates[:top_k], start=1):
        cats, sols, rgs, _tbucket = key
        temps = temps_for_key[key]
        mean_temp = sum(temps) / len(temps) if temps else None
        aggregated.append(
            AggregatedConditionsPrediction(
                catalysts=list(cats),
                solvents=list(sols),
                reagents=list(rgs),
                temperature_c=mean_temp,
                consensus_score=min(1.0, weight / max_weight),
                rank=rank,
                vote_count=len(voters[key]),
                contributing_models=sorted(voters[key]),
            )
        )
    return aggregated
