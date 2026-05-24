"""Shared utilities for LLM-based predictors (prompt templates, JSON parsing,
similarity-based few-shot example selection).

The predictor modules build prompts using these helpers and post the results
to whichever LLM backend they target. Currently used by the Claude predictors
(`predictors/forward/claude.py`, `predictors/conditions/claude.py`).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Few-shot example library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotExample:
    reactants: str
    product: str
    catalysts: tuple[str, ...] = ()
    solvents: tuple[str, ...] = ()
    reagents: tuple[str, ...] = ()
    temperature_c: float | None = None
    note: str = ""


# A small built-in corpus of canonical reactions covering common transformations.
# This is used for zero-shot grounding when no external index (DRFP-on-ORD) is
# configured. Curated to span polar, radical, and metal-catalysed regimes.
BUILTIN_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        reactants="CC(=O)Cl.Nc1ccccc1",
        product="CC(=O)Nc1ccccc1",
        solvents=("ClCCl",),
        reagents=("CCN(CC)CC",),  # triethylamine
        temperature_c=0.0,
        note="Schotten-Baumann amide formation: acetyl chloride + aniline -> acetanilide.",
    ),
    FewShotExample(
        reactants="CC(=O)O.CCO",
        product="CCOC(C)=O",
        reagents=("OS(=O)(=O)O",),  # sulfuric acid
        temperature_c=78.0,
        note="Fischer esterification: acetic acid + ethanol -> ethyl acetate.",
    ),
    FewShotExample(
        reactants="Brc1ccccc1.OB(O)c1ccccc1",
        product="c1ccc(-c2ccccc2)cc1",
        catalysts=("[Pd]",),  # generic Pd(0)
        solvents=("C1CCOC1", "O"),  # THF/water
        reagents=("[K+].[O-]C(=O)[O-].[K+]",),  # K2CO3
        temperature_c=80.0,
        note="Suzuki coupling: bromobenzene + phenylboronic acid -> biphenyl.",
    ),
    FewShotExample(
        reactants="CC(C)=O.[Na+].[BH4-]",
        product="CC(C)O",
        solvents=("CO",),  # methanol
        temperature_c=0.0,
        note="Carbonyl reduction with NaBH4: acetone -> 2-propanol.",
    ),
    FewShotExample(
        reactants="CC#N.O",
        product="CC(N)=O",
        reagents=("[H+]",),
        temperature_c=100.0,
        note="Nitrile partial hydrolysis: acetonitrile -> acetamide.",
    ),
    FewShotExample(
        reactants="CCBr.[Na+].[N-]=[N+]=[N-]",
        product="CCN=[N+]=[N-]",
        solvents=("CS(C)=O",),  # DMSO
        temperature_c=25.0,
        note="SN2 azidation: ethyl bromide + sodium azide -> ethyl azide.",
    ),
    FewShotExample(
        reactants="c1ccc2ccccc2c1.[O-][N+](=O)O",
        product="O=[N+]([O-])c1ccc2ccccc2c1",
        reagents=("OS(=O)(=O)O",),  # sulfuric acid
        temperature_c=0.0,
        note="Electrophilic aromatic nitration of naphthalene.",
    ),
)


def select_examples(reactants: str, n: int = 3) -> list[FewShotExample]:
    """Pick `n` few-shot examples most similar to `reactants`.

    Uses DRFP fingerprints if available, else falls back to the first `n`
    examples in the built-in corpus. Returns at most `n` examples.
    """
    if n <= 0:
        return []
    pool = list(BUILTIN_EXAMPLES)

    try:
        from drfp import DrfpEncoder  # type: ignore  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        query_rxn = f"{reactants}>>"
        pool_rxns = [f"{e.reactants}>>{e.product}" for e in pool]
        encoder = DrfpEncoder()
        query_fp = np.array(encoder.encode(query_rxn))  # shape (1, D)
        pool_fp = np.array(encoder.encode(pool_rxns))  # shape (N, D)
        # Tanimoto over binary fingerprints
        intersect = (query_fp & pool_fp).sum(axis=1)
        union = (query_fp | pool_fp).sum(axis=1)
        sims = intersect / np.maximum(union, 1)
        order = np.argsort(-sims)
        return [pool[i] for i in order[:n]]
    except Exception:  # noqa: BLE001
        return pool[:n]


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_FORWARD_SYSTEM = """You are an expert organic chemist. \
Given the reactants of a chemical reaction, predict the most likely products. \
Return only valid SMILES strings (RDKit-canonical preferred). \
Order predictions by likelihood. \
Output strictly the JSON object the user requests — no markdown, no commentary."""

_CONDITIONS_SYSTEM = """You are an expert organic chemist. \
Given the reactants and product of a chemical reaction, propose suitable \
reaction conditions (catalyst, solvent, reagent, temperature in °C). \
Use SMILES for any chemical species. Order proposals by likelihood. \
Output strictly the JSON object the user requests — no markdown, no commentary."""


def build_forward_prompt(reactants: str, top_k: int, n_examples: int = 3) -> tuple[str, str]:
    """Return (system, user) prompt strings for forward reaction prediction."""
    examples = select_examples(reactants, n=n_examples)
    ex_blocks = "\n\n".join(
        f"REACTANTS: {e.reactants}\nPRODUCT: {e.product}\nNOTE: {e.note}"
        for e in examples
    )
    user = (
        (f"Reference reactions (canonical, retrieved by similarity):\n\n{ex_blocks}\n\n"
         if ex_blocks else "")
        + f"Now predict the {top_k} most likely products for these reactants:\n"
        + f"REACTANTS: {reactants}\n\n"
        + "Respond as a JSON object with this exact shape:\n"
        + '{"predictions": [{"product_smiles": "<canonical SMILES>", "score": <float 0..1>}, ...]}\n'
        + f'The list must have at most {top_k} items, ordered by descending score.'
    )
    return _FORWARD_SYSTEM, user


def build_conditions_prompt(
    reactants: str, product: str, top_k: int, n_examples: int = 3
) -> tuple[str, str]:
    """Return (system, user) prompt strings for condition prediction."""
    examples = select_examples(reactants, n=n_examples)
    ex_blocks = "\n\n".join(
        (
            f"REACTANTS: {e.reactants}\nPRODUCT: {e.product}\n"
            f"CATALYSTS: {list(e.catalysts)}\nSOLVENTS: {list(e.solvents)}\n"
            f"REAGENTS: {list(e.reagents)}\nTEMPERATURE_C: {e.temperature_c}\n"
            f"NOTE: {e.note}"
        )
        for e in examples
    )
    user = (
        (f"Reference reactions with known conditions:\n\n{ex_blocks}\n\n" if ex_blocks else "")
        + f"Now propose the {top_k} most likely condition sets for this reaction:\n"
        + f"REACTANTS: {reactants}\nPRODUCT: {product}\n\n"
        + "Respond as a JSON object with this exact shape:\n"
        + '{"predictions": [{"catalysts": [<smiles>...], "solvents": [<smiles>...], '
        + '"reagents": [<smiles>...], "temperature_c": <float or null>, '
        + '"score": <float 0..1>}, ...]}\n'
        + f"The list must have at most {top_k} items, ordered by descending score."
    )
    return _CONDITIONS_SYSTEM, user


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def parse_json_payload(raw: str) -> dict[str, Any]:
    """Robustly extract a JSON object from an LLM response.

    Handles three forms: bare JSON, fenced ```json ... ```, and prose with an
    embedded JSON object. Raises ValueError if nothing parses.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("empty LLM response")

    # Try strict parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try fenced block
    match = _JSON_FENCE_RE.search(raw)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Last resort: scan for the first {...} balanced object
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"could not parse JSON from LLM response: {raw[:200]!r}...")


def load_corpus_jsonl(path: Path) -> list[FewShotExample]:
    """Optional helper to load a larger few-shot corpus from a JSONL file.

    Each line should be a JSON object with keys matching FewShotExample fields.
    Useful for swapping the built-in corpus for an ORD subset.
    """
    out: list[FewShotExample] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out.append(
                FewShotExample(
                    reactants=rec["reactants"],
                    product=rec["product"],
                    catalysts=tuple(rec.get("catalysts", [])),
                    solvents=tuple(rec.get("solvents", [])),
                    reagents=tuple(rec.get("reagents", [])),
                    temperature_c=rec.get("temperature_c"),
                    note=rec.get("note", ""),
                )
            )
    return out
