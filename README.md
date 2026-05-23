# chemclaw2_forward

A meta-model **MCP server** for **organic-chemistry forward reaction prediction** and **reaction condition prediction**, designed to plug into [ChemClaw2](#chemclaw2-integration) as an MCP service.

The server runs **many open-source reaction-prediction models in parallel** and aggregates their outputs via Borda-weighted voting. This produces richer top-K candidates and more robust rankings than any single model alone — each architecture has different blind spots (Molecular Transformer is weaker on stereochemistry, T5Chem drifts on rare reaction classes, etc.), and consensus filters them out.

---

## What's inside

### Forward reaction prediction (reactants → products)

| Predictor ID | Source | Architecture | Status |
|---|---|---|---|
| `reaction_t5_v2` | `sagawa/ReactionT5v2-forward` (HF) | T5, pretrained | Phase A — HF download, easiest install |
| `t5chem` | `HelloJocelynLu/t5chem` | T5 multi-task | Phase B — needs checkpoint |
| `molecular_transformer` | `pschwllr/MolecularTransformer` | OpenNMT-py seq2seq | Phase B — needs checkpoint + legacy torch |
| `chemformer` | `MolecularAI/MolBART` | BART | Phase B — needs fine-tuned checkpoint |
| `megan` | `molecule-one/megan` | Graph-edit attention | Phase B — needs repo + checkpoint |
| `graphrxn` | `jidushanbojue/GraphRXN` | Graph NN | Phase B — needs repo + checkpoint |

### Reaction condition prediction (reactants + product → catalyst, solvent, reagent, temperature)

| Predictor ID | Source | Approach | Status |
|---|---|---|---|
| `rxn_insight` | `mrodobbe/Rxn-INSIGHT` | Rule + similarity | Phase A — pip install, no GPU |
| `parrot` | `wangxr0526/Parrot` | Transformer | Phase B — needs checkpoint |
| `two_stage_dnn` | Chen & Li 2024 | Multi-label clf + ranker | Phase B — needs checkpoint |
| `reagents_mt` | `Academich/reagents` | Molecular Transformer fine-tune | Phase B — needs checkpoint |
| `askcos_condition` | MIT ASKCOS | NN trained on Reaxys | Phase B — needs `askcos-core` |

### Meta-model (aggregator)

Borda-weighted rank voting:
- Each candidate's weight = Σ over predictors of `model_trust_prior × predictor_score / rank`.
- Candidates ranked by weight; ties broken by vote count.
- For conditions, the voting unit is the whole `(catalysts, solvents, reagents, temp_bucket)` tuple — temperature is bucketed to 10 °C bins so trivial mismatches don't drown out agreement.
- Trust priors are seeded from published benchmarks (`ReactionT5 v2 = 1.0`, MT = 0.9, …) and are overridable via config.

### Out of scope for this branch

- **LLM-based predictor**. The plan reserves `chem_llm.py` for a future Anthropic-API-backed `ClaudePredictor`. Local chemistry LLMs (ChemDFM, ChemLLM, Chemma) were researched but intentionally not implemented here per project decision.
- **Benchmark-driven trust priors**. Current priors are seeded from published numbers; an automatic re-tuning step from `scripts/benchmark.py` results is left as future work.
- **Mixture-of-Experts gating by reaction class**. A natural Phase D extension once trust priors are calibrated.

---

## Install

The core server installs with no model dependencies:

```bash
pip install -e .
```

Then add the optional extras for the predictors you want. The Phase A pair are the lightest and recommended defaults:

```bash
pip install -e '.[reaction_t5,rxn_insight]'
```

To install every model that co-installs cleanly:

```bash
pip install -e '.[all]'
```

`molecular_transformer` and `reagents_mt` pin to legacy PyTorch via OpenNMT-py and **cannot** be co-installed with `reaction_t5` in the same env. Use a separate venv (or subprocess worker) for those.

Some predictors also need manually-downloaded checkpoints. Run:

```bash
python scripts/download_models.py
```

to warm the HuggingFace cache and print the download URLs + env vars for the rest.

---

## Run

```bash
cp .env.example .env  # edit as needed
chemclaw2-forward-server
```

The server boots on `http://0.0.0.0:8765` by default, with REST endpoints and an MCP endpoint at `/mcp`. Missing predictors are skipped at startup with a warning — the server stays up as long as at least one predictor in each category loads.

Quick sanity check:

```bash
curl -s http://localhost:8765/health | jq
curl -s http://localhost:8765/models | jq
curl -s -X POST http://localhost:8765/predict/forward \
  -H 'Content-Type: application/json' \
  -d '{"reactants": "CC(=O)Cl.Nc1ccccc1", "top_k": 5}' | jq
```

---

## REST / MCP endpoints

| HTTP | Path | MCP `operation_id` | Purpose |
|---|---|---|---|
| POST | `/predict/forward` | `predict_forward_reaction` | Meta-model forward prediction |
| POST | `/predict/conditions` | `predict_reaction_conditions` | Meta-model condition prediction |
| POST | `/predict/forward/{model}` | `predict_forward_single_model` | Query one forward predictor (debug) |
| POST | `/predict/conditions/{model}` | `predict_conditions_single_model` | Query one conditions predictor (debug) |
| GET | `/models` | `list_available_models` | Enumerate registered & unavailable predictors |
| GET | `/health` | `health_check` | Liveness probe |

All routes are auto-exposed as MCP tools by [`fastapi-mcp`](https://github.com/tadata-org/fastapi_mcp). The MCP transport is mounted at `/mcp`.

---

## ChemClaw2 integration

ChemClaw2 consumes this server as an MCP service. Two transports are supported:

### HTTP (recommended for production / shared deployments)

```bash
chemclaw2-forward-server  # default 0.0.0.0:8765
```

Point ChemClaw2 at `http://<host>:8765/mcp` in its MCP-server configuration.

### stdio (for local development)

The HTTP server can also be invoked over MCP stdio by ChemClaw2 spawning it as a subprocess; configure ChemClaw2 to launch `chemclaw2-forward-server` with appropriate environment variables.

### Discovery

ChemClaw2 should call `list_available_models` at session start to learn which predictors are alive. This lets the agent reason about disagreement, ask follow-up questions when only weak predictors are available, and short-circuit when no model can handle a query.

---

## Testing

```bash
pip install -e '.[dev]'
pytest
```

The test suite (preprocessing, aggregator, server smoke) runs without any model dependencies — all RDKit-only.

For a small live benchmark of registered predictors on a handful of canonical reactions:

```bash
python scripts/benchmark.py
```

---

## Project layout

```
src/chemclaw2_forward/
  server.py                 # FastAPI app + fastapi-mcp mount
  config.py                 # Pydantic Settings
  schemas.py                # Request / response / Prediction models
  preprocessing.py          # RDKit canonicalisation, reaction parsing
  predictors/
    base.py                 # BasePredictor ABCs
    __init__.py             # Plugin registry + auto-discovery
    forward/                # Forward-prediction model wrappers
    conditions/             # Conditions-prediction model wrappers
  meta/
    aggregator.py           # Borda-weighted voting
tests/
scripts/
  download_models.py
  benchmark.py
```

---

## References

Selected papers and repos behind each predictor (full citations live in each predictor module's `citation` field, surfaced through `GET /models`):

- ReactionT5 v2 — Sagawa & Kojima 2024, `sagawa/ReactionT5v2-forward`
- T5Chem — Lu & Zhang, J. Chem. Inf. Model. 2022, `HelloJocelynLu/t5chem`
- Molecular Transformer — Schwaller et al., ACS Cent. Sci. 2019, `pschwllr/MolecularTransformer`
- Chemformer / MolBART — Irwin et al., Mach. Learn. Sci. Technol. 2022, `MolecularAI/MolBART`
- MEGAN — Sacha et al., J. Chem. Inf. Model. 2021, `molecule-one/megan`
- GraphRXN — Yan et al., J. Cheminform. 2023, `jidushanbojue/GraphRXN`
- Rxn-INSIGHT — Rodobbe et al., J. Cheminform. 2024, `mrodobbe/Rxn-INSIGHT`
- Parrot — Wang et al., Research 2023, `wangxr0526/Parrot`
- ASKCOS condition recommender — Gao, Struble, Coley et al., ACS Cent. Sci. 2018
- Two-stage DNN — Chen & Li, J. Cheminform. 2024
- Reagents-MT — Andronov et al., `Academich/reagents`
- RXNMapper (preprocessing) — Schwaller et al., Sci. Adv. 2021, `rxn4chemistry/rxnmapper`

License: MIT.
