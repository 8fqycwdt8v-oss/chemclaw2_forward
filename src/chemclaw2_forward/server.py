"""FastAPI server exposing forward & condition meta-prediction as MCP tools.

Routes are auto-converted to MCP tools by fastapi-mcp, keyed on each route's
`operation_id`. ChemClaw2 connects to /mcp.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import __version__
from .cache import get_cache
from .config import get_settings
from .meta.aggregator import aggregate_conditions, aggregate_forward
from .meta.classifier import classify_reaction
from .predictors import (
    discover_predictors,
    list_conditions,
    list_forward,
    unavailable,
)
from .preprocessing import canonical_multi_smiles, canonical_smiles
from .schemas import (
    CacheClearResponse,
    ClassifyRequest,
    ClassifyResponse,
    ConditionsPrediction,
    ConditionsRequest,
    ConditionsResponse,
    ForwardPrediction,
    ForwardRequest,
    ForwardResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    discover_predictors()
    settings = get_settings()
    logger.info(
        "Forward predictors registered: %s",
        [p.name for p in list_forward()],
    )
    logger.info(
        "Conditions predictors registered: %s",
        [p.name for p in list_conditions()],
    )
    logger.info(
        "Unavailable predictors: %s",
        list(unavailable().keys()),
    )
    app.state.settings = settings
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ChemClaw2 Forward + Conditions Meta-Model",
        version=__version__,
        description=(
            "Meta-model MCP server combining open-source organic-chemistry "
            "forward-reaction and reaction-condition prediction models. "
            "Aggregates heterogeneous predictors (Molecular Transformer, T5Chem, "
            "ReactionT5, MEGAN, Chemformer, GraphRXN, Parrot, Rxn-INSIGHT, "
            "two-stage DNN, reagents-MT, ASKCOS) via Borda-weighted voting."
        ),
        lifespan=lifespan,
    )

    @app.post(
        "/predict/forward",
        response_model=ForwardResponse,
        operation_id="predict_forward_reaction",
        summary="Predict products from reactants (meta-model consensus).",
    )
    async def predict_forward(request: ForwardRequest) -> ForwardResponse:
        return await _run_forward(request)

    @app.post(
        "/predict/conditions",
        response_model=ConditionsResponse,
        operation_id="predict_reaction_conditions",
        summary="Predict catalyst, solvent, reagent, temperature for a reaction (meta-model).",
    )
    async def predict_conditions(request: ConditionsRequest) -> ConditionsResponse:
        return await _run_conditions(request)

    @app.post(
        "/predict/forward/{model_name}",
        response_model=list[ForwardPrediction],
        operation_id="predict_forward_single_model",
        summary="Query a single forward predictor (debug / introspection).",
    )
    async def predict_forward_single(model_name: str, request: ForwardRequest) -> list[ForwardPrediction]:
        return await _run_forward_single(model_name, request)

    @app.post(
        "/predict/conditions/{model_name}",
        response_model=list[ConditionsPrediction],
        operation_id="predict_conditions_single_model",
        summary="Query a single conditions predictor.",
    )
    async def predict_conditions_single(
        model_name: str, request: ConditionsRequest
    ) -> list[ConditionsPrediction]:
        return await _run_conditions_single(model_name, request)

    @app.get(
        "/models",
        response_model=ModelsResponse,
        operation_id="list_available_models",
        summary="List registered predictors and their availability.",
    )
    async def models() -> ModelsResponse:
        return _build_models_response()

    @app.get(
        "/health",
        response_model=HealthResponse,
        operation_id="health_check",
        summary="Liveness probe.",
    )
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            n_forward_models=len(list_forward()),
            n_conditions_models=len(list_conditions()),
        )

    @app.post(
        "/classify",
        response_model=ClassifyResponse,
        operation_id="classify_reaction",
        summary="Assign a coarse reaction class via SMARTS rules (used for MoE gating).",
    )
    async def classify(request: ClassifyRequest) -> ClassifyResponse:
        canon_r = _safe_canon_reactants(request.reactants)
        canon_p = _safe_canon_single(request.product) if request.product else None
        klass = classify_reaction(request.reactants, product=request.product)
        return ClassifyResponse(
            reaction_class=klass,
            canonical_reactants=canon_r,
            canonical_product=canon_p,
        )

    @app.post(
        "/cache/clear",
        response_model=CacheClearResponse,
        operation_id="clear_prediction_cache",
        summary="Drop every cached prediction result.",
    )
    async def clear_cache() -> CacheClearResponse:
        cache = get_cache()
        n = cache.clear()
        return CacheClearResponse(cleared_entries=n, enabled=cache.enabled)

    _mount_mcp(app)
    return app


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------


def _select_forward_predictors(request_models: list[str] | None):
    settings = get_settings()
    disabled = settings.parse_disabled()
    enabled = settings.parse_enabled(settings.enabled_forward_models)

    selected = []
    for p in list_forward():
        if p.name in disabled:
            continue
        if enabled is not None and p.name not in enabled:
            continue
        if request_models is not None and p.name not in request_models:
            continue
        selected.append(p)
    return selected


def _select_conditions_predictors(request_models: list[str] | None):
    settings = get_settings()
    disabled = settings.parse_disabled()
    enabled = settings.parse_enabled(settings.enabled_conditions_models)

    selected = []
    for p in list_conditions():
        if p.name in disabled:
            continue
        if enabled is not None and p.name not in enabled:
            continue
        if request_models is not None and p.name not in request_models:
            continue
        selected.append(p)
    return selected


async def _run_forward(request: ForwardRequest) -> ForwardResponse:
    settings = get_settings()
    predictors = _select_forward_predictors(request.models)
    if not predictors:
        raise HTTPException(503, "No forward predictors available with current configuration.")

    canon_reactants = _safe_canon_reactants(request.reactants)
    results = await asyncio.gather(
        *(p.predict(request.reactants, request.top_k) for p in predictors),
        return_exceptions=True,
    )

    per_model: dict[str, list[ForwardPrediction]] = {}
    succeeded = 0
    for p, res in zip(predictors, results, strict=True):
        if isinstance(res, Exception):
            logger.warning("Forward predictor %s failed: %r", p.name, res)
            continue
        per_model[p.name] = res
        succeeded += 1

    consensus = aggregate_forward(
        per_model, settings, request.top_k, reactants=request.reactants
    )
    return ForwardResponse(
        consensus=consensus,
        per_model=per_model,
        canonical_reactants=canon_reactants,
        n_models_queried=len(predictors),
        n_models_succeeded=succeeded,
    )


async def _run_conditions(request: ConditionsRequest) -> ConditionsResponse:
    settings = get_settings()
    predictors = _select_conditions_predictors(request.models)
    if not predictors:
        raise HTTPException(503, "No conditions predictors available with current configuration.")

    canon_reactants = _safe_canon_reactants(request.reactants)
    canon_product = _safe_canon_single(request.product)

    results = await asyncio.gather(
        *(p.predict(request.reactants, request.product, request.top_k) for p in predictors),
        return_exceptions=True,
    )
    per_model: dict[str, list[ConditionsPrediction]] = {}
    succeeded = 0
    for p, res in zip(predictors, results, strict=True):
        if isinstance(res, Exception):
            logger.warning("Conditions predictor %s failed: %r", p.name, res)
            continue
        per_model[p.name] = res
        succeeded += 1

    consensus = aggregate_conditions(
        per_model,
        settings,
        request.top_k,
        reactants=request.reactants,
        product=request.product,
    )
    return ConditionsResponse(
        consensus=consensus,
        per_model=per_model,
        canonical_reactants=canon_reactants,
        canonical_product=canon_product,
        n_models_queried=len(predictors),
        n_models_succeeded=succeeded,
    )


async def _run_forward_single(name: str, request: ForwardRequest) -> list[ForwardPrediction]:
    matches = [p for p in list_forward() if p.name == name]
    if not matches:
        raise HTTPException(404, f"Forward predictor not found: {name}")
    return await matches[0].predict(request.reactants, request.top_k)


async def _run_conditions_single(name: str, request: ConditionsRequest) -> list[ConditionsPrediction]:
    matches = [p for p in list_conditions() if p.name == name]
    if not matches:
        raise HTTPException(404, f"Conditions predictor not found: {name}")
    return await matches[0].predict(request.reactants, request.product, request.top_k)


def _safe_canon_reactants(s: str) -> str:
    try:
        # strip any > parts before canonicalising
        left = s.split(">")[0]
        return canonical_multi_smiles(left)
    except ValueError:
        return s


def _safe_canon_single(s: str) -> str:
    try:
        return canonical_smiles(s)
    except ValueError:
        return s


def _build_models_response() -> ModelsResponse:
    unavail = unavailable()

    fwd_infos: list[ModelInfo] = []
    for p in list_forward():
        fwd_infos.append(
            ModelInfo(
                name=p.name,
                kind="forward",
                available=True,
                description=p.description,
                citation=p.citation,
                extras_install=p.extras_install,
            )
        )
    for name, (kind, reason) in unavail.items():
        if kind != "forward":
            continue
        fwd_infos.append(
            ModelInfo(
                name=name,
                kind="forward",
                available=False,
                description="(not loaded)",
                citation=None,
                extras_install=None,
                unavailable_reason=reason,
            )
        )

    cond_infos: list[ModelInfo] = []
    for p in list_conditions():
        cond_infos.append(
            ModelInfo(
                name=p.name,
                kind="conditions",
                available=True,
                description=p.description,
                citation=p.citation,
                extras_install=p.extras_install,
            )
        )
    for name, (kind, reason) in unavail.items():
        if kind != "conditions":
            continue
        cond_infos.append(
            ModelInfo(
                name=name,
                kind="conditions",
                available=False,
                description="(not loaded)",
                citation=None,
                extras_install=None,
                unavailable_reason=reason,
            )
        )

    return ModelsResponse(forward=fwd_infos, conditions=cond_infos)


def _mount_mcp(app: FastAPI) -> None:
    """Mount fastapi-mcp on the app, exposing all routes as MCP tools at /mcp."""
    try:
        from fastapi_mcp import FastApiMCP  # noqa: PLC0415

        mcp = FastApiMCP(
            app,
            name="chemclaw2-forward",
            description=(
                "Forward reaction prediction and reaction condition prediction "
                "meta-model. Aggregates many open-source organic chemistry models."
            ),
        )
        # Use mount_http() (the modern API); fall back to mount() for older versions.
        mount = getattr(mcp, "mount_http", None) or getattr(mcp, "mount")
        mount()
    except Exception as exc:  # noqa: BLE001
        logger.warning("fastapi-mcp not available; MCP endpoint disabled (%r)", exc)


app = create_app()


def main() -> None:
    """Console-script entrypoint: launch uvicorn on configured host/port."""
    import uvicorn  # noqa: PLC0415

    settings = get_settings()
    uvicorn.run(
        "chemclaw2_forward.server:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
