"""Smoke tests: server boots, routes are wired, health/models endpoints reply."""

from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from chemclaw2_forward.server import create_app  # noqa: E402


@pytest.fixture()
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert "forward" in body
    assert "conditions" in body
    # Each model entry has required keys
    for entry in body["forward"] + body["conditions"]:
        assert "name" in entry
        assert "kind" in entry
        assert "available" in entry


def test_predict_forward_no_models_returns_503(client):
    """If no predictors are available (e.g. clean install), endpoint reports 503 cleanly."""
    r = client.post(
        "/predict/forward",
        json={"reactants": "CCO", "top_k": 3},
    )
    # Either it works (some predictor installed) or 503 (none installed). Both are valid;
    # we just want it not to crash with 500.
    assert r.status_code in {200, 503}


def test_predict_single_model_404_unknown(client):
    r = client.post(
        "/predict/forward/this_model_does_not_exist",
        json={"reactants": "CCO", "top_k": 3},
    )
    assert r.status_code == 404
