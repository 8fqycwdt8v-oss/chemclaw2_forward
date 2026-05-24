"""Tests for the ClaudeForward and ClaudeConditions predictors with mocked SDK."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

pytest.importorskip("rdkit")


def _install_fake_anthropic(monkeypatch):
    """Install a fake `anthropic` module in sys.modules so the predictor's
    `import anthropic` succeeds without network deps. Returns the MagicMock
    client created by `anthropic.Anthropic(api_key=...)`."""
    fake_client = MagicMock()
    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
    return fake_client


def _make_message(text: str):
    block = types.SimpleNamespace(text=text)
    return types.SimpleNamespace(content=[block])


def test_claude_forward_parses_json_response(monkeypatch):
    client = _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    # Import lazily so the patched anthropic is what the module sees.
    from chemclaw2_forward.predictors.forward.claude import ClaudeForward

    predictor = ClaudeForward()
    client.messages.create.return_value = _make_message(
        '{"predictions": [{"product_smiles": "CC(=O)Nc1ccccc1", "score": 0.92}, '
        '{"product_smiles": "CCO", "score": 0.4}]}'
    )

    preds = predictor.predict_sync.__wrapped__(predictor, "CC(=O)Cl.Nc1ccccc1", 5) \
        if hasattr(predictor.predict_sync, "__wrapped__") else None
    # No wrapping in our base; just call directly after load.
    predictor.load()
    preds = predictor.predict_sync("CC(=O)Cl.Nc1ccccc1", 5)
    assert len(preds) == 2
    assert preds[0].product_smiles == "CC(=O)Nc1ccccc1"
    assert preds[0].rank == 1
    assert 0.9 <= preds[0].score <= 1.0
    assert preds[0].source_model == "claude"


def test_claude_forward_handles_unparseable_response(monkeypatch):
    client = _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from chemclaw2_forward.predictors.forward.claude import ClaudeForward

    predictor = ClaudeForward()
    client.messages.create.return_value = _make_message("the model refuses to comply")
    predictor.load()
    preds = predictor.predict_sync("CC(=O)Cl.Nc1ccccc1", 5)
    assert preds == []


def test_claude_forward_skips_invalid_smiles(monkeypatch):
    client = _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from chemclaw2_forward.predictors.forward.claude import ClaudeForward

    predictor = ClaudeForward()
    client.messages.create.return_value = _make_message(
        '{"predictions": [{"product_smiles": "totally_invalid", "score": 0.99}, '
        '{"product_smiles": "CCO", "score": 0.5}]}'
    )
    predictor.load()
    preds = predictor.predict_sync("CCO", 5)
    # Only the valid SMILES survives canonicalisation
    assert len(preds) == 1
    assert preds[0].product_smiles == "CCO"


def test_claude_forward_load_fails_without_api_key(monkeypatch):
    _install_fake_anthropic(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Also make sure Settings doesn't pick up a key from .env
    monkeypatch.setattr(
        "chemclaw2_forward.config.get_settings",
        lambda: type("S", (), {"anthropic_api_key": None, "claude_model_id": "x",
                                "claude_max_tokens": 1, "claude_n_examples": 0})(),
    )

    from chemclaw2_forward.predictors.forward.claude import ClaudeForward

    predictor = ClaudeForward()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        predictor.load()


def test_claude_conditions_parses_response(monkeypatch):
    client = _install_fake_anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from chemclaw2_forward.predictors.conditions.claude import ClaudeConditions

    predictor = ClaudeConditions()
    client.messages.create.return_value = _make_message(
        '{"predictions": [{"catalysts": [], "solvents": ["ClCCl"], '
        '"reagents": ["CCN(CC)CC"], "temperature_c": 0, "score": 0.95}]}'
    )
    predictor.load()
    preds = predictor.predict_sync("CC(=O)Cl.Nc1ccccc1", "CC(=O)Nc1ccccc1", 5)
    assert len(preds) == 1
    p = preds[0]
    assert p.solvents == ["ClCCl"]
    assert p.reagents == ["CCN(CC)CC"]
    assert p.temperature_c == 0.0
    assert p.source_model == "claude"
