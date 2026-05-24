"""Tests for the shared LLM prompt / parsing helpers."""

from __future__ import annotations

import json

import pytest

from chemclaw2_forward.predictors.llm_prompts import (
    build_conditions_prompt,
    build_forward_prompt,
    parse_json_payload,
    select_examples,
)


def test_select_examples_returns_at_most_n():
    examples = select_examples("CCO", n=2)
    assert len(examples) <= 2


def test_select_examples_zero_returns_empty():
    assert select_examples("CCO", n=0) == []


def test_build_forward_prompt_includes_reactants_and_schema():
    system, user = build_forward_prompt("CCO.CC(=O)Cl", top_k=3, n_examples=0)
    assert "expert organic chemist" in system.lower()
    assert "CCO.CC(=O)Cl" in user
    assert '"predictions"' in user
    assert "3" in user  # top_k


def test_build_conditions_prompt_includes_both_sides():
    system, user = build_conditions_prompt("CCO.CC(=O)O", "CCOC(C)=O", top_k=2, n_examples=0)
    assert "CCO.CC(=O)O" in user
    assert "CCOC(C)=O" in user
    assert "catalysts" in user
    assert "temperature_c" in user


def test_parse_json_bare():
    payload = parse_json_payload('{"predictions": [{"product_smiles": "CCO", "score": 0.9}]}')
    assert payload["predictions"][0]["product_smiles"] == "CCO"


def test_parse_json_in_fenced_block():
    text = "Here's the result:\n```json\n{\"predictions\": [{\"product_smiles\": \"CCO\"}]}\n```\n"
    payload = parse_json_payload(text)
    assert payload["predictions"][0]["product_smiles"] == "CCO"


def test_parse_json_in_prose():
    text = "Some preamble {\"predictions\": [{\"product_smiles\": \"CCO\"}]} trailing"
    payload = parse_json_payload(text)
    assert payload["predictions"][0]["product_smiles"] == "CCO"


def test_parse_json_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json_payload("definitely not JSON at all")


def test_parse_json_handles_nested_braces():
    obj = {"predictions": [{"data": {"nested": True}}, {"data": {"x": 1}}]}
    text = f"prefix {json.dumps(obj)} suffix"
    parsed = parse_json_payload(text)
    assert parsed == obj
