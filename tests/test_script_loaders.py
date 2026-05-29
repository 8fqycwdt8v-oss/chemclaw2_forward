"""Tests for dataset loaders in the scripts (JSON + JSONL handling)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def benchmark_mod():
    pytest.importorskip("rdkit")
    return _load_module("_benchmark_under_test", _SCRIPTS / "benchmark.py")


def test_benchmark_loads_json_list(benchmark_mod, tmp_path: Path):
    p = tmp_path / "data.json"
    p.write_text('[{"reactants": "CCO", "expected_product": "CCO"}]')
    recs = benchmark_mod._load_records(p)
    assert recs == [{"reactants": "CCO", "expected_product": "CCO"}]


def test_benchmark_loads_jsonl(benchmark_mod, tmp_path: Path):
    """Regression: benchmark.py used to crash on the JSONL format it advertised."""
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"reactants": "CCO", "expected_product": "CCO"}\n'
        '{"reactants": "CC", "expected_product": "CC"}\n'
    )
    recs = benchmark_mod._load_records(p)
    assert len(recs) == 2
    assert recs[1]["reactants"] == "CC"


def test_benchmark_jsonl_skips_blank_lines(benchmark_mod, tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"reactants": "CCO", "expected_product": "CCO"}\n\n   \n')
    recs = benchmark_mod._load_records(p)
    assert len(recs) == 1
