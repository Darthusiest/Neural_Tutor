"""Eval suite JSON loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.dataset import EvalCase, case_is_critical_from_behavior, load_eval_dataset

_SUITE = Path(__file__).resolve().parent.parent / "data" / "eval" / "l487_eval_suite.json"


def test_load_eval_dataset_roundtrip():
    meta, cases = load_eval_dataset(_SUITE)
    assert meta.get("name") == "l487_eval_suite"
    assert len(cases) >= 5
    assert all(isinstance(c, EvalCase) for c in cases)
    crit = [c for c in cases if c.critical]
    assert any(c.id == "md_softmax_001" for c in crit)


def test_load_eval_dataset_missing_cases(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"name": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="cases"):
        load_eval_dataset(p)


def test_load_eval_dataset_missing_id(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"cases": [{"query": "q"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        load_eval_dataset(p)


def test_case_is_critical_from_behavior():
    assert case_is_critical_from_behavior({"critical": True})
    assert case_is_critical_from_behavior({"error_tags": ["critical", "x"]})
    assert not case_is_critical_from_behavior({"error_tags": ["mode"]})
