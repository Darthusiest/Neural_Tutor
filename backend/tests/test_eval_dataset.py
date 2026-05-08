"""Eval suite JSON loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.dataset import (
    EvalCase,
    case_is_critical_from_behavior,
    effective_intent,
    load_eval_dataset,
)

_SUITE = Path(__file__).resolve().parent.parent / "data" / "eval" / "l487_eval_suite.json"


def test_load_eval_dataset_roundtrip():
    meta, cases = load_eval_dataset(_SUITE)
    assert meta.get("name") == "l487_eval_suite"
    assert len(cases) >= 5
    assert all(isinstance(c, EvalCase) for c in cases)
    assert all(c.intent for c in cases)
    assert {effective_intent(c) for c in cases} >= {"definition", "compare", "step_by_step"}
    crit = [c for c in cases if c.critical]
    assert any(c.id == "md_softmax_001" for c in crit)


def test_effective_intent_falls_back_to_category():
    case = EvalCase(id="x", query="q", expected_mode="chat", category="retrieval_purity")
    assert effective_intent(case) == "retrieval_grounded"
    assert effective_intent({"category": "synthesis"}) == "synthesis"
    assert effective_intent({"intent": "compare", "category": "definitions"}) == "compare"


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
