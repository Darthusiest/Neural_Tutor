from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import json

import pytest

from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun
from app.services.answers.answer_validation import ValidationResult
from app.services.eval_run import load_eval_suite, run_eval_suite, score_eval_case
from app.services.reasoning_pipeline import PipelineResult
from app.services.retrieval_v2 import EnhancedRetrievalResult


def test_evaluation_run_roundtrip(app):
    with app.app_context():
        r = EvaluationRun(
            run_name="t1",
            git_commit="abc",
            branch_name="main",
            dataset_name="l487_eval_suite:1",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            overall_score=1.0,
            notes_json="{}",
        )
        db.session.add(r)
        db.session.commit()
        cid = r.id
        c = EvaluationCaseResult(
            evaluation_run_id=cid,
            test_id="x1",
            query_text="q",
            expected_mode="chat",
            detected_mode="chat",
            effective_mode="chat",
            expected_behavior_json="{}",
            actual_response="ok",
            pass_bool=True,
            score=1.0,
            error_categories_json="[]",
            validation_failures_json="{}",
            retrieval_chunk_ids_json="[]",
            latency_ms=10,
        )
        db.session.add(c)
        db.session.commit()
        out = db.session.get(EvaluationRun, cid)
        assert out is not None
        assert out.passed_cases == 1
        assert out.case_results.count() == 1


def test_load_eval_suite_default_path():
    path = Path(__file__).resolve().parent.parent / "data" / "eval" / "l487_eval_suite.json"
    s = load_eval_suite(path)
    assert s["name"] == "l487_eval_suite"
    assert "cases" in s
    assert len(s["cases"]) >= 1


def test_score_eval_case_mode_and_strings():
    enhanced = EnhancedRetrievalResult(
        chunks=[],
        confidence=0.0,
        detected_topic=None,
        mode_routing={"detected_mode": "chat", "effective_mode": "chat"},
    )
    pr = PipelineResult(
        enhanced_result=enhanced,
        structured_query=MagicMock(),
        answer_plan=MagicMock(),
        course_answer="Softmax is a probability distribution over classes.",
        validation=ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={},
            severity="pass",
        ),
        used_llm_for_answer=False,
        primary_model="rule_based",
        query_complexity="simple",
    )
    case = {
        "expected_mode": "chat",
        "must_include": ["softmax", "probabilit"],
        "must_not_include": ["mfcc"],
    }
    out = score_eval_case(case, pr)
    assert out["pass_bool"] is True
    assert out["score"] == pytest.approx(1.0, abs=0.01)


def test_run_eval_suite_smoke(app):
    """Full pipeline; requires course chunks in Test DB (conftest has empty db)."""
    # Empty DB: pipeline may no-op; we only check persistence and no exception.
    path = Path(__file__).resolve().parent.parent / "data" / "eval" / "l487_eval_suite.json"
    with app.app_context():
        er = run_eval_suite(path, "smoke-test", top_k=3)
        assert er.id is not None
        assert er.total_cases > 0
        rows = EvaluationCaseResult.query.filter_by(evaluation_run_id=er.id).all()
        assert len(rows) == er.total_cases
        for row in rows:
            assert row.query_text
            assert row.validation_failures_json
            v = json.loads(row.validation_failures_json)
            assert "severity" in v
