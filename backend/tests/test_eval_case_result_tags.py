"""Canonical tags reconstructed from DB rows (CLI eval path)."""

from __future__ import annotations

import json

from app.eval.case_result_tags import canonical_failure_tags_for_row, case_raw_from_evaluation_row
from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun


def test_canonical_tags_empty_when_passed(app):
    with app.app_context():
        r = EvaluationRun(
            run_name="t",
            dataset_name="ds",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            overall_score=1.0,
            notes_json="{}",
        )
        db.session.add(r)
        db.session.flush()
        row = EvaluationCaseResult(
            evaluation_run_id=r.id,
            test_id="a",
            query_text="q",
            expected_mode="chat",
            detected_mode="chat",
            effective_mode="chat",
            expected_behavior_json=json.dumps(
                {"must_include": ["x"], "expected_mode": "chat", "category": ""}
            ),
            actual_response="has x",
            pass_bool=True,
            score=1.0,
            error_categories_json="[]",
            validation_failures_json=json.dumps(
                {"passed": True, "checks_failed": [], "flags": {}, "severity": "pass"}
            ),
            retrieval_chunk_ids_json="[]",
            latency_ms=1,
        )
        db.session.add(row)
        db.session.commit()
        assert canonical_failure_tags_for_row(row) == []


def test_canonical_tags_critic_overlay_uses_stored_categories_only():
    class _Overlay:
        is_critic_eval_overlay = True
        pass_bool = False
        error_categories_json = json.dumps(["http_429"])

    assert canonical_failure_tags_for_row(_Overlay()) == ["http_429"]


def test_canonical_tags_critic_overlay_empty_when_critic_passed():
    class _Overlay:
        is_critic_eval_overlay = True
        pass_bool = True
        error_categories_json = json.dumps(["ignored"])

    assert canonical_failure_tags_for_row(_Overlay()) == []


def test_case_raw_merges_expected_behavior(app):
    with app.app_context():
        r = EvaluationRun(
            run_name="raw",
            dataset_name="ds",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            overall_score=0.0,
            notes_json="{}",
        )
        db.session.add(r)
        db.session.flush()
        row = EvaluationCaseResult(
            evaluation_run_id=r.id,
            test_id="tid",
            query_text="hello",
            expected_mode="chat",
            detected_mode=None,
            effective_mode=None,
            expected_behavior_json=json.dumps(
                {
                    "expected_mode": "compare",
                    "must_include": ["A"],
                    "category": "compare",
                }
            ),
            actual_response="",
            pass_bool=False,
            score=0.0,
            error_categories_json="[]",
            validation_failures_json=None,
            retrieval_chunk_ids_json="[]",
            latency_ms=None,
        )
        d = case_raw_from_evaluation_row(row)
        assert d["id"] == "tid"
        assert d["query"] == "hello"
        assert d["expected_mode"] == "compare"
        assert d["must_include"] == ["A"]
