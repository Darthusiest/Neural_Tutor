from __future__ import annotations

import json

from app.eval.capability_analytics import (
    build_analytics_payload,
    derive_primary_error_type,
    primary_error_type_for_row,
    retrieval_diagnostics,
    summarize_boost,
    summarize_capability,
    summarize_coverage,
    summarize_retrieval,
    summarize_structure,
    top_three_issues,
)
from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun
from app.models.content import LectureChunk


def _run() -> EvaluationRun:
    run = EvaluationRun(
        run_name="cap",
        dataset_name="ds",
        total_cases=0,
        passed_cases=0,
        failed_cases=0,
        overall_score=0.0,
        notes_json="{}",
    )
    db.session.add(run)
    db.session.flush()
    return run


def _case(
    run: EvaluationRun,
    *,
    test_id: str,
    intent: str,
    pass_bool: bool,
    actual_response: str = "Course Answer:\n\nThe key idea: softmax maps scores to probabilities.",
    must_include: list[str] | None = None,
    errors: list[str] | None = None,
    primary_error_type: str | None = None,
    chunk_ids: list[int] | None = None,
    boost_metrics: dict | None = None,
) -> EvaluationCaseResult:
    row = EvaluationCaseResult(
        evaluation_run_id=run.id,
        test_id=test_id,
        query_text=f"query {test_id}",
        expected_mode="chat",
        detected_mode="chat",
        effective_mode="chat",
        expected_behavior_json=json.dumps(
            {
                "category": "definitions",
                "intent": intent,
                "must_include": must_include or ["softmax"],
            }
        ),
        actual_response=actual_response,
        pass_bool=pass_bool,
        score=1.0 if pass_bool else 0.5,
        error_categories_json=json.dumps(errors or []),
        primary_error_type=primary_error_type,
        validation_failures_json=json.dumps({"checks_failed": [], "flags": {}}),
        retrieval_chunk_ids_json=json.dumps(chunk_ids or []),
        boost_metrics_json=json.dumps(boost_metrics) if boost_metrics else None,
        latency_ms=10,
    )
    db.session.add(row)
    db.session.flush()
    return row


def test_derive_primary_error_type_priority_ladder():
    assert (
        derive_primary_error_type(
            {"intent": "definition", "must_include": ["softmax"]},
            [],
            [],
            {},
            "Course Answer: probability",
            retrieval_text="",
        )
        == "retrieval_miss"
    )
    assert (
        derive_primary_error_type(
            {"intent": "definition", "must_not_include": ["mfcc"]},
            [],
            [],
            {},
            "MFCC leaked into the answer",
            retrieval_text="softmax",
        )
        == "hallucination"
    )
    assert (
        derive_primary_error_type(
            {"intent": "definition", "must_not_include": ["mfcc"]},
            ["retrieval_leakage"],
            [],
            {},
            "softmax only",
            retrieval_text="mfcc",
        )
        == "retrieval_noise"
    )
    assert (
        derive_primary_error_type(
            {"intent": "compare"},
            ["compare_entity_collapse"],
            [],
            {},
            "Course Answer",
        )
        == "template_misuse"
    )
    assert (
        derive_primary_error_type(
            {"intent": "definition"},
            [],
            ["structure_summary_no_header"],
            {},
            "Course Answer",
        )
        == "structure_failure"
    )
    assert (
        derive_primary_error_type({"intent": "step_by_step"}, [], [], {}, "Explain process")
        == "missing_steps"
    )
    assert derive_primary_error_type({"intent": "definition"}, [], [], {}, "short") == "shallow_explanation"


def test_capability_and_error_summaries(app):
    with app.app_context():
        run = _run()
        _case(run, test_id="a", intent="definition", pass_bool=True)
        _case(
            run,
            test_id="b",
            intent="compare",
            pass_bool=False,
            errors=["compare_entity_collapse"],
            primary_error_type="template_misuse",
        )
        cases = (
            EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id)
            .order_by(EvaluationCaseResult.test_id)
            .all()
        )

        capability = summarize_capability(cases)
        assert capability["total_cases"] == 2
        assert capability["definition_accuracy"] == 1.0
        assert capability["compare_accuracy"] == 0.0

        payload = build_analytics_payload(cases)
        assert payload["error_breakdown"]["by_error_type"]["template_misuse"]["count"] == 1
        assert top_three_issues(payload)


def test_retrieval_structure_coverage_and_boost(app):
    with app.app_context():
        run = _run()
        chunk = LectureChunk(
            chunk_key="softmax-test",
            lecture_number=1,
            topic="Softmax",
            keywords=json.dumps(["softmax", "probability"]),
            source_excerpt="Softmax converts scores to probabilities.",
            clean_explanation="Softmax converts scores to probabilities.",
        )
        db.session.add(chunk)
        db.session.flush()
        _case(
            run,
            test_id="a",
            intent="definition",
            pass_bool=True,
            chunk_ids=[chunk.id],
            boost_metrics={
                "boost_triggered": True,
                "boost_latency_ms": 25,
                "latency_without_boost_ms": 10,
                "latency_with_boost_ms": 35,
                "score_without_boost": 0.5,
                "score_with_boost": 1.0,
                "boost_improved": True,
            },
        )
        _case(
            run,
            test_id="b",
            intent="step_by_step",
            pass_bool=False,
            actual_response="No ordered steps here.",
            must_include=["softmax"],
            primary_error_type="missing_steps",
            chunk_ids=[chunk.id],
        )
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        diags = retrieval_diagnostics(cases)
        assert diags[0].top_1_correct is True
        assert summarize_retrieval(cases)["top_k_recall"] == 1.0
        assert summarize_structure(cases)["violations"]["missing_steps_format"] == 1
        assert "softmax" in summarize_coverage(cases)["under_tested_concepts"]
        assert summarize_boost(cases)["boost_added_value_rate"] == 1.0
        assert primary_error_type_for_row(cases[1]) == "missing_steps"
