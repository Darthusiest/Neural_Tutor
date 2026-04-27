"""Tests for canonical eval failure tags in ``app.services.eval_run``."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.answers.answer_validation import ValidationResult
from app.services.eval_run import (
    ERROR_CATEGORY_TAGS,
    failure_tags_for_case,
    score_eval_case,
)
from app.services.reasoning_pipeline import PipelineResult
from app.services.retrieval_v2 import EnhancedRetrievalResult


def _pr(
    *,
    answer: str,
    mode_routing: dict | None,
    chunks: list[dict],
    supporting_chunks: list[dict] | None = None,
    validation: ValidationResult | None = None,
) -> PipelineResult:
    er = EnhancedRetrievalResult(
        chunks=chunks,
        confidence=0.5,
        detected_topic=None,
        diagnostics=None,
        supporting_chunks=list(supporting_chunks or []),
        mode_routing=mode_routing,
    )
    val = validation or ValidationResult(
        passed=True,
        checks_run=[],
        checks_passed=[],
        checks_failed=[],
        flags={},
        severity="pass",
    )
    sq = MagicMock()
    plan = MagicMock()
    return PipelineResult(
        enhanced_result=er,
        structured_query=sq,
        answer_plan=plan,
        course_answer=answer,
        validation=val,
        used_llm_for_answer=False,
        primary_model="",
        query_complexity="simple",
    )


def test_error_category_tags_count():
    assert len(ERROR_CATEGORY_TAGS) == 16


def test_failure_tags_empty_on_pass():
    case = {"id": "x", "query": "q", "expected_mode": "chat", "must_include": ["ok"]}
    pr = _pr(
        answer="has ok",
        mode_routing={"detected_mode": "chat", "effective_mode": "chat"},
        chunks=[],
        validation=ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={},
            severity="pass",
        ),
    )
    assert failure_tags_for_case(case, pr, pass_bool=True) == []


def test_mode_misclassification_and_routing_failure_split():
    case = {"id": "m", "query": "q", "expected_mode": "chat", "must_include": ["x"]}
    pr = _pr(
        answer="x",
        mode_routing={"detected_mode": "quiz", "effective_mode": "summary"},
        chunks=[],
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "mode_misclassification" in tags
    assert "mode_routing_failure" in tags


def test_mode_misclassification_only_effective_matches():
    case = {"id": "m2", "query": "q", "expected_mode": "chat", "must_include": ["x"]}
    pr = _pr(
        answer="x",
        mode_routing={"detected_mode": "quiz", "effective_mode": "chat"},
        chunks=[],
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "mode_misclassification" in tags
    assert "mode_routing_failure" not in tags


def test_retrieval_leakage_without_answer_leak():
    case = {
        "id": "r",
        "query": "q",
        "must_not_include": ["mfcc"],
        "must_include": [],
    }
    pr = _pr(
        answer="clean answer",
        mode_routing=None,
        chunks=[{"clean_explanation": "discusses mfcc features", "id": 1}],
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "retrieval_leakage" in tags
    assert "forbidden_topic_leakage" not in tags


def test_forbidden_topic_leakage_in_answer():
    case = {"id": "f", "query": "q", "must_not_include": ["mfcc"], "must_include": []}
    pr = _pr(
        answer="MFCC is here",
        mode_routing=None,
        chunks=[{"clean_explanation": "other", "id": 1}],
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "forbidden_topic_leakage" in tags


def test_validation_missed_error_when_validator_passes_but_suite_fails():
    case = {"id": "v", "query": "q", "must_include": ["needle"], "expected_mode": "chat"}
    pr = _pr(
        answer="unrelated text",
        mode_routing={"detected_mode": "chat", "effective_mode": "chat"},
        chunks=[],
        validation=ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={},
            severity="pass",
        ),
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "missing_required_concept" in tags
    assert "validation_missed_error" in tags


def test_validation_check_maps_to_quiz_not_rendered():
    case = {"id": "q", "query": "q", "must_include": []}
    pr = _pr(
        answer="bad",
        mode_routing=None,
        chunks=[],
        validation=ValidationResult(
            passed=False,
            checks_run=["must_match_quiz_contract"],
            checks_passed=[],
            checks_failed=["must_match_quiz_contract"],
            flags={},
            severity="fail",
        ),
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "quiz_not_rendered" in tags


def test_unmapped_validation_check_gets_validation_missed_error():
    case = {"id": "u", "query": "q", "must_include": []}
    pr = _pr(
        answer="x",
        mode_routing=None,
        chunks=[],
        validation=ValidationResult(
            passed=False,
            checks_run=["must_be_course_grounded"],
            checks_passed=[],
            checks_failed=["must_be_course_grounded"],
            flags={},
            severity="fail",
        ),
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "validation_missed_error" in tags


def test_clarification_category_adds_tag():
    case = {
        "id": "c",
        "query": "Compare these",
        "category": "clarification",
        "must_include": [],
    }
    pr = _pr(answer="x", mode_routing=None, chunks=[])
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "clarification_missing" in tags


def test_generic_filler_flag():
    case = {"id": "g", "query": "q", "must_include": []}
    pr = _pr(
        answer="x",
        mode_routing=None,
        chunks=[],
        validation=ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={"generic_filler": True},
            severity="pass",
        ),
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "generic_filler" in tags


def test_score_eval_case_pass_has_empty_error_categories():
    case = {
        "id": "p",
        "query": "q",
        "expected_mode": "chat",
        "must_include": ["ok"],
    }
    pr = _pr(
        answer="ok",
        mode_routing={"detected_mode": "chat", "effective_mode": "chat"},
        chunks=[],
    )
    out = score_eval_case(case, pr)
    assert out["pass_bool"] is True
    assert out["error_categories"] == []


def test_compare_validation_tags():
    case = {"id": "cmp", "query": "q", "must_include": []}
    pr = _pr(
        answer="x",
        mode_routing=None,
        chunks=[],
        validation=ValidationResult(
            passed=False,
            checks_run=[],
            checks_passed=[],
            checks_failed=[
                "must_have_distinct_compare_evidence",
                "must_cover_both_sides",
            ],
            flags={},
            severity="fail",
        ),
    )
    tags = failure_tags_for_case(case, pr, pass_bool=False)
    assert "compare_entity_collapse" in tags
    assert "compare_asymmetry" in tags
