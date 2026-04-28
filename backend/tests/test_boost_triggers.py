"""Unit tests for Gemini boost gating (secondary explanation only)."""

from __future__ import annotations

from app.services.answers.answer_validation import ValidationResult
from app.services.generation.boost_triggers import should_use_gemini_boost
from app.services.query_understanding import QueryType


def _v(*, passed: bool, severity: str) -> ValidationResult:
    return ValidationResult(
        passed=passed,
        checks_run=["x"],
        checks_passed=[] if not passed else ["x"],
        checks_failed=[] if passed else ["x"],
        severity=severity,
    )


def test_user_toggle_runs_boost():
    use, reason = should_use_gemini_boost(
        user_query="what is softmax",
        confidence=0.99,
        validation=_v(passed=True, severity="pass"),
        confidence_threshold=0.35,
        boost_toggle=True,
        mode="chat",
        query_type=QueryType.DEFINITION,
        answer_intent="direct_definition",
        subquestion_count=1,
    )
    assert use and reason == "user_toggle"


def test_no_boost_when_toggle_off_validation_weak():
    """Boost is strictly opt-in: weak validation alone does not auto-trigger."""
    use, reason = should_use_gemini_boost(
        user_query="x",
        confidence=0.99,
        validation=_v(passed=False, severity="weak"),
        confidence_threshold=0.35,
        boost_toggle=False,
        mode="chat",
        query_type=QueryType.DEFINITION,
        answer_intent="direct_definition",
        subquestion_count=1,
    )
    assert not use and reason == "boost_disabled"


def test_no_boost_when_toggle_off_legacy_compare_mode():
    """Boost is strictly opt-in: compare/summary mode does not auto-trigger."""
    use, reason = should_use_gemini_boost(
        user_query="compare a and b",
        confidence=0.99,
        validation=None,
        confidence_threshold=0.35,
        boost_toggle=False,
        mode="compare",
        query_type=None,
        answer_intent=None,
        subquestion_count=0,
    )
    assert not use and reason == "boost_disabled"


def test_no_boost_when_toggle_off_low_confidence():
    """Boost is strictly opt-in: low confidence alone does not auto-trigger."""
    use, reason = should_use_gemini_boost(
        user_query="define softmax",
        confidence=0.05,
        validation=_v(passed=True, severity="pass"),
        confidence_threshold=0.35,
        boost_toggle=False,
        mode="chat",
        query_type=QueryType.DEFINITION,
        answer_intent="direct_definition",
        subquestion_count=1,
    )
    assert not use and reason == "boost_disabled"


def test_no_boost_when_toggle_off_pass_and_high_confidence():
    use, reason = should_use_gemini_boost(
        user_query="define softmax",
        confidence=0.99,
        validation=_v(passed=True, severity="pass"),
        confidence_threshold=0.35,
        boost_toggle=False,
        mode="chat",
        query_type=QueryType.DEFINITION,
        answer_intent="direct_definition",
        subquestion_count=1,
    )
    assert not use and reason == "boost_disabled"
