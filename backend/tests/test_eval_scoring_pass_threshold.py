"""CLI eval rubric: lenient pass at 0.75; forbidden remains hard-fail."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.eval.dataset import EvalCase
from app.eval.scoring import PASS_THRESHOLD, score_eval_case
from app.services.answers.answer_validation import ValidationResult
from app.services.eval_run import score_eval_case as score_eval_case_db
from app.services.reasoning_pipeline import PipelineResult
from app.services.retrieval_v2 import EnhancedRetrievalResult


def test_pass_threshold_constant():
    assert PASS_THRESHOLD == 0.75


def test_cli_score_075_structure_only_fail_passes_lenient():
    """Compare with mode+content+forbidden OK but no contrast cue → 0.75, pass OK."""
    case = EvalCase(
        id="lenient-cmp",
        query="Compare CNN and attention",
        expected_mode="compare",
        must_include=["cnn", "attention"],
        category="compare",
    )
    text = """Course Answer:

### Direct Answer
cnn handles locality; attention is global for this course.
"""
    mm = {"effective": "compare", "detected": "compare"}
    pipeline = {"validation": {"flags": {"missing_comparison_side": False}}}
    out = score_eval_case(case, text, mm, pipeline)
    assert out.score == 0.75
    assert out.mode_ok and out.content_ok and out.forbidden_ok
    assert out.structure_ok is False
    assert out.pass_ok is True


def test_cli_forbidden_leak_fails_even_at_075_aggregate():
    """must_not_include hit forces pass_ok=False."""
    case = EvalCase(
        id="forbid",
        query="What is ok?",
        expected_mode="chat",
        must_include=["ok"],
        must_not_include=["secret"],
        category="definitions",
    )
    text = """Course Answer:

### Direct Answer
ok and secret phrase
"""
    mm = {"effective": "chat", "detected": "chat"}
    out = score_eval_case(case, text, mm, {"answer_mode": "direct_definition"})
    assert out.forbidden_ok is False
    assert out.pass_ok is False


def _pr_db(answer: str, mode_routing: dict, **kwargs) -> PipelineResult:
    er = EnhancedRetrievalResult(
        chunks=[],
        confidence=0.5,
        detected_topic=None,
        diagnostics=None,
        supporting_chunks=[],
        mode_routing=mode_routing,
    )
    val = kwargs.get("validation") or ValidationResult(
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
        primary_llm_usage={},
    )


def test_db_eval_run_forbidden_hard_fails_despite_mean_score():
    case = {
        "expected_mode": "chat",
        "must_include": ["alpha"],
        "must_not_include": ["BAD"],
    }
    pr = _pr_db(
        "alpha BAD",
        {"detected_mode": "chat", "effective_mode": "chat"},
    )
    out = score_eval_case_db(case, pr)
    assert out["pass_bool"] is False
    assert "forbidden_topic_leakage" in out["error_categories"]
