"""Deterministic API mode detection and routing."""

from __future__ import annotations

import pytest

from app.services.query_mode import (
    apply_effective_api_mode,
    detect_query_mode,
    resolve_effective_mode,
)
from app.services.query_understanding import QueryType, analyze_query


def test_detect_softmax_is_chat():
    r = detect_query_mode("What is softmax?")
    assert r.mode == "chat"


def test_detect_compare_cnn_mlp():
    r = detect_query_mode("Compare CNN and MLP")
    assert r.mode == "compare"
    assert r.confidence >= 0.32


def test_detect_summarize_lecture():
    r = detect_query_mode("Summarize lecture 10")
    assert r.mode == "summary"


def test_detect_quiz_me():
    r = detect_query_mode("Quiz me on MFCCs")
    assert r.mode == "quiz"


def test_resolve_auto_uses_detection():
    det = detect_query_mode("difference between bias and variance")
    eff, overridden = resolve_effective_mode("auto", det)
    assert eff == det.mode
    assert overridden is False


def test_resolve_manual_overrides():
    det = detect_query_mode("What is softmax?")
    eff, overridden = resolve_effective_mode("compare", det)
    assert eff == "compare"
    assert overridden is True


def test_apply_effective_chat_preserves_analyze_query():
    base = analyze_query("What is softmax?")
    out = apply_effective_api_mode(base, "What is softmax?", "chat")
    assert out.query_type == base.query_type


def test_apply_effective_quiz_forces_type():
    base = analyze_query("What is softmax?")
    out = apply_effective_api_mode(base, "What is softmax?", "quiz")
    assert out.query_type == QueryType.QUIZ


@pytest.mark.parametrize(
    "raw,expected_mode",
    [
        ("Compare CNN and MLP", "compare"),
        ("COMPARE CNN AND MLP", "compare"),
        ("compare cnn vs mlp", "compare"),
        ("Quiz me on MFCCs", "quiz"),
        ("QUIZ ME ON MFCCS", "quiz"),
        ("Summarize lecture 10", "summary"),
        ("summarize LECTURE 10", "summary"),
        ("What is softmax?", "chat"),
    ],
)
def test_detect_matrix_normalizes_case(raw, expected_mode):
    r = detect_query_mode(raw)
    assert r.mode == expected_mode


def test_detect_mixed_summary_and_quiz_is_ambiguous():
    r = detect_query_mode("summarize lecture 10 and quiz me on it")
    assert r.mode == "quiz"
    assert r.ambiguous is True
    assert r.candidate_modes is not None
    assert set(r.candidate_modes) == {"quiz", "summary"}


def test_resolve_compare_overrides_summary_detection():
    det = detect_query_mode("Summarize lecture 10")
    assert det.mode == "summary"
    eff, overridden = resolve_effective_mode("compare", det)
    assert eff == "compare"
    assert overridden is True


def test_resolve_auto_chat_detection_stays_chat():
    det = detect_query_mode("What is softmax?")
    assert det.mode == "chat"
    eff, overridden = resolve_effective_mode("auto", det)
    assert eff == "chat"
    assert overridden is False
