"""Deterministic API mode detection and routing."""

from __future__ import annotations

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
