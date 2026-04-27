"""Unit tests for deterministic eval scoring (no full chat pipeline)."""

from __future__ import annotations

from app.eval.dataset import EvalCase
from app.eval.scoring import score_eval_case

GOOD_MODE = {
    "detected": "chat",
    "effective": "chat",
    "confidence": 0.8,
    "signals": [],
    "overridden": False,
    "ambiguous": False,
}


def _case(**kw) -> EvalCase:
    d = {
        "id": "t",
        "query": "q",
        "expected_mode": "chat",
        "must_include": [],
        "must_not_include": [],
        "expected_sections": [],
        "forbidden_sections": [],
        "category": "",
        "error_tags": [],
        "mode": "auto",
    }
    d.update(kw)
    return EvalCase(
        id=d["id"],
        query=d["query"],
        expected_mode=d["expected_mode"],
        must_include=d["must_include"],
        must_not_include=d["must_not_include"],
        expected_sections=d["expected_sections"],
        forbidden_sections=d["forbidden_sections"],
        category=d["category"],
        error_tags=d["error_tags"],
        mode=d["mode"],
    )


def test_mode_detection_uses_detected():
    c = _case(
        id="m",
        category="mode_detection",
        expected_mode="quiz",
    )
    r = score_eval_case(
        c,
        "Quiz: Lecture 1\n",
        {**GOOD_MODE, "detected": "quiz", "effective": "quiz"},
        {"answer_mode": "teaching_plus_check", "answer_plan": {}},
    )
    assert r.mode_ok
    r2 = score_eval_case(
        c,
        "x",
        {**GOOD_MODE, "detected": "chat", "effective": "chat"},
        None,
    )
    assert not r2.mode_ok
    assert "mode_detected_mismatch" in r2.error_categories


def test_effective_mode_quarter():
    c = _case(expected_mode="compare")
    r = score_eval_case(
        c,
        "Course Answer:\nA vs B while different",
        {**GOOD_MODE, "detected": "compare", "effective": "chat"},
        {"answer_mode": "compare", "answer_plan": {}},
    )
    assert not r.mode_ok


def test_must_include_and_forbidden():
    c = _case(
        must_include=["softmax", "probabilit"],
        must_not_include=["mfcc"],
    )
    text = "Softmax is a probability map."
    r = score_eval_case(c, text, GOOD_MODE, None)
    assert r.content_ok and r.forbidden_ok
    r2 = score_eval_case(c, text + " mention mfcc", GOOD_MODE, None)
    assert not r2.forbidden_ok


def test_quiz_structure():
    c = _case(
        id="q",
        expected_mode="quiz",
    )
    good = (
        "Quiz: softmax\n\n1. Q?\n2. A) x\n   B) y\n\n"
        "Answer Key:\n\n1. z\n2. A) softmax\n"
    )
    r = score_eval_case(
        c,
        good,
        {**GOOD_MODE, "effective": "quiz", "detected": "quiz"},
        {"answer_mode": "teaching_plus_check", "answer_plan": {}},
    )
    assert r.structure_ok

    r2 = score_eval_case(
        c,
        "Quiz: x\n",
        {**GOOD_MODE, "effective": "quiz"},
        {"answer_mode": "teaching_plus_check", "answer_plan": {}},
    )
    assert not r2.structure_ok
