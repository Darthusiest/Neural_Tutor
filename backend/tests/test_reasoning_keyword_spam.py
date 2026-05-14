"""Regression: incoherent keyword-spam gate before retrieval-heavy answer paths."""

from __future__ import annotations

import dataclasses

from app.services.query_understanding import analyze_query
from app.services.reasoning_pipeline import (
    _blocklisted_physics_explain_without_course_anchor,
    _is_incoherent_keyword_spam,
)
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import build_structured_query


def test_adv_v3_kwspam_triggers_repeated_question_marks(app):
    with app.app_context():
        intent = analyze_query("temperature rvq attention ???")
        assert _is_incoherent_keyword_spam("temperature rvq attention ???", intent) is True


def test_plain_what_question_not_spam(app):
    with app.app_context():
        intent = analyze_query("What is softmax?")
        assert _is_incoherent_keyword_spam("What is softmax?", intent) is False


def test_short_quiz_followup_not_spam(app):
    with app.app_context():
        intent = analyze_query("Quiz me??")
        assert _is_incoherent_keyword_spam("Quiz me??", intent) is False


def test_general_relativity_explain_blocked_when_no_kb_anchor(app):
    with app.app_context():
        q = "Explain general relativity in detail."
        intent = analyze_query(q)
        sq = build_structured_query(intent, kb=get_kb(), mode_routing={})
        assert _blocklisted_physics_explain_without_course_anchor(q, sq, []) is True
        # Spurious KB concept bindings must NOT disable the physics gate unless chunks discuss GR.

        sq2 = dataclasses.replace(sq, concept_ids=["softmax"])
        assert _blocklisted_physics_explain_without_course_anchor(q, sq2, []) is True

        anchored = [
            {
                "topic": "Gravity",
                "keywords": "",
                "clean_explanation": "General relativity models gravity as curvature of spacetime.",
                "source_excerpt": "",
            },
        ]
        assert _blocklisted_physics_explain_without_course_anchor(q, sq2, anchored) is False

