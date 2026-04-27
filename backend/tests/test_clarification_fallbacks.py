"""Mode-aware clarification fallbacks (Task 8).

Covers two layers:

- Direct unit tests on :func:`is_underspecified_for_mode` and
  :func:`clarification_for_mode`.
- End-to-end ``POST /api/chat`` tests verifying that obviously
  underspecified queries (``Compare these`` / ``Quiz me`` / ``Summarize
  this``) hit the orchestrator pre-check and emit the templated
  clarification text rather than running retrieval.

Also locks down the existing ``asf`` no-match path so the helpful
``varied_no_chunk_course_answer`` response keeps surfacing.
"""

from __future__ import annotations

import json

from app.extensions import db
from app.models import LectureChunk
from app.services.answers.clarification import (
    clarification_for_mode,
    is_underspecified_for_mode,
)
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import build_structured_query
from app.services.query_understanding import QueryIntent, QueryType, analyze_query
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

from tests.conftest import register_user

_PW = "Abcd1234!"


def _login(client, email: str) -> None:
    register_user(client, email, _PW)
    client.post(
        "/api/auth/login",
        json={"email": email, "password": _PW},
        content_type="application/json",
    )


def _seed_minimal_chunks(app) -> None:
    """Seed one chunk so the no-match `asf` path still routes through the
    `classify_no_match_query` branch (which only fires when retrieval
    finds zero hits — easy to ensure with a single off-topic chunk)."""
    with app.app_context():
        db.session.add(
            LectureChunk(
                chunk_key="clarify-fallback-mfcc",
                lecture_number=10,
                topic="MFCCs — Core Idea",
                keywords=json.dumps(["mfcc", "speech"]),
                source_excerpt="MFCCs summarize the spectrum of speech.",
                clean_explanation="MFCCs summarize the spectrum of speech.",
                sample_questions="[]",
                sample_answer=None,
            )
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()


def _open_chat_session(client) -> int:
    return client.post(
        "/api/sessions",
        json={"title": "t"},
        content_type="application/json",
    ).get_json()["session"]["id"]


def _post_chat(client, sid: int, message: str, **extra) -> dict:
    payload = {"session_id": sid, "message": message, **extra}
    response = client.post(
        "/api/chat",
        json=payload,
        content_type="application/json",
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


# ---------------------------------------------------------------------------
# Unit tests on the helper module
# ---------------------------------------------------------------------------


def test_is_underspecified_compare_no_entities():
    intent = analyze_query("Compare these")
    sq = build_structured_query(intent, kb=get_kb())
    assert is_underspecified_for_mode("Compare these", sq, "compare") is True


def test_is_underspecified_compare_with_entities_passes():
    intent = QueryIntent(
        query_type=QueryType.COMPARE,
        original_query="Compare CNN and MLP",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[],
        detected_concepts=[],
        compare_concepts=("CNN", "MLP"),
        compare_entities=["CNN", "MLP"],
    )
    sq = build_structured_query(intent, kb=get_kb())
    assert is_underspecified_for_mode("Compare CNN and MLP", sq, "compare") is False


def test_is_underspecified_quiz_no_lecture_or_concept():
    intent = analyze_query("Quiz me")
    sq = build_structured_query(intent, kb=get_kb())
    assert is_underspecified_for_mode("Quiz me", sq, "quiz") is True


def test_is_underspecified_quiz_with_concept_passes():
    intent = analyze_query("Quiz me on MFCCs")
    sq = build_structured_query(intent, kb=get_kb())
    assert is_underspecified_for_mode("Quiz me on MFCCs", sq, "quiz") is False


def test_is_underspecified_summary_no_lecture_or_concept():
    intent = analyze_query("Summarize this")
    sq = build_structured_query(intent, kb=get_kb())
    assert is_underspecified_for_mode("Summarize this", sq, "summary") is True


def test_is_underspecified_summary_with_lecture_passes():
    intent = analyze_query("Summarize Lecture 10")
    sq = build_structured_query(intent, kb=get_kb())
    assert is_underspecified_for_mode("Summarize Lecture 10", sq, "summary") is False


def test_is_underspecified_chat_always_false():
    intent = analyze_query("hi")
    sq = build_structured_query(intent, kb=get_kb())
    assert is_underspecified_for_mode("hi", sq, "chat") is False


def test_clarification_copy_is_mode_specific():
    sq = build_structured_query(analyze_query("Compare these"), kb=get_kb())
    compare_text = clarification_for_mode("Compare these", sq, "compare")
    quiz_text = clarification_for_mode("Quiz me", sq, "quiz")
    summary_text = clarification_for_mode("Summarize this", sq, "summary")

    assert "two concepts" in compare_text
    assert "topic" in quiz_text and "lecture" in quiz_text
    assert "lecture" in summary_text and "topic" in summary_text
    assert compare_text != quiz_text != summary_text


# ---------------------------------------------------------------------------
# Orchestrator-layer end-to-end fallbacks
# ---------------------------------------------------------------------------


def test_compare_these_asks_for_concepts(client, app):
    _login(client, "fallback-compare@test.dev")
    _seed_minimal_chunks(app)
    sid = _open_chat_session(client)
    body = _post_chat(client, sid, "Compare these", mode_override="compare")

    assert body["mode"]["effective"] == "compare"
    answer = body["answer"]
    # Must include the templated copy from clarification_for_mode("compare").
    assert "two concepts" in answer
    # Must NOT slip into Course Answer scaffolding.
    assert "### Direct Answer" not in answer
    assert "Course Answer:" not in answer


def test_quiz_me_asks_for_topic_or_lecture(client, app):
    _login(client, "fallback-quiz@test.dev")
    _seed_minimal_chunks(app)
    sid = _open_chat_session(client)
    body = _post_chat(client, sid, "Quiz me", mode_override="quiz")

    assert body["mode"]["effective"] == "quiz"
    answer = body["answer"]
    assert "topic" in answer.lower()
    assert "lecture" in answer.lower()
    # Must NOT contain Course Answer scaffolding.
    assert "### Direct Answer" not in answer
    assert "Course Answer:" not in answer


def test_summarize_this_asks_for_lecture_or_topic(client, app):
    _login(client, "fallback-summary@test.dev")
    _seed_minimal_chunks(app)
    sid = _open_chat_session(client)
    body = _post_chat(client, sid, "Summarize this", mode_override="summary")

    assert body["mode"]["effective"] == "summary"
    answer = body["answer"]
    assert "lecture" in answer.lower()
    assert "topic" in answer.lower()
    assert "### Direct Answer" not in answer
    assert "Course Answer:" not in answer


def test_asf_returns_helpful_no_match_response(client, app):
    """Random gibberish ``asf`` keeps hitting the existing varied no-match path."""
    _login(client, "fallback-asf@test.dev")
    _seed_minimal_chunks(app)
    sid = _open_chat_session(client)
    body = _post_chat(client, sid, "asf")

    answer = body["answer"]
    assert isinstance(answer, str) and len(answer.strip()) > 0
    # Should not hallucinate a Direct Answer when nothing matched.
    assert "### Direct Answer" not in answer
    # The existing varied response uses the marker "Course coverage" or
    # asks the user to rephrase — accept either as long as it's a real
    # helpful message rather than empty / boilerplate.
    helpful_markers = (
        "course",
        "rephrase",
        "ask",
        "topic",
        "lecture",
    )
    assert any(m in answer.lower() for m in helpful_markers)
