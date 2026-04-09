"""Compose retrieval, course answer, optional boost, and persistence for one chat turn."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from flask import current_app

from app.extensions import db
from app.models import (
    ChatSession,
    Message,
    MessageOutcome,
    ResponseVariant,
    RetrievalChunkHit,
    RetrievalLog,
)
from app.services.llm import generate_boosted_explanation
from app.services.retrieval import (
    format_course_answer,
    retrieve,
    tokenize_query_terms,
)

# Keywords that suggest the user wants a simpler/different explanation
_CLARIFY_KEYWORDS = frozenset(
    "clarify explain again confused unclear rephrase repeat elaborate".split()
)
_SIMPLER_KEYWORDS = frozenset("simpler simple easier basic dumb layman".split())
_DEEPER_KEYWORDS = frozenset("deeper detail more expand further depth".split())
_EXAMPLE_KEYWORDS = frozenset("example instance sample show demonstrate".split())


def _classify_follow_up(text: str) -> str | None:
    """Heuristic follow-up type from user message text."""
    tokens = set(tokenize_query_terms(text))
    if tokens & _SIMPLER_KEYWORDS:
        return "simpler"
    if tokens & _CLARIFY_KEYWORDS:
        return "clarify"
    if tokens & _DEEPER_KEYWORDS:
        return "deeper"
    if tokens & _EXAMPLE_KEYWORDS:
        return "example"
    return None


def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard-style overlap of query tokens between two strings."""
    ta = set(tokenize_query_terms(a))
    tb = set(tokenize_query_terms(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _populate_previous_outcome(session: ChatSession, current_user_text: str) -> None:
    """Fill in MessageOutcome for the most recent assistant message (if missing)."""
    prev_messages = (
        Message.query.filter_by(session_id=session.id)
        .order_by(Message.created_at.desc())
        .limit(3)
        .all()
    )
    if len(prev_messages) < 2:
        return

    prev_assistant = None
    prev_user_text = None
    for m in prev_messages:
        if prev_assistant is None and m.role == "assistant":
            prev_assistant = m
        elif prev_assistant is not None and m.role == "user":
            prev_user_text = m.content_text
            break

    if prev_assistant is None or prev_assistant.message_outcome is not None:
        return

    follow_up_type = _classify_follow_up(current_user_text)
    was_rephrased = False
    if prev_user_text:
        was_rephrased = _token_overlap_ratio(prev_user_text, current_user_text) > 0.6

    topic_changed = follow_up_type is None and not was_rephrased

    outcome = MessageOutcome(
        message_id=prev_assistant.id,
        had_follow_up=True,
        follow_up_count=1,
        follow_up_type=follow_up_type or ("rephrase" if was_rephrased else "new_topic"),
        was_rephrased=was_rephrased,
        user_changed_topic_after=topic_changed,
        answer_resolved=None,
    )
    db.session.add(outcome)


def _response_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def handle_chat_turn(
    session: ChatSession,
    text: str,
    boost_toggle: bool,
    mode: str,
) -> dict[str, Any]:
    """
    Run retrieval, build course / boosted answers, persist messages and logs.
    Caller must not commit before this; this function commits on success.
    """
    threshold = float(current_app.config.get("CONFIDENCE_THRESHOLD", 0.35))

    session.mode = mode

    # Retroactively populate outcome for the previous assistant message
    _populate_previous_outcome(session, text)

    db.session.add(
        Message(session_id=session.id, role="user", content_text=text, payload_json=None)
    )
    db.session.flush()

    t0 = time.perf_counter()
    r = retrieve(text)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if not r.chunks:
        course_answer = (
            "Course Answer:\nThis question appears outside the LING 487 materials "
            "I can access, or retrieval found no relevant lecture chunk. "
            "Try rephrasing using terms from the course, or ask about a specific lecture topic."
        )
    else:
        course_answer = format_course_answer(r.chunks)

    low_confidence = r.confidence < threshold
    need_boost = boost_toggle or low_confidence or mode in ("compare", "summary")

    boosted = None
    boost_reason = None
    if need_boost:
        ctx = json.dumps(r.chunks) if r.chunks else "[]"
        boosted, _usage = generate_boosted_explanation(text, ctx)
        boost_reason = (
            "toggle" if boost_toggle else ("low_confidence" if low_confidence else "mode")
        )
        if not boosted:
            boosted = None
            boost_reason = None

    assistant = Message(
        session_id=session.id,
        role="assistant",
        content_text=None,
        payload_json=json.dumps(
            {
                "course_answer": course_answer,
                "boosted_explanation": boosted,
                "confidence": r.confidence,
            }
        ),
    )
    db.session.add(assistant)
    db.session.flush()

    # --- RetrievalLog (enriched) ---
    diag = r.diagnostics
    log = RetrievalLog(
        session_id=session.id,
        message_id=assistant.id,
        user_question=text,
        normalized_query=" ".join(diag.query_tokens) if diag else None,
        query_tokens_json=json.dumps(diag.query_tokens) if diag else None,
        detected_topic=r.detected_topic,
        lecture_numbers_detected_json=(
            json.dumps(diag.lecture_numbers_detected) if diag else None
        ),
        retrieval_backend=diag.retrieval_backend if diag else "keyword",
        top_k_requested=diag.top_k_requested if diag else None,
        num_chunks_scored=diag.num_chunks_scored if diag else None,
        num_chunks_hit=diag.num_chunks_hit if diag else None,
        confidence=r.confidence,
        top_score=diag.top_score if diag else None,
        second_score=diag.second_score if diag else None,
        score_margin=diag.score_margin if diag else None,
        query_coverage=diag.query_coverage if diag else None,
        is_low_confidence=low_confidence,
        is_off_topic=len(r.chunks) == 0,
        latency_ms=latency_ms,
        token_usage_json=None,
    )
    db.session.add(log)
    db.session.flush()

    # --- RetrievalChunkHit (one per selected chunk) ---
    if diag:
        for hit in diag.chunk_hits:
            db.session.add(
                RetrievalChunkHit(
                    retrieval_log_id=log.id,
                    lecture_chunk_id=hit.chunk_id,
                    rank=hit.rank,
                    score=hit.score,
                    selected_for_answer=True,
                    token_score=hit.token_score,
                    phrase_score=hit.phrase_score,
                    lecture_bonus=hit.lecture_bonus,
                    strong_field_token_score=hit.strong_field_token_score,
                    matched_query_terms=hit.matched_query_terms,
                    phrase_events=hit.phrase_events,
                    field_scores_json=json.dumps(hit.field_scores) if hit.field_scores else None,
                )
            )

    # --- ResponseVariant (enriched) ---
    boost_used = boosted is not None
    rv = ResponseVariant(
        message_id=assistant.id,
        retrieval_log_id=log.id,
        course_answer=course_answer,
        boosted_explanation=boosted,
        boost_used=boost_used,
        boost_reason=boost_reason,
        boost_auto_triggered=(low_confidence or mode in ("compare", "summary")) and not boost_toggle,
        boost_toggle_user_selected=boost_toggle,
        model_name=None,
        provider_name=None,
        course_answer_prompt_version=None,
        boost_prompt_version=None,
        token_usage_json=None,
        course_answer_length=len(course_answer),
        boosted_answer_length=len(boosted) if boosted else None,
        response_fingerprint=_response_fingerprint(course_answer),
    )
    db.session.add(rv)
    db.session.commit()

    return {
        "assistant_message_id": assistant.id,
        "course_answer": course_answer,
        "boosted_explanation": boosted,
        "retrieval_confidence": r.confidence,
        "boost_applied": boost_used,
    }
