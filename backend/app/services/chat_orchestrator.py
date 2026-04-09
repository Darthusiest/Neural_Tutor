"""Compose retrieval, course answer, optional boost, and persistence for one chat turn."""

from __future__ import annotations

import json
import time

from flask import current_app

from app.extensions import db
from app.models import ChatSession, Message, ResponseVariant, RetrievalLog
from app.services.llm import generate_boosted_explanation
from app.services.retrieval import format_course_answer, retrieve


def handle_chat_turn(
    session: ChatSession,
    text: str,
    boost_toggle: bool,
    mode: str,
) -> dict:
    """
    Run retrieval, build course / boosted answers, persist messages and logs.
    Caller must not commit before this; this function commits on success.
    """
    threshold = float(current_app.config.get("CONFIDENCE_THRESHOLD", 0.35))

    session.mode = mode
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

    rv = ResponseVariant(
        message_id=assistant.id,
        course_answer=course_answer,
        boosted_explanation=boosted,
        boost_reason=boost_reason,
        model_name=None,
        token_usage_json=None,
    )
    db.session.add(rv)

    log = RetrievalLog(
        session_id=session.id,
        message_id=assistant.id,
        user_question=text,
        detected_topic=r.detected_topic,
        retrieved_chunk_ids=json.dumps([c.get("id") for c in r.chunks]),
        confidence=r.confidence,
        latency_ms=latency_ms,
        token_usage_json=None,
    )
    db.session.add(log)
    db.session.commit()

    return {
        "assistant_message_id": assistant.id,
        "course_answer": course_answer,
        "boosted_explanation": boosted,
        "retrieval_confidence": r.confidence,
        "boost_applied": boosted is not None,
    }
