import json
import time

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required
from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db, limiter
from app.models import ChatSession, Feedback, Message, ResponseVariant, RetrievalLog
from app.services.llm import generate_boosted_explanation
from app.services.retrieval import format_course_answer, retrieve
from app.utils.security import parse_request_json

bp = Blueprint("chat", __name__)

CONFIDENCE_THRESHOLD = 0.35


def _require_user_session(sid: int) -> ChatSession | None:
    return ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()


@bp.route("/sessions", methods=["GET"])
@login_required
def list_sessions():
    rows = (
        ChatSession.query.filter_by(user_id=current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    out = []
    for s in rows:
        out.append(
            {
                "id": s.id,
                "title": s.title,
                "mode": s.mode,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
        )
    return jsonify({"sessions": out})


@bp.route("/sessions", methods=["POST"])
@login_required
@limiter.limit("45 per minute")
def create_session():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    title = (data.get("title") or "New chat").strip() or "New chat"
    mode = (data.get("mode") or "chat").strip()
    s = ChatSession(user_id=current_user.id, title=title, mode=mode)
    db.session.add(s)
    try:
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("create_session failed")
        return jsonify({"error": "could not create session"}), 500
    return jsonify({"session": {"id": s.id, "title": s.title, "mode": s.mode}}), 201


@bp.route("/sessions/<int:sid>", methods=["GET"])
@login_required
def get_session(sid: int):
    s = _require_user_session(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    return jsonify(
        {
            "session": {
                "id": s.id,
                "title": s.title,
                "mode": s.mode,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
        }
    )


@bp.route("/sessions/<int:sid>/messages", methods=["GET"])
@login_required
def list_messages(sid: int):
    s = _require_user_session(sid)
    if not s:
        return jsonify({"error": "not found"}), 404
    msgs = (
        Message.query.filter_by(session_id=s.id).order_by(Message.created_at.asc()).all()
    )
    out = []
    for m in msgs:
        item = {
            "id": m.id,
            "role": m.role,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        if m.role == "user":
            item["content"] = m.content_text or ""
        else:
            rv = m.response_variant
            item["course_answer"] = rv.course_answer if rv else ""
            item["boosted_explanation"] = rv.boosted_explanation if rv else None
            item["payload_json"] = m.payload_json
        out.append(item)
    return jsonify({"messages": out})


@bp.route("/chat", methods=["POST"])
@login_required
@limiter.limit("90 per minute")
def chat():
    """
    Retrieve lecture chunks, assemble a grounded Course Answer, optionally
    request a Boosted Explanation from the LLM layer.
    """
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    session_id = data.get("session_id")
    text = (data.get("message") or "").strip()
    boost_toggle = bool(data.get("boost_toggle"))
    mode = (data.get("mode") or "chat").strip()

    if not session_id or not text:
        return jsonify({"error": "session_id and message required"}), 400

    s = _require_user_session(int(session_id))
    if not s:
        return jsonify({"error": "session not found"}), 404

    s.mode = mode
    db.session.add(
        Message(session_id=s.id, role="user", content_text=text, payload_json=None)
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

    low_confidence = r.confidence < CONFIDENCE_THRESHOLD
    need_boost = boost_toggle or low_confidence or mode in ("compare", "summary")

    boosted = None
    boost_reason = None
    if need_boost:
        ctx = json.dumps(r.chunks) if r.chunks else "[]"
        boosted, _usage = generate_boosted_explanation(text, ctx)
        boost_reason = "toggle" if boost_toggle else ("low_confidence" if low_confidence else "mode")
        if not boosted:
            boosted = None
            boost_reason = None

    assistant = Message(
        session_id=s.id,
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
        session_id=s.id,
        message_id=assistant.id,
        user_question=text,
        detected_topic=r.detected_topic,
        retrieved_chunk_ids=json.dumps([c.get("id") for c in r.chunks]),
        confidence=r.confidence,
        latency_ms=latency_ms,
        token_usage_json=None,
    )
    db.session.add(log)
    try:
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("chat commit failed")
        return jsonify({"error": "failed to save chat turn"}), 500

    return jsonify(
        {
            "assistant_message_id": assistant.id,
            "course_answer": course_answer,
            "boosted_explanation": boosted,
            "retrieval_confidence": r.confidence,
            "boost_applied": boosted is not None,
        }
    )


@bp.route("/feedback", methods=["POST"])
@login_required
@limiter.limit("90 per minute")
def feedback():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    message_id = data.get("message_id")
    if not message_id:
        return jsonify({"error": "message_id required"}), 400
    m = db.session.get(Message, int(message_id))
    if not m or m.session.user_id != current_user.id:
        return jsonify({"error": "not found"}), 404
    fb = Feedback.query.filter_by(message_id=m.id).first()
    if not fb:
        fb = Feedback(message_id=m.id)
        db.session.add(fb)
    fb.course_thumb = data.get("course_thumb")
    fb.boost_thumb = data.get("boost_thumb")
    fb.preferred = data.get("preferred")
    try:
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("feedback commit failed")
        return jsonify({"error": "could not save feedback"}), 500
    return jsonify({"ok": True})
