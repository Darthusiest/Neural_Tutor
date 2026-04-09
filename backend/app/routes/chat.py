import json

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required
from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db, limiter
from app.models import ChatSession, Feedback, Message
from app.services.chat_orchestrator import handle_chat_turn
from app.utils.security import parse_request_json

bp = Blueprint("chat", __name__)


def _require_user_session(sid: int) -> ChatSession | None:
    return ChatSession.query.filter_by(id=sid, user_id=current_user.id).first()


@bp.route("/sessions", methods=["GET"])
@login_required
def list_sessions():
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = max(int(request.args.get("offset", 0)), 0)
    q = ChatSession.query.filter_by(user_id=current_user.id).order_by(
        ChatSession.updated_at.desc()
    )
    total = q.count()
    rows = q.offset(offset).limit(limit).all()
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
    return jsonify({"sessions": out, "total": total, "limit": limit, "offset": offset})


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
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = max(int(request.args.get("offset", 0)), 0)
    q = Message.query.filter_by(session_id=s.id).order_by(Message.created_at.asc())
    total = q.count()
    msgs = q.offset(offset).limit(limit).all()
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
    return jsonify({"messages": out, "total": total, "limit": limit, "offset": offset})


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

    try:
        out = handle_chat_turn(s, text, boost_toggle, mode)
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("chat commit failed")
        return jsonify({"error": "failed to save chat turn"}), 500

    return jsonify(out)


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

    if "helpfulness_rating" in data:
        val = data["helpfulness_rating"]
        if isinstance(val, int) and 1 <= val <= 5:
            fb.helpfulness_rating = val
    if "resolved" in data:
        fb.resolved = bool(data["resolved"])
    if "follow_up_required" in data:
        fb.follow_up_required = bool(data["follow_up_required"])
    if "follow_up_type" in data:
        fb.follow_up_type = data["follow_up_type"]
    if "explicit_confusion_flag" in data:
        fb.explicit_confusion_flag = bool(data["explicit_confusion_flag"])
    if "feedback_note" in data:
        note = (data["feedback_note"] or "").strip()
        fb.feedback_note = note[:2000] if note else None
    if "preference_strength" in data:
        fb.preference_strength = data["preference_strength"]
    try:
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("feedback commit failed")
        return jsonify({"error": "could not save feedback"}), 500
    return jsonify({"ok": True})
