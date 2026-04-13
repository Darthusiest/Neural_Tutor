"""Study modes: quiz, compare, summary (grounded in lecture chunks)."""

from __future__ import annotations

import json
import time
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required

from app.extensions import db, limiter
from app.models import ChatSession, Message
from app.services.generation.llm import generate_comparison_boost
from app.services.study import (
    build_quiz_next,
    build_quiz_reveal,
    normalize_compare_labels,
    run_compare,
    run_summary_by_lecture,
    run_summary_by_topic,
)
from app.utils.security import parse_request_json

bp = Blueprint("study", __name__)


def _require_session(sid: int | None) -> ChatSession | None:
    if not sid:
        return None
    return ChatSession.query.filter_by(id=int(sid), user_id=current_user.id).first()


def _persist_turn(
    session_id: int | None,
    *,
    user_line: str,
    course_answer: str,
    boosted: str | None,
    study_meta: dict[str, Any],
) -> None:
    if not session_id:
        return
    s = _require_session(session_id)
    if not s:
        return
    s.mode = study_meta.get("mode", s.mode)
    db.session.add(
        Message(session_id=s.id, role="user", content_text=user_line, payload_json=None)
    )
    payload = {
        "course_answer": course_answer,
        "boosted_explanation": boosted,
        "confidence": None,
        "study": study_meta,
    }
    db.session.add(
        Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps(payload),
        )
    )
    db.session.commit()


@bp.route("/quiz/next", methods=["POST"])
@login_required
@limiter.limit("60 per minute")
def quiz_next():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    qtype = (data.get("question_type") or "mc").strip().lower()
    if qtype not in ("mc", "short"):
        return jsonify({"error": "question_type must be 'mc' or 'short'"}), 400
    topic = (data.get("topic") or "").strip() or None
    session_id = data.get("session_id")

    out = build_quiz_next(question_type=qtype, topic=topic)
    if "error" in out:
        return jsonify(out), 400
    return jsonify(out)


@bp.route("/quiz/answer", methods=["POST"])
@login_required
@limiter.limit("90 per minute")
def quiz_answer():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    chunk_id = data.get("chunk_id")
    qtype = (data.get("question_type") or "").strip().lower()
    token = (data.get("quiz_token") or "").strip()
    user_answer = (data.get("user_answer") or "").strip() or None
    selected_index = data.get("selected_index")
    session_id = data.get("session_id")

    if not chunk_id or qtype not in ("mc", "short"):
        return jsonify({"error": "chunk_id and question_type required"}), 400

    try:
        cid = int(chunk_id)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid chunk_id"}), 400

    if qtype == "mc" and selected_index is not None:
        try:
            selected_index = int(selected_index)
        except (TypeError, ValueError):
            return jsonify({"error": "selected_index must be an integer"}), 400

    mc_options = data.get("options")
    if isinstance(mc_options, list):
        mc_options = [str(x) for x in mc_options]
    else:
        mc_options = None

    reveal = build_quiz_reveal(
        cid,
        qtype,
        quiz_token=token,
        user_answer=user_answer,
        selected_index=selected_index,
        mc_options=mc_options,
    )
    if "error" in reveal:
        return jsonify(reveal), 400

    course = reveal.get("course_answer") or ""
    extra = reveal.get("reveal") or ""
    combined = course
    if reveal.get("feedback_lines"):
        combined = f"{reveal['feedback_lines']}\n\n{extra}"
    else:
        combined = f"{course}\n\n{extra}"

    if user_answer:
        user_line = f"[Quiz {qtype}] {user_answer}"
    elif selected_index is not None:
        user_line = f"[Quiz {qtype}] selected option {selected_index}"
    else:
        user_line = f"[Quiz {qtype}] submitted"

    _persist_turn(
        int(session_id) if session_id else None,
        user_line=user_line[:4000],
        course_answer=combined,
        boosted=None,
        study_meta={"kind": "quiz", "mode": "quiz", "chunk_id": cid, "question_type": qtype},
    )

    return jsonify(
        {
            "course_answer": combined,
            "reveal": reveal,
        }
    )


@bp.route("/compare", methods=["POST"])
@login_required
@limiter.limit("45 per minute")
def compare():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    a = normalize_compare_labels(data.get("concept_a") or "")
    b = normalize_compare_labels(data.get("concept_b") or "")
    expand = bool(data.get("expand"))
    session_id = data.get("session_id")

    out = run_compare(a, b)
    if "error" in out:
        return jsonify(out), 400

    course_answer = out["course_answer"]
    boosted = None
    usage = {}
    t0 = time.perf_counter()
    if expand and current_app.config.get("OPENAI_API_KEY"):
        boosted, usage = generate_comparison_boost(a, b, course_answer)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    user_line = f"[Compare] {a} vs {b}"

    _persist_turn(
        int(session_id) if session_id else None,
        user_line=user_line,
        course_answer=course_answer,
        boosted=boosted,
        study_meta={
            "kind": "compare",
            "mode": "compare",
            "concept_a": a,
            "concept_b": b,
            "latency_ms": latency_ms,
            "token_usage": usage,
        },
    )

    return jsonify(
        {
            "course_answer": course_answer,
            "boosted_explanation": boosted,
            "confidence_a": out.get("confidence_a"),
            "confidence_b": out.get("confidence_b"),
        }
    )


@bp.route("/summary", methods=["POST"])
@login_required
@limiter.limit("45 per minute")
def summary():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    kind = (data.get("kind") or "topic").strip().lower()
    session_id = data.get("session_id")

    if kind == "lecture":
        try:
            n = int(data.get("lecture_number"))
        except (TypeError, ValueError):
            return jsonify({"error": "lecture_number required for kind=lecture"}), 400
        out = run_summary_by_lecture(n)
    else:
        topic = (data.get("topic") or "").strip()
        out = run_summary_by_topic(topic)

    if "error" in out:
        return jsonify(out), 400

    course_answer = out["course_answer"]
    user_line = (
        f"[Summary lecture {out['lecture_number']}]"
        if kind == "lecture"
        else f"[Summary topic] {out.get('topic', '')}"
    )

    _persist_turn(
        int(session_id) if session_id else None,
        user_line=user_line,
        course_answer=course_answer,
        boosted=None,
        study_meta={
            "kind": "summary",
            "mode": "summary",
            "summary_kind": kind,
            "lecture_number": out.get("lecture_number"),
            "topic": out.get("topic"),
            "confidence": out.get("confidence"),
        },
    )

    return jsonify(
        {
            "course_answer": course_answer,
            "lecture_number": out.get("lecture_number"),
            "topic": out.get("topic"),
            "confidence": out.get("confidence"),
        }
    )
