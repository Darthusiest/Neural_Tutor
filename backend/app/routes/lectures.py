"""Read-only lecture catalog and keyword retrieval API."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request
from flask_login import login_required

from app.extensions import limiter
from app.services.lecture_data import (
    get_lecture_summary,
    list_topics_catalog,
    search_lecture_chunks,
)
from app.utils.security import parse_request_json

bp = Blueprint("lectures", __name__)

_MAX_TOP_K = 20


@bp.route("/topics", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def list_topics():
    return jsonify(list_topics_catalog())


@bp.route("/<int:lecture_number>/summary", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def lecture_summary(lecture_number: int):
    data = get_lecture_summary(lecture_number)
    if data is None:
        return jsonify({"error": "lecture not found"}), 404
    return jsonify(data)


@bp.route("/retrieve", methods=["POST"])
@login_required
@limiter.limit("90 per minute")
def lecture_retrieve():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    q = (data.get("query") or "").strip()
    if not q:
        return jsonify({"error": "query required"}), 400

    raw_k = data.get("top_k", 5)
    try:
        top_k = int(raw_k)
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400
    top_k = max(1, min(top_k, _MAX_TOP_K))

    backend = (data.get("backend") or "keyword").strip().lower()
    if backend not in ("keyword", "embedding", "hybrid"):
        return jsonify({"error": "backend must be 'keyword', 'embedding', or 'hybrid'"}), 400

    try:
        if backend == "keyword":
            r = search_lecture_chunks(q, top_k=top_k, backend="keyword")
        elif backend == "embedding":
            r = search_lecture_chunks(q, top_k=top_k, backend="embedding")
        else:
            r = search_lecture_chunks(q, top_k=top_k, backend="hybrid")
    except NotImplementedError:
        return jsonify({"error": "embedding/hybrid retrieval is not implemented yet"}), 501
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    payload: dict[str, Any] = {
        "chunks": r.chunks,
        "confidence": r.confidence,
        "detected_topic": r.detected_topic,
    }
    if hasattr(r, "supporting_chunks") and r.supporting_chunks:
        payload["supporting_chunks"] = r.supporting_chunks
    if hasattr(r, "query_intent") and r.query_intent is not None:
        payload["query_type"] = r.query_intent.query_type.value
        if r.query_intent.typo_corrections:
            payload["typo_corrections"] = r.query_intent.typo_corrections
    if hasattr(r, "related_topics") and r.related_topics:
        payload["related_topics"] = r.related_topics
    return jsonify(payload)
