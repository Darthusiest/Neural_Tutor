"""Read-only lecture catalog and keyword retrieval API."""

from __future__ import annotations

from collections import defaultdict

from flask import Blueprint, jsonify, request
from flask_login import login_required

from app.extensions import limiter
from app.models import LectureChunk
from app.services.retrieval import retrieve_chunks
from app.utils.security import parse_request_json

bp = Blueprint("lectures", __name__)

_MAX_TOP_K = 20


def _lecture_title_from_chunk(chunk: LectureChunk) -> str:
    return chunk.topic.split("—", 1)[0].strip() if chunk.topic else ""


@bp.route("/topics", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def list_topics():
    chunks = LectureChunk.query.order_by(
        LectureChunk.lecture_number, LectureChunk.id
    ).all()
    by_num: dict[int, list[LectureChunk]] = defaultdict(list)
    for c in chunks:
        by_num[c.lecture_number].append(c)

    lectures = []
    for lec_num in sorted(by_num.keys()):
        rows = by_num[lec_num]
        lectures.append(
            {
                "lecture_number": lec_num,
                "title": _lecture_title_from_chunk(rows[0]) if rows else "",
                "chunk_count": len(rows),
                "section_topics": [r.topic for r in rows],
            }
        )
    return jsonify({"lectures": lectures})


@bp.route("/<int:lecture_number>/summary", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def lecture_summary(lecture_number: int):
    chunks = (
        LectureChunk.query.filter_by(lecture_number=lecture_number)
        .order_by(LectureChunk.id)
        .all()
    )
    if not chunks:
        return jsonify({"error": "lecture not found"}), 404

    return jsonify(
        {
            "lecture_number": lecture_number,
            "title": _lecture_title_from_chunk(chunks[0]),
            "chunk_count": len(chunks),
            "sections": [{"id": c.id, "topic": c.topic} for c in chunks],
        }
    )


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
    if backend not in ("keyword", "embedding"):
        return jsonify({"error": "backend must be 'keyword' or 'embedding'"}), 400

    try:
        if backend == "keyword":
            r = retrieve_chunks(q, top_k=top_k, backend="keyword")
        else:
            r = retrieve_chunks(q, top_k=top_k, backend="embedding")
    except NotImplementedError:
        return jsonify({"error": "embedding retrieval is not implemented yet"}), 501
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify(
        {
            "chunks": r.chunks,
            "confidence": r.confidence,
            "detected_topic": r.detected_topic,
        }
    )
