"""
Lecture corpus access for the LING 487 tutor (v2: multi-strategy retrieval).

- **Catalog:** ``list_topics_catalog``, ``get_lecture_summary``
- **Search:** ``search_lecture_chunks`` → :class:`EnhancedRetrievalResult`
  (backward-compatible superset of ``RetrievalResult``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from app.models import LectureChunk
from app.services.retrieval_v2 import EnhancedRetrievalResult, retrieve_enhanced


def _lecture_title_from_topic(topic: str) -> str:
    return topic.split("—", 1)[0].strip() if topic else ""


def list_topics_catalog() -> dict[str, Any]:
    """
    Structured topic list: one entry per lecture_number with title and section topics.

    Shape matches ``GET /api/lectures/topics`` JSON body (without the Flask wrapper).
    """
    chunks = LectureChunk.query.order_by(LectureChunk.lecture_number, LectureChunk.id).all()
    by_num: dict[int, list[LectureChunk]] = defaultdict(list)
    for c in chunks:
        by_num[c.lecture_number].append(c)

    lectures = []
    for lec_num in sorted(by_num.keys()):
        rows = by_num[lec_num]
        lectures.append(
            {
                "lecture_number": lec_num,
                "title": _lecture_title_from_topic(rows[0].topic) if rows else "",
                "chunk_count": len(rows),
                "section_topics": [r.topic for r in rows],
            }
        )
    return {"lectures": lectures}


def get_lecture_summary(lecture_number: int) -> dict[str, Any] | None:
    """
    Summary for one lecture, or None if no chunks exist.

    Matches ``GET /api/lectures/<n>/summary`` payload (without HTTP).
    """
    chunks = (
        LectureChunk.query.filter_by(lecture_number=lecture_number)
        .order_by(LectureChunk.id)
        .all()
    )
    if not chunks:
        return None
    return {
        "lecture_number": lecture_number,
        "title": _lecture_title_from_topic(chunks[0].topic),
        "chunk_count": len(chunks),
        "sections": [
            {"id": c.id, "chunk_key": c.chunk_key, "topic": c.topic} for c in chunks
        ],
    }


def search_lecture_chunks(
    query: str,
    *,
    top_k: int = 5,
    backend: Literal["keyword", "embedding", "hybrid"] = "keyword",
) -> EnhancedRetrievalResult:
    """
    Multi-strategy retrieval (v2).

    Returns :class:`EnhancedRetrievalResult`, a backward-compatible superset of
    ``RetrievalResult`` with additional fields: ``supporting_chunks``,
    ``query_intent``, ``related_topics``.
    """
    return retrieve_enhanced(query, top_k=top_k, backend=backend)
