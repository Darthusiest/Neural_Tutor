"""Keyword retrieval over `lecture_chunks` (v1)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.models import LectureChunk


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    confidence: float
    detected_topic: str | None


def _tokenize_query(q: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 2]


def _lecture_numbers_mentioned(q: str) -> set[int]:
    nums = set()
    for m in re.finditer(r"(?:lecture|lec\.?|week)\s*#?\s*(\d+)", q.lower()):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"\blec\s*(\d+)\b", q.lower()):
        nums.add(int(m.group(1)))
    return nums


def _chunk_blob(row: LectureChunk) -> str:
    kws: list[str] = []
    try:
        kws = json.loads(row.keywords or "[]")
    except json.JSONDecodeError:
        kws = []
    parts = [row.topic, row.explanation, " ".join(kws)]
    return " ".join(parts).lower()


def _row_to_dict(row: LectureChunk) -> dict[str, Any]:
    return {
        "id": row.id,
        "lecture_number": row.lecture_number,
        "topic": row.topic,
        "explanation": row.explanation,
        "keywords": row.keywords,
    }


def retrieve(query: str, top_k: int = 5) -> RetrievalResult:
    rows = LectureChunk.query.order_by(LectureChunk.id).all()
    if not rows:
        return RetrievalResult(chunks=[], confidence=0.0, detected_topic=None)

    q_tokens = _tokenize_query(query)
    lec_boost = _lecture_numbers_mentioned(query)

    scores: list[tuple[float, LectureChunk]] = []
    for row in rows:
        blob = _chunk_blob(row)
        hits = 0.0
        for t in q_tokens:
            if t in blob:
                hits += 1.0
        if row.lecture_number in lec_boost:
            hits += 3.0
        scores.append((hits, row))

    scores.sort(key=lambda x: x[0], reverse=True)
    best, second = scores[0][0], scores[1][0] if len(scores) > 1 else 0.0

    if best <= 0:
        return RetrievalResult(chunks=[], confidence=0.0, detected_topic=None)

    top = [r for s, r in scores if s > 0][:top_k]
    margin = best - second

    confidence = min(1.0, best / 5.0)
    if margin >= 2:
        confidence = min(1.0, confidence + 0.12)
    if margin >= 4:
        confidence = min(1.0, confidence + 0.1)
    if not q_tokens and lec_boost:
        confidence = max(confidence, 0.45)

    detected = top[0].topic.split("—")[0].strip() if top else None
    return RetrievalResult(
        chunks=[_row_to_dict(r) for r in top],
        confidence=float(confidence),
        detected_topic=detected,
    )


def format_course_answer(chunks: list[dict[str, Any]]) -> str:
    """Build the mandatory Course Answer block from retrieved chunks only."""
    lines: list[str] = ["Course Answer:", ""]
    for c in chunks:
        num = c.get("lecture_number")
        topic = c.get("topic", "")
        expl = (c.get("explanation") or "").strip()
        lines.append(f"Lecture {num} — {topic}")
        for part in expl.split("\n"):
            part = part.strip()
            if part:
                lines.append(f"- {part}")
        lines.append("")
    return "\n".join(lines).rstrip()
