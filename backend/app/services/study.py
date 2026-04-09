"""Quiz, compare, and summary study flows grounded in lecture chunks."""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import re
from types import SimpleNamespace
from typing import Any, Literal

from flask import current_app

from app.extensions import db
from app.models import LectureChunk
from app.services.retrieval import (
    _row_to_public_dict,
    format_course_answer,
    retrieve_chunks,
)

QuizType = Literal["mc", "short"]


def _secret() -> str:
    return current_app.config.get("SECRET_KEY") or "dev-secret"


def sign_quiz_payload(chunk_id: int, qtype: str) -> str:
    """HMAC so clients cannot forge quiz answers for arbitrary chunks."""
    raw = f"{chunk_id}:{qtype}".encode()
    return hmac.new(_secret().encode(), raw, hashlib.sha256).hexdigest()[:32]


def verify_quiz_token(chunk_id: int, qtype: str, token: str) -> bool:
    if not token:
        return False
    return hmac.compare_digest(sign_quiz_payload(chunk_id, qtype), token)


def _chunk_to_public(row: LectureChunk) -> dict[str, Any]:
    d = {
        "id": row.id,
        "chunk_key": row.chunk_key,
        "lecture_number": row.lecture_number,
        "topic": row.topic,
        "keywords": row.keywords,
        "source_excerpt": row.source_excerpt,
        "clean_explanation": row.clean_explanation,
        "sample_questions": row.sample_questions,
        "sample_answer": row.sample_answer,
        "chunk_type": getattr(row, "chunk_type", None),
        "concept_family": getattr(row, "concept_family", None),
    }
    return _row_to_public_dict(d)


def _pick_chunks_for_topic(topic: str | None, *, limit: int = 80) -> list[LectureChunk]:
    q = LectureChunk.query.order_by(LectureChunk.id)
    rows = q.limit(limit).all()
    if not topic or not str(topic).strip():
        return rows
    needle = str(topic).strip().lower()
    filtered = [r for r in rows if needle in (r.topic or "").lower()]
    return filtered or rows


def _sample_questions_list(chunk: LectureChunk) -> list[str]:
    raw = chunk.sample_questions or "[]"
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return []


def build_quiz_next(
    *,
    question_type: QuizType,
    topic: str | None = None,
) -> dict[str, Any]:
    """
    One quiz step: pick a chunk and build a course-grounded question.

    - **mc**: match an excerpt to the correct section title (4 options).
    - **short**: open prompt tied to one section (reveal uses stored explanation).
    """
    pool = _pick_chunks_for_topic(topic)
    if not pool:
        return {"error": "no_lecture_material", "message": "No course chunks loaded."}

    chunk = random.choice(pool)
    pub = _chunk_to_public(chunk)

    token = sign_quiz_payload(chunk.id, question_type)

    sq = _sample_questions_list(chunk)
    excerpt = (pub.get("source_excerpt") or "")[:420].strip()
    if len((pub.get("source_excerpt") or "")) > 420:
        excerpt += "…"

    if question_type == "mc":
        others = [c for c in pool if c.id != chunk.id]
        random.shuffle(others)
        distractors: list[Any] = others[:3]
        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            attempts += 1
            extra = random.choice(pool)
            if extra.id != chunk.id and extra not in distractors:
                distractors.append(extra)
        pad = 0
        decoys = (
            "Different section (review another lecture)",
            "Unrelated course topic",
            "Placeholder distractor",
        )
        while len(distractors) < 3:
            distractors.append(SimpleNamespace(topic=decoys[pad % len(decoys)]))
            pad += 1
        options = [chunk.topic] + [d.topic for d in distractors[:3]]
        random.shuffle(options)

        if sq:
            question = (
                f"{sq[0]}\n\n"
                f"Choose the best answer based on LING 487 lecture materials."
            )
        else:
            question = (
                "Which section title best matches the following excerpt from the course?\n\n"
                f"\"{excerpt}\""
            )

        return {
            "question_type": "mc",
            "chunk_id": chunk.id,
            "quiz_token": token,
            "question": question,
            "options": options,
            "lecture_number": chunk.lecture_number,
        }

    # short answer
    if sq:
        question = (
            f"{sq[0]}\n\n"
            f"Answer briefly using ideas from the course (Lecture {chunk.lecture_number})."
        )
    else:
        question = (
            f"In your own words, what is the main idea of this section?\n\n"
            f"**{chunk.topic}**\n\n"
            f"Excerpt:\n\"{excerpt}\""
        )

    return {
        "question_type": "short",
        "chunk_id": chunk.id,
        "quiz_token": token,
        "question": question,
        "lecture_number": chunk.lecture_number,
    }


def build_quiz_reveal(
    chunk_id: int,
    question_type: QuizType,
    *,
    quiz_token: str,
    user_answer: str | None = None,
    selected_index: int | None = None,
) -> dict[str, Any]:
    if not verify_quiz_token(chunk_id, question_type, quiz_token):
        return {"error": "invalid_token", "message": "Invalid or expired quiz token."}

    chunk = db.session.get(LectureChunk, chunk_id)
    if not chunk:
        return {"error": "not_found", "message": "Chunk not found."}

    pub = _chunk_to_public(chunk)
    course_block = format_course_answer([pub])

    lines = [
        "Course Answer:",
        "",
        "### Quiz reveal (course material)",
        "",
        f"Lecture {chunk.lecture_number} — {chunk.topic}",
        "",
    ]
    expl = (chunk.clean_explanation or chunk.source_excerpt or "").strip()
    for part in expl.split("\n"):
        p = part.strip()
        if p:
            lines.append(f"- {p}")

    if chunk.sample_answer:
        lines.extend(["", f"**Sample answer (from materials):** {chunk.sample_answer.strip()}"])

    if question_type == "mc":
        lines.extend(["", f"**Correct option:** {chunk.topic}"])

    reveal = "\n".join(lines).strip()

    feedback = []
    if question_type == "mc" and selected_index is not None:
        feedback.append(
            "Your choice has been recorded. Below is the authoritative explanation from the lecture chunk."
        )
    elif question_type == "short" and user_answer:
        feedback.append("Your answer (not graded):")
        feedback.append(user_answer.strip())
        feedback.append("")
        feedback.append("Below is the course-grounded explanation.")

    return {
        "chunk_id": chunk_id,
        "question_type": question_type,
        "course_answer": course_block,
        "reveal": reveal,
        "feedback_lines": "\n".join(feedback).strip() if feedback else None,
    }


def format_compare_answer(
    label_a: str,
    label_b: str,
    chunks_a: list[dict[str, Any]],
    chunks_b: list[dict[str, Any]],
) -> str:
    """Side-by-side comparison grounded in retrieved chunks only."""
    lines: list[str] = [
        "Course Answer:",
        "",
        f"Compare: **{label_a}** vs **{label_b}**",
        "",
        f"### {label_a}",
        "",
    ]
    lines.extend(_format_chunk_bullets(chunks_a))
    lines.extend(["", f"### {label_b}", ""])
    lines.extend(_format_chunk_bullets(chunks_b))
    lines.extend(["", "### Takeaways (from course excerpts)", ""])
    for lab, chs in ((label_a, chunks_a), (label_b, chunks_b)):
        first = chs[0] if chs else {}
        t = (first.get("topic") or "").split("—")[-1].strip() or lab
        expl = (first.get("clean_explanation") or first.get("source_excerpt") or "").strip()
        first_line = expl.split("\n")[0][:240] if expl else ""
        if first_line:
            lines.append(f"- **{t}:** {first_line}")
    return "\n".join(lines).rstrip()


def _format_chunk_bullets(chunks: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for c in chunks:
        num = c.get("lecture_number")
        topic = c.get("topic", "")
        expl = (c.get("clean_explanation") or "").strip() or (c.get("source_excerpt") or "").strip()
        out.append(f"Lecture {num} — {topic}")
        for part in expl.split("\n"):
            p = part.strip()
            if p:
                out.append(f"- {p}")
        out.append("")
    return out


def run_compare(
    concept_a: str,
    concept_b: str,
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    a = (concept_a or "").strip()
    b = (concept_b or "").strip()
    if not a or not b:
        return {"error": "both_concepts_required", "message": "concept_a and concept_b are required."}

    ra = retrieve_chunks(a, top_k=top_k)
    rb = retrieve_chunks(b, top_k=top_k)
    if not ra.chunks and not rb.chunks:
        return {
            "error": "off_topic",
            "message": "No course material matched these concepts. Try different phrasing.",
        }

    course_answer = format_compare_answer(a, b, ra.chunks, rb.chunks)
    return {
        "concept_a": a,
        "concept_b": b,
        "chunks_a": ra.chunks,
        "chunks_b": rb.chunks,
        "confidence_a": ra.confidence,
        "confidence_b": rb.confidence,
        "course_answer": course_answer,
    }


def run_summary_by_lecture(lecture_number: int) -> dict[str, Any]:
    rows = (
        LectureChunk.query.filter_by(lecture_number=lecture_number)
        .order_by(LectureChunk.id)
        .all()
    )
    if not rows:
        return {"error": "not_found", "message": f"No sections for lecture {lecture_number}."}
    chunks = [_chunk_to_public(r) for r in rows]
    title = (rows[0].topic or "").split("—")[0].strip()
    inner = format_course_answer(chunks)
    rest = inner[len("Course Answer:\n\n") :] if inner.startswith("Course Answer:") else inner
    head = f"Course Answer:\n\n### Summary — Lecture {lecture_number}"
    if title:
        head += f" — {title}"
    return {
        "lecture_number": lecture_number,
        "title": title,
        "course_answer": f"{head}\n\n{rest}",
    }


def run_summary_by_topic(topic: str, *, top_k: int = 8) -> dict[str, Any]:
    t = (topic or "").strip()
    if len(t) < 2:
        return {"error": "topic_required", "message": "topic must be a non-empty string."}

    r = retrieve_chunks(t, top_k=top_k)
    if not r.chunks:
        return {
            "error": "off_topic",
            "message": "No course material matched that topic. Try a keyword from the syllabus.",
        }
    inner = format_course_answer(r.chunks)
    rest = inner[len("Course Answer:\n\n") :] if inner.startswith("Course Answer:") else inner
    course_answer = f"Course Answer:\n\n### Summary — topic: {t}\n\n{rest}"
    return {
        "topic": t,
        "confidence": r.confidence,
        "course_answer": course_answer,
    }


def normalize_compare_labels(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())
