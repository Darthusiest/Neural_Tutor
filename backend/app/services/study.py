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

from app.services.reasoning_pipeline import run_reasoning_pipeline

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
    mc_options: list[str] | None = None,
) -> dict[str, Any]:
    if not verify_quiz_token(chunk_id, question_type, quiz_token):
        return {"error": "invalid_token", "message": "Invalid or expired quiz token."}

    chunk = db.session.get(LectureChunk, chunk_id)
    if not chunk:
        return {"error": "not_found", "message": "Chunk not found."}

    pub = _chunk_to_public(chunk)
    course_block = format_course_answer([pub])

    judgment = ""
    if question_type == "mc" and selected_index is not None and mc_options and 0 <= int(selected_index) < len(
        mc_options
    ):
        chosen = str(mc_options[int(selected_index)]).strip()
        target = str(chunk.topic).strip()
        if chosen.lower() == target.lower():
            judgment = "correct"
        elif chosen and target and (chosen.lower() in target.lower() or target.lower() in chosen.lower()):
            judgment = "partially_correct"
        else:
            judgment = "incorrect"
    elif question_type == "short" and user_answer:
        ua = user_answer.strip().lower()
        key_bits = [w for w in (chunk.clean_explanation or "").lower().split() if len(w) > 5][:8]
        overlap = sum(1 for w in key_bits if w in ua)
        if overlap >= 3:
            judgment = "partially_correct"
        elif overlap == 0:
            judgment = "needs_work"
        else:
            judgment = "partially_correct"

    lines = [
        "Course Answer:",
        "",
        "### Quiz feedback (course material)",
        "",
    ]
    if judgment:
        lines.extend(["### Result", "", f"**Judgment:** {judgment.replace('_', ' ')}", ""])
    lines.extend(
        [
            "### Correct idea (from this section)",
            "",
            f"**Section title:** {chunk.topic}",
            "",
        ]
    )
    expl = (chunk.clean_explanation or chunk.source_excerpt or "").strip()
    for part in expl.split("\n")[:12]:
        p = part.strip()
        if p:
            lines.append(f"- {p}")

    if chunk.sample_answer:
        lines.extend(["", f"**Sample answer (from materials):** {chunk.sample_answer.strip()}"])

    if question_type == "mc":
        lines.extend(["", f"**Authoritative match:** {chunk.topic}"])

    lines.extend(
        [
            "",
            "### Why this matters",
            "",
            "- Quizzes check whether you can **name the idea** and **tie it to the section’s own wording**—use the bullets above as the reference.",
            "",
            "### Short teaching note",
            "",
            "- Read the bullets once for **definitions**, then again while asking: **what would I say on an exam without looking?**",
        ]
    )

    reveal = "\n".join(lines).strip()

    feedback = []
    if question_type == "mc" and selected_index is not None and mc_options:
        feedback.append(
            f"You selected: **{mc_options[int(selected_index)]}** (see **Result** above)."
        )
    elif question_type == "short" and user_answer:
        feedback.append("### Your answer")
        feedback.append("")
        feedback.append(user_answer.strip())
        feedback.append("")
        feedback.append("Compare with **Correct idea** above.")

    return {
        "chunk_id": chunk_id,
        "question_type": question_type,
        "course_answer": course_block,
        "reveal": reveal,
        "feedback_lines": "\n".join(feedback).strip() if feedback else None,
        "judgment": judgment or None,
    }


def format_compare_answer(
    label_a: str,
    label_b: str,
    chunks_a: list[dict[str, Any]],
    chunks_b: list[dict[str, Any]],
) -> str:
    """Side-by-side comparison: concepts, similarities, differences, when each matters (chunk-grounded)."""
    lines: list[str] = [
        "Course Answer:",
        "",
        f"### Compare: **{label_a}** vs **{label_b}**",
        "",
        f"### Concept A — {label_a}",
        "",
    ]
    lines.extend(_format_chunk_bullets(chunks_a))
    lines.extend(["", f"### Concept B — {label_b}", ""])
    lines.extend(_format_chunk_bullets(chunks_b))

    ka = _keyword_set(chunks_a)
    kb = _keyword_set(chunks_b)
    overlap = sorted(ka & kb)[:12]
    only_a = sorted(ka - kb)[:10]
    only_b = sorted(kb - ka)[:10]

    lines.extend(["", "### Similarities", ""])
    if overlap:
        lines.append(
            "Both ideas connect through these course themes or terms (from retrieved sections):"
        )
        for t in overlap:
            lines.append(f"- {t}")
    else:
        lines.append(
            "- The materials describe each idea in different vocabulary; both are grounded in the same course, "
            "but the excerpts emphasize different aspects (see the concept sections above)."
        )

    lines.extend(["", "### Differences", ""])
    if only_a:
        lines.append(f"- **{label_a}** is associated more with: {', '.join(only_a[:8])}.")
    if only_b:
        lines.append(f"- **{label_b}** is associated more with: {', '.join(only_b[:8])}.")
    if not only_a and not only_b:
        lines.append(
            "- Use the **Concept A** and **Concept B** sections above: contrast the definitions, "
            "examples, and lecture contexts line by line."
        )

    lines.extend(["", "### When each matters", ""])
    lines.append(
        f"- Choose **{label_a}** when the question is about the first cluster of ideas above "
        f"(definitions, examples, and lecture hooks in that column)."
    )
    lines.append(
        f"- Choose **{label_b}** when the question targets the second cluster—its definitions, "
        "examples, and typical exam prompts shown there."
    )
    lines.append(
        "- If the task asks you to **relate** them, start from similarities, then spell out the contrast using the **Differences** bullets."
    )
    return "\n".join(lines).rstrip()


def _keyword_set(chunks: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for c in chunks:
        raw = c.get("keywords") or "[]"
        try:
            arr = json.loads(raw)
            if isinstance(arr, list):
                for x in arr:
                    s = str(x).strip().lower()
                    if len(s) > 2:
                        out.add(s)
        except json.JSONDecodeError:
            pass
    return out


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
    out = {
        "concept_a": a,
        "concept_b": b,
        "chunks_a": ra.chunks,
        "chunks_b": rb.chunks,
        "confidence_a": ra.confidence,
        "confidence_b": rb.confidence,
        "course_answer": course_answer,
    }
    if current_app.config.get("STRUCTURED_STUDY_PIPELINE_ENABLED") and current_app.config.get(
        "STRUCTURED_PIPELINE_ENABLED"
    ):
        try:
            pr = run_reasoning_pipeline(
                f"Compare {a} and {b} in this course.",
                top_k=top_k,
                backend="keyword",
            )
            out["course_answer"] = pr.course_answer
        except Exception:
            current_app.logger.exception("structured study compare failed; using lexical compare answer")
    return out


def _format_summary_recap(chunks: list[dict[str, Any]], title: str) -> str:
    """Tight recap: main idea, supporting ideas, why it matters, connections (chunk-grounded)."""
    lines: list[str] = ["Course Answer:", "", f"### Summary — {title}", "", "### Main idea", ""]
    first = chunks[0] if chunks else {}
    expl = (first.get("clean_explanation") or first.get("source_excerpt") or "").strip()
    if expl:
        lines.append(expl.split("\n")[0][:720].strip())
    else:
        lines.append("_No explanation text in the top section._")

    lines.extend(["", "### Supporting ideas", ""])
    for c in chunks[:10]:
        t = (c.get("topic") or "").strip()
        bit = (c.get("clean_explanation") or c.get("source_excerpt") or "").strip()
        one = bit.split("\n")[0][:320] if bit else ""
        if t and one:
            lines.append(f"- **{t}:** {one}")

    lines.extend(["", "### Why it matters", ""])
    lines.append(
        "- This lecture thread builds the definitions and examples your assessments assume—use it to anchor "
        "new terms and to check intuitions before moving to later lectures."
    )

    lec_nums = sorted({int(c.get("lecture_number") or 0) for c in chunks if c.get("lecture_number")})
    lines.extend(["", "### Key connections", ""])
    if len(lec_nums) > 1:
        lines.append(f"- Sections span lecture material in **L{lec_nums[0]}** through the listed topics above.")
    else:
        lines.append("- Connections are **within-lecture**: headings above follow the order of ideas in the notes.")
    topics = [str(c.get("topic") or "").split("—")[0].strip() for c in chunks[:5]]
    topics = [t for t in topics if t]
    if topics:
        lines.append(f"- Topics to cross-link while studying: {', '.join(topics[:5])}.")
    return "\n".join(lines).rstrip()


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
    heading = f"Lecture {lecture_number}" + (f" — {title}" if title else "")
    out = {
        "lecture_number": lecture_number,
        "title": title,
        "course_answer": _format_summary_recap(chunks, heading),
    }
    if current_app.config.get("STRUCTURED_STUDY_PIPELINE_ENABLED") and current_app.config.get(
        "STRUCTURED_PIPELINE_ENABLED"
    ):
        try:
            pr = run_reasoning_pipeline(
                f"Summarize lecture {lecture_number} for this course.",
                top_k=8,
                backend="keyword",
            )
            out["course_answer"] = pr.course_answer
        except Exception:
            current_app.logger.exception("structured study lecture summary failed; using recap template")
    return out


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
    course_answer = _format_summary_recap(r.chunks, f"topic: {t}")
    out = {
        "topic": t,
        "confidence": r.confidence,
        "course_answer": course_answer,
    }
    if current_app.config.get("STRUCTURED_STUDY_PIPELINE_ENABLED") and current_app.config.get(
        "STRUCTURED_PIPELINE_ENABLED"
    ):
        try:
            pr = run_reasoning_pipeline(
                f"Summarize the topic {t} for this course.",
                top_k=8,
                backend="keyword",
            )
            out["course_answer"] = pr.course_answer
        except Exception:
            current_app.logger.exception("structured study topic summary failed; using recap template")
    return out


def normalize_compare_labels(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())
