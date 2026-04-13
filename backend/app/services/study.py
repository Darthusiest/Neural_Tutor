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

# Filter junk "overlap" tokens in compare keyword sets (stops similarity lists like "defining, different, distinct").
_KEYWORD_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if as of at by for from in into is it its no not on onto or such than that the their
    them then these they this those to too up via was were what when where which while who whom whose why
    will with within without both each few more most other same some such than very can could should would
    about above after again against all also any around as be been being below between both
    defining different distinct have has had having do does did done get gets got how however
    key keys like likely mainly many may might must need needs new next often once only onto our out over
    own per perhaps quite rather really said say says see seen she should since so some such than that
    their them then there these they this those though through throughout thus to too toward under until
    up upon us use used uses using very via want wants was way we well were what when where whether which
    while who whom whose why will with within without would yet you your
    analogy analogies definitions definition examples example
    """.split()
)


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
    overlap = sorted(ka & kb)
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)

    lines.extend(["", "### Similarities", ""])
    overlap = [t for t in overlap if t not in _KEYWORD_STOPWORDS][:10]
    if len(overlap) >= 3:
        lines.append(
            "Both ideas show up in related course material. Shared themes you will see across the excerpts above "
            f"include: {', '.join(overlap[:8])}."
        )
    else:
        lines.append(
            "- The notes above ground each term in concrete definitions and examples; read **Concept A** and "
            "**Concept B** side by side and name what stays the same (e.g. both are part of how we represent "
            "speech or models in this course)."
        )

    lines.extend(["", "### Differences", ""])
    only_a = [t for t in only_a if t not in _KEYWORD_STOPWORDS][:8]
    only_b = [t for t in only_b if t not in _KEYWORD_STOPWORDS][:8]
    if only_a:
        lines.append(
            f"- **{label_a}** is tied more strongly in the keywords to: {', '.join(only_a)}."
        )
    if only_b:
        lines.append(
            f"- **{label_b}** is tied more strongly in the keywords to: {', '.join(only_b)}."
        )
    if not only_a and not only_b:
        lines.append(
            f"- Contrast the **Concept A** and **Concept B** blocks above: pick one distinction per paragraph "
            f"({label_a} vs {label_b}) using the course wording, not generic ML vocabulary."
        )

    lines.extend(["", "### When each matters", ""])
    lines.append(
        f"- Use **{label_a}** when the prompt asks about the ideas in the first column; use **{label_b}** when "
        "the prompt targets the second."
    )
    lines.append(
        "- If you must relate them, state one clean contrast sentence, then support it with a phrase from each column above."
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
                    if len(s) > 2 and s not in _KEYWORD_STOPWORDS:
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
    topics: list[str] = []
    seen_topic: set[str] = set()
    for c in chunks[:8]:
        t = str(c.get("topic") or "").split("—")[0].strip()
        if t and t not in seen_topic:
            seen_topic.add(t)
            topics.append(t)
    if topics:
        lines.append(f"- Topics to cross-link while studying: {', '.join(topics[:6])}.")
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
