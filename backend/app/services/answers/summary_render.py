"""Deterministic Course Answer markdown for summary mode (lecture or topic).

Two entry points share one public function:

- ``format_summary_markdown`` — chooses the lecture-scoped or topic-scoped layout
  based on ``StructuredQuery.intent.lecture_numbers``.

The output is **distinct from** the standard four-block Course Answer
(``### Direct Answer`` / ``### Explanation`` / ``### Example / Intuition`` /
``### Why it matters``). Summary mode uses its own headings so the test
``summary query uses summary renderer`` and the broader rule "summary mode
must use a dedicated summary renderer" can be enforced structurally.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.knowledge.domain_knowledge import (
    get_concept_family_for_lecture,
    get_related_lectures,
)
from app.services.knowledge.structured_query import StructuredQuery


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TOPIC_DELIMITER_RE = re.compile(r"\s*[—\-:|]\s*")


def _topic_head(topic: str | None) -> str:
    """Return the leading heading portion of a chunk topic (before any em-dash sub-section)."""
    if not topic:
        return ""
    head = _TOPIC_DELIMITER_RE.split(str(topic), maxsplit=1)[0].strip()
    return head


def _first_sentence(text: str, *, max_len: int = 320) -> str:
    """First sentence (period/?/!) or first line, capped."""
    body = (text or "").strip()
    if not body:
        return ""
    match = re.match(r"([^.!?\n]+[.!?])", body)
    if match:
        return match.group(1).strip()[:max_len]
    first_line = body.split("\n", 1)[0].strip()
    return first_line[:max_len]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _topic_label(structured_query: StructuredQuery, fallback_chunks: list[dict[str, Any]]) -> str:
    """Pick a concise topic label for the topic-scoped header (no lecture number)."""
    intent = structured_query.intent
    if intent.detected_concepts:
        return str(intent.detected_concepts[0]).strip()
    for chunk in fallback_chunks:
        head = _topic_head(chunk.get("topic"))
        if head:
            return head
    raw = (intent.original_query or "").strip()
    return raw[:60] if raw else "course topic"


# ---------------------------------------------------------------------------
# Lecture-scoped summary
# ---------------------------------------------------------------------------

def _ordered_lecture_chunks(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    lecture_number: int,
) -> list[dict[str, Any]]:
    """Lecture-filtered chunks, plan-primary-first then any remaining lecture chunks.

    Order is preserved by ``id`` to mirror lecture order; primary chunks come first
    so that the renderer's "Main idea" anchor stays close to what retrieval ranked top.
    """
    in_lecture = [
        c for c in all_chunks
        if c.get("lecture_number") == lecture_number and c.get("id") is not None
    ]
    if not in_lecture:
        return []
    primary_ids = list(plan.primary_chunk_ids or [])
    primary_set = set(primary_ids)
    primary_in_lecture = [c for c in chunks_by_ids(in_lecture, primary_ids) if c.get("id") is not None]
    remainder = sorted(
        (c for c in in_lecture if c.get("id") not in primary_set),
        key=lambda c: c.get("id") or 0,
    )
    return primary_in_lecture + remainder


def _connect_sentence(lecture_number: int, key_topics: list[str]) -> str:
    """Single concept-specific sentence describing how the listed topics fit together."""
    family = get_concept_family_for_lecture(lecture_number)
    related = [n for n in get_related_lectures(lecture_number) if n != lecture_number]
    if family and key_topics:
        anchor = key_topics[0]
        return (
            f"These sections build the **{family.replace('_', ' ')}** thread: starting from "
            f"{anchor} and layering the remaining topics on top of the same vocabulary."
        )
    if related and key_topics:
        related_str = ", ".join(f"Lecture {n}" for n in related[:3])
        return (
            f"The sections above set up vocabulary you'll re-use in {related_str}; "
            f"keep the order in mind when reviewing."
        )
    if len(key_topics) >= 2:
        return (
            f"The lecture moves from {key_topics[0]} to {key_topics[-1]}; the middle topics "
            "tie the opening definition to the closing application."
        )
    return "The sections above are sequential — each one builds on the previous heading."


def _study_focus_lines(key_topics: list[str]) -> list[str]:
    """Concept-specific study suggestions (2 bullets, no boilerplate filler)."""
    out: list[str] = []
    if key_topics:
        out.append(f"Re-read the section on **{key_topics[0]}** before the next lecture.")
    if len(key_topics) >= 2:
        out.append(
            f"Make sure you can explain **{key_topics[1]}** in your own words without checking the slides."
        )
    elif key_topics:
        out.append(
            f"Practice restating **{key_topics[0]}** in your own words; that's the anchor "
            "for the rest of the thread."
        )
    if not out:
        out.append("Pick one heading above and try to teach it back from memory.")
    return out


def _format_lecture_summary(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
    lecture_number: int,
) -> str:
    ordered = _ordered_lecture_chunks(plan, all_chunks, lecture_number)
    if not ordered:
        return (
            f"Summary: Lecture {lecture_number}\n\n"
            "I couldn't pull sections for that lecture from the notes.\n\n"
            "Try a more specific question (e.g. a concept name from that lecture) or "
            "double-check the lecture number."
        )

    main_idea_source = (
        ordered[0].get("clean_explanation") or ordered[0].get("source_excerpt") or ""
    )
    main_idea = _first_sentence(main_idea_source) or _topic_head(ordered[0].get("topic"))
    if not main_idea:
        main_idea = f"Lecture {lecture_number} introduces a connected set of course topics."

    key_topics: list[str] = []
    for chunk in ordered:
        head = _topic_head(chunk.get("topic"))
        if head:
            key_topics.append(head)
    key_topics = _dedupe_preserve_order(key_topics)[:6]

    connect_sentence = _connect_sentence(lecture_number, key_topics)
    study_focus = _study_focus_lines(key_topics)

    parts: list[str] = [
        f"Summary: Lecture {lecture_number}",
        "",
        "### Main idea",
        "",
        main_idea,
        "",
        "### Key topics",
        "",
    ]
    if key_topics:
        parts.extend(f"- {topic}" for topic in key_topics)
    else:
        parts.append("- (No section headings found for this lecture.)")
    parts.extend(
        [
            "",
            "### How the topics connect",
            "",
            connect_sentence,
            "",
            "### Study focus",
            "",
        ]
    )
    parts.extend(f"- {line}" for line in study_focus)
    return "\n".join(parts).rstrip()


# ---------------------------------------------------------------------------
# Topic-scoped summary
# ---------------------------------------------------------------------------

def _format_topic_summary(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
) -> str:
    primary = chunks_by_ids(all_chunks, list(plan.primary_chunk_ids or [])) or list(all_chunks)[:6]
    if not primary:
        topic = _topic_label(structured_query, [])
        return (
            f"Summary: {topic}\n\n"
            "I couldn't pull course material for that topic. Try a keyword from the "
            "syllabus or name a specific concept."
        )

    topic_label = _topic_label(structured_query, primary)

    main_idea_source = (
        primary[0].get("clean_explanation") or primary[0].get("source_excerpt") or ""
    )
    main_idea = _first_sentence(main_idea_source) or _topic_head(primary[0].get("topic"))
    if not main_idea:
        main_idea = f"{topic_label} appears across the listed sections of the course notes."

    key_points: list[str] = []
    seen_heads: set[str] = set()
    for chunk in primary:
        head = _topic_head(chunk.get("topic"))
        head_key = head.lower()
        if not head or head_key in seen_heads:
            continue
        sentence = _first_sentence(
            chunk.get("clean_explanation") or chunk.get("source_excerpt") or "",
            max_len=240,
        )
        if not sentence:
            continue
        seen_heads.add(head_key)
        key_points.append(f"**{head}:** {sentence}")
        if len(key_points) >= 4:
            break

    parts: list[str] = [
        f"Summary: {topic_label}",
        "",
        "### Main idea",
        "",
        main_idea,
        "",
        "### Key points",
        "",
    ]
    if key_points:
        parts.extend(f"- {line}" for line in key_points)
    else:
        parts.append(
            f"- The notes describe **{topic_label}** mostly through one section; ask a follow-up "
            "for a deeper breakdown."
        )

    parts.extend(
        [
            "",
            "### Study focus",
            "",
            f"- Be able to define **{topic_label}** without looking at the slides, then walk through "
            "one example from the notes.",
        ]
    )
    return "\n".join(parts).rstrip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def format_summary_markdown(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
) -> str:
    """Return summary markdown driven by lecture vs topic scope.

    Lecture scope: ``len(intent.lecture_numbers) == 1`` -> lecture-scoped layout
    (Main idea / Key topics / How the topics connect / Study focus).

    Topic scope: otherwise -> compact topic recap (Main idea / Key points / Study focus).
    """
    lecture_numbers = list(structured_query.intent.lecture_numbers or [])
    if len(lecture_numbers) == 1:
        return _format_lecture_summary(plan, all_chunks, structured_query, lecture_numbers[0])
    return _format_topic_summary(plan, all_chunks, structured_query)
