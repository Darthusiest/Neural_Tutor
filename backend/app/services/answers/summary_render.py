"""Deterministic Course Answer markdown for summary mode (lecture or topic).

Two entry points share one public function:

- ``format_summary_markdown`` — chooses the lecture-scoped or topic-scoped layout
  based on ``StructuredQuery.intent.lecture_numbers``.

The output is **distinct from** the standard four-block Course Answer
(``### Direct Answer`` / ``### Explanation`` / ``### Example / Intuition`` /
``### Why it matters``). Summary mode uses its own headings so the test
``summary query uses summary renderer`` and the broader rule "summary mode
must use a dedicated summary renderer" can be enforced structurally.

Lecture-scoped layout (``Summary: Lecture N``):

- **Main idea** — single-sentence lecture overview drawn from the strongest
  retrieved chunk.
- **Key topics** — up to 6 deduped section heading prefixes (one bullet each).
- **How the topics connect** — concept-family or related-lecture sentence.
- **Study focus** — 1–2 bullets pointing at what to re-read / restate.

Topic-scoped layout (``Summary: <topic>``):

- **Core idea** — first sentence of the strongest topic-filtered chunk.
- **Key points** — up to 4 ``- **<head>:** <sentence>`` bullets from distinct
  topic heads, only from chunks that mention the target topic or one of its
  KB aliases.
- **Study focus** — single restate-without-slides bullet.

Topic scope filters chunks down to those that mention the target topic (or
one of its KB aliases) somewhere in ``topic`` / ``keywords`` /
``clean_explanation``, so unrelated nearby concepts (e.g. formants surfacing
under a "Recap of MFCCs" query) are dropped before rendering.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.knowledge.concept_kb import ConceptKB, get_kb
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

def _lecture_chunk_sort_key(chunk: dict[str, Any]) -> tuple[int, int]:
    """Stable order for lecture-filtered chunks.

    Prefers an explicit ordering field on the chunk dict (``chunk_order`` /
    ``position`` / ``order``) when present so future schema additions feed
    through naturally; otherwise falls back to ``id`` ascending. The
    ``LectureChunk`` ORM does not currently carry an order column, so the
    metadata branch is forward-compatible only.
    """
    for key in ("chunk_order", "position", "order"):
        raw = chunk.get(key)
        if isinstance(raw, int):
            return (0, raw)
        if isinstance(raw, str) and raw.isdigit():
            return (0, int(raw))
    return (1, int(chunk.get("id") or 0))


def _ordered_lecture_chunks(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    lecture_number: int,
) -> list[dict[str, Any]]:
    """Lecture-filtered chunks, plan-primary-first then any remaining lecture chunks.

    Order is preserved by ``chunk_order`` / ``position`` / ``order`` metadata
    when available, falling back to ``id`` ascending. Primary chunks come
    first so that the renderer's "Main idea" anchor stays close to what
    retrieval ranked top.
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
        key=_lecture_chunk_sort_key,
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
    """Concept-specific study suggestions (1–2 bullets, no boilerplate filler)."""
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
    return out[:2]


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

def _topic_term_set(structured_query: StructuredQuery, kb: ConceptKB) -> tuple[str, list[str]]:
    """Return ``(label, terms)`` for the topic — KB aliases when available, else label tokens.

    ``label`` is the canonical display string (used in ``Summary: <label>``).
    ``terms`` is a lowercased list used to match chunks; the returned terms
    always include the label itself (lowercased) so that a chunk whose ``topic``
    contains the user-typed string still passes the filter when KB lookup
    fails (e.g. user typed an unindexed surface form).
    """
    intent = structured_query.intent
    label = ""
    terms: list[str] = []
    if intent.detected_concepts:
        label = str(intent.detected_concepts[0]).strip()
    elif intent.original_query:
        label = intent.original_query.strip()[:60]

    seen: set[str] = set()

    def _add(term: str | None) -> None:
        if not term:
            return
        clean = term.strip().lower()
        if not clean or len(clean) < 2 or clean in seen:
            return
        seen.add(clean)
        terms.append(clean)

    _add(label)
    # KB lookup by detected concept (handles canonical names / aliases).
    if intent.detected_concepts:
        meta = kb.get_concept(intent.detected_concepts[0])
        if meta:
            label = meta.name or label
            _add(meta.name)
            for alias in meta.aliases[:12]:
                _add(alias)
    # Fall back to any concept_ids the structured query already resolved.
    for cid in structured_query.concept_ids[:2]:
        meta = kb.get_concept_by_id(cid)
        if meta:
            if not label:
                label = meta.name
            _add(meta.name)
            for alias in meta.aliases[:12]:
                _add(alias)

    if not label:
        label = "course topic"
    return label, terms


def _chunk_blob_for_topic_filter(chunk: dict[str, Any]) -> str:
    parts = [
        str(chunk.get("topic", "")),
        str(chunk.get("keywords", "")),
        str(chunk.get("clean_explanation", "")),
    ]
    return " ".join(parts).lower()


def _term_appears(term: str, blob: str) -> bool:
    if len(term) < 2:
        return False
    if " " in term:
        return term in blob
    return re.search(r"\b" + re.escape(term) + r"\b", blob) is not None


def _topic_scoped_chunks(
    primary: list[dict[str, Any]], topic_terms: list[str]
) -> list[dict[str, Any]]:
    """Keep only chunks whose searchable blob contains at least one topic term."""
    if not topic_terms:
        return list(primary)
    out: list[dict[str, Any]] = []
    for chunk in primary:
        blob = _chunk_blob_for_topic_filter(chunk)
        if any(_term_appears(t, blob) for t in topic_terms):
            out.append(chunk)
    return out


def _format_topic_summary(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
    kb: ConceptKB,
) -> str:
    primary = chunks_by_ids(all_chunks, list(plan.primary_chunk_ids or [])) or list(all_chunks)[:8]
    topic_label, topic_terms = _topic_term_set(structured_query, kb)
    scoped = _topic_scoped_chunks(primary, topic_terms)
    if not scoped:
        return (
            f"Summary: {topic_label}\n\n"
            "I couldn't pull course material for that topic. Try a keyword from the "
            "syllabus or name a specific concept."
        )

    main_idea_source = (
        scoped[0].get("clean_explanation") or scoped[0].get("source_excerpt") or ""
    )
    core_idea = _first_sentence(main_idea_source) or _topic_head(scoped[0].get("topic"))
    if not core_idea:
        core_idea = f"{topic_label} appears across the listed sections of the course notes."

    key_points: list[str] = []
    seen_heads: set[str] = set()
    for chunk in scoped:
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
        "### Core idea",
        "",
        core_idea,
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
    kb: ConceptKB | None = None,
) -> str:
    """Return summary markdown driven by lecture vs topic scope.

    Lecture scope: ``len(intent.lecture_numbers) == 1`` -> lecture-scoped layout
    (Main idea / Key topics / How the topics connect / Study focus).

    Topic scope: otherwise -> compact topic recap (Core idea / Key points /
    Study focus). Chunks are filtered to those mentioning the target topic
    or one of its KB aliases, so unrelated retrievals don't pollute the recap.
    """
    lecture_numbers = list(structured_query.intent.lecture_numbers or [])
    if len(lecture_numbers) == 1:
        return _format_lecture_summary(plan, all_chunks, structured_query, lecture_numbers[0])
    return _format_topic_summary(plan, all_chunks, structured_query, kb or get_kb())
