"""Unit tests for ``app.services.answers.summary_render.format_summary_markdown``.

Covers:
- Lecture-scoped layout (``Summary: Lecture N`` + Main idea / Key topics / How the topics
  connect / Study focus).
- Topic dedupe by heading prefix.
- Topic order preservation (primary chunks first, then chunk-id order).
- Topic-scoped layout (``Summary: <topic>`` + Main idea / Key points / Study focus) when no
  single lecture is detected.
- Forbidden Course Answer headings never leak into summary output.
"""

from __future__ import annotations

from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.summary_render import format_summary_markdown
from app.services.knowledge.structured_query import StructuredQuery
from app.services.query_understanding import QueryIntent, QueryType


_FORBIDDEN_HEADINGS = (
    "### Direct Answer",
    "### Explanation",
    "### Example / Intuition",
    "### Why it matters",
    "Course Answer:",
)


def _intent(query: str, *, lecture_numbers: list[int] | None = None, concepts: list[str] | None = None) -> QueryIntent:
    return QueryIntent(
        query_type=QueryType.SUMMARY,
        original_query=query,
        expanded_query=query.lower(),
        query_tokens=query.lower().split(),
        expanded_tokens=query.lower().split(),
        lecture_numbers=list(lecture_numbers or []),
        detected_concepts=list(concepts or []),
    )


def _structured_query(intent: QueryIntent) -> StructuredQuery:
    return StructuredQuery(
        intent=intent,
        concept_ids=[],
        answer_intent="lecture_summary",
        sub_questions=[],
        retrieval_hints=[],
        lecture_scope=list(intent.lecture_numbers),
        answer_style="teaching",
        decomposition_template=[],
        effective_mode="summary",
        detected_mode="summary",
    )


def _plan(primary_ids: list[int]) -> AnswerPlan:
    return AnswerPlan(
        answer_mode="lecture_summary",
        sections=[],
        primary_chunk_ids=list(primary_ids),
        supporting_chunk_ids=[],
        include_example=False,
        include_analogy=False,
        include_prerequisites=False,
        include_related_concepts=[],
        comparison_axes=[],
        lecture_scope=[],
    )


def _chunk(cid: int, *, lecture: int, topic: str, explanation: str) -> dict:
    return {
        "id": cid,
        "lecture_number": lecture,
        "topic": topic,
        "clean_explanation": explanation,
        "source_excerpt": explanation,
        "sample_questions": "[]",
        "sample_answer": None,
    }


def _no_course_answer_headings(text: str) -> None:
    for marker in _FORBIDDEN_HEADINGS:
        assert marker not in text, f"summary output unexpectedly contains '{marker}'"


# ---------------------------------------------------------------------------
# Lecture-scoped summary
# ---------------------------------------------------------------------------

def test_summary_render_lecture_emits_required_sections():
    """Summarize Lecture 10 -> Summary header + Main idea + Key topics + How they connect + Study focus."""
    chunks = [
        _chunk(1, lecture=10, topic="MFCCs — Core Idea", explanation="MFCCs summarize the spectrum of speech."),
        _chunk(2, lecture=10, topic="MFCCs — Pipeline", explanation="The MFCC pipeline filterbank, log, DCT."),
        _chunk(3, lecture=10, topic="Formants — Core Idea", explanation="Formants are spectral peaks."),
    ]
    plan = _plan([1, 2, 3])
    sq = _structured_query(_intent("Summarize Lecture 10", lecture_numbers=[10]))
    out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: Lecture 10")
    for heading in ("### Main idea", "### Key topics", "### How the topics connect", "### Study focus"):
        assert heading in out, f"missing heading {heading} in: {out}"
    _no_course_answer_headings(out)


def test_summary_render_lecture_dedupes_topic_heads():
    """Multiple chunks with the same heading prefix collapse to one bullet."""
    chunks = [
        _chunk(1, lecture=10, topic="MFCCs — Core Idea", explanation="A."),
        _chunk(2, lecture=10, topic="MFCCs — Pipeline", explanation="B."),
        _chunk(3, lecture=10, topic="MFCCs — Filterbank step", explanation="C."),
        _chunk(4, lecture=10, topic="Formants — Core Idea", explanation="D."),
    ]
    plan = _plan([1, 2, 3, 4])
    sq = _structured_query(_intent("Summarize Lecture 10", lecture_numbers=[10]))
    out = format_summary_markdown(plan, chunks, sq)

    assert out.count("- MFCCs") == 1
    assert "- Formants" in out


def test_summary_render_lecture_filters_other_lectures():
    """Chunks from other lectures must not appear in a single-lecture summary."""
    chunks = [
        _chunk(1, lecture=10, topic="MFCCs — Core Idea", explanation="A."),
        _chunk(2, lecture=11, topic="Backprop — Core Idea", explanation="Should not appear."),
    ]
    plan = _plan([1, 2])
    sq = _structured_query(_intent("Summarize Lecture 10", lecture_numbers=[10]))
    out = format_summary_markdown(plan, chunks, sq)

    assert "Backprop" not in out
    assert "Should not appear" not in out


def test_summary_render_lecture_with_no_evidence_falls_back():
    """Single-lecture summary with no chunks for that lecture -> short fallback message."""
    chunks = [
        _chunk(1, lecture=12, topic="Other — Core Idea", explanation="Other lecture."),
    ]
    plan = _plan([1])
    sq = _structured_query(_intent("Summarize Lecture 99", lecture_numbers=[99]))
    out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: Lecture 99")
    assert "couldn't pull sections" in out.lower() or "i couldn't" in out.lower()
    _no_course_answer_headings(out)


# ---------------------------------------------------------------------------
# Topic-scoped summary
# ---------------------------------------------------------------------------

def test_summary_render_topic_uses_topic_layout():
    """No lecture number -> topic-scoped layout with Main idea + Key points + Study focus."""
    chunks = [
        _chunk(
            1,
            lecture=14,
            topic="Softmax — Core Idea",
            explanation="Softmax turns logits into a probability distribution.",
        ),
        _chunk(
            2,
            lecture=14,
            topic="Softmax — Use",
            explanation="Softmax is applied to the final layer of a classifier.",
        ),
    ]
    plan = _plan([1, 2])
    sq = _structured_query(_intent("Recap softmax", concepts=["softmax"]))
    out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: softmax")
    assert "### Main idea" in out
    assert "### Key points" in out
    assert "### Study focus" in out
    # Lecture-only sections must not appear in the topic-scoped layout.
    assert "### Key topics" not in out
    assert "### How the topics connect" not in out
    _no_course_answer_headings(out)
