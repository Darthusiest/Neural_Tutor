"""Unit tests for ``app.services.answers.summary_render.format_summary_markdown``.

Covers:

- Lecture-scoped layout (``Summary: Lecture N`` + Main idea / Key topics / How the topics
  connect / Study focus) — exercised against the spec queries: *Summarize Lecture 10*,
  *Main takeaways from Lecture 10*, *What are the main ideas of Lecture 16?*.
- Topic dedupe by heading prefix.
- Topic order preservation (primary chunks first, then chunk-id order or explicit
  ``chunk_order`` / ``position`` metadata when present).
- Lecture filter is hard — chunks from other lectures never appear.
- Topic-scoped layout (``Summary: <topic>`` + Core idea / Key points / Study focus) —
  exercised against the spec query *Give me a recap of MFCCs*; chunks for unrelated
  concepts (e.g. softmax) are dropped before rendering.
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


def _structured_query(
    intent: QueryIntent, *, concept_ids: list[str] | None = None
) -> StructuredQuery:
    return StructuredQuery(
        intent=intent,
        concept_ids=list(concept_ids or []),
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


def _chunk(
    cid: int,
    *,
    lecture: int,
    topic: str,
    explanation: str,
    chunk_order: int | None = None,
) -> dict:
    out: dict = {
        "id": cid,
        "lecture_number": lecture,
        "topic": topic,
        "clean_explanation": explanation,
        "source_excerpt": explanation,
        "sample_questions": "[]",
        "sample_answer": None,
        "keywords": "",
    }
    if chunk_order is not None:
        out["chunk_order"] = chunk_order
    return out


def _no_course_answer_headings(text: str) -> None:
    for marker in _FORBIDDEN_HEADINGS:
        assert marker not in text, f"summary output unexpectedly contains '{marker}'"


# ---------------------------------------------------------------------------
# Lecture-scoped summary
# ---------------------------------------------------------------------------

def test_summary_render_summarize_lecture_10(app):
    """*Summarize Lecture 10* -> Summary header + Main idea + Key topics + How they connect + Study focus."""
    chunks = [
        _chunk(1, lecture=10, topic="MFCCs — Core Idea", explanation="MFCCs summarize the spectrum of speech."),
        _chunk(2, lecture=10, topic="MFCCs — Pipeline", explanation="The MFCC pipeline filterbank, log, DCT."),
        _chunk(3, lecture=10, topic="Formants — Core Idea", explanation="Formants are spectral peaks."),
    ]
    plan = _plan([1, 2, 3])
    sq = _structured_query(_intent("Summarize Lecture 10", lecture_numbers=[10]))
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: Lecture 10")
    for heading in ("### Main idea", "### Key topics", "### How the topics connect", "### Study focus"):
        assert heading in out, f"missing heading {heading} in: {out}"
    _no_course_answer_headings(out)


def test_summary_render_main_takeaways_lecture_10(app):
    """*Main takeaways from Lecture 10* uses the same lecture-scoped shape."""
    chunks = [
        _chunk(1, lecture=10, topic="MFCCs — Core Idea", explanation="MFCCs summarize the spectrum of speech."),
        _chunk(2, lecture=10, topic="MFCCs — Pipeline", explanation="The MFCC pipeline filterbank, log, DCT."),
        _chunk(3, lecture=10, topic="Formants — Core Idea", explanation="Formants are spectral peaks."),
    ]
    plan = _plan([1, 2, 3])
    sq = _structured_query(_intent("Main takeaways from Lecture 10", lecture_numbers=[10]))
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: Lecture 10")
    for heading in ("### Main idea", "### Key topics", "### How the topics connect", "### Study focus"):
        assert heading in out
    _no_course_answer_headings(out)


def test_summary_render_main_ideas_lecture_16_drops_other_lectures(app):
    """*What are the main ideas of Lecture 16?* respects the lecture filter strictly."""
    chunks = [
        _chunk(
            1,
            lecture=16,
            topic="Attention — Core Idea",
            explanation="Attention reweights tokens by relevance.",
        ),
        _chunk(
            2,
            lecture=16,
            topic="Attention — QKV",
            explanation="Queries, keys, values shape attention.",
        ),
        _chunk(
            3,
            lecture=14,
            topic="Softmax — Should not appear",
            explanation="Softmax turns logits into probabilities.",
        ),
    ]
    plan = _plan([1, 2, 3])
    sq = _structured_query(
        _intent("What are the main ideas of Lecture 16?", lecture_numbers=[16])
    )
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: Lecture 16")
    assert "Softmax" not in out
    assert "Should not appear" not in out
    for heading in ("### Main idea", "### Key topics", "### How the topics connect", "### Study focus"):
        assert heading in out
    _no_course_answer_headings(out)


def test_summary_render_lecture_dedupes_topic_heads(app):
    """Multiple chunks with the same heading prefix collapse to one bullet."""
    chunks = [
        _chunk(1, lecture=10, topic="MFCCs — Core Idea", explanation="A."),
        _chunk(2, lecture=10, topic="MFCCs — Pipeline", explanation="B."),
        _chunk(3, lecture=10, topic="MFCCs — Filterbank step", explanation="C."),
        _chunk(4, lecture=10, topic="Formants — Core Idea", explanation="D."),
    ]
    plan = _plan([1, 2, 3, 4])
    sq = _structured_query(_intent("Summarize Lecture 10", lecture_numbers=[10]))
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    assert out.count("- MFCCs") == 1
    assert "- Formants" in out


def test_summary_render_lecture_uses_chunk_order_metadata(app):
    """Lecture-scoped order respects ``chunk_order`` metadata when present."""
    chunks = [
        _chunk(
            10,
            lecture=10,
            topic="MFCCs — Pipeline",
            explanation="Pipeline first.",
            chunk_order=5,
        ),
        _chunk(
            11,
            lecture=10,
            topic="MFCCs — Core Idea",
            explanation="Core idea first.",
            chunk_order=1,
        ),
        _chunk(
            12,
            lecture=10,
            topic="Formants — Core Idea",
            explanation="Formants follow.",
            chunk_order=10,
        ),
    ]
    plan = _plan([])
    sq = _structured_query(_intent("Summarize Lecture 10", lecture_numbers=[10]))
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    core_pos = out.find("- MFCCs")
    formants_pos = out.find("- Formants")
    assert 0 < core_pos < formants_pos, out


def test_summary_render_lecture_with_no_evidence_falls_back(app):
    """Single-lecture summary with no chunks for that lecture -> short fallback message."""
    chunks = [
        _chunk(1, lecture=12, topic="Other — Core Idea", explanation="Other lecture."),
    ]
    plan = _plan([1])
    sq = _structured_query(_intent("Summarize Lecture 99", lecture_numbers=[99]))
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: Lecture 99")
    assert "couldn't pull sections" in out.lower() or "i couldn't" in out.lower()
    _no_course_answer_headings(out)


# ---------------------------------------------------------------------------
# Topic-scoped summary
# ---------------------------------------------------------------------------

def test_summary_render_recap_mfccs_topic_only(app):
    """*Give me a recap of MFCCs* -> ``Summary: MFCCs`` with Core idea / Key points / Study focus.

    Unrelated retrieved chunks (e.g. softmax) are dropped before rendering — the
    topic-scope path filters chunks to those mentioning the topic (or a KB
    alias) before the renderer reads them.
    """
    chunks = [
        {
            "id": 1,
            "lecture_number": 10,
            "topic": "MFCCs — Core Idea",
            "clean_explanation": "MFCCs summarize the spectrum of speech.",
            "source_excerpt": "MFCCs summarize the spectrum of speech.",
            "keywords": "mfcc, spectrum",
            "sample_questions": "[]",
        },
        {
            "id": 2,
            "lecture_number": 10,
            "topic": "MFCCs — Pipeline",
            "clean_explanation": "The MFCC pipeline applies a filterbank, log, and DCT.",
            "source_excerpt": "The MFCC pipeline applies a filterbank, log, and DCT.",
            "keywords": "mfcc, filterbank, dct",
            "sample_questions": "[]",
        },
        {
            "id": 3,
            "lecture_number": 14,
            "topic": "Softmax — Should not appear",
            "clean_explanation": "Softmax turns logits into a probability distribution.",
            "source_excerpt": "Softmax turns logits into a probability distribution.",
            "keywords": "softmax, logits",
            "sample_questions": "[]",
        },
    ]
    plan = _plan([1, 2, 3])
    sq = _structured_query(
        _intent("Give me a recap of MFCCs", concepts=["MFCCs"]),
        concept_ids=[],
    )
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: MFCCs") or out.startswith("Summary: MFCC")
    assert "### Core idea" in out
    assert "### Key points" in out
    assert "### Study focus" in out
    # Lecture-only sections must not appear in the topic-scoped layout.
    assert "### Key topics" not in out
    assert "### How the topics connect" not in out
    # Softmax chunk filtered out before rendering.
    assert "Softmax" not in out
    assert "Should not appear" not in out
    _no_course_answer_headings(out)


def test_summary_render_topic_uses_topic_layout(app):
    """No lecture number -> topic-scoped layout with Core idea + Key points + Study focus."""
    chunks = [
        {
            "id": 1,
            "lecture_number": 14,
            "topic": "Softmax — Core Idea",
            "clean_explanation": "Softmax turns logits into a probability distribution.",
            "source_excerpt": "Softmax turns logits into a probability distribution.",
            "keywords": "softmax",
            "sample_questions": "[]",
        },
        {
            "id": 2,
            "lecture_number": 14,
            "topic": "Softmax — Use",
            "clean_explanation": "Softmax is applied to the final layer of a classifier.",
            "source_excerpt": "Softmax is applied to the final layer of a classifier.",
            "keywords": "softmax, classifier",
            "sample_questions": "[]",
        },
    ]
    plan = _plan([1, 2])
    sq = _structured_query(_intent("Recap softmax", concepts=["softmax"]))
    with app.app_context():
        out = format_summary_markdown(plan, chunks, sq)

    assert out.startswith("Summary: softmax") or out.startswith("Summary: Softmax")
    assert "### Core idea" in out
    assert "### Key points" in out
    assert "### Study focus" in out
    # Lecture-only sections must not appear in the topic-scoped layout.
    assert "### Key topics" not in out
    assert "### How the topics connect" not in out
    # ``Main idea`` is reserved for lecture-scoped layout; topic uses Core idea.
    assert "### Main idea" not in out
    _no_course_answer_headings(out)
