"""Unit tests for ``app.services.answers.quiz_render.format_quiz_markdown``.

Tests build :class:`AnswerPlan` and :class:`StructuredQuery` directly so the
renderer can be exercised without retrieval / DB plumbing. Every assertion
mirrors a requirement from the deterministic-pipeline-pass spec:

- 3 retrieved chunks -> 3 questions (short / MC / T-F) + answer key
- Lecture-scoped query -> ``Quiz: Lecture N`` header and lecture-only evidence
- Topic-scoped query -> ``Quiz: <topic>`` header
- Thin / empty evidence -> graceful degrade (fewer questions or fallback)
- No Course Answer headings ever appear in quiz output
"""

from __future__ import annotations

from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.quiz_render import format_quiz_markdown
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
        query_type=QueryType.QUIZ,
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
        answer_intent="teaching_plus_check",
        sub_questions=[],
        retrieval_hints=[],
        lecture_scope=list(intent.lecture_numbers),
        answer_style="quiz",
        decomposition_template=[],
        effective_mode="quiz",
        detected_mode="quiz",
    )


def _plan(primary_ids: list[int]) -> AnswerPlan:
    return AnswerPlan(
        answer_mode="teaching_plus_check",
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
    sample_questions: str = "[]",
    sample_answer: str | None = None,
) -> dict:
    return {
        "id": cid,
        "lecture_number": lecture,
        "topic": topic,
        "clean_explanation": explanation,
        "source_excerpt": explanation,
        "sample_questions": sample_questions,
        "sample_answer": sample_answer,
    }


def _no_course_answer_headings(text: str) -> None:
    for marker in _FORBIDDEN_HEADINGS:
        assert marker not in text, f"quiz output unexpectedly contains '{marker}': {text[:200]}"


# ---------------------------------------------------------------------------
# Topic queries
# ---------------------------------------------------------------------------

def test_quiz_render_topic_mfccs_emits_three_questions_and_answer_key():
    """Quiz me on MFCCs: 3 questions (short/MC/T-F) + answer key, header Quiz: MFCCs."""
    chunks = [
        _chunk(
            1,
            lecture=10,
            topic="MFCCs — Core Idea",
            explanation="MFCCs summarize the spectrum of speech as a small vector.",
            sample_questions='["What do MFCCs summarize?"]',
        ),
        _chunk(
            2,
            lecture=10,
            topic="MFCCs — Pipeline",
            explanation="The MFCC pipeline applies a filterbank, takes logs, and runs a DCT.",
        ),
        _chunk(
            3,
            lecture=10,
            topic="Formants — Core Idea",
            explanation="Formants are spectral peaks tied to vocal tract shape.",
        ),
        _chunk(
            4,
            lecture=11,
            topic="Spectrogram — Overview",
            explanation="A spectrogram shows energy across time and frequency.",
        ),
    ]
    plan = _plan([1, 2, 3, 4])
    sq = _structured_query(_intent("Quiz me on MFCCs", concepts=["MFCCs"]))
    out = format_quiz_markdown(plan, chunks, sq)

    assert out.startswith("Quiz: MFCCs")
    assert "Answer Key:" in out
    # Three numbered question slots.
    for marker in ("1.", "2.", "3.", "True or false:"):
        assert marker in out, f"missing marker {marker!r} in: {out}"
    # MC slot has 4 lettered options.
    for letter in ("A)", "B)", "C)", "D)"):
        assert letter in out
    # Answer key rows for each question.
    assert "\n1. " in out
    assert "\n2. " in out
    assert "\n3. " in out
    _no_course_answer_headings(out)


def test_quiz_render_topic_softmax_uses_topic_header():
    """Give me a practice quiz on softmax -> Quiz: softmax."""
    chunks = [
        _chunk(
            10,
            lecture=14,
            topic="Softmax — Core Idea",
            explanation="Softmax turns logits into a probability distribution.",
        ),
        _chunk(
            11,
            lecture=14,
            topic="Hardmax — Core Idea",
            explanation="Hardmax picks the argmax and returns a one-hot vector.",
        ),
        _chunk(
            12,
            lecture=15,
            topic="Cross Entropy — Definition",
            explanation="Cross-entropy compares two probability distributions.",
        ),
    ]
    plan = _plan([10, 11, 12])
    sq = _structured_query(_intent("Give me a practice quiz on softmax", concepts=["softmax"]))
    out = format_quiz_markdown(plan, chunks, sq)

    assert out.startswith("Quiz: softmax")
    assert "Answer Key:" in out
    _no_course_answer_headings(out)


# ---------------------------------------------------------------------------
# Lecture queries
# ---------------------------------------------------------------------------

def test_quiz_render_lecture_eleven_uses_only_lecture_chunks():
    """Test me on Lecture 11: header is Quiz: Lecture 11; evidence drawn only from lecture 11."""
    chunks = [
        _chunk(
            20,
            lecture=11,
            topic="Backpropagation — Core Idea",
            explanation="Backprop chains gradients through the layers.",
        ),
        _chunk(
            21,
            lecture=11,
            topic="Gradient Descent — Step",
            explanation="Gradient descent updates weights against the gradient.",
        ),
        _chunk(
            22,
            lecture=11,
            topic="Loss — Definition",
            explanation="A loss measures the gap between prediction and truth.",
        ),
        _chunk(
            23,
            lecture=12,
            topic="Attention — Should not appear",
            explanation="Attention reweights tokens by relevance.",
        ),
    ]
    plan = _plan([20, 21, 22, 23])
    sq = _structured_query(_intent("Test me on Lecture 11", lecture_numbers=[11]))
    out = format_quiz_markdown(plan, chunks, sq)

    assert out.startswith("Quiz: Lecture 11")
    assert "Answer Key:" in out
    # Lecture 12 evidence must not be used in stems or answer key.
    assert "Attention" not in out
    assert "reweights tokens" not in out
    _no_course_answer_headings(out)


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_quiz_render_thin_evidence_one_chunk_yields_one_question():
    """Single chunk -> one question, no MC slot, no T-F slot."""
    chunks = [
        _chunk(
            30,
            lecture=20,
            topic="Lonely Topic — Definition",
            explanation="The lonely topic stands by itself in retrieved evidence.",
        ),
    ]
    plan = _plan([30])
    sq = _structured_query(_intent("Quiz me on the lonely topic", concepts=["lonely topic"]))
    out = format_quiz_markdown(plan, chunks, sq)

    assert "Answer Key:" in out
    assert "1." in out
    # No second question or T-F line.
    assert "2." not in out
    assert "True or false:" not in out
    _no_course_answer_headings(out)


def test_quiz_render_no_evidence_returns_clarification_fallback():
    """Empty evidence -> falls back to clarification, never fabricates questions."""
    plan = _plan([])
    sq = _structured_query(_intent("Quiz me on something", concepts=[]))
    out = format_quiz_markdown(plan, [], sq)

    assert "Quiz:" in out
    # No Q1/Q2/Q3 numbered prompts when evidence is empty.
    assert "Answer Key:" not in out
    assert "1." not in out
    assert "True or false:" not in out
    _no_course_answer_headings(out)
