"""Renderer contract validation — cross-mode contamination checks.

Each mode's rendered output must NOT leak headings that belong to other modes.
This file tests that invariant with parametrized queries and deliberate
mode_override / query-intent mismatches.
"""

from __future__ import annotations

import json

import pytest

from app.extensions import db
from app.models import LectureChunk
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

from tests.conftest import register_user

_PW = "Abcd1234!"

_counter = 0


def _unique_email(prefix: str) -> str:
    global _counter
    _counter += 1
    return f"{prefix}-{_counter}@contract.dev"


def _login(client, email: str) -> None:
    register_user(client, email, _PW)
    client.post(
        "/api/auth/login",
        json={"email": email, "password": _PW},
        content_type="application/json",
    )


def _open_chat_session(client) -> int:
    return client.post(
        "/api/sessions",
        json={"title": "t"},
        content_type="application/json",
    ).get_json()["session"]["id"]


def _post_chat(client, sid: int, message: str, **extra) -> dict:
    payload = {"session_id": sid, "message": message, **extra}
    resp = client.post(
        "/api/chat",
        json=payload,
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


# ---------------------------------------------------------------------------
# Shared fixture — seed chunks once per test class
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed_chunks(app):
    """Seed lecture chunks for all renderer-contract tests."""
    with app.app_context():
        db.session.add_all(
            [
                # Lecture 10 — MFCC / formant / spectrum (3 chunks)
                LectureChunk(
                    chunk_key="rc-mfcc-1",
                    lecture_number=10,
                    topic="MFCCs — Core Idea",
                    keywords=json.dumps(["mfcc", "speech", "spectrum"]),
                    source_excerpt="MFCCs summarize the spectrum of speech as a small vector.",
                    clean_explanation="MFCCs summarize the spectrum of speech as a small vector.",
                    sample_questions=json.dumps(["What do MFCCs summarize?"]),
                    sample_answer="The spectrum of speech.",
                ),
                LectureChunk(
                    chunk_key="rc-mfcc-2",
                    lecture_number=10,
                    topic="MFCCs — Pipeline",
                    keywords=json.dumps(["mfcc", "filterbank", "log"]),
                    source_excerpt="The MFCC pipeline applies a filterbank, takes logs, and runs a DCT.",
                    clean_explanation="The MFCC pipeline applies a filterbank, takes logs, and runs a DCT.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                LectureChunk(
                    chunk_key="rc-formant-1",
                    lecture_number=10,
                    topic="Formants — Core Idea",
                    keywords=json.dumps(["formant", "vowel", "spectrum"]),
                    source_excerpt="Formants are spectral peaks tied to vocal tract shape.",
                    clean_explanation="Formants are spectral peaks tied to vocal tract shape.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                # Lecture 14 — softmax / hardmax (2 chunks)
                LectureChunk(
                    chunk_key="rc-softmax-1",
                    lecture_number=14,
                    topic="Softmax — Core Idea",
                    keywords=json.dumps(["softmax", "probability", "logits"]),
                    source_excerpt="Softmax turns logits into a probability distribution.",
                    clean_explanation="Softmax turns logits into a probability distribution.",
                    sample_questions=json.dumps(["What does softmax produce?"]),
                    sample_answer="A probability distribution over classes.",
                ),
                LectureChunk(
                    chunk_key="rc-hardmax-1",
                    lecture_number=14,
                    topic="Hardmax — Core Idea",
                    keywords=json.dumps(["hardmax", "argmax", "one-hot"]),
                    source_excerpt="Hardmax picks the argmax and returns a one-hot vector.",
                    clean_explanation="Hardmax picks the argmax and returns a one-hot vector.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                # Lecture 15 — CNN / MLP (2 chunks)
                LectureChunk(
                    chunk_key="rc-cnn-1",
                    lecture_number=15,
                    topic="CNN — Core Idea",
                    keywords=json.dumps(["cnn", "convolution", "kernel"]),
                    source_excerpt="A CNN slides convolutional kernels across the input to extract spatial features.",
                    clean_explanation="A CNN slides convolutional kernels across the input to extract spatial features.",
                    sample_questions=json.dumps(["What does a CNN do?"]),
                    sample_answer="Extract spatial features with shared kernels.",
                ),
                LectureChunk(
                    chunk_key="rc-mlp-1",
                    lecture_number=15,
                    topic="MLP — Core Idea",
                    keywords=json.dumps(["mlp", "feedforward", "fully connected"]),
                    source_excerpt="An MLP is a fully connected feedforward network of dense layers.",
                    clean_explanation="An MLP is a fully connected feedforward network of dense layers.",
                    sample_questions=json.dumps(["What is an MLP?"]),
                    sample_answer="A fully connected feedforward network.",
                ),
                # Lecture 8 — backprop (1 chunk)
                LectureChunk(
                    chunk_key="rc-backprop-1",
                    lecture_number=8,
                    topic="Backpropagation — Core Idea",
                    keywords=json.dumps(["backpropagation", "gradient", "chain rule"]),
                    source_excerpt="Backpropagation computes gradients via the chain rule.",
                    clean_explanation="Backpropagation computes gradients via the chain rule.",
                    sample_questions=json.dumps(["What does backpropagation compute?"]),
                    sample_answer="Gradients of the loss with respect to each weight.",
                ),
                # Lecture 14 — attention (1 chunk)
                LectureChunk(
                    chunk_key="rc-attention-1",
                    lecture_number=14,
                    topic="Attention — Core Idea",
                    keywords=json.dumps(["attention", "query", "key", "value"]),
                    source_excerpt="Attention computes a weighted sum of values using query-key similarity.",
                    clean_explanation="Attention computes a weighted sum of values using query-key similarity.",
                    sample_questions=json.dumps(["How does attention work?"]),
                    sample_answer="It weights values by query-key similarity.",
                ),
                # Lecture 7 — dynamic programming (1 chunk)
                LectureChunk(
                    chunk_key="rc-dp-1",
                    lecture_number=7,
                    topic="Dynamic Programming — Core Idea",
                    keywords=json.dumps(["dynamic programming", "dp", "memoization"]),
                    source_excerpt="Dynamic programming solves overlapping subproblems by caching results.",
                    clean_explanation="Dynamic programming solves overlapping subproblems by caching results.",
                    sample_questions=json.dumps(["What is dynamic programming?"]),
                    sample_answer="An algorithm design technique that caches subproblem results.",
                ),
            ]
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()


# ---------------------------------------------------------------------------
# Forbidden-heading sets per mode
# ---------------------------------------------------------------------------

QUIZ_FORBIDDEN = (
    "Course Answer:",
    "### Direct Answer",
    "### Explanation",
    "### Example / Intuition",
    "### Why it matters",
)

SUMMARY_FORBIDDEN = (
    "Course Answer:",
    "### Direct Answer",
    "### Explanation",
)

COMPARE_FORBIDDEN = (
    "Quiz:",
    "Summary:",
    "### Example / Intuition",
    "### Why it matters",
)

CHAT_FORBIDDEN_START = (
    "Quiz:",
    "Summary:",
)


# ===================================================================
# Quiz renderer contract
# ===================================================================

class TestQuizRendererContract:

    @pytest.mark.parametrize("query", [
        "Quiz me on MFCCs",
        "Test me on softmax",
        "Quiz me on CNNs",
    ])
    def test_quiz_output_has_correct_headings_and_no_contamination(
        self, client, app, query,
    ):
        email = _unique_email("quiz-contract")
        _login(client, email)
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, query, mode_override="quiz")

        answer = body["answer"]
        assert "Quiz:" in answer, f"quiz output missing 'Quiz:' header: {answer[:300]}"
        assert "Answer Key:" in answer, f"quiz output missing 'Answer Key:': {answer[:300]}"

        for heading in QUIZ_FORBIDDEN:
            assert heading not in answer, (
                f"quiz output contaminated with '{heading}': {answer[:400]}"
            )


# ===================================================================
# Summary renderer contract
# ===================================================================

class TestSummaryRendererContract:

    @pytest.mark.parametrize("query", [
        "Summarize Lecture 10",
        "Summary of Lecture 15",
        "Recap of backpropagation",
    ])
    def test_summary_output_has_correct_heading_and_no_contamination(
        self, client, app, query,
    ):
        email = _unique_email("summary-contract")
        _login(client, email)
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, query, mode_override="summary")

        answer = body["answer"]
        assert "Summary:" in answer, f"summary output missing 'Summary:' header: {answer[:300]}"

        for heading in SUMMARY_FORBIDDEN:
            assert heading not in answer, (
                f"summary output contaminated with '{heading}': {answer[:400]}"
            )


# ===================================================================
# Compare renderer contract
# ===================================================================

class TestCompareRendererContract:

    @pytest.mark.parametrize("query,entity_a,entity_b", [
        ("Compare CNN and MLP", "CNN", "MLP"),
        ("Compare softmax and hardmax", "softmax", "hardmax"),
        ("Difference between MFCCs and formants", "MFCC", "formant"),
    ])
    def test_compare_output_mentions_entities_and_no_contamination(
        self, client, app, query, entity_a, entity_b,
    ):
        email = _unique_email("compare-contract")
        _login(client, email)
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, query, mode_override="compare")

        answer = body["answer"]
        answer_lower = answer.lower()
        assert entity_a.lower() in answer_lower, (
            f"compare output missing '{entity_a}': {answer[:400]}"
        )
        assert entity_b.lower() in answer_lower, (
            f"compare output missing '{entity_b}': {answer[:400]}"
        )

        for heading in COMPARE_FORBIDDEN:
            assert heading not in answer, (
                f"compare output contaminated with '{heading}': {answer[:400]}"
            )


# ===================================================================
# Chat renderer contract
# ===================================================================

class TestChatRendererContract:

    @pytest.mark.parametrize("query", [
        "What is backpropagation?",
        "How does softmax work?",
        "Explain attention.",
    ])
    def test_chat_output_has_course_answer_format_and_no_contamination(
        self, client, app, query,
    ):
        email = _unique_email("chat-contract")
        _login(client, email)
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, query)

        answer = body["answer"]
        assert (
            "Course Answer:" in answer or "### Direct Answer" in answer
        ), f"chat output missing Course Answer format: {answer[:400]}"

        for marker in CHAT_FORBIDDEN_START:
            assert not answer.startswith(marker), (
                f"chat output starts with forbidden '{marker}': {answer[:300]}"
            )


# ===================================================================
# Cross-contamination (mismatched mode_override vs query intent)
# ===================================================================

class TestCrossContamination:

    def test_quiz_override_on_compare_query_follows_quiz_contract(
        self, client, app,
    ):
        _login(client, _unique_email("cross-quiz"))
        sid = _open_chat_session(client)
        body = _post_chat(
            client, sid, "Compare CNN and MLP", mode_override="quiz",
        )
        answer = body["answer"]

        assert "Quiz:" in answer, f"expected Quiz: header: {answer[:300]}"
        for heading in QUIZ_FORBIDDEN:
            assert heading not in answer, (
                f"quiz-override output contaminated with '{heading}': {answer[:400]}"
            )

    def test_summary_override_on_quiz_query_follows_summary_contract(
        self, client, app,
    ):
        _login(client, _unique_email("cross-summary"))
        sid = _open_chat_session(client)
        body = _post_chat(
            client, sid, "Quiz me on MFCCs", mode_override="summary",
        )
        answer = body["answer"]

        assert "Summary:" in answer, f"expected Summary: header: {answer[:300]}"
        for heading in SUMMARY_FORBIDDEN:
            assert heading not in answer, (
                f"summary-override output contaminated with '{heading}': {answer[:400]}"
            )

    def test_compare_override_on_summary_query_follows_compare_contract(
        self, client, app,
    ):
        _login(client, _unique_email("cross-compare"))
        sid = _open_chat_session(client)
        body = _post_chat(
            client, sid, "Summarize Lecture 10", mode_override="compare",
        )
        answer = body["answer"]

        for heading in COMPARE_FORBIDDEN:
            assert heading not in answer, (
                f"compare-override output contaminated with '{heading}': {answer[:400]}"
            )

    def test_chat_override_on_quiz_query_follows_chat_contract(
        self, client, app,
    ):
        _login(client, _unique_email("cross-chat"))
        sid = _open_chat_session(client)
        body = _post_chat(
            client, sid, "Quiz me on MFCCs", mode_override="chat",
        )
        answer = body["answer"]

        assert (
            "Course Answer:" in answer or "### Direct Answer" in answer
        ), f"chat-override output missing Course Answer format: {answer[:400]}"
        assert not answer.startswith("Quiz:"), (
            f"chat-override output starts with Quiz:: {answer[:300]}"
        )
