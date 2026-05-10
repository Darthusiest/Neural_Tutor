"""Comprehensive mode-routing tests for ``POST /api/chat``.

Each test verifies that a mode trigger produces the correct renderer output.
No knowledge coverage, security, or edge-case tests — purely routing validation.

Modes tested:
- quiz   -> Quiz: + Answer Key, no Course Answer headings
- compare -> both entity labels present, no quiz/summary markers
- summary -> Summary: header + topic layout, no Course Answer headings
- chat    -> Course Answer / Direct Answer, no quiz/summary markers
"""

from __future__ import annotations

import json

import pytest

from app.extensions import db
from app.models import LectureChunk
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

from tests.conftest import register_user

_PW = "Abcd1234!"


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
    response = client.post(
        "/api/chat",
        json=payload,
        content_type="application/json",
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _seed_route_chunks(app) -> None:
    """Seed minimal lecture chunks for retrieval to find evidence for routing tests."""
    with app.app_context():
        db.session.add_all(
            [
                # --- MFCC chunks (lecture 10) ---
                LectureChunk(
                    chunk_key="comp-mfcc-1",
                    lecture_number=10,
                    topic="MFCCs — Core Idea",
                    keywords=json.dumps(["mfcc", "speech", "spectrum"]),
                    source_excerpt="MFCCs summarize the spectrum of speech as a small vector.",
                    clean_explanation="MFCCs summarize the spectrum of speech as a small vector.",
                    sample_questions=json.dumps(["What do MFCCs summarize?"]),
                    sample_answer="The spectrum of speech.",
                ),
                LectureChunk(
                    chunk_key="comp-mfcc-2",
                    lecture_number=10,
                    topic="MFCCs — Pipeline",
                    keywords=json.dumps(["mfcc", "filterbank", "log"]),
                    source_excerpt="The MFCC pipeline applies a filterbank, takes logs, and runs a DCT.",
                    clean_explanation="The MFCC pipeline applies a filterbank, takes logs, and runs a DCT.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                LectureChunk(
                    chunk_key="comp-formant-1",
                    lecture_number=10,
                    topic="Formants — Core Idea",
                    keywords=json.dumps(["formant", "vowel", "spectrum"]),
                    source_excerpt="Formants are spectral peaks tied to vocal tract shape.",
                    clean_explanation="Formants are spectral peaks tied to vocal tract shape.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                # --- Softmax / Hardmax chunks (lecture 14) ---
                LectureChunk(
                    chunk_key="comp-softmax-1",
                    lecture_number=14,
                    topic="Softmax — Core Idea",
                    keywords=json.dumps(["softmax", "probability", "logits"]),
                    source_excerpt="Softmax turns logits into a probability distribution.",
                    clean_explanation="Softmax turns logits into a probability distribution.",
                    sample_questions=json.dumps(["What does softmax produce?"]),
                    sample_answer="A probability distribution over classes.",
                ),
                LectureChunk(
                    chunk_key="comp-hardmax-1",
                    lecture_number=14,
                    topic="Hardmax — Core Idea",
                    keywords=json.dumps(["hardmax", "argmax", "one-hot"]),
                    source_excerpt="Hardmax picks the argmax and returns a one-hot vector.",
                    clean_explanation="Hardmax picks the argmax and returns a one-hot vector.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                # --- CNN / MLP chunks (lecture 15) ---
                LectureChunk(
                    chunk_key="comp-cnn-1",
                    lecture_number=15,
                    topic="CNN — Core Idea",
                    keywords=json.dumps(["cnn", "convolution", "kernel"]),
                    source_excerpt="A CNN slides convolutional kernels across the input to extract spatial features.",
                    clean_explanation="A CNN slides convolutional kernels across the input to extract spatial features.",
                    sample_questions=json.dumps(["What does a CNN do?"]),
                    sample_answer="Extract spatial features with shared kernels.",
                ),
                LectureChunk(
                    chunk_key="comp-mlp-1",
                    lecture_number=15,
                    topic="MLP — Core Idea",
                    keywords=json.dumps(["mlp", "feedforward", "fully connected"]),
                    source_excerpt="An MLP is a fully connected feedforward network of dense layers.",
                    clean_explanation="An MLP is a fully connected feedforward network of dense layers.",
                    sample_questions=json.dumps(["What is an MLP?"]),
                    sample_answer="A fully connected feedforward network.",
                ),
                # --- Backprop chunk (lecture 8) ---
                LectureChunk(
                    chunk_key="comp-backprop-1",
                    lecture_number=8,
                    topic="Backpropagation — Core Idea",
                    keywords=json.dumps(["backpropagation", "gradient", "chain rule"]),
                    source_excerpt="Backpropagation computes gradients via the chain rule to update weights.",
                    clean_explanation="Backpropagation computes gradients via the chain rule to update weights.",
                    sample_questions=json.dumps(["How does backprop compute gradients?"]),
                    sample_answer="By applying the chain rule layer by layer.",
                ),
                # --- Attention chunk (lecture 14) ---
                LectureChunk(
                    chunk_key="comp-attention-1",
                    lecture_number=14,
                    topic="Attention — Core Idea",
                    keywords=json.dumps(["attention", "query", "key", "value"]),
                    source_excerpt="Attention computes a weighted sum of values using query-key similarity.",
                    clean_explanation="Attention computes a weighted sum of values using query-key similarity.",
                    sample_questions=json.dumps(["What does attention compute?"]),
                    sample_answer="A weighted sum of values based on query-key similarity.",
                ),
                # --- Dynamic programming chunk (lecture 7) ---
                LectureChunk(
                    chunk_key="comp-dp-1",
                    lecture_number=7,
                    topic="Dynamic Programming — Core Idea",
                    keywords=json.dumps(["dynamic programming", "dp", "subproblem"]),
                    source_excerpt="Dynamic programming solves overlapping subproblems by caching results.",
                    clean_explanation="Dynamic programming solves overlapping subproblems by caching results.",
                    sample_questions=json.dumps(["What is dynamic programming?"]),
                    sample_answer="A method for solving overlapping subproblems via memoization.",
                ),
                # --- Dropout / Layer norm chunks (lecture 17) ---
                LectureChunk(
                    chunk_key="comp-dropout-1",
                    lecture_number=17,
                    topic="Dropout — Core Idea",
                    keywords=json.dumps(["dropout", "regularization", "random"]),
                    source_excerpt="Dropout randomly zeroes activations during training to reduce overfitting.",
                    clean_explanation="Dropout randomly zeroes activations during training to reduce overfitting.",
                    sample_questions=json.dumps(["What does dropout do?"]),
                    sample_answer="Randomly zeroes activations to regularize the network.",
                ),
                LectureChunk(
                    chunk_key="comp-layernorm-1",
                    lecture_number=17,
                    topic="Layer Normalization — Core Idea",
                    keywords=json.dumps(["layer normalization", "norm", "mean", "variance"]),
                    source_excerpt="Layer normalization normalizes activations across features for each sample.",
                    clean_explanation="Layer normalization normalizes activations across features for each sample.",
                    sample_questions=json.dumps(["What does layer norm do?"]),
                    sample_answer="Normalizes across features per sample.",
                ),
                # --- Bias / Variance chunks (lecture 16) ---
                LectureChunk(
                    chunk_key="comp-bias-1",
                    lecture_number=16,
                    topic="Bias — Core Idea",
                    keywords=json.dumps(["bias", "underfitting", "model capacity"]),
                    source_excerpt="High bias means the model is too simple and underfits the data.",
                    clean_explanation="High bias means the model is too simple and underfits the data.",
                    sample_questions=json.dumps(["What does high bias indicate?"]),
                    sample_answer="The model is too simple and underfits.",
                ),
                LectureChunk(
                    chunk_key="comp-variance-1",
                    lecture_number=16,
                    topic="Variance — Core Idea",
                    keywords=json.dumps(["variance", "overfitting", "generalization"]),
                    source_excerpt="High variance means the model overfits training data and generalizes poorly.",
                    clean_explanation="High variance means the model overfits training data and generalizes poorly.",
                    sample_questions=json.dumps(["What does high variance indicate?"]),
                    sample_answer="The model overfits and generalizes poorly.",
                ),
                # --- Transformer chunk (lecture 14, for compare with CNN) ---
                LectureChunk(
                    chunk_key="comp-transformer-1",
                    lecture_number=14,
                    topic="Transformer — Core Idea",
                    keywords=json.dumps(["transformer", "self-attention", "encoder"]),
                    source_excerpt="A transformer uses self-attention instead of recurrence to model sequences.",
                    clean_explanation="A transformer uses self-attention instead of recurrence to model sequences.",
                    sample_questions=json.dumps(["What is a transformer?"]),
                    sample_answer="A model that uses self-attention instead of recurrence.",
                ),
                # --- Lecture 11 (quiz routing tests expect lecture-scoped hits) ---
                LectureChunk(
                    chunk_key="comp-lec11-1",
                    lecture_number=11,
                    topic="Lecture 11 — Review",
                    keywords=json.dumps(["lecture 11", "review"]),
                    source_excerpt="Lecture 11 reviews core neural methods covered earlier in the term.",
                    clean_explanation="Lecture 11 reviews core neural methods covered earlier in the term.",
                    sample_questions=json.dumps(["What does Lecture 11 cover?"]),
                    sample_answer="A review of core neural methods.",
                ),
            ]
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()


_QUIZ_FORBIDDEN = (
    "Course Answer:",
    "### Direct Answer",
    "### Explanation",
    "### Example / Intuition",
    "### Why it matters",
)

_COMPARE_FORBIDDEN = (
    "Quiz:",
    "### Example / Intuition",
    "### Why it matters",
)

_SUMMARY_FORBIDDEN = (
    "Course Answer:",
    "### Direct Answer",
    "### Explanation",
)

_CHAT_FORBIDDEN = (
    "Quiz:",
    "Summary:",
)


# ---------------------------------------------------------------------------
# Quiz routing
# ---------------------------------------------------------------------------

class TestQuizRouting:
    """Quiz triggers must produce Quiz: + Answer Key and never Course Answer headings."""

    @pytest.fixture(autouse=True)
    def seed(self, app):
        _seed_route_chunks(app)

    def test_quiz_me_on_mfccs(self, client):
        _login(client, "comp-quiz-mfcc@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Quiz me on MFCCs", mode_override="quiz")

        assert body["mode"]["effective"] == "quiz"
        answer = body["answer"]
        assert "Quiz:" in answer
        assert "Answer Key:" in answer
        assert "1." in answer
        for marker in _QUIZ_FORBIDDEN:
            assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"

    def test_quiz_lecture_11_mode_override(self, client):
        _login(client, "comp-quiz-lec11@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Test me on Lecture 11", mode_override="quiz")

        assert body["mode"]["effective"] == "quiz"
        answer = body["answer"]
        assert "Quiz:" in answer
        assert "Answer Key:" in answer
        for marker in _QUIZ_FORBIDDEN:
            assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"

    def test_practice_quiz_softmax(self, client):
        _login(client, "comp-quiz-softmax@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Give me a practice quiz on softmax", mode_override="quiz")

        assert body["mode"]["effective"] == "quiz"
        answer = body["answer"]
        assert "Quiz:" in answer
        assert "Answer Key:" in answer
        for marker in _QUIZ_FORBIDDEN:
            assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"

    def test_quiz_backpropagation(self, client):
        _login(client, "comp-quiz-backprop@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Ask me questions about backpropagation", mode_override="quiz")

        assert body["mode"]["effective"] == "quiz"
        answer = body["answer"]
        assert "Quiz:" in answer
        assert "Answer Key:" in answer
        for marker in _QUIZ_FORBIDDEN:
            assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"

    def test_quiz_attention(self, client):
        _login(client, "comp-quiz-attn@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Check my understanding of attention", mode_override="quiz")

        assert body["mode"]["effective"] == "quiz"
        answer = body["answer"]
        assert "Quiz:" in answer
        assert "Answer Key:" in answer
        for marker in _QUIZ_FORBIDDEN:
            assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"

    def test_quiz_cnns(self, client):
        _login(client, "comp-quiz-cnn@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Pop quiz on CNNs", mode_override="quiz")

        assert body["mode"]["effective"] == "quiz"
        answer = body["answer"]
        assert "Quiz:" in answer
        assert "Answer Key:" in answer
        for marker in _QUIZ_FORBIDDEN:
            assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"

    def test_quiz_dynamic_programming(self, client):
        _login(client, "comp-quiz-dp@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Drill me on dynamic programming", mode_override="quiz")

        assert body["mode"]["effective"] == "quiz"
        answer = body["answer"]
        assert "Quiz:" in answer
        assert "Answer Key:" in answer
        for marker in _QUIZ_FORBIDDEN:
            assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"


# ---------------------------------------------------------------------------
# Compare routing
# ---------------------------------------------------------------------------

class TestCompareRouting:
    """Compare triggers must surface both entity labels and never quiz/summary markers."""

    @pytest.fixture(autouse=True)
    def seed(self, app):
        _seed_route_chunks(app)

    def test_compare_cnn_and_transformer(self, client):
        _login(client, "comp-cmp-cnn-trans@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Compare CNN and transformer", mode_override="compare")

        assert body["mode"]["effective"] == "compare"
        answer = body["answer"]
        assert "CNN" in answer, f"compare output missing 'CNN': {answer[:400]}"
        for marker in _COMPARE_FORBIDDEN:
            assert marker not in answer, f"compare output unexpectedly contains '{marker}'"

    def test_compare_mfccs_and_formants(self, client):
        _login(client, "comp-cmp-mfcc-form@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Difference between MFCCs and formants", mode_override="compare")

        assert body["mode"]["effective"] == "compare"
        answer = body["answer"]
        assert "MFCC" in answer or "mfcc" in answer.lower(), f"compare output missing 'MFCC': {answer[:400]}"
        assert "formant" in answer.lower(), f"compare output missing 'formant': {answer[:400]}"
        for marker in _COMPARE_FORBIDDEN:
            assert marker not in answer, f"compare output unexpectedly contains '{marker}'"

    def test_compare_softmax_and_hardmax(self, client):
        _login(client, "comp-cmp-soft-hard@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "How is softmax different from hardmax?", mode_override="compare")

        assert body["mode"]["effective"] == "compare"
        answer = body["answer"]
        assert "softmax" in answer.lower(), f"compare output missing 'softmax': {answer[:400]}"
        assert "hardmax" in answer.lower(), f"compare output missing 'hardmax': {answer[:400]}"
        for marker in _COMPARE_FORBIDDEN:
            assert marker not in answer, f"compare output unexpectedly contains '{marker}'"

    def test_compare_bias_and_variance(self, client):
        _login(client, "comp-cmp-bias-var@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Contrast bias and variance", mode_override="compare")

        assert body["mode"]["effective"] == "compare"
        answer = body["answer"]
        assert "bias" in answer.lower(), f"compare output missing 'bias': {answer[:400]}"
        assert "variance" in answer.lower(), f"compare output missing 'variance': {answer[:400]}"
        for marker in _COMPARE_FORBIDDEN:
            assert marker not in answer, f"compare output unexpectedly contains '{marker}'"

    def test_compare_cnn_vs_mlp(self, client):
        _login(client, "comp-cmp-cnn-mlp@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "CNN vs MLP", mode_override="compare")

        assert body["mode"]["effective"] == "compare"
        answer = body["answer"]
        assert "CNN" in answer, f"compare output missing 'CNN': {answer[:400]}"
        assert "MLP" in answer, f"compare output missing 'MLP': {answer[:400]}"
        for marker in _COMPARE_FORBIDDEN:
            assert marker not in answer, f"compare output unexpectedly contains '{marker}'"

    def test_compare_dropout_and_layer_norm(self, client):
        _login(client, "comp-cmp-drop-ln@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Distinguish dropout from layer normalization", mode_override="compare")

        assert body["mode"]["effective"] == "compare"
        answer = body["answer"]
        assert "dropout" in answer.lower(), f"compare output missing 'dropout': {answer[:400]}"
        assert "normalization" in answer.lower() or "norm" in answer.lower(), (
            f"compare output missing 'layer norm': {answer[:400]}"
        )
        for marker in _COMPARE_FORBIDDEN:
            assert marker not in answer, f"compare output unexpectedly contains '{marker}'"


# ---------------------------------------------------------------------------
# Summary routing
# ---------------------------------------------------------------------------

class TestSummaryRouting:
    """Summary triggers must produce Summary: header + topic layout, never Course Answer headings."""

    @pytest.fixture(autouse=True)
    def seed(self, app):
        _seed_route_chunks(app)

    def test_summarize_lecture_10(self, client):
        _login(client, "comp-sum-lec10@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Summarize Lecture 10", mode_override="summary")

        assert body["mode"]["effective"] == "summary"
        answer = body["answer"]
        assert "Summary: Lecture 10" in answer
        assert "### Main idea" in answer
        assert "### Key topics" in answer
        for marker in _SUMMARY_FORBIDDEN:
            assert marker not in answer, f"summary output unexpectedly contains '{marker}'"

    def test_summary_recap_mfccs(self, client):
        _login(client, "comp-sum-mfcc@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Give me a recap of MFCCs", mode_override="summary")

        assert body["mode"]["effective"] == "summary"
        answer = body["answer"]
        assert "Summary:" in answer
        assert "### Core idea" in answer or "### Main idea" in answer
        for marker in _SUMMARY_FORBIDDEN:
            assert marker not in answer, f"summary output unexpectedly contains '{marker}'"

    def test_summary_lecture_15(self, client):
        _login(client, "comp-sum-lec15@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Summary of Lecture 15", mode_override="summary")

        assert body["mode"]["effective"] == "summary"
        answer = body["answer"]
        assert "Summary:" in answer
        for marker in _SUMMARY_FORBIDDEN:
            assert marker not in answer, f"summary output unexpectedly contains '{marker}'"

    def test_summary_lecture_16(self, client):
        _login(client, "comp-sum-lec16@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Main ideas of Lecture 16", mode_override="summary")

        assert body["mode"]["effective"] == "summary"
        answer = body["answer"]
        assert "Summary:" in answer
        for marker in _SUMMARY_FORBIDDEN:
            assert marker not in answer, f"summary output unexpectedly contains '{marker}'"

    def test_summary_attention(self, client):
        _login(client, "comp-sum-attn@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Overview of attention", mode_override="summary")

        assert body["mode"]["effective"] == "summary"
        answer = body["answer"]
        assert "Summary:" in answer
        for marker in _SUMMARY_FORBIDDEN:
            assert marker not in answer, f"summary output unexpectedly contains '{marker}'"

    def test_summary_lecture_8(self, client):
        _login(client, "comp-sum-lec8@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "TL;DR Lecture 8", mode_override="summary")

        assert body["mode"]["effective"] == "summary"
        answer = body["answer"]
        assert "Summary:" in answer
        for marker in _SUMMARY_FORBIDDEN:
            assert marker not in answer, f"summary output unexpectedly contains '{marker}'"


# ---------------------------------------------------------------------------
# Chat fallback
# ---------------------------------------------------------------------------

class TestChatFallback:
    """Plain questions with no mode_override should route to chat and produce Course Answer format."""

    @pytest.fixture(autouse=True)
    def seed(self, app):
        _seed_route_chunks(app)

    def test_chat_backpropagation(self, client):
        _login(client, "comp-chat-backprop@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "What is backpropagation?")

        assert body["mode"]["effective"] == "chat"
        answer = body["answer"]
        assert "Course Answer:" in answer or "### Direct Answer" in answer
        for marker in _CHAT_FORBIDDEN:
            assert marker not in answer, f"chat output unexpectedly contains '{marker}'"

    def test_chat_learning_in_neural_networks(self, client):
        _login(client, "comp-chat-learning@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "How does learning work in neural networks?")

        assert body["mode"]["effective"] == "chat"
        answer = body["answer"]
        assert "Course Answer:" in answer or "### Direct Answer" in answer
        for marker in _CHAT_FORBIDDEN:
            assert marker not in answer, f"chat output unexpectedly contains '{marker}'"

    def test_chat_chain_rule(self, client):
        _login(client, "comp-chat-chain@test.dev")
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, "Explain the chain rule.")

        assert body["mode"]["effective"] == "chat"
        answer = body["answer"]
        assert "Course Answer:" in answer or "### Direct Answer" in answer
        for marker in _CHAT_FORBIDDEN:
            assert marker not in answer, f"chat output unexpectedly contains '{marker}'"
