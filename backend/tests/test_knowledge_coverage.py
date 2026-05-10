"""Knowledge-coverage tests for the LING 487 tutor.

Verifies that every concept in the knowledge base produces a grounded,
concept-pure answer through the full chat pipeline.  One parametrized
test class covers all 47 concepts.

Markers: ``@pytest.mark.slow`` — these hit the complete retrieval +
compose pipeline with the real corpus.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.extensions import db
from app.models import LectureChunk
from app.services.knowledge.kb_chunk_audit import audit_kb_chunk_coverage
from app.services.knowledge.concept_kb import reset_kb_for_tests
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

from tests.conftest import register_user

_DATA = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"
_PW = "Abcd1234!"


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        import_lecture_json(_DATA, upsert=False)
        invalidate_lecture_cache()
        load_lecture_cache()
    yield
    reset_kb_for_tests()


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


def _post_chat(client, sid: int, message: str) -> dict:
    payload = {"session_id": sid, "message": message}
    response = client.post(
        "/api/chat",
        json=payload,
        content_type="application/json",
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def _ci_contains(haystack: str, needle: str) -> bool:
    """Case-insensitive substring check, tolerating hyphen/space variants."""
    return bool(re.search(re.escape(needle).replace(r"\ ", r"[\s-]"), haystack, re.IGNORECASE))


# ---------------------------------------------------------------------------
# 47 concept cases: (concept_id, query, expected_terms, forbidden_terms)
# ---------------------------------------------------------------------------

_CONCEPTS = [
    # Lectures 4-8 — Foundations
    ("weights_biases", "What are weights and biases in a neural network?", ["weight", "bias"], ["softmax", "attention"]),
    ("linear_nn", "Explain how a linear neural network works.", ["linear"], ["convolution", "attention"]),
    ("sigmoid", "What does the sigmoid function do?", ["sigmoid"], ["softmax", "attention"]),
    ("universal_approximation", "What is the universal approximation theorem?", ["universal", "approximation"], ["convolution"]),
    ("inference", "How does inference work in a neural network?", ["inference", "forward"], ["diffusion"]),
    ("learning", "How does a neural network learn?", ["learn"], ["diffusion", "distillation"]),
    ("inner_product", "What is an inner product?", ["inner product"], ["convolution"]),
    ("representations", "What are representations in neural networks?", ["representation"], []),
    ("speech_vector_prediction", "Explain speech vector prediction.", ["speech", "vector"], ["diffusion"]),
    ("sgd", "What is stochastic gradient descent?", ["gradient"], ["attention", "transformer"]),
    ("loss", "What is a loss function?", ["loss"], ["attention"]),
    ("gradient", "What is a gradient?", ["gradient"], ["attention", "transformer"]),
    ("chain_rule", "Explain the chain rule in backpropagation.", ["chain rule"], ["attention"]),
    ("backpropagation", "How does backpropagation work?", ["backprop"], ["attention", "convolution"]),
    ("dynamic_programming", "What is dynamic programming?", ["dynamic programming"], ["neural network", "backprop"]),
    ("value", "What is value in dynamic programming?", ["value"], ["attention", "transformer"]),
    ("greedy_algorithm", "Explain the greedy algorithm.", ["greedy"], ["gradient", "transformer"]),
    ("exhaustive_algorithm", "What is an exhaustive algorithm?", ["exhaustive"], ["gradient", "transformer"]),
    # Lectures 9-13 — Applications & LLMs
    ("classification", "What is classification in neural networks?", ["classif"], ["diffusion"]),
    ("softmax", "How does softmax work?", ["softmax", "probability"], ["hardmax"]),
    ("hardmax", "What is hardmax?", ["hardmax"], []),
    ("temperature", "What is temperature in softmax?", ["temperature"], ["diffusion"]),
    ("spectrum", "What is a spectrum in speech processing?", ["spectrum"], ["transformer"]),
    ("formants", "What are formants?", ["formant"], ["softmax", "transformer"]),
    ("mfcc", "What are MFCCs?", ["mfcc"], ["softmax", "transformer"]),
    ("bias_variance", "Explain bias and variance.", ["bias", "variance"], ["attention"]),
    ("train_test_split", "What is train/test split?", ["train", "test"], ["diffusion"]),
    ("autoencoder", "How does an autoencoder work?", ["autoencoder"], ["transformer", "attention"]),
    ("llm", "What is a large language model?", ["language model"], ["formant", "mfcc"]),
    ("phonotactics", "What are phonotactics?", ["phonotactic"], ["convolution"]),
    # Lectures 14-18 — Transformers
    ("attention", "Explain the attention mechanism.", ["attention"], ["convolution", "mfcc"]),
    ("qkv", "What are query, key, and value in attention?", ["query", "key", "value"], ["mfcc", "formant"]),
    ("multi_head_attention", "What is multi-head attention?", ["multi-head", "attention"], ["mfcc"]),
    ("transformer", "Explain the transformer architecture.", ["transformer"], ["mfcc", "formant"]),
    ("feedforward", "What are feedforward layers in a transformer?", ["feedforward"], ["mfcc"]),
    ("cnn", "What is a convolutional neural network?", ["convolution"], ["self-attention", "transformer"]),
    ("residual_stream", "What is a residual connection?", ["residual"], ["mfcc", "formant"]),
    ("layer_norm", "What is layer normalization?", ["normalization"], ["mfcc"]),
    ("dropout", "What is dropout?", ["dropout"], ["mfcc", "formant"]),
    ("positional_encoding", "What is positional encoding?", ["positional"], ["mfcc", "formant"]),
    # Lectures 19-20 — Advanced
    ("vector_quantization", "What is vector quantization?", ["vector quantization"], ["attention", "transformer"]),
    ("rvq", "Explain residual vector quantization.", ["residual", "quantization"], ["attention"]),
    ("mimi", "What is the Mimi audio codec?", ["mimi"], ["transformer"]),
    ("distillation", "How does knowledge distillation work?", ["distillation"], ["mfcc", "formant"]),
    ("diffusion", "What is a diffusion model?", ["diffusion"], ["mfcc", "formant"]),
    ("generative_ai", "What is generative AI?", ["generative"], ["mfcc", "formant"]),
    ("structure_correlation", "Explain correlation and structure.", ["correlation"], ["mfcc"]),
]


def test_kb_chunk_audit_min_two_hits(app, corpus):
    """Every KB concept must appear in at least two lecture chunks (CI guard)."""
    from app.services.knowledge.concept_kb import get_kb

    with app.app_context():
        kb = get_kb()
        rows = LectureChunk.query.all()
        result = audit_kb_chunk_coverage(kb, rows, min_chunks=2)
    assert result.ok(), (
        "KB concepts with fewer than 2 chunk hits — expand LING487_SUPER_TUTOR "
        f"or aliases: {result.below_threshold}"
    )


# ---------------------------------------------------------------------------
# Parametrized test class
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestKnowledgeCoverage:
    """Each concept in the KB must produce a grounded, concept-pure answer."""

    @pytest.mark.parametrize(
        "concept_id, query, expected_terms, forbidden_terms",
        _CONCEPTS,
        ids=[c[0] for c in _CONCEPTS],
    )
    def test_concept(
        self,
        client,
        app,
        corpus,
        concept_id,
        query,
        expected_terms,
        forbidden_terms,
    ):
        email = f"kb-{concept_id}@test.dev"
        _login(client, email)
        sid = _open_chat_session(client)
        body = _post_chat(client, sid, query)
        answer = body["answer"]

        assert "Course Answer:" in answer, (
            f"[{concept_id}] missing 'Course Answer:' marker in response"
        )

        answer_lower = answer.lower()
        for term in expected_terms:
            assert _ci_contains(answer_lower, term), (
                f"[{concept_id}] expected '{term}' not found in answer"
            )

        opening = answer[:300].lower()
        for term in forbidden_terms:
            assert not _ci_contains(opening, term), (
                f"[{concept_id}] forbidden '{term}' dominates opening 300 chars"
            )
