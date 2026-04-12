"""Comprehensive golden tests for retrieval-v2 multi-strategy pipeline.

Tests realistic student queries: paraphrases, typos, aliases, compare,
summary, synthesis, and vague wording.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extensions import db
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache
from app.services.retrieval_v2 import EnhancedRetrievalResult, retrieve_enhanced

_DATA = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"


@pytest.fixture
def corpus(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        import_lecture_json(_DATA, upsert=False)
        invalidate_lecture_cache()
        load_lecture_cache()
    yield


# ---------------------------------------------------------------------------
# A. Definition queries
# ---------------------------------------------------------------------------

class TestDefinition:
    def test_what_is_backpropagation(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is backpropagation?")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 8
            assert isinstance(r, EnhancedRetrievalResult)

    def test_what_is_softmax(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is softmax?")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] in (9, 12)

    def test_what_is_autoencoder(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is an autoencoder?")
            assert r.chunks[0]["lecture_number"] == 9

    def test_explain_dynamic_programming(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("Explain dynamic programming")
            assert r.chunks[0]["lecture_number"] == 7


# ---------------------------------------------------------------------------
# B. Alias / acronym expansion
# ---------------------------------------------------------------------------

class TestAlias:
    def test_dp_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is DP?")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 7

    def test_backprop_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("explain backprop")
            assert r.chunks[0]["lecture_number"] == 8

    def test_cnn_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("How do CNNs work?")
            assert r.chunks[0]["lecture_number"] == 16

    def test_llm_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What are LLMs?")
            assert r.chunks[0]["lecture_number"] in (13, 18)

    def test_mfcc_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What are MFCCs?")
            assert r.chunks[0]["lecture_number"] == 10

    def test_qkv_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is QKV in attention?")
            assert r.chunks[0]["lecture_number"] == 14

    def test_sgd_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("How does SGD update weights?")
            assert r.chunks[0]["lecture_number"] in (4, 8)

    def test_asr_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("ASR speech recognition tasks")
            assert r.chunks[0]["lecture_number"] in (10, 13)

    def test_vq_resolves(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is VQ?")
            assert r.chunks[0]["lecture_number"] == 19


# ---------------------------------------------------------------------------
# C. Typo handling
# ---------------------------------------------------------------------------

class TestTypo:
    def test_backpropagtion(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("backpropagtion gradient computation")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 8

    def test_normalzation(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("normalzation techniques")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 17


# ---------------------------------------------------------------------------
# D. Compare queries — should pull from BOTH concept sides
# ---------------------------------------------------------------------------

class TestCompare:
    def test_bias_vs_variance(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("difference between bias and variance", top_k=4)
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 11
            assert r.query_intent is not None
            assert r.query_intent.query_type.value == "compare"

    def test_cnn_vs_residual(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("compare CNN and residual connections", top_k=4)
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 16

    def test_mfcc_vs_formants(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("difference between MFCCs and formants", top_k=4)
            assert r.chunks
            lecs = {c["lecture_number"] for c in r.chunks}
            assert 10 in lecs

    def test_softmax_classifier_vs_softmax_output(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("compare softmax classifier and softmax output layer", top_k=4)
            assert r.chunks
            lecs = {c["lecture_number"] for c in r.chunks}
            assert lecs & {9, 12}


# ---------------------------------------------------------------------------
# E. Summary queries — single-lecture recap (ranked, capped) from that lecture
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_lecture_10(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("summary of lecture 10", top_k=10)
            assert r.chunks
            assert all(c["lecture_number"] == 10 for c in r.chunks)
            assert len(r.chunks) >= 3

    def test_summarize_lecture_8(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("summarize lecture 8", top_k=10)
            assert r.chunks
            assert all(c["lecture_number"] == 8 for c in r.chunks)

    def test_overview_lecture_15(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("overview of lecture 15", top_k=10)
            assert r.chunks
            assert all(c["lecture_number"] == 15 for c in r.chunks)


# ---------------------------------------------------------------------------
# F. Cross-lecture synthesis
# ---------------------------------------------------------------------------

class TestSynthesis:
    def test_lectures_13_through_15(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("How do lectures 13 through 15 connect?", top_k=6)
            assert r.chunks
            lecs = {c["lecture_number"] for c in r.chunks}
            assert lecs & {13, 14, 15}

    def test_attention_and_transformer(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced(
                "How does attention relate to the transformer architecture?", top_k=5
            )
            lecs = {c["lecture_number"] for c in r.chunks}
            assert lecs & {14, 15}


# ---------------------------------------------------------------------------
# G. Paraphrases / student wording
# ---------------------------------------------------------------------------

class TestParaphrase:
    def test_compressing_input_autoencoder(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("compressing input into a small code then rebuilding it")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 9

    def test_weights_learn(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("how do neural nets learn from data")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] in (4, 8)

    def test_teacher_student_model(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("teacher model training a student model")
            assert r.chunks[0]["lecture_number"] == 19


# ---------------------------------------------------------------------------
# H. Enhanced result shape
# ---------------------------------------------------------------------------

class TestResultShape:
    def test_has_query_intent(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is softmax?")
            assert r.query_intent is not None
            assert r.query_intent.query_type.value in (
                "definition", "general", "lecture_specific",
            )

    def test_backward_compatible(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("backpropagation")
            assert hasattr(r, "chunks")
            assert hasattr(r, "confidence")
            assert hasattr(r, "detected_topic")
            assert isinstance(r.confidence, float)

    def test_synthesis_has_supporting(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("How do lectures 13 through 15 connect?", top_k=5)
            assert isinstance(r.supporting_chunks, list)

    def test_chunk_type_in_output(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is backpropagation?")
            assert r.chunks
            assert "chunk_type" in r.chunks[0]

    def test_concept_family_in_output(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("What is backpropagation?")
            assert "concept_family" in r.chunks[0]


# ---------------------------------------------------------------------------
# I. Lecture-specific with number
# ---------------------------------------------------------------------------

class TestLectureSpecific:
    def test_lecture_8_gradients(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("lecture 8 gradient computation")
            assert r.chunks
            assert r.chunks[0]["lecture_number"] == 8
            assert r.confidence >= 0.40

    def test_week_7_dp(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("week 7 DP subproblems")
            assert r.chunks[0]["lecture_number"] == 7


# ---------------------------------------------------------------------------
# J. Retrieval v2 hardening (regression)
# ---------------------------------------------------------------------------


class TestSummaryHardening:
    def test_single_lecture_summary_has_diagnostics(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("summary of lecture 10", top_k=10)
            assert r.diagnostics is not None
            assert r.diagnostics.chunk_hits
            assert all(c["lecture_number"] == 10 for c in r.chunks)


class TestCompareHardening:
    def test_compare_has_merged_diagnostics_and_side_tuple(self, corpus, app):
        with app.app_context():
            r = retrieve_enhanced("difference between bias and variance", top_k=4)
            assert r.query_intent is not None
            assert r.query_intent.compare_concepts is not None
            assert r.compare_side_diagnostics is not None
            assert r.diagnostics is not None
            assert len(r.diagnostics.chunk_hits) <= len(r.chunks)
