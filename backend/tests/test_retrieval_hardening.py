"""Small hardening checks: field weights, diagnostics logging, compare/summary queries."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.extensions import db
from app.models import LectureChunk
from app.services.lecture_loader import import_lecture_json
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache, retrieve_chunks

_CORPUS = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"


@pytest.fixture
def loaded_corpus(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        import_lecture_json(_CORPUS, upsert=False)
        invalidate_lecture_cache()
        load_lecture_cache()
    yield


def test_topic_field_weight_beats_body_only(app):
    """Same rare token in topic should outrank the same token only in source_excerpt (FIELD_WEIGHTS)."""
    with app.app_context():
        db.drop_all()
        db.create_all()
        tok = "zzzweighttokunique"
        db.session.add_all(
            [
                LectureChunk(
                    chunk_key="test-weight-a",
                    lecture_number=1,
                    topic=f"Course A — {tok} in title",
                    keywords=json.dumps([tok]),
                    source_excerpt="filler filler filler",
                    clean_explanation="filler",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                LectureChunk(
                    chunk_key="test-weight-b",
                    lecture_number=2,
                    topic="Course B — other",
                    keywords=json.dumps(["other"]),
                    source_excerpt=f"paragraph mentioning {tok} once",
                    clean_explanation="filler",
                    sample_questions="[]",
                    sample_answer=None,
                ),
            ]
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()
        r = retrieve_chunks(tok, top_k=2)
        assert r.chunks and r.chunks[0]["lecture_number"] == 1


def test_retrieval_logs_scores_at_debug(app, caplog, loaded_corpus):
    with app.app_context():
        caplog.set_level(logging.DEBUG, logger="app.services.retrieval")
        retrieve_chunks("backpropagation", top_k=2)
    assert "top_score=" in caplog.text
    assert "score_margin=" in caplog.text
    assert "query_coverage=" in caplog.text


def test_compare_style_cnn_residual(loaded_corpus, app):
    """Compare is a stopword; content terms should still hit CNN / residual lecture."""
    with app.app_context():
        r = retrieve_chunks("compare CNN and residual connections for speech", top_k=2)
        assert r.chunks
        assert r.chunks[0]["lecture_number"] == 16


def test_summary_style_lecture_13(loaded_corpus, app):
    """Summary/summarize are stopwords; LLM + lecture hint should still land on 13."""
    with app.app_context():
        r = retrieve_chunks("summarize lecture 13 LLM autoregressive token prediction", top_k=2)
        assert r.chunks[0]["lecture_number"] == 13
        assert r.confidence >= 0.44


def test_week_query_dynamic_programming(loaded_corpus, app):
    with app.app_context():
        r = retrieve_chunks("week 7 DP subproblems", top_k=2)
        assert r.chunks[0]["lecture_number"] == 7


def test_ambiguous_softmax_top_two_are_softmax_related(loaded_corpus, app):
    """Generic softmax query: rank 1–2 should stay on softmax lectures (9 or 12); later ranks may drift."""
    with app.app_context():
        r = retrieve_chunks("softmax output probabilities distribution", top_k=4)
        top_two = [c["lecture_number"] for c in r.chunks[:2]]
        assert set(top_two).issubset({9, 12})
