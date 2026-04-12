"""
Golden checks for LING487_SUPER_TUTOR import + keyword retrieval.

Sample queries and expected behavior (after ``import_lecture_json``):

+-------------------------------+------------------+--------------------------------+
| Query (student-style)         | Top lecture #    | Notes                          |
+===============================+==================+================================+
| backpropagation weights       | 8                | Core Idea / Steps              |
| Explain lecture 8 gradients   | 8                | conf ≥ 0.44 (lecture hint)     |
| dynamic programming subproblem| 7              | DP lecture                     |
| inner product similarity      | 6                | Inner product                  |
| MFCC speech recognition       | 10               | MFCCs section                  |
| softmax probabilities lec 12  | 12               | Explicit lecture disambiguates |
| softmax classifier            | 9 or 12          | Both mention softmax (OK)      |
| bias vs variance              | 11               | Either Bias or Variance chunk  |
| multi-head attention          | 15               | Transformer                    |
| autoencoder compression       | 9                | NN Applications                |
+-------------------------------+------------------+--------------------------------+

Weak / ambiguous cases (lexical v1): generic "softmax" without lecture number may
rank lecture 9 (Softmax Classifier) before 12 (Softmax lecture); prefer explicit
"lecture 12" or "softmax lecture" in the query.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extensions import db
from app.services.lecture_data import get_lecture_summary, list_topics_catalog
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache, retrieve_chunks

_DATA = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"


@pytest.fixture
def loaded_corpus(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        n = import_lecture_json(_DATA, upsert=False)
        assert n > 0
        invalidate_lecture_cache()
        load_lecture_cache()
    yield


def test_import_row_count(loaded_corpus, app):
    from app.models import LectureChunk

    with app.app_context():
        assert LectureChunk.query.count() == 51


def test_list_topics_and_summary(loaded_corpus, app):
    with app.app_context():
        cat = list_topics_catalog()
        nums = {x["lecture_number"] for x in cat["lectures"]}
        assert nums == set(range(4, 21))
        s = get_lecture_summary(8)
        assert s is not None
        assert s["lecture_number"] == 8
        assert "Backpropagation" in s["title"]
        assert len(s["sections"]) == 3


@pytest.mark.parametrize(
    "query,expected_lec",
    [
        ("What is backpropagation and how does it update weights?", 8),
        ("Explain dynamic programming for decision making", 7),
        ("How does inner product relate to cosine similarity?", 6),
        ("What are MFCCs used for in speech recognition?", 10),
        ("What is the difference between bias and variance in machine learning?", 11),
        ("How does multi-head attention work in transformers?", 15),
        ("What is an autoencoder?", 9),
        ("Neural network weights and forward pass", 4),
        ("ASR and text to speech tasks", 13),
        ("backprop chain rule", 8),
    ],
)
def test_retrieval_top_lecture(loaded_corpus, app, query, expected_lec):
    with app.app_context():
        r = retrieve_chunks(query, top_k=3)
        assert r.chunks, query
        assert r.confidence > 0.2, query
        assert r.diagnostics is not None
        assert r.diagnostics.retrieval_backend == "keyword"
        top = r.chunks[0]
        assert top["lecture_number"] == expected_lec, (query, top["topic"])


def test_lecture_number_query_confidence_floor(loaded_corpus, app):
    with app.app_context():
        r = retrieve_chunks("Explain lecture 8 gradient computation", top_k=2)
        assert r.chunks
        assert r.chunks[0]["lecture_number"] == 8
        assert r.confidence >= 0.44


def test_softmax_explicit_lecture_disambiguates(loaded_corpus, app):
    with app.app_context():
        r = retrieve_chunks("softmax probabilities lecture 12", top_k=2)
        assert r.chunks[0]["lecture_number"] == 12


def test_backprop_alias(loaded_corpus, app):
    with app.app_context():
        r = retrieve_chunks("backprop", top_k=1)
        assert r.chunks and r.chunks[0]["lecture_number"] == 8
