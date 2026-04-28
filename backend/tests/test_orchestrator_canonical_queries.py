"""End-to-end pipeline checks for canonical concept-purity queries."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extensions import db
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.reasoning_pipeline import run_reasoning_pipeline
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache
from app.services.knowledge.concept_kb import reset_kb_for_tests

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
    reset_kb_for_tests()


def test_what_is_cnn_pipeline(corpus, app):
    with app.app_context():
        r = run_reasoning_pipeline("What is CNN?", top_k=5)
        assert r.course_answer
        low = r.course_answer.lower()
        assert "transformer" not in low and "self-attention" not in low


def test_cnn_do_not_mention_transformers(corpus, app):
    with app.app_context():
        r = run_reasoning_pipeline(
            "What is CNN? Do not mention transformers or residuals.", top_k=5
        )
        assert r.course_answer
        low = r.course_answer.lower()
        assert "transformer" not in low
        assert "residual" not in low


def test_mfcc_do_not_mention_softmax(corpus, app):
    with app.app_context():
        r = run_reasoning_pipeline(
            "What is MFCC? Do not mention softmax.", top_k=5
        )
        assert r.course_answer
        assert "softmax" not in r.course_answer.lower()


def test_dynamic_programming_no_neural_networks(corpus, app):
    with app.app_context():
        r = run_reasoning_pipeline(
            "What is dynamic programming? Do not mention neural networks.", top_k=5
        )
        assert r.course_answer
        low = r.course_answer.lower()
        assert "neural network" not in low
