from __future__ import annotations

import json
import pytest

from app.extensions import db
from app.models import LectureChunk
from app.services.retrieval import (
    format_course_answer,
    invalidate_lecture_cache,
    load_lecture_cache,
    retrieve,
    retrieve_chunks,
)


def _sample_chunk_kwargs(topic="Syntax — Trees", **overrides):
    base = {
        "topic": topic,
        "lecture_number": 3,
        "keywords": json.dumps(["syntax", "trees", "xbar"]),
        "source_excerpt": "X-bar theory explains phrase structure.",
        "clean_explanation": "X-bar theory explains phrase structure.",
        "sample_questions": "[]",
        "sample_answer": None,
    }
    base.update(overrides)
    return base


def test_retrieve_empty_db(app):
    with app.app_context():
        invalidate_lecture_cache()
        load_lecture_cache()
        r = retrieve("phonology")
    assert r.chunks == []
    assert r.confidence == 0.0
    assert r.detected_topic is None


def test_retrieve_keyword_match(app):
    with app.app_context():
        db.session.add(LectureChunk(**_sample_chunk_kwargs()))
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()
        r = retrieve("syntax phrase")
    assert r.chunks
    assert any("syntax" in (c.get("topic") or "").lower() for c in r.chunks)


def test_confidence_scoring_ranks_hits(app):
    with app.app_context():
        db.session.add(
            LectureChunk(
                **_sample_chunk_kwargs(
                    topic="L1 — Noise",
                    lecture_number=1,
                    keywords=json.dumps(["noise"]),
                    source_excerpt="Unrelated filler content about widgets.",
                    clean_explanation="Unrelated filler content about widgets.",
                )
            )
        )
        db.session.add(
            LectureChunk(
                **_sample_chunk_kwargs(
                    topic="L2 — Target",
                    lecture_number=2,
                    keywords=json.dumps(["optimality", "constraints"]),
                    source_excerpt="Optimality Theory uses ranked constraints.",
                    clean_explanation="Optimality Theory uses ranked constraints.",
                )
            )
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()
        r = retrieve("optimality constraints ranked")
    assert r.chunks
    assert r.chunks[0].get("topic", "").startswith("L2")


def test_format_course_answer():
    chunks = [
        {
            "id": 1,
            "lecture_number": 1,
            "topic": "Intro — Sounds",
            "keywords": "[]",
            "source_excerpt": "fallback",
            "clean_explanation": "Phones are segmented units.\nDistinction matters.",
            "sample_questions": None,
            "sample_answer": None,
        }
    ]
    out = format_course_answer(chunks)
    assert out.startswith("Course Answer:")
    assert "Sounds" in out
    assert "Phones" in out


def test_retrieve_chunks_embedding_not_implemented():
    with pytest.raises(NotImplementedError):
        retrieve_chunks("test", backend="embedding")
