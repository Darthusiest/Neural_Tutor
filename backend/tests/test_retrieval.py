from __future__ import annotations

import json

from app.extensions import db
from app.models import LectureChunk
from app.services.retrieval import (
    format_course_answer,
    invalidate_lecture_cache,
    load_lecture_cache,
    retrieve,
)


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
        db.session.add(
            LectureChunk(
                topic="Syntax — Trees",
                lecture_number=3,
                keywords=json.dumps(["syntax", "trees", "xbar"]),
                explanation="X-bar theory explains phrase structure.",
                example_qa=None,
            )
        )
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
                topic="L1 — Noise",
                lecture_number=1,
                keywords=json.dumps(["noise"]),
                explanation="Unrelated filler content about widgets.",
                example_qa=None,
            )
        )
        db.session.add(
            LectureChunk(
                topic="L2 — Target",
                lecture_number=2,
                keywords=json.dumps(["optimality", "constraints"]),
                explanation="Optimality Theory uses ranked constraints.",
                example_qa=None,
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
            "explanation": "Phones are segmented units.\nDistinction matters.",
            "keywords": "[]",
        }
    ]
    out = format_course_answer(chunks)
    assert out.startswith("Course Answer:")
    assert "Sounds" in out
    assert "Phones" in out
