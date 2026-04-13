from __future__ import annotations

import json

from app.extensions import db
from app.models import LectureChunk
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

from tests.conftest import register_user

_PW = "Abcd1234!"


def _auth(client):
    register_user(client, "lecture-api@test.dev", _PW)


def test_topics_empty_authenticated(client):
    _auth(client)
    r = client.get("/api/lectures/topics")
    assert r.status_code == 200
    assert r.get_json() == {"lectures": []}


def test_topics_unauthenticated(client):
    assert client.get("/api/lectures/topics").status_code == 401


def test_topics_with_data(client, app):
    _auth(client)
    with app.app_context():
        db.session.add(
            LectureChunk(
                chunk_key="test-foundations-core",
                lecture_number=4,
                topic="Foundations — Core Idea",
                keywords=json.dumps(["net"]),
                source_excerpt="Neural networks learn weights.",
                clean_explanation="Neural networks learn weights.",
                sample_questions="[]",
                sample_answer=None,
            )
        )
        db.session.commit()

    r = client.get("/api/lectures/topics")
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["lectures"]) == 1
    lec = body["lectures"][0]
    assert lec["lecture_number"] == 4
    assert "Foundations" in lec["title"]
    assert lec["chunk_count"] == 1


def test_summary_404(client):
    _auth(client)
    assert client.get("/api/lectures/99/summary").status_code == 404


def test_summary_ok(client, app):
    _auth(client)
    with app.app_context():
        db.session.add(
            LectureChunk(
                chunk_key="test-inner-core",
                lecture_number=6,
                topic="Inner Product — Core Idea",
                keywords=json.dumps(["dot"]),
                source_excerpt="Inner product measures similarity.",
                clean_explanation="Inner product measures similarity.",
                sample_questions="[]",
                sample_answer=None,
            )
        )
        db.session.commit()

    r = client.get("/api/lectures/6/summary")
    assert r.status_code == 200
    data = r.get_json()
    assert data["lecture_number"] == 6
    assert data["chunk_count"] == 1
    assert len(data["sections"]) == 1
    assert data["sections"][0]["topic"] == "Inner Product — Core Idea"
    assert data["sections"][0]["chunk_key"] == "test-inner-core"


def test_retrieve_keyword(client, app):
    _auth(client)
    with app.app_context():
        db.session.add(
            LectureChunk(
                chunk_key="test-unique-section",
                lecture_number=10,
                topic="Test — Section",
                keywords=json.dumps(["uniquekwxyz"]),
                source_excerpt="The uniquekwxyz concept is important.",
                clean_explanation="The uniquekwxyz concept is important.",
                sample_questions="[]",
                sample_answer=None,
            )
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()

    r = client.post(
        "/api/lectures/retrieve",
        json={"query": "uniquekwxyz concept", "top_k": 3},
        content_type="application/json",
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["confidence"] > 0
    assert data["chunks"]
    assert any("uniquekwxyz" in json.dumps(c).lower() for c in data["chunks"])
    chunk0 = data["chunks"][0]
    assert chunk0.get("source_excerpt") == chunk0.get("source_text")


def test_retrieve_embedding_disabled_returns_400(client):
    _auth(client)
    r = client.post(
        "/api/lectures/retrieve",
        json={"query": "anything", "backend": "embedding"},
        content_type="application/json",
    )
    assert r.status_code == 400
    assert "disabled" in r.get_json().get("error", "").lower()
