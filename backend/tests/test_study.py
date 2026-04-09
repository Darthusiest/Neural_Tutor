"""Study mode API: quiz, compare, summary."""

from __future__ import annotations

import json

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


def _seed_chunks(app):
    with app.app_context():
        db.session.add_all(
            [
                LectureChunk(
                    chunk_key="t-a",
                    lecture_number=4,
                    topic="Foundations — Core Idea",
                    keywords=json.dumps(["neural", "weights"]),
                    source_excerpt="Neural networks learn weights from data.",
                    clean_explanation="Neural networks learn weights from data.",
                    sample_questions=json.dumps(["What do neural networks learn?"]),
                    sample_answer="Patterns via weights.",
                ),
                LectureChunk(
                    chunk_key="t-b",
                    lecture_number=5,
                    topic="Speech — Core Idea",
                    keywords=json.dumps(["vector", "speech"]),
                    source_excerpt="Speech is represented as vectors.",
                    clean_explanation="Speech is represented as vectors.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                LectureChunk(
                    chunk_key="t-c",
                    lecture_number=6,
                    topic="Inner Product — Core Idea",
                    keywords=json.dumps(["dot", "similarity"]),
                    source_excerpt="Inner product measures similarity.",
                    clean_explanation="Inner product measures similarity.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
            ]
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()


def test_quiz_next_mc(client, app):
    _login(client, "study-quiz@test.dev")
    _seed_chunks(app)
    r = client.post(
        "/api/study/quiz/next",
        json={"question_type": "mc"},
        content_type="application/json",
    )
    assert r.status_code == 200
    data = r.get_json()
    assert data["question_type"] == "mc"
    assert "options" in data and len(data["options"]) == 4
    assert "quiz_token" in data
    assert "correct_index" not in data


def test_quiz_answer_short(client, app):
    _login(client, "study-quiz2@test.dev")
    _seed_chunks(app)
    n = client.post(
        "/api/study/quiz/next",
        json={"question_type": "short"},
        content_type="application/json",
    ).get_json()
    r = client.post(
        "/api/study/quiz/answer",
        json={
            "chunk_id": n["chunk_id"],
            "question_type": "short",
            "quiz_token": n["quiz_token"],
            "user_answer": "my guess",
        },
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert "course_answer" in body
    assert "reveal" in body


def test_compare(client, app):
    _login(client, "study-cmp@test.dev")
    _seed_chunks(app)
    r = client.post(
        "/api/study/compare",
        json={"concept_a": "neural", "concept_b": "vector"},
        content_type="application/json",
    )
    assert r.status_code == 200
    data = r.get_json()
    assert "Course Answer" in data["course_answer"]
    assert data.get("boosted_explanation") is None


def test_summary_lecture(client, app):
    _login(client, "study-sum@test.dev")
    _seed_chunks(app)
    r = client.post(
        "/api/study/summary",
        json={"kind": "lecture", "lecture_number": 4},
        content_type="application/json",
    )
    assert r.status_code == 200
    assert "Lecture 4" in r.get_json()["course_answer"]


def test_summary_topic(client, app):
    _login(client, "study-sum2@test.dev")
    _seed_chunks(app)
    r = client.post(
        "/api/study/summary",
        json={"kind": "topic", "topic": "inner product"},
        content_type="application/json",
    )
    assert r.status_code == 200
    assert "inner product" in r.get_json()["course_answer"].lower()


def test_messages_payload_fallback(client, app):
    """Assistant messages without ResponseVariant still show course_answer from payload_json."""
    _login(client, "study-msg@test.dev")
    _seed_chunks(app)
    sid = client.post(
        "/api/sessions",
        json={"title": "t", "mode": "quiz"},
        content_type="application/json",
    ).get_json()["session"]["id"]
    client.post(
        "/api/study/summary",
        json={"kind": "lecture", "lecture_number": 4, "session_id": sid},
        content_type="application/json",
    )
    r = client.get(f"/api/sessions/{sid}/messages")
    assert r.status_code == 200
    msgs = r.get_json()["messages"]
    assistant = [m for m in msgs if m["role"] == "assistant"][-1]
    assert assistant.get("course_answer")
