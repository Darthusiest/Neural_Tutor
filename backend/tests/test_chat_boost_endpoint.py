"""POST /api/chat/boost/<message_id> (deferred constrained boost)."""

from __future__ import annotations

import json

from app.extensions import db
from app.models import ChatSession, Message, User

from tests.conftest import register_user

_PW = "Abcd1234!"


def _login(client, email: str) -> None:
    client.post(
        "/api/auth/login",
        json={"email": email, "password": _PW},
        content_type="application/json",
    )


def test_boost_endpoint_idempotent_skipped(client, app):
    register_user(client, "boost_skip@test.dev", _PW)
    _login(client, "boost_skip@test.dev")
    with app.app_context():
        u = User.query.filter_by(email="boost_skip@test.dev").first()
        s = ChatSession(user_id=u.id, title="t", mode="auto")
        db.session.add(s)
        db.session.flush()
        payload = {
            "course_answer": "Course Answer:\n\nok",
            "answer": "Course Answer:\n\nok",
            "boosted_explanation": None,
            "boost_status": "skipped",
            "boost_skip_reason": "no_retrieval_log",
            "mode": {"effective": "chat"},
            "mode_routing": {},
        }
        m = Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps(payload),
        )
        db.session.add(m)
        db.session.commit()
        mid = m.id

    r = client.post(f"/api/chat/boost/{mid}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["boost_status"] == "skipped"


def test_boost_endpoint_forbidden_wrong_user(client, app):
    register_user(client, "boost_a@test.dev", _PW)
    register_user(client, "boost_b@test.dev", _PW)
    with app.app_context():
        ua = User.query.filter_by(email="boost_a@test.dev").first()
        ub = User.query.filter_by(email="boost_b@test.dev").first()
        s = ChatSession(user_id=ua.id, title="t", mode="auto")
        db.session.add(s)
        db.session.flush()
        m = Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps({"boost_status": "pending"}),
        )
        db.session.add(m)
        db.session.commit()
        mid = m.id

    _login(client, "boost_b@test.dev")
    r = client.post(f"/api/chat/boost/{mid}")
    assert r.status_code == 403
