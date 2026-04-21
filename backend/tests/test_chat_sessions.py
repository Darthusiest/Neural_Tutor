"""Chat session API: list, create, patch, delete."""

from __future__ import annotations

import time

from app.extensions import db
from app.models import ChatSession, Feedback, Message

from tests.conftest import register_user

_PW = "Abcd1234!"


def _login(client, email: str) -> None:
    register_user(client, email, _PW)
    client.post(
        "/api/auth/login",
        json={"email": email, "password": _PW},
        content_type="application/json",
    )


def test_patch_session_requires_auth(client):
    r = client.patch(
        "/api/sessions/1",
        json={"title": "x"},
        content_type="application/json",
    )
    assert r.status_code == 401


def test_patch_session_updates_title(client, app):
    _login(client, "patch-ok@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "old", "mode": "chat"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    with app.app_context():
        before = db.session.get(ChatSession, sid)
        assert before is not None
        t0 = before.updated_at
    time.sleep(0.02)

    r = client.patch(
        f"/api/sessions/{sid}",
        json={"title": "  new title  "},
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["session"]["title"] == "new title"
    assert body["session"].get("updated_at")

    with app.app_context():
        s = db.session.get(ChatSession, sid)
        assert s is not None
        assert s.title == "new title"
        assert s.updated_at and t0 and s.updated_at > t0


def test_patch_session_requires_title(client):
    _login(client, "patch-need-title@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "t", "mode": "chat"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    r = client.patch(
        f"/api/sessions/{sid}",
        json={"mode": "chat"},
        content_type="application/json",
    )
    assert r.status_code == 400
    assert "title" in (r.get_json().get("error") or "").lower()


def test_patch_other_users_session_returns_404(client):
    _login(client, "patch-a@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "t", "mode": "chat"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    client.post("/api/auth/logout", content_type="application/json")

    _login(client, "patch-b@test.dev")
    r = client.patch(
        f"/api/sessions/{sid}",
        json={"title": "hacked"},
        content_type="application/json",
    )
    assert r.status_code == 404


def test_delete_session_requires_auth(client):
    r = client.delete("/api/sessions/1")
    assert r.status_code == 401


def test_delete_session_success(client, app):
    _login(client, "del-ok@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "t", "mode": "chat"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 204
    assert r.data == b""

    with app.app_context():
        assert db.session.get(ChatSession, sid) is None


def test_delete_session_removes_feedback(client, app):
    _login(client, "del-fb@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "t", "mode": "chat"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    with app.app_context():
        m = Message(session_id=sid, role="assistant", content_text="hi")
        db.session.add(m)
        db.session.commit()
        fb = Feedback(message_id=m.id)
        db.session.add(fb)
        db.session.commit()
        mid = m.id

    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 204

    with app.app_context():
        assert db.session.get(ChatSession, sid) is None
        assert db.session.get(Message, mid) is None
        assert Feedback.query.filter_by(message_id=mid).first() is None


def test_chat_accepts_message_without_mode_and_returns_mode_block(client, app):
    """POST /api/chat: ``mode`` is optional; response includes ``mode`` + ``answer``."""
    _login(client, "chat-contract@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "t"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    r = client.post(
        "/api/chat",
        json={"session_id": sid, "message": "compare CNN and MLP"},
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert "answer" in body and body["answer"] == body.get("course_answer")
    assert "mode" in body
    mode = body["mode"]
    assert mode["detected"] in ("compare", "chat", "quiz", "summary")
    assert mode["effective"] in ("compare", "chat", "quiz", "summary")
    assert "confidence" in mode and isinstance(mode["confidence"], (int, float))
    assert "signals" in mode and isinstance(mode["signals"], list)
    assert "overridden" in mode and isinstance(mode["overridden"], bool)
    assert "ambiguous" in mode and isinstance(mode["ambiguous"], bool)
    assert "mode_routing" in body


def test_chat_mode_override_wins_over_mode(client, app):
    _login(client, "chat-override@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "t"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    r = client.post(
        "/api/chat",
        json={
            "session_id": sid,
            "message": "hello",
            "mode": "chat",
            "mode_override": "compare",
        },
        content_type="application/json",
    )
    assert r.status_code == 200
    assert r.get_json()["mode"]["effective"] == "compare"
    assert r.get_json()["mode"]["overridden"] is True


def test_delete_other_users_session_returns_404(client):
    _login(client, "del-a@test.dev")
    sid = client.post(
        "/api/sessions",
        json={"title": "t", "mode": "chat"},
        content_type="application/json",
    ).get_json()["session"]["id"]

    client.post("/api/auth/logout", content_type="application/json")

    _login(client, "del-b@test.dev")
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 404
