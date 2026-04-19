"""Chat session API: list, create, patch, delete."""

from __future__ import annotations

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

    r = client.patch(
        f"/api/sessions/{sid}",
        json={"title": "  new title  "},
        content_type="application/json",
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["session"]["title"] == "new title"

    with app.app_context():
        s = db.session.get(ChatSession, sid)
        assert s is not None
        assert s.title == "new title"


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
