from __future__ import annotations

from tests.conftest import register_user

_VALID_PASSWORD = "Abcd1234!"


def test_register_and_login(client):
    r = register_user(client, "u@example.com", _VALID_PASSWORD)
    assert r.status_code == 201
    r2 = client.post(
        "/api/auth/login",
        json={"email": "u@example.com", "password": _VALID_PASSWORD},
        content_type="application/json",
    )
    assert r2.status_code == 200
    assert r2.get_json()["user"]["email"] == "u@example.com"


def test_register_duplicate(client):
    register_user(client, "dup@example.com", _VALID_PASSWORD)
    r = register_user(client, "dup@example.com", _VALID_PASSWORD)
    assert r.status_code == 409


def test_login_wrong_password(client):
    register_user(client, "wp@example.com", _VALID_PASSWORD)
    r = client.post(
        "/api/auth/login",
        json={"email": "wp@example.com", "password": "Wrongpass1!"},
        content_type="application/json",
    )
    assert r.status_code == 401


def test_logout_unauthenticated(client):
    r = client.post("/api/auth/logout", content_type="application/json")
    assert r.status_code == 401


def test_forgot_password_returns_200_regardless(client):
    msg = "If an account exists for this email, a reset link has been sent."
    r1 = client.post(
        "/api/auth/forgot-password",
        json={"email": "nonexistent-xyz@example.com"},
        content_type="application/json",
    )
    assert r1.status_code == 200
    assert r1.get_json().get("message") == msg

    register_user(client, "exists@example.com", _VALID_PASSWORD)
    r2 = client.post(
        "/api/auth/forgot-password",
        json={"email": "exists@example.com"},
        content_type="application/json",
    )
    assert r2.status_code == 200
    assert r2.get_json().get("message") == msg
