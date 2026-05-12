"""ADMIN_EMAILS syncs User.is_admin on register and login."""

from __future__ import annotations

import pytest

from app import create_app
from app.config import TestConfig
from app.extensions import db
from tests.conftest import register_user

_PW = "Abcd1234!"


class _AllowlistConfig(TestConfig):
    ADMIN_EMAILS = frozenset({"owner@usc.edu", "other@test.dev"})


@pytest.fixture
def app_allowlist():
    application = create_app(_AllowlistConfig)
    with application.app_context():
        db.create_all()
    yield application
    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client_allowlist(app_allowlist):
    return app_allowlist.test_client()


def test_register_admin_when_email_in_allowlist(client_allowlist):
    r = register_user(client_allowlist, "owner@usc.edu", _PW)
    assert r.status_code == 201
    assert r.get_json()["user"]["is_admin"] is True


def test_register_not_admin_when_not_in_allowlist(client_allowlist):
    r = register_user(client_allowlist, "plain@usc.edu", _PW)
    assert r.status_code == 201
    assert r.get_json()["user"]["is_admin"] is False


def test_login_grants_admin_normalized_email(client_allowlist):
    register_user(client_allowlist, "owner@usc.edu", _PW)
    client_allowlist.post("/api/auth/logout")
    r = client_allowlist.post(
        "/api/auth/login",
        json={"email": "OWNER@USC.EDU", "password": _PW},
    )
    assert r.status_code == 200
    body = r.get_json()
    assert body["user"]["email"] == "owner@usc.edu"
    assert body["user"]["is_admin"] is True


def test_login_syncs_allowlist_membership(client_allowlist):
    register_user(client_allowlist, "other@test.dev", _PW)
    client_allowlist.post("/api/auth/logout")
    r = client_allowlist.post(
        "/api/auth/login",
        json={"email": "OTHER@TEST.DEV", "password": _PW},
    )
    assert r.status_code == 200
    assert r.get_json()["user"]["is_admin"] is True
