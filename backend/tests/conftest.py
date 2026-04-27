from __future__ import annotations

import pytest

from app import create_app
from app.config import TestConfig
from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun  # noqa: F401 - tables for create_all


@pytest.fixture
def app():
    application = create_app(TestConfig)
    with application.app_context():
        db.create_all()
    yield application
    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def register_user(client, email: str, password: str):
    return client.post(
        "/api/auth/register",
        json={"email": email, "password": password},
        content_type="application/json",
    )
