"""Admin insights aggregates API."""

from __future__ import annotations

from app.extensions import db
from app.models import ChatSession, Message, ResponseVariant, RetrievalLog, User
from app.services.admin_insights import compute_insights_summary

from tests.conftest import register_user

_PW = "Abcd1234!"


def test_compute_insights_empty_db(app):
    with app.app_context():
        data = compute_insights_summary(7)
    assert data["insufficient_data"] is True
    assert data["volume"]["retrieval_events"] == 0
    assert "models_and_tokens" in data
    assert data["models_and_tokens"]["response_variants_in_window"] == 0


def test_insights_forbidden_for_non_admin(client):
    register_user(client, "plain@test.dev", _PW)
    client.post(
        "/api/auth/login",
        json={"email": "plain@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights")
    assert r.status_code == 403


def test_insights_ok_for_admin(client, app):
    register_user(client, "admin@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="admin@test.dev").first()
        assert u is not None
        u.is_admin = True
        db.session.commit()

    client.post(
        "/api/auth/login",
        json={"email": "admin@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights?days=14")
    assert r.status_code == 200
    body = r.get_json()
    assert "volume" in body
    assert "retrieval" in body
    assert "pipeline" in body
    assert body["window"]["days"] == 14
    assert body["insufficient_data"] is True


def test_insights_with_retrieval_log(client, app):
    register_user(client, "admin2@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="admin2@test.dev").first()
        u.is_admin = True
        db.session.commit()

    with app.app_context():
        u = User.query.filter_by(email="admin2@test.dev").first()
        s = ChatSession(user_id=u.id, title="s")
        db.session.add(s)
        db.session.flush()
        m = Message(session_id=s.id, role="assistant", content_text="a")
        db.session.add(m)
        db.session.flush()
        log = RetrievalLog(
            session_id=s.id,
            message_id=m.id,
            user_question="What is softmax?",
            confidence=0.55,
            latency_ms=100,
            num_chunks_hit=2,
            is_off_topic=False,
            is_low_confidence=False,
            query_type_v2="direct_definition",
            answer_mode="direct_definition",
            validation_passed=True,
            validation_checks_json='{"severity": "pass", "passed": true}',
        )
        db.session.add(log)
        db.session.commit()

    client.post(
        "/api/auth/login",
        json={"email": "admin2@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights")
    assert r.status_code == 200
    body = r.get_json()
    assert body["insufficient_data"] is False
    assert body["volume"]["retrieval_events"] >= 1
    assert body["pipeline"]["by_query_type_v2"].get("direct_definition", 0) >= 1
    sev = body["pipeline"].get("validation_severity") or {}
    assert sev.get("pass", 0) >= 1 or "pass" in sev


def test_compute_severity_rollup(app):
    with app.app_context():
        u = User(email="solo@test.dev", is_admin=False)
        u.set_password(_PW)
        db.session.add(u)
        db.session.flush()
        s = ChatSession(user_id=u.id, title="s")
        db.session.add(s)
        db.session.flush()
        m = Message(session_id=s.id, role="assistant", content_text="x")
        db.session.add(m)
        db.session.flush()
        db.session.add(
            RetrievalLog(
                session_id=s.id,
                message_id=m.id,
                user_question="q",
                validation_checks_json='{"severity":"weak","passed":false}',
            )
        )
        db.session.commit()

        data = compute_insights_summary(7)
    sev = data["pipeline"]["validation_severity"]
    assert sev.get("weak", 0) >= 1


def test_low_confidence_drill_down_and_csv(client, app):
    register_user(client, "adm3@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="adm3@test.dev").first()
        u.is_admin = True
        db.session.commit()

    with app.app_context():
        u = User.query.filter_by(email="adm3@test.dev").first()
        s = ChatSession(user_id=u.id, title="s")
        db.session.add(s)
        db.session.flush()
        m = Message(session_id=s.id, role="assistant", content_text="a")
        db.session.add(m)
        db.session.flush()
        db.session.add(
            RetrievalLog(
                session_id=s.id,
                message_id=m.id,
                user_question="weak?",
                confidence=0.1,
                is_low_confidence=True,
            )
        )
        db.session.commit()

    client.post(
        "/api/auth/login",
        json={"email": "adm3@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights/low-confidence?limit=10&offset=0")
    assert r.status_code == 200
    body = r.get_json()
    assert body["total"] >= 1
    assert len(body["items"]) >= 1
    assert body["items"][0]["user_question"].startswith("weak")

    csv_r = client.get("/api/admin/insights/low-confidence.csv")
    assert csv_r.status_code == 200
    assert "weak?" in csv_r.get_data(as_text=True)


def test_chunk_analytics_endpoint(client, app):
    register_user(client, "adm4@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="adm4@test.dev").first()
        u.is_admin = True
        db.session.commit()

    client.post(
        "/api/auth/login",
        json={"email": "adm4@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights/chunks?limit=5")
    assert r.status_code == 200
    b = r.get_json()
    assert "top_chunks_in_low_confidence_retrievals" in b
    assert "top_chunks_overall" in b


def test_token_rollup_from_response_variant(app):
    import json

    with app.app_context():
        u = User(email="tok@test.dev", is_admin=False)
        u.set_password(_PW)
        db.session.add(u)
        db.session.flush()
        s = ChatSession(user_id=u.id, title="s")
        db.session.add(s)
        db.session.flush()
        m = Message(session_id=s.id, role="assistant", content_text="a")
        db.session.add(m)
        db.session.flush()
        db.session.add(
            ResponseVariant(
                message_id=m.id,
                course_answer="Course Answer:\n\nx",
                token_usage_json=json.dumps(
                    {
                        "primary": {
                            "usage": {"total_tokens": 42},
                            "model": "gpt-4o-mini",
                            "provider": "openai",
                        }
                    }
                ),
                model_name="gpt-4o-mini",
                provider_name="openai",
            )
        )
        db.session.commit()
        data = compute_insights_summary(7)
    mt = data["models_and_tokens"]
    assert mt["sum_total_tokens_estimated"] == 42
    assert mt["by_provider"].get("openai", 0) >= 1
