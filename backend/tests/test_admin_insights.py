"""Admin insights aggregates API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.extensions import db
from app.models import ChatSession, Message, ResponseVariant, RetrievalLog, User
from app.services.admin_insights import compute_insights_summary, compute_tokens_by_day

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


def test_compute_tokens_by_day_groups_by_utc_date(app):
    import json

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    day_a = (now - timedelta(days=4)).replace(hour=8, minute=0, second=0, microsecond=0)
    day_b = (now - timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
    key_a = day_a.date().isoformat()
    key_b = day_b.date().isoformat()

    with app.app_context():
        u = User(email="dayroll@test.dev", is_admin=False)
        u.set_password(_PW)
        db.session.add(u)
        db.session.flush()
        s = ChatSession(user_id=u.id, title="s")
        db.session.add(s)
        db.session.flush()
        m1 = Message(session_id=s.id, role="assistant", content_text="a")
        m2 = Message(session_id=s.id, role="assistant", content_text="b")
        db.session.add_all([m1, m2])
        db.session.flush()
        db.session.add(
            ResponseVariant(
                message_id=m1.id,
                course_answer="Course Answer:\n\nx",
                created_at=day_a,
                token_usage_json=json.dumps(
                    {"primary": {"usage": {"total_tokens": 10}, "model": "gpt-4o-mini"}}
                ),
            )
        )
        db.session.add(
            ResponseVariant(
                message_id=m2.id,
                course_answer="Course Answer:\n\ny",
                created_at=day_b,
                token_usage_json=json.dumps(
                    {
                        "primary": {"usage": {"total_tokens": 20}},
                        "boost": {"usage": {"total_tokens": 5}},
                    }
                ),
            )
        )
        db.session.commit()

        out = compute_tokens_by_day(30)

    assert len(out["days"]) == 2
    assert out["days"][0]["date"] == key_a
    assert out["days"][0]["sum_tokens_estimated"] == 10
    assert out["days"][0]["response_variants"] == 1
    assert out["days"][1]["date"] == key_b
    assert out["days"][1]["sum_tokens_estimated"] == 25
    assert out["days"][1]["variants_with_token_totals"] == 1


def test_tokens_by_day_endpoint_forbidden(client):
    register_user(client, "noadm_tok@test.dev", _PW)
    client.post(
        "/api/auth/login",
        json={"email": "noadm_tok@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights/tokens-by-day?days=7")
    assert r.status_code == 403


def test_cost_summary_and_content_quality_endpoints(client, app):
    register_user(client, "adm_cost@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="adm_cost@test.dev").first()
        u.is_admin = True
        db.session.commit()

    client.post(
        "/api/auth/login",
        json={"email": "adm_cost@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights/cost-summary?days=7")
    assert r.status_code == 200
    b = r.get_json()
    assert "sum_tokens_estimated" in b
    assert "over_cap" in b

    r2 = client.get("/api/admin/insights/content-quality?days=7")
    assert r2.status_code == 200
    assert "weak_chunks_by_low_confidence_hits" in r2.get_json()


def test_tokens_by_day_endpoint_ok(client, app):
    register_user(client, "adm_tokday@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="adm_tokday@test.dev").first()
        u.is_admin = True
        db.session.commit()

    client.post(
        "/api/auth/login",
        json={"email": "adm_tokday@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/insights/tokens-by-day?days=14")
    assert r.status_code == 200
    body = r.get_json()
    assert "window" in body
    assert "days" in body
    assert isinstance(body["days"], list)
