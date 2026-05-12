"""Admin API: Gemini critic routes."""

from __future__ import annotations

from unittest.mock import patch

from sqlalchemy.exc import OperationalError

from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun, User

from tests.conftest import register_user

_PW = "Abcd1234!"


def test_critic_routes_forbidden_non_admin(client):
    register_user(client, "nc@test.dev", _PW)
    client.post(
        "/api/auth/login",
        json={"email": "nc@test.dev", "password": _PW},
        content_type="application/json",
    )
    assert client.post("/api/admin/eval/runs/1/critic", json={}).status_code == 403
    assert client.get("/api/admin/eval/runs/1/critic").status_code == 403


def test_critic_post_and_get_summary(client, app, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.eval_critic._repo_root", lambda: tmp_path)
    monkeypatch.setattr("app.routes.admin._repo_root", lambda: tmp_path)

    register_user(client, "gc@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="gc@test.dev").first()
        u.is_admin = True
        db.session.commit()
        run = EvaluationRun(
            run_name="cr",
            dataset_name="d@v",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
        )
        db.session.add(run)
        db.session.flush()
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="t1",
                query_text="q",
                pass_bool=True,
                score=1.0,
                error_categories_json="[]",
                expected_behavior_json="{}",
                expected_mode="chat",
                effective_mode="chat",
                retrieval_chunk_ids_json="[]",
            )
        )
        db.session.commit()
        rid = run.id

    from app.services.critic.gemini_critic import CriticVerdict

    def _fake(**_k):
        return (
            CriticVerdict(
                score=1.0,
                passed=True,
                dimensions={
                    "grounded": 1.0,
                    "accurate": 1.0,
                    "complete": 1.0,
                    "mode_compliant": 1.0,
                    "no_hallucination": 1.0,
                },
                error_categories=[],
                rationale="ok",
            ),
            {"model": "m"},
        )

    client.post(
        "/api/auth/login",
        json={"email": "gc@test.dev", "password": _PW},
        content_type="application/json",
    )
    with patch("app.services.eval_critic.run_gemini_critic", side_effect=_fake):
        pr = client.post(f"/api/admin/eval/runs/{rid}/critic", json={"force": True})
    assert pr.status_code == 200
    body = pr.get_json()
    assert body.get("status") in ("ok", "partial")
    assert body.get("critic_batch_id")

    gr = client.get(f"/api/admin/eval/runs/{rid}/critic")
    assert gr.status_code == 200
    gjson = gr.get_json()
    assert gjson.get("critic_batch_id")
    assert "evaluation_summary.png" in (gjson.get("artifact_urls") or {})


def test_critic_post_rejects_non_list_modes(client, app):
    register_user(client, "md@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="md@test.dev").first()
        u.is_admin = True
        db.session.commit()
        run = EvaluationRun(
            run_name="m",
            dataset_name="x",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
        )
        db.session.add(run)
        db.session.flush()
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="t1",
                query_text="q",
                pass_bool=True,
                score=1.0,
                error_categories_json="[]",
                expected_behavior_json="{}",
                effective_mode="chat",
                retrieval_chunk_ids_json="[]",
            )
        )
        db.session.commit()
        rid = run.id

    client.post(
        "/api/auth/login",
        json={"email": "md@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.post(f"/api/admin/eval/runs/{rid}/critic", json={"force": True, "modes": "chat"})
    assert r.status_code == 400


def test_critic_post_no_cases_in_scope(client, app):
    register_user(client, "ns@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="ns@test.dev").first()
        u.is_admin = True
        db.session.commit()
        run = EvaluationRun(
            run_name="ns",
            dataset_name="x",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
        )
        db.session.add(run)
        db.session.flush()
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="t1",
                query_text="q",
                pass_bool=False,
                score=0.0,
                error_categories_json="[]",
                expected_behavior_json="{}",
                effective_mode="quiz",
                retrieval_chunk_ids_json="[]",
            )
        )
        db.session.commit()
        rid = run.id

    client.post(
        "/api/auth/login",
        json={"email": "ns@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.post(f"/api/admin/eval/runs/{rid}/critic", json={"force": True})
    assert r.status_code == 422
    assert r.get_json().get("error") == "no_cases_in_scope"


def test_critic_image_whitelist(client, app):
    register_user(client, "im@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="im@test.dev").first()
        u.is_admin = True
        db.session.commit()
        run = EvaluationRun(
            run_name="i",
            dataset_name="x",
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
        )
        db.session.add(run)
        db.session.commit()
        rid = run.id

    client.post(
        "/api/auth/login",
        json={"email": "im@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get(f"/api/admin/eval/critic-image/{rid}/evil.png")
    assert r.status_code == 404


def test_critic_get_schema_outdated_json(client, app):
    register_user(client, "schema@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="schema@test.dev").first()
        u.is_admin = True
        db.session.commit()

    client.post(
        "/api/auth/login",
        json={"email": "schema@test.dev", "password": _PW},
        content_type="application/json",
    )

    def _boom(_run_id):
        raise OperationalError(
            "stmt",
            {},
            Exception("no such table: evaluation_critic_results"),
        )

    with patch("app.routes.admin.critic_summary_for_run", side_effect=_boom):
        r = client.get("/api/admin/eval/runs/1/critic")
    assert r.status_code == 503
    body = r.get_json()
    assert body.get("error") == "schema_outdated"
