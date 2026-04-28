"""Admin eval analytics API."""

from __future__ import annotations

import json

from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun, User

from tests.conftest import register_user

_PW = "Abcd1234!"


def test_eval_runs_forbidden_non_admin(client):
    register_user(client, "u1@test.dev", _PW)
    client.post(
        "/api/auth/login",
        json={"email": "u1@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/eval/runs")
    assert r.status_code == 403


def test_eval_runs_list_and_detail(client, app):
    register_user(client, "adm_eval@test.dev", _PW)
    with app.app_context():
        u = User.query.filter_by(email="adm_eval@test.dev").first()
        u.is_admin = True
        db.session.commit()
        run = EvaluationRun(
            run_name="rn",
            dataset_name="l487_eval_suite@1",
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            overall_score=0.75,
            notes_json=json.dumps({"reports_dir": "/tmp/reports/eval_runs/x"}),
        )
        db.session.add(run)
        db.session.flush()
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="t1",
                query_text="q1",
                expected_mode="chat",
                detected_mode="chat",
                effective_mode="chat",
                expected_behavior_json=json.dumps({"category": "quiz"}),
                actual_response="ok",
                pass_bool=True,
                score=1.0,
                error_categories_json="[]",
                validation_failures_json="{}",
                retrieval_chunk_ids_json="[]",
                latency_ms=5,
            )
        )
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="t2",
                query_text="q2",
                expected_mode="chat",
                detected_mode="quiz",
                effective_mode="chat",
                expected_behavior_json=json.dumps({"category": "definitions"}),
                actual_response="bad",
                pass_bool=False,
                score=0.5,
                error_categories_json=json.dumps(["mode_mismatch"]),
                validation_failures_json="{}",
                retrieval_chunk_ids_json="[]",
                latency_ms=6,
            )
        )
        db.session.commit()
        rid = run.id

    client.post(
        "/api/auth/login",
        json={"email": "adm_eval@test.dev", "password": _PW},
        content_type="application/json",
    )
    r = client.get("/api/admin/eval/runs?limit=10")
    assert r.status_code == 200
    body = r.get_json()
    assert "runs" in body
    assert len(body["runs"]) >= 1
    hit = next(x for x in body["runs"] if x["id"] == rid)
    assert hit["pass_rate"] == 0.5
    assert hit["report_files"]["examples_md"]

    d = client.get(f"/api/admin/eval/runs/{rid}")
    assert d.status_code == 200
    detail = d.get_json()
    assert detail["counts"]["failed_cases"] == 1
    assert len(detail["category_breakdown"]) >= 1
    assert len(detail["failure_table_preview"]) == 1

    f = client.get(f"/api/admin/eval/runs/{rid}/failures")
    assert f.status_code == 200
    fails = f.get_json()
    assert fails["failures"][0]["test_id"] == "t2"
    assert "canonical_failure_tags" in fails["failures"][0]
