"""Eval critic orchestration (mocked Gemini)."""

from __future__ import annotations

import json
from unittest.mock import patch

from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun
from app.services.critic.gemini_critic import CriticVerdict
from app.models.evaluation import EvaluationCriticResult


def test_run_critic_for_eval_run_writes_rows_and_charts(app, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.services.eval_critic._repo_root",
        lambda: tmp_path,
    )

    v = CriticVerdict(
        score=0.85,
        passed=True,
        dimensions={
            "grounded": 0.9,
            "accurate": 0.9,
            "complete": 0.8,
            "mode_compliant": 0.8,
            "no_hallucination": 0.8,
        },
        error_categories=[],
        rationale="ok",
    )

    def _fake_critic(**_kwargs):
        return v, {"model": "gemini-test", "usage": {"totalTokenCount": 10}}

    with app.app_context():
        run = EvaluationRun(
            run_name="t",
            dataset_name="ds@1",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            overall_score=0.9,
        )
        db.session.add(run)
        db.session.flush()
        case = EvaluationCaseResult(
            evaluation_run_id=run.id,
            test_id="c1",
            query_text="What is test?",
            expected_mode="chat",
            detected_mode="chat",
            effective_mode="chat",
            expected_behavior_json=json.dumps({"category": "definitions"}),
            actual_response="A test is …",
            pass_bool=True,
            score=1.0,
            error_categories_json="[]",
            retrieval_chunk_ids_json="[]",
            assistant_message_id=None,
        )
        db.session.add(case)
        db.session.commit()
        rid = run.id

        with patch("app.services.eval_critic.run_gemini_critic", side_effect=_fake_critic):
            from app.services.eval_critic import run_critic_for_eval_run

            out = run_critic_for_eval_run(rid, force=True)

    assert out.get("status") == "ok"
    assert out.get("critic_batch_id")
    batch = out["critic_batch_id"]
    with app.app_context():
        rows = EvaluationCriticResult.query.filter_by(critic_batch_id=batch).all()
        assert len(rows) == 1
        assert rows[0].critic_pass is True

    out_dir = tmp_path / "evaluation_outputs" / "critic" / batch
    assert (out_dir / "pipeline_diagram.png").is_file()


def test_run_critic_cached_without_force(app, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.eval_critic._repo_root", lambda: tmp_path)

    def _fake_critic(**_kwargs):
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
                rationale="x",
            ),
            {"model": "m"},
        )

    with app.app_context():
        run = EvaluationRun(
            run_name="t2",
            dataset_name="ds2",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
        )
        db.session.add(run)
        db.session.flush()
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="c1",
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
        with patch("app.services.eval_critic.run_gemini_critic", side_effect=_fake_critic):
            from app.services.eval_critic import run_critic_for_eval_run

            run_critic_for_eval_run(rid, force=True)
            out2 = run_critic_for_eval_run(rid, force=False)
    assert out2.get("status") == "cached"


def test_run_critic_no_verdict_maps_pipeline_primary(app, tmp_path, monkeypatch):
    """Rows without a parsed verdict must not pretend to be shallow_explanation in charts."""
    monkeypatch.setattr("app.services.eval_critic._repo_root", lambda: tmp_path)

    def _fake_critic(**_kwargs):
        return None, {"model": "gemini-test", "error": "critic_malformed_json"}

    with app.app_context():
        run = EvaluationRun(
            run_name="t3",
            dataset_name="ds@1",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
        )
        db.session.add(run)
        db.session.flush()
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="c1",
                query_text="q",
                pass_bool=False,
                score=0.0,
                error_categories_json="[]",
                expected_behavior_json="{}",
                expected_mode="chat",
                effective_mode="chat",
                retrieval_chunk_ids_json="[]",
            )
        )
        db.session.commit()
        rid = run.id

        with patch("app.services.eval_critic.run_gemini_critic", side_effect=_fake_critic):
            from app.services.eval_critic import _map_critic_primary, run_critic_for_eval_run

            run_critic_for_eval_run(rid, force=True)
            row = EvaluationCriticResult.query.filter_by(evaluation_run_id=rid).first()

    assert row is not None
    assert row.critic_score is None
    assert json.loads(row.error_categories_json) == ["critic_malformed_json"]
    assert _map_critic_primary(row) == "critic_pipeline_error"


def test_run_critic_skips_quiz_and_writes_manifest(app, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.eval_critic._repo_root", lambda: tmp_path)

    modes_seen: list[str] = []

    def _fake_critic(**kwargs):
        modes_seen.append(str(kwargs.get("mode") or ""))
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
            {"model": "gemini-test", "usage": {"totalTokenCount": 5}},
        )

    with app.app_context():
        run = EvaluationRun(
            run_name="mixed",
            dataset_name="ds@1",
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
        )
        db.session.add(run)
        db.session.flush()
        for tid, em in (("c_chat", "chat"), ("c_quiz", "quiz")):
            db.session.add(
                EvaluationCaseResult(
                    evaluation_run_id=run.id,
                    test_id=tid,
                    query_text="q",
                    pass_bool=True,
                    score=1.0,
                    error_categories_json="[]",
                    expected_behavior_json="{}",
                    effective_mode=em,
                    retrieval_chunk_ids_json="[]",
                )
            )
        db.session.commit()
        rid = run.id

        with patch("app.services.eval_critic.run_gemini_critic", side_effect=_fake_critic):
            from app.services.eval_critic import critic_batch_complete, run_critic_for_eval_run

            out = run_critic_for_eval_run(rid, force=True)
    assert out.get("status") == "ok"
    batch = out["critic_batch_id"]
    assert modes_seen == ["chat"]
    with app.app_context():
        n = EvaluationCriticResult.query.filter_by(critic_batch_id=batch).count()
        assert n == 1
        assert critic_batch_complete(rid, batch) is True

    man = tmp_path / "evaluation_outputs" / "critic" / batch / "manifest.json"
    assert man.is_file()
    man_d = json.loads(man.read_text())
    assert man_d.get("evaluation_run_id") == rid
    assert man_d.get("modes_filter") == ["chat", "compare", "summary"]
    assert len(man_d.get("case_result_ids") or []) == 1


def test_effective_mode_fallback_expected_behavior(app, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.eval_critic._repo_root", lambda: tmp_path)

    def _fake_critic(**kwargs):
        assert kwargs.get("mode") == "summary"
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
            {"model": "m", "usage": {"totalTokenCount": 2}},
        )

    with app.app_context():
        run = EvaluationRun(
            run_name="fb",
            dataset_name="ds",
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
                expected_behavior_json=json.dumps({"expected_mode": "summary"}),
                effective_mode="auto",
                retrieval_chunk_ids_json="[]",
            )
        )
        db.session.commit()
        rid = run.id
        with patch("app.services.eval_critic.run_gemini_critic", side_effect=_fake_critic):
            from app.services.eval_critic import run_critic_for_eval_run

            run_critic_for_eval_run(rid, force=True)


def test_run_critic_no_cases_in_scope(app, tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.eval_critic._repo_root", lambda: tmp_path)

    with app.app_context():
        run = EvaluationRun(
            run_name="q",
            dataset_name="ds",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
        )
        db.session.add(run)
        db.session.flush()
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=run.id,
                test_id="z",
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
        with patch("app.services.eval_critic.run_gemini_critic") as m:
            from app.services.eval_critic import run_critic_for_eval_run

            out = run_critic_for_eval_run(rid, force=True)
        m.assert_not_called()
    assert out.get("error") == "no_cases_in_scope"
