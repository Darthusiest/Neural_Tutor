from __future__ import annotations

import json
from pathlib import Path

from app.eval.analytics_common import suite_category
from app.eval.export_analytics import (
    export_error_counts,
    export_mode_accuracy_by_run,
    export_overall_by_run,
    export_score_by_category,
)
from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun


def test_plot_eval_metrics_writes_pngs(app, tmp_path):
    from app.eval import plot_eval_metrics

    with app.app_context():
        r = EvaluationRun(
            run_name="plot",
            dataset_name="ds",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            overall_score=0.9,
            notes_json="{}",
        )
        db.session.add(r)
        db.session.flush()
        rid = r.id
        db.session.add(
            EvaluationCaseResult(
                evaluation_run_id=rid,
                test_id="only",
                query_text="q",
                expected_mode="chat",
                detected_mode="chat",
                effective_mode="chat",
                expected_behavior_json=json.dumps({"category": "definitions"}),
                actual_response="",
                pass_bool=True,
                score=1.0,
                error_categories_json="[]",
                validation_failures_json=None,
                retrieval_chunk_ids_json="[]",
                latency_ms=50,
            )
        )
        db.session.commit()

        code = plot_eval_metrics.main(["--out-dir", str(tmp_path), "--run-ids", str(rid)])
        assert code == 0
        for name in (
            "overall_score_over_time.png",
            "pass_rate_by_category_over_time.png",
            "error_categories_over_time.png",
            "mode_accuracy_over_time.png",
            "retrieval_leakage_over_time.png",
            "latency_over_time.png",
        ):
            assert (tmp_path / name).is_file()


def test_suite_category_fallback(app):
    with app.app_context():
        c = EvaluationCaseResult(
            evaluation_run_id=1,
            test_id="t",
            query_text="q",
            expected_mode="compare",
            detected_mode="compare",
            effective_mode="compare",
            expected_behavior_json=json.dumps({}),
            actual_response="",
            pass_bool=True,
            score=1.0,
            error_categories_json="[]",
            validation_failures_json=None,
            retrieval_chunk_ids_json="[]",
            latency_ms=1,
        )
        assert suite_category(c) == "compare"

        c2 = EvaluationCaseResult(
            evaluation_run_id=1,
            test_id="t2",
            query_text="q",
            expected_mode=None,
            detected_mode=None,
            effective_mode=None,
            expected_behavior_json=json.dumps({"category": "quiz"}),
            actual_response="",
            pass_bool=False,
            score=0.5,
            error_categories_json="[]",
            validation_failures_json=None,
            retrieval_chunk_ids_json="[]",
            latency_ms=None,
        )
        assert suite_category(c2) == "quiz"


def test_export_aggregates(app):
    with app.app_context():
        r = EvaluationRun(
            run_name="a",
            dataset_name="ds",
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            overall_score=0.75,
            notes_json="{}",
        )
        db.session.add(r)
        db.session.flush()
        rid = r.id
        db.session.add_all(
            [
                EvaluationCaseResult(
                    evaluation_run_id=rid,
                    test_id="c1",
                    query_text="q1",
                    expected_mode="chat",
                    detected_mode="chat",
                    effective_mode="chat",
                    expected_behavior_json=json.dumps({"category": "compare"}),
                    actual_response="",
                    pass_bool=True,
                    score=1.0,
                    error_categories_json=json.dumps(["mode_mismatch"]),
                    validation_failures_json=None,
                    retrieval_chunk_ids_json="[]",
                    latency_ms=100,
                ),
                EvaluationCaseResult(
                    evaluation_run_id=rid,
                    test_id="c2",
                    query_text="q2",
                    expected_mode="chat",
                    detected_mode="summary",
                    effective_mode="summary",
                    expected_behavior_json=json.dumps({"category": "compare"}),
                    actual_response="",
                    pass_bool=False,
                    score=0.5,
                    error_categories_json=json.dumps(
                        ["mode_mismatch", "retrieval_leakage"]
                    ),
                    validation_failures_json=None,
                    retrieval_chunk_ids_json="[]",
                    latency_ms=200,
                ),
            ]
        )
        db.session.commit()

        runs = [r]
        cases = (
            EvaluationCaseResult.query.filter_by(evaluation_run_id=rid)
            .order_by(EvaluationCaseResult.test_id)
            .all()
        )

        overall = export_overall_by_run(runs)
        assert overall[0]["run_id"] == rid
        assert overall[0]["pass_rate"] == 0.5

        by_cat = export_score_by_category(cases)
        assert len(by_cat) == 1
        assert by_cat[0]["category"] == "compare"
        assert by_cat[0]["total"] == 2
        assert by_cat[0]["passed"] == 1
        assert by_cat[0]["failed"] == 1

        errs = export_error_counts(cases)
        assert {e["error_category"]: e["count"] for e in errs} == {
            "mode_mismatch": 2,
            "retrieval_leakage": 1,
        }

        mode_rows = export_mode_accuracy_by_run(runs, cases)
        assert mode_rows[0]["cases_with_expected_mode"] == 2
        assert mode_rows[0]["effective_matches"] == 1


def test_run_eval_suite_persists_category_in_expected_behavior(app):
    path = Path(__file__).resolve().parent.parent / "data" / "eval" / "l487_eval_suite.json"
    from app.services.eval_run import run_eval_suite

    with app.app_context():
        er = run_eval_suite(path, "analytics-category-test", top_k=2)
        row = EvaluationCaseResult.query.filter_by(evaluation_run_id=er.id).first()
        assert row is not None
        beh = json.loads(row.expected_behavior_json)
        assert "category" in beh
        assert beh["category"]
