"""Regression comparison between two persisted eval runs."""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.regression import (
    RegressionFinding,
    compare_eval_runs,
    render_regression_markdown,
    write_regression_report,
)
from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun


def _row(
    run_id: int,
    tid: str,
    *,
    passed: bool,
    score: float,
    detected: str = "chat",
    effective: str = "chat",
    critical: bool = False,
    must_not: list | None = None,
):
    must_not = must_not or []
    return EvaluationCaseResult(
        evaluation_run_id=run_id,
        test_id=tid,
        query_text="q",
        expected_mode="chat",
        detected_mode=detected,
        effective_mode=effective,
        expected_behavior_json=json.dumps(
            {
                "expected_mode": "chat",
                "must_include": [],
                "must_not_include": must_not,
                "critical": critical,
                "error_tags": [],
                "category": "chat",
            }
        ),
        actual_response="ok",
        pass_bool=passed,
        score=score,
        error_categories_json="[]",
        validation_failures_json=json.dumps(
            {"passed": True, "checks_failed": [], "flags": {}, "severity": "pass"}
        ),
        retrieval_chunk_ids_json="[]",
        latency_ms=1,
    )


def test_compare_overall_score_regression(app):
    with app.app_context():
        p = EvaluationRun(
            run_name="p",
            dataset_name="ds-reg",
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            overall_score=1.0,
            notes_json="{}",
        )
        c = EvaluationRun(
            run_name="c",
            dataset_name="ds-reg",
            total_cases=2,
            passed_cases=0,
            failed_cases=2,
            overall_score=0.85,
            notes_json="{}",
        )
        db.session.add_all([p, c])
        db.session.commit()
        prev_cases = [
            _row(p.id, "a", passed=True, score=1.0),
            _row(p.id, "b", passed=True, score=1.0),
        ]
        curr_cases = [
            _row(c.id, "a", passed=True, score=1.0),
            _row(c.id, "b", passed=True, score=0.7),
        ]
        for x in prev_cases + curr_cases:
            db.session.add(x)
        db.session.commit()
        f = compare_eval_runs(p, c, prev_cases, curr_cases)
        assert f.overall_score_regression is True
        assert f.overall_rel_drop is not None
        assert f.overall_rel_drop > 0.05


def test_newly_failing_critical(app):
    with app.app_context():
        p = EvaluationRun(
            run_name="p2",
            dataset_name="ds-crit",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            overall_score=1.0,
            notes_json="{}",
        )
        c = EvaluationRun(
            run_name="c2",
            dataset_name="ds-crit",
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            overall_score=0.0,
            notes_json="{}",
        )
        db.session.add_all([p, c])
        db.session.commit()
        prev_cases = [_row(p.id, "crit1", passed=True, score=1.0, critical=True)]
        curr_cases = [_row(c.id, "crit1", passed=False, score=0.0, critical=True)]
        for x in prev_cases + curr_cases:
            db.session.add(x)
        db.session.commit()
        f = compare_eval_runs(p, c, prev_cases, curr_cases)
        assert "crit1" in f.newly_failing_critical


def test_render_regression_markdown_includes_dataset():
    prev = EvaluationRun(
        run_name="a",
        dataset_name="ds",
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        overall_score=1.0,
        notes_json="{}",
    )
    prev.id = 1
    curr = EvaluationRun(
        run_name="b",
        dataset_name="ds",
        total_cases=1,
        passed_cases=0,
        failed_cases=1,
        overall_score=0.5,
        notes_json="{}",
    )
    curr.id = 2
    md = render_regression_markdown(
        dataset_name="ds",
        prev=prev,
        curr=curr,
        finding=RegressionFinding(
            overall_score_regression=True,
            overall_prev=1.0,
            overall_curr=0.5,
            overall_rel_drop=0.5,
        ),
    )
    assert "Regression report" in md
    assert "Overall score regression" in md


def test_write_regression_report_skips_without_previous(app, tmp_path):
    with app.app_context():
        c = EvaluationRun(
            run_name="only",
            dataset_name="ds-solo",
            total_cases=0,
            passed_cases=0,
            failed_cases=0,
            overall_score=None,
            notes_json="{}",
        )
        db.session.add(c)
        db.session.commit()
        rid = c.id
        out = write_regression_report(tmp_path, rid, "ds-solo")
        assert out is None


def test_write_regression_report_writes_when_previous_exists(app, tmp_path):
    with app.app_context():
        p = EvaluationRun(
            run_name="old",
            dataset_name="ds-two",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            overall_score=1.0,
            notes_json="{}",
        )
        c = EvaluationRun(
            run_name="new",
            dataset_name="ds-two",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            overall_score=1.0,
            notes_json="{}",
        )
        db.session.add_all([p, c])
        db.session.commit()
        # Ensure 'current' is newer — flush order: p first, c second; same second resolution ok
        db.session.add(_row(p.id, "t", passed=True, score=1.0))
        db.session.add(_row(c.id, "t", passed=True, score=1.0))
        db.session.commit()
        path = write_regression_report(tmp_path, c.id, "ds-two")
        assert path is not None
        assert path.name == "regression_report.md"
        text = Path(path).read_text(encoding="utf-8")
        assert "Regression report" in text
        assert "ds-two" in text
