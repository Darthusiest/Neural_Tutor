"""Admin JSON for batch evaluation runs (dashboard + API)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import asc, desc

from app.eval.analytics_common import parse_json_list, suite_category
from app.eval.case_result_tags import canonical_failure_tags_for_row
from app.extensions import db
from app.models.evaluation import EvaluationCaseResult, EvaluationRun


def _notes_dict(run: EvaluationRun) -> dict[str, Any]:
    if not run.notes_json:
        return {}
    try:
        data = json.loads(run.notes_json)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _report_file_paths(notes: dict[str, Any]) -> dict[str, str | None]:
    base = notes.get("reports_dir")
    if not base:
        return {"reports_dir": None, "examples_md": None, "error_analysis_md": None, "regression_md": None}
    p = Path(base)
    return {
        "reports_dir": str(p),
        "examples_md": str(p / "examples.md"),
        "error_analysis_md": str(p / "error_analysis.md"),
        "regression_md": str(p / "regression_report.md"),
    }


def serialize_run_summary(run: EvaluationRun) -> dict[str, Any]:
    t = max(1, run.total_cases or 0)
    passed = run.passed_cases or 0
    notes = _notes_dict(run)
    out: dict[str, Any] = {
        "id": run.id,
        "run_name": run.run_name,
        "dataset_name": run.dataset_name,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "total_cases": run.total_cases,
        "passed_cases": passed,
        "failed_cases": run.failed_cases,
        "overall_score": run.overall_score,
        "pass_rate": round(passed / t, 4),
        "git_commit": run.git_commit,
        "branch_name": run.branch_name,
        "report_files": _report_file_paths(notes),
    }
    return out


def list_evaluation_runs(*, limit: int = 100, dataset_substring: str | None = None) -> dict[str, Any]:
    q = EvaluationRun.query.order_by(desc(EvaluationRun.created_at), desc(EvaluationRun.id))
    if dataset_substring:
        q = q.filter(EvaluationRun.dataset_name.contains(dataset_substring))
    rows = q.limit(max(1, min(limit, 500))).all()
    return {"runs": [serialize_run_summary(r) for r in rows]}


def _category_breakdown(cases: list[EvaluationCaseResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for c in cases:
        cat = suite_category(c)
        bucket = out.setdefault(
            cat,
            {"category": cat, "n": 0, "passed": 0, "failed": 0, "score_sum": 0.0},
        )
        bucket["n"] += 1
        if c.pass_bool:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
        bucket["score_sum"] += float(c.score or 0.0)
    for b in out.values():
        n = b["n"]
        b["mean_score"] = round(b["score_sum"] / max(1, n), 4)
        b["pass_rate"] = round(b["passed"] / max(1, n), 4)
        del b["score_sum"]
    return out


def evaluation_run_detail(run_id: int) -> dict[str, Any] | None:
    run = db.session.get(EvaluationRun, run_id)
    if run is None:
        return None
    cases = (
        EvaluationCaseResult.query.filter_by(evaluation_run_id=run_id)
        .order_by(asc(EvaluationCaseResult.test_id))
        .all()
    )
    t = max(1, run.total_cases or len(cases))
    passed = run.passed_cases or sum(1 for c in cases if c.pass_bool)
    summary = serialize_run_summary(run)
    failed_n = run.failed_cases or sum(1 for c in cases if not c.pass_bool)
    summary["failure_table_preview"] = [
        {
            "test_id": c.test_id,
            "query": (c.query_text or "")[:200],
            "score": c.score,
            "error_categories": parse_json_list(c.error_categories_json),
            "canonical_tags": canonical_failure_tags_for_row(c),
        }
        for c in cases
        if not c.pass_bool
    ][:200]
    summary["category_breakdown"] = list(_category_breakdown(cases).values())
    summary["counts"] = {
        "total_cases": len(cases),
        "passed_cases": passed,
        "failed_cases": failed_n,
        "pass_rate": round(passed / t, 4),
    }
    return summary


def evaluation_run_failures(run_id: int) -> dict[str, Any] | None:
    run = db.session.get(EvaluationRun, run_id)
    if run is None:
        return None
    cases = (
        EvaluationCaseResult.query.filter_by(evaluation_run_id=run_id, pass_bool=False)
        .order_by(asc(EvaluationCaseResult.test_id))
        .all()
    )
    rows = []
    for c in cases:
        rows.append(
            {
                "test_id": c.test_id,
                "query_text": c.query_text,
                "expected_mode": c.expected_mode,
                "detected_mode": c.detected_mode,
                "effective_mode": c.effective_mode,
                "score": c.score,
                "scoring_errors": parse_json_list(c.error_categories_json),
                "canonical_failure_tags": canonical_failure_tags_for_row(c),
                "latency_ms": c.latency_ms,
            }
        )
    return {
        "evaluation_run_id": run_id,
        "dataset_name": run.dataset_name,
        "run_name": run.run_name,
        "failures": rows,
        "report_files": _report_file_paths(_notes_dict(run)),
    }
