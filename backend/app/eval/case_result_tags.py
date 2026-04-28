"""Reconstruct canonical failure tags from persisted :class:`EvaluationCaseResult` rows."""

from __future__ import annotations

import json
from typing import Any

from app.eval.analytics_common import parse_expected_behavior
from app.eval.report_markdown import build_pipeline_result_for_tags
from app.models.evaluation import EvaluationCaseResult
from app.services.eval_run import failure_tags_for_case


def _chunk_ids_from_json(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[int] = []
    for x in data:
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, str) and x.isdigit():
            out.append(int(x))
    return out


def case_raw_from_evaluation_row(row: EvaluationCaseResult) -> dict[str, Any]:
    beh = parse_expected_behavior(row.expected_behavior_json)
    return {
        "id": row.test_id,
        "query": row.query_text,
        "expected_mode": beh.get("expected_mode") or row.expected_mode or "chat",
        "must_include": list(beh.get("must_include") or []),
        "must_not_include": list(beh.get("must_not_include") or []),
        "expected_sections": list(beh.get("expected_sections") or []),
        "forbidden_sections": list(beh.get("forbidden_sections") or []),
        "category": beh.get("category") or "",
        "error_tags": list(beh.get("error_tags") or []),
    }


def canonical_failure_tags_for_row(row: EvaluationCaseResult) -> list[str]:
    """Canonical tags for analytics/regression; empty when the suite marked the case as passing."""
    if row.pass_bool:
        return []
    case_raw = case_raw_from_evaluation_row(row)
    payload: dict[str, Any] = {
        "mode": {
            "detected": row.detected_mode,
            "effective": row.effective_mode,
        }
    }
    pl_diag: dict[str, Any] | None = None
    if row.validation_failures_json:
        try:
            v = json.loads(row.validation_failures_json)
            if isinstance(v, dict):
                pl_diag = {"validation": v}
        except json.JSONDecodeError:
            pl_diag = None
    pr = build_pipeline_result_for_tags(
        course_answer=row.actual_response or "",
        payload=payload,
        pl_diag=pl_diag,
        chunk_ids=_chunk_ids_from_json(row.retrieval_chunk_ids_json),
    )
    return failure_tags_for_case(case_raw, pr, pass_bool=False)
