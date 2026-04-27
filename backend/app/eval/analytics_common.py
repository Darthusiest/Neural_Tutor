"""Shared helpers for eval DB analytics (CSV export and plots)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import asc

from app.models import EvaluationCaseResult, EvaluationRun


def parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if x is not None]


def parse_expected_behavior(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def suite_category(case_row: EvaluationCaseResult) -> str:
    """Group label from suite JSON: category, else expected_mode, else unknown."""
    beh = parse_expected_behavior(case_row.expected_behavior_json)
    cat = (beh.get("category") or "").strip()
    if cat:
        return cat
    em = (case_row.expected_mode or beh.get("expected_mode") or "").strip()
    if em:
        return em
    return "unknown"


@dataclass(frozen=True)
class RunFilter:
    dataset_substring: str | None = None
    run_ids: frozenset[int] | None = None
    last_n_runs: int | None = None


def fetch_ordered_runs(rf: RunFilter) -> list[EvaluationRun]:
    q = EvaluationRun.query
    if rf.dataset_substring:
        sub = rf.dataset_substring
        q = q.filter(EvaluationRun.dataset_name.contains(sub))
    if rf.run_ids:
        q = q.filter(EvaluationRun.id.in_(rf.run_ids))
    q = q.order_by(asc(EvaluationRun.created_at), asc(EvaluationRun.id))
    runs = q.all()
    if rf.last_n_runs is not None and rf.last_n_runs > 0:
        runs = runs[-rf.last_n_runs :]
    return runs


def fetch_case_rows_for_runs(run_ids: list[int]) -> list[EvaluationCaseResult]:
    if not run_ids:
        return []
    return (
        EvaluationCaseResult.query.filter(
            EvaluationCaseResult.evaluation_run_id.in_(run_ids)
        )
        .order_by(
            asc(EvaluationCaseResult.evaluation_run_id),
            asc(EvaluationCaseResult.test_id),
        )
        .all()
    )


def percentile(sorted_vals: list[float], p: float) -> float | None:
    """p in [0, 100]; sorted_vals must be sorted ascending."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))
