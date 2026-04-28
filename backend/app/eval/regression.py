"""Compare the latest eval run to the previous run on the same dataset; write markdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import asc

from app.eval.analytics_common import parse_expected_behavior, suite_category
from app.eval.case_result_tags import canonical_failure_tags_for_row
from app.eval.dataset import case_is_critical_from_behavior
from app.extensions import db
from app.models.evaluation import EvaluationCaseResult, EvaluationRun

RETRIEVAL_LEAKAGE = "retrieval_leakage"
MODE_TAGS = frozenset({"mode_misclassification", "mode_routing_failure"})

OVERALL_DROP_THRESHOLD = 0.05
CATEGORY_DROP_THRESHOLD = 0.10
RATE_INCREASE_EPS = 0.001


@dataclass
class RegressionFinding:
    overall_score_regression: bool = False
    overall_prev: float | None = None
    overall_curr: float | None = None
    overall_rel_drop: float | None = None
    category_regressions: list[dict[str, Any]] = field(default_factory=list)
    newly_failing_critical: list[str] = field(default_factory=list)
    retrieval_leakage_increase: bool = False
    leakage_rate_prev: float | None = None
    leakage_rate_curr: float | None = None
    mode_failure_increase: bool = False
    mode_fail_rate_prev: float | None = None
    mode_fail_rate_curr: float | None = None


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _cases_by_test_id(rows: list[EvaluationCaseResult]) -> dict[str, EvaluationCaseResult]:
    return {r.test_id: r for r in rows}


def _category_means(rows: list[EvaluationCaseResult]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for r in rows:
        cat = suite_category(r)
        buckets.setdefault(cat, []).append(float(r.score or 0.0))
    return {c: _mean(vals) or 0.0 for c, vals in buckets.items() if vals}


def _leakage_rate(rows: list[EvaluationCaseResult]) -> float:
    if not rows:
        return 0.0
    n = sum(1 for r in rows if RETRIEVAL_LEAKAGE in canonical_failure_tags_for_row(r))
    return n / len(rows)


def _mode_fail_rate(rows: list[EvaluationCaseResult]) -> float:
    if not rows:
        return 0.0
    n = 0
    for r in rows:
        tags = set(canonical_failure_tags_for_row(r))
        if tags & MODE_TAGS:
            n += 1
    return n / len(rows)


def compare_eval_runs(
    prev: EvaluationRun,
    curr: EvaluationRun,
    prev_cases: list[EvaluationCaseResult],
    curr_cases: list[EvaluationCaseResult],
) -> RegressionFinding:
    out = RegressionFinding()
    pv, cv = prev.overall_score, curr.overall_score
    out.overall_prev = pv
    out.overall_curr = cv
    if pv is not None and cv is not None and pv > 1e-9:
        rel = (pv - cv) / pv
        out.overall_rel_drop = rel
        if rel > OVERALL_DROP_THRESHOLD:
            out.overall_score_regression = True

    prev_cm = _category_means(prev_cases)
    curr_cm = _category_means(curr_cases)
    for cat, p_mean in prev_cm.items():
        c_mean = curr_cm.get(cat)
        if c_mean is None or p_mean <= 1e-9:
            continue
        rel_drop = (p_mean - c_mean) / p_mean
        if rel_drop > CATEGORY_DROP_THRESHOLD:
            out.category_regressions.append(
                {
                    "category": cat,
                    "mean_prev": round(p_mean, 4),
                    "mean_curr": round(c_mean, 4),
                    "relative_drop": round(rel_drop, 4),
                }
            )

    prev_map = _cases_by_test_id(prev_cases)
    curr_map = _cases_by_test_id(curr_cases)
    for tid, crow in curr_map.items():
        beh = parse_expected_behavior(crow.expected_behavior_json)
        if not case_is_critical_from_behavior(beh):
            continue
        prow = prev_map.get(tid)
        if prow and prow.pass_bool and not crow.pass_bool:
            out.newly_failing_critical.append(tid)

    out.leakage_rate_prev = _leakage_rate(prev_cases)
    out.leakage_rate_curr = _leakage_rate(curr_cases)
    if (
        out.leakage_rate_curr is not None
        and out.leakage_rate_prev is not None
        and (out.leakage_rate_curr - out.leakage_rate_prev) > RATE_INCREASE_EPS
    ):
        out.retrieval_leakage_increase = True

    out.mode_fail_rate_prev = _mode_fail_rate(prev_cases)
    out.mode_fail_rate_curr = _mode_fail_rate(curr_cases)
    if (
        out.mode_fail_rate_curr is not None
        and out.mode_fail_rate_prev is not None
        and (out.mode_fail_rate_curr - out.mode_fail_rate_prev) > RATE_INCREASE_EPS
    ):
        out.mode_failure_increase = True

    return out


def _fetch_previous_run(current_id: int, dataset_name: str) -> EvaluationRun | None:
    return (
        EvaluationRun.query.filter(
            EvaluationRun.dataset_name == dataset_name,
            EvaluationRun.id != current_id,
        )
        .order_by(EvaluationRun.created_at.desc())
        .first()
    )


def _fetch_cases(run_id: int) -> list[EvaluationCaseResult]:
    return (
        EvaluationCaseResult.query.filter_by(evaluation_run_id=run_id)
        .order_by(asc(EvaluationCaseResult.test_id))
        .all()
    )


def render_regression_markdown(
    *,
    dataset_name: str,
    prev: EvaluationRun,
    curr: EvaluationRun,
    finding: RegressionFinding,
) -> str:
    lines = [
        "# Regression report",
        "",
        f"- **Dataset:** `{dataset_name}`",
        f"- **Previous run:** id={prev.id} name={prev.run_name!r} score={prev.overall_score}",
        f"- **Current run:** id={curr.id} name={curr.run_name!r} score={curr.overall_score}",
        "",
        "## Flags",
        "",
    ]
    any_flag = False
    if finding.overall_score_regression:
        any_flag = True
        lines.append(
            f"- **Overall score regression:** relative drop "
            f"{(finding.overall_rel_drop or 0) * 100:.1f}% "
            f"(threshold {OVERALL_DROP_THRESHOLD * 100:.0f}%)"
        )
    if finding.category_regressions:
        any_flag = True
        lines.append("- **Category mean score regression** (>10% drop vs previous):")
        for c in finding.category_regressions:
            lines.append(
                f"  - `{c['category']}`: {c['mean_prev']} → {c['mean_curr']} "
                f"(Δ {c['relative_drop'] * 100:.1f}%)"
            )
    if finding.newly_failing_critical:
        any_flag = True
        lines.append(
            "- **Newly failing critical tests:** " + ", ".join(f"`{x}`" for x in finding.newly_failing_critical)
        )
    if finding.retrieval_leakage_increase:
        any_flag = True
        lines.append(
            f"- **Retrieval leakage rate increased:** "
            f"{finding.leakage_rate_prev:.4f} → {finding.leakage_rate_curr:.4f}"
        )
    if finding.mode_failure_increase:
        any_flag = True
        lines.append(
            f"- **Mode routing failure rate increased:** "
            f"{finding.mode_fail_rate_prev:.4f} → {finding.mode_fail_rate_curr:.4f}"
        )
    if not any_flag:
        lines.append("- No regression thresholds tripped (see numeric snapshot below).")
    lines.extend(
        [
            "",
            "## Numeric snapshot",
            "",
            f"| Metric | Previous | Current |",
            f"|--------|----------|---------|",
            f"| Mean overall score | {finding.overall_prev} | {finding.overall_curr} |",
            f"| Retrieval leakage share | {finding.leakage_rate_prev} | {finding.leakage_rate_curr} |",
            f"| Mode misroute share | {finding.mode_fail_rate_prev} | {finding.mode_fail_rate_curr} |",
            "",
        ]
    )
    return "\n".join(lines)


def write_regression_report(out_dir: Path, current_run_id: int, dataset_name: str) -> Path | None:
    """
    Write ``regression_report.md`` under ``out_dir`` when a prior run exists for ``dataset_name``.
    Returns the path written, or ``None`` if there is nothing to compare.
    """
    prev = _fetch_previous_run(current_run_id, dataset_name)
    if prev is None:
        return None
    curr = db.session.get(EvaluationRun, current_run_id)
    if curr is None:
        return None
    prev_cases = _fetch_cases(prev.id)
    curr_cases = _fetch_cases(current_run_id)
    finding = compare_eval_runs(prev, curr, prev_cases, curr_cases)
    body = render_regression_markdown(
        dataset_name=dataset_name,
        prev=prev,
        curr=curr,
        finding=finding,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "regression_report.md"
    path.write_text(body, encoding="utf-8")
    return path
