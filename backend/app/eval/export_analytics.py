"""Export eval run / case analytics as CSV files from the database."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import create_app
from app.eval.analytics_common import (
    RunFilter,
    fetch_case_rows_for_runs,
    fetch_ordered_runs,
    parse_expected_behavior,
    parse_json_list,
    percentile,
    suite_category,
)
from app.eval.capability_analytics import (
    retrieval_diagnostics,
    summarize_boost,
    summarize_capability,
    summarize_coverage,
    summarize_coverage_phase_buckets,
    summarize_errors,
    summarize_structure,
)
from app.models import EvaluationCaseResult, EvaluationRun

RETRIEVAL_LEAKAGE_TAG = "retrieval_leakage"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _default_out_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = _repo_root() / "reports" / "eval_analytics" / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _norm_mode(m: str | None) -> str:
    return (m or "").strip().lower()


def export_overall_by_run(runs: list[EvaluationRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in runs:
        total = max(1, r.total_cases or 0)
        passed = r.passed_cases or 0
        rows.append(
            {
                "run_id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "run_name": r.run_name,
                "dataset_name": r.dataset_name,
                "git_commit": r.git_commit or "",
                "total_cases": r.total_cases,
                "passed_cases": passed,
                "failed_cases": r.failed_cases,
                "overall_score": "" if r.overall_score is None else r.overall_score,
                "pass_rate": round(passed / total, 6),
            }
        )
    return rows


def export_score_by_category(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    by_cat: dict[str, dict[str, Any]] = {}
    for c in cases:
        cat = suite_category(c)
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "passed": 0, "failed": 0, "score_sum": 0.0}
        by_cat[cat]["total"] += 1
        if c.pass_bool:
            by_cat[cat]["passed"] += 1
        else:
            by_cat[cat]["failed"] += 1
        if c.score is not None:
            by_cat[cat]["score_sum"] += float(c.score)
    rows: list[dict[str, Any]] = []
    for cat in sorted(by_cat.keys()):
        agg = by_cat[cat]
        t = agg["total"]
        avg = round(agg["score_sum"] / max(1, t), 4)
        rows.append(
            {
                "category": cat,
                "total": t,
                "passed": agg["passed"],
                "failed": agg["failed"],
                "avg_score": avg,
            }
        )
    return rows


def export_pass_fail_by_case(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in cases:
        errs = parse_json_list(c.error_categories_json)
        rows.append(
            {
                "evaluation_run_id": c.evaluation_run_id,
                "test_id": c.test_id,
                "query_text": c.query_text,
                "category": suite_category(c),
                "expected_mode": c.expected_mode or "",
                "detected_mode": c.detected_mode or "",
                "effective_mode": c.effective_mode or "",
                "pass_bool": c.pass_bool,
                "score": "" if c.score is None else c.score,
                "error_categories": ";".join(errs),
                "latency_ms": "" if c.latency_ms is None else c.latency_ms,
            }
        )
    return rows


def export_error_counts(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for c in cases:
        for e in parse_json_list(c.error_categories_json):
            if e:
                counts[e] += 1
    rows = [{"error_category": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    return rows


def export_mode_accuracy_by_run(
    runs: list[EvaluationRun], cases: list[EvaluationCaseResult]
) -> list[dict[str, Any]]:
    by_run: dict[int, list[EvaluationCaseResult]] = defaultdict(list)
    for c in cases:
        by_run[c.evaluation_run_id].append(c)

    rows: list[dict[str, Any]] = []
    run_ids = {r.id for r in runs}
    for rid in sorted(run_ids):
        rc = by_run.get(rid, [])
        eff_total = 0
        eff_ok = 0
        det_total = 0
        det_ok = 0
        for c in rc:
            exp = _norm_mode(c.expected_mode) or _norm_mode(
                str(parse_expected_behavior(c.expected_behavior_json).get("expected_mode") or "")
            )
            if not exp:
                continue
            eff_total += 1
            if _norm_mode(c.effective_mode) == exp:
                eff_ok += 1
            beh = parse_expected_behavior(c.expected_behavior_json)
            cat = (beh.get("category") or "").strip().lower()
            if cat == "mode_detection":
                det_total += 1
                if _norm_mode(c.detected_mode) == exp:
                    det_ok += 1

        row: dict[str, Any] = {
            "run_id": rid,
            "cases_with_expected_mode": eff_total,
            "effective_matches": eff_ok,
            "effective_accuracy": ""
            if eff_total == 0
            else round(eff_ok / eff_total, 6),
            "mode_detection_cases": det_total,
            "detected_matches": det_ok,
            "detected_accuracy": ""
            if det_total == 0
            else round(det_ok / det_total, 6),
        }
        rows.append(row)
    return rows


def export_retrieval_leakage(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in cases:
        errs = parse_json_list(c.error_categories_json)
        had = RETRIEVAL_LEAKAGE_TAG in errs
        rows.append(
            {
                "evaluation_run_id": c.evaluation_run_id,
                "test_id": c.test_id,
                "query_text": c.query_text,
                "had_retrieval_leakage": had,
                "error_categories": ";".join(errs),
            }
        )
    return rows


def export_worst_queries(
    cases: list[EvaluationCaseResult], *, per_run_limit: int
) -> list[dict[str, Any]]:
    by_run: dict[int, list[EvaluationCaseResult]] = defaultdict(list)
    for c in cases:
        by_run[c.evaluation_run_id].append(c)

    rows: list[dict[str, Any]] = []
    for rid in sorted(by_run.keys()):
        rc = sorted(
            by_run[rid],
            key=lambda x: (
                float(x.score) if x.score is not None else float("-inf"),
                x.test_id,
            ),
        )
        for c in rc[:per_run_limit]:
            errs = parse_json_list(c.error_categories_json)
            rows.append(
                {
                    "evaluation_run_id": c.evaluation_run_id,
                    "test_id": c.test_id,
                    "query_text": c.query_text,
                    "score": "" if c.score is None else c.score,
                    "pass_bool": c.pass_bool,
                    "error_categories": ";".join(errs),
                }
            )
    return rows


def export_latency_by_run(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    by_run: dict[int, list[int]] = defaultdict(list)
    for c in cases:
        if c.latency_ms is not None:
            by_run[c.evaluation_run_id].append(int(c.latency_ms))

    rows: list[dict[str, Any]] = []
    for rid in sorted(by_run.keys()):
        vals = sorted(by_run[rid])
        n = len(vals)
        mean = sum(vals) / n
        med = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        p95 = percentile(vals, 95.0)
        rows.append(
            {
                "run_id": rid,
                "mean_latency_ms": round(mean, 2),
                "median_latency_ms": round(med, 2),
                "p95_latency_ms": "" if p95 is None else round(p95, 2),
            }
        )
    return rows


def export_capability_breakdown(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    cap = summarize_capability(cases)["by_intent"]
    return [
        {
            "intent": intent,
            "total_cases": data["total_cases"],
            "correct_cases": data["correct_cases"],
            "accuracy": data["accuracy"],
        }
        for intent, data in sorted(cap.items())
    ]


def export_primary_error_breakdown(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    errors = summarize_errors(cases)["by_error_type"]
    return [
        {
            "error_type": error_type,
            "count": data["count"],
            "percentage": data["percentage"],
        }
        for error_type, data in errors.items()
    ]


def export_retrieval_diagnostics(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diag in retrieval_diagnostics(cases):
        d = diag.to_dict()
        rows.append(
            {
                "test_id": d["test_id"],
                "query_text": d["query_text"],
                "concept": d["concept"],
                "top_1_correct": d["top_1_correct"],
                "top_k_contains_correct": d["top_k_contains_correct"],
                "retrieval_noise": d["retrieval_noise"],
                "retrieved_chunk_ids": ";".join(str(x) for x in d["retrieved_chunk_ids"]),
                "concept_match_score": ""
                if d["concept_match_score"] is None
                else d["concept_match_score"],
            }
        )
    return rows


def export_structure_compliance(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    structure = summarize_structure(cases)
    rows: list[dict[str, Any]] = []
    violations = structure.get("violations", {})
    for intent, data in structure["by_intent"].items():
        rows.append(
            {
                "intent": intent,
                "total_cases": data["total_cases"],
                "compliant_cases": data["compliant_cases"],
                "compliance_rate": data["compliance_rate"],
                "violations": ";".join(
                    f"{name}:{count}" for name, count in sorted(violations.items())
                ),
            }
        )
    return rows


def export_coverage(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    coverage = summarize_coverage(cases)
    under_tested = set(coverage["under_tested_concepts"])
    return [
        {
            "concept": concept,
            "case_count": coverage["count_per_concept"][concept],
            "accuracy": coverage["accuracy_per_concept"][concept],
            "under_tested": concept in under_tested,
        }
        for concept in sorted(coverage["count_per_concept"])
    ]


def export_coverage_phase_plan(
    cases: list[EvaluationCaseResult],
    *,
    min_cases: int = 3,
    sort_mode: str = "failure_first",
) -> list[dict[str, Any]]:
    """Ordered remediation phases: one row per coverage bucket (see :func:`summarize_coverage_phase_buckets`)."""
    return summarize_coverage_phase_buckets(
        cases, min_cases=min_cases, sort_mode=sort_mode
    )


def export_boost_effectiveness(cases: list[EvaluationCaseResult]) -> list[dict[str, Any]]:
    boost = summarize_boost(cases)
    if not boost:
        return []
    return [boost]


def _parse_run_ids(s: str | None) -> frozenset[int] | None:
    if not s or not s.strip():
        return None
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return frozenset(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export eval analytics CSVs from the database.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: reports/eval_analytics/<utc-timestamp>/)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Substring filter on EvaluationRun.dataset_name",
    )
    parser.add_argument(
        "--run-ids",
        default=None,
        help="Comma-separated evaluation run ids (optional)",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Keep only the last N runs after ordering by created_at",
    )
    parser.add_argument(
        "--worst-k",
        type=int,
        default=20,
        help="Per-run worst queries to include (by ascending score)",
    )
    parser.add_argument(
        "--coverage-phase-sort",
        choices=("failure_first", "chart_volume"),
        default="failure_first",
        help=(
            "Ordering for coverage_phase_plan.csv: failure_first (default) or "
            "chart_volume (matches coverage_by_concept.png bucket order)."
        ),
    )
    parser.add_argument(
        "--coverage-min-cases",
        type=int,
        default=3,
        help="Minimum cases before a bucket is treated as adequately sampled in coverage_phase_plan.csv.",
    )
    args = parser.parse_args(argv)

    rf = RunFilter(
        dataset_substring=args.dataset,
        run_ids=_parse_run_ids(args.run_ids),
        last_n_runs=args.last_n,
    )

    app = create_app()
    with app.app_context():
        runs = fetch_ordered_runs(rf)
        if not runs:
            total = EvaluationRun.query.count()
            if total == 0:
                print(
                    "No evaluation runs in the database; nothing written.\n"
                    "Create runs first, for example:\n"
                    "  PYTHONPATH=. python -m app.eval.run_eval "
                    '--dataset data/eval/l487_eval_suite.json --run-name "baseline"'
                )
            else:
                print(
                    "No evaluation runs matched filters; nothing written. "
                    f"(Found {total} run(s) in DB — try different --dataset / --run-ids / --last-n.)"
                )
            return 1
        out_dir = args.out_dir or _default_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_ids = [r.id for r in runs]
        cases = fetch_case_rows_for_runs(run_ids)

        _write_csv(
            out_dir / "overall_score_by_run.csv",
            export_overall_by_run(runs),
            [
                "run_id",
                "created_at",
                "run_name",
                "dataset_name",
                "git_commit",
                "total_cases",
                "passed_cases",
                "failed_cases",
                "overall_score",
                "pass_rate",
            ],
        )
        _write_csv(
            out_dir / "score_by_category.csv",
            export_score_by_category(cases),
            ["category", "total", "passed", "failed", "avg_score"],
        )
        _write_csv(
            out_dir / "pass_fail_by_test_case.csv",
            export_pass_fail_by_case(cases),
            [
                "evaluation_run_id",
                "test_id",
                "query_text",
                "category",
                "expected_mode",
                "detected_mode",
                "effective_mode",
                "pass_bool",
                "score",
                "error_categories",
                "latency_ms",
            ],
        )
        _write_csv(
            out_dir / "error_count_by_error_category.csv",
            export_error_counts(cases),
            ["error_category", "count"],
        )
        _write_csv(
            out_dir / "mode_accuracy.csv",
            export_mode_accuracy_by_run(runs, cases),
            [
                "run_id",
                "cases_with_expected_mode",
                "effective_matches",
                "effective_accuracy",
                "mode_detection_cases",
                "detected_matches",
                "detected_accuracy",
            ],
        )
        _write_csv(
            out_dir / "retrieval_leakage.csv",
            export_retrieval_leakage(cases),
            [
                "evaluation_run_id",
                "test_id",
                "query_text",
                "had_retrieval_leakage",
                "error_categories",
            ],
        )
        _write_csv(
            out_dir / "worst_performing_queries.csv",
            export_worst_queries(cases, per_run_limit=max(1, args.worst_k)),
            [
                "evaluation_run_id",
                "test_id",
                "query_text",
                "score",
                "pass_bool",
                "error_categories",
            ],
        )
        _write_csv(
            out_dir / "latency_by_run.csv",
            export_latency_by_run(cases),
            ["run_id", "mean_latency_ms", "median_latency_ms", "p95_latency_ms"],
        )
        _write_csv(
            out_dir / "capability_breakdown.csv",
            export_capability_breakdown(cases),
            ["intent", "total_cases", "correct_cases", "accuracy"],
        )
        _write_csv(
            out_dir / "error_breakdown.csv",
            export_primary_error_breakdown(cases),
            ["error_type", "count", "percentage"],
        )
        _write_csv(
            out_dir / "retrieval_diagnostics.csv",
            export_retrieval_diagnostics(cases),
            [
                "test_id",
                "query_text",
                "concept",
                "top_1_correct",
                "top_k_contains_correct",
                "retrieval_noise",
                "retrieved_chunk_ids",
                "concept_match_score",
            ],
        )
        _write_csv(
            out_dir / "structure_compliance.csv",
            export_structure_compliance(cases),
            ["intent", "total_cases", "compliant_cases", "compliance_rate", "violations"],
        )
        _write_csv(
            out_dir / "coverage.csv",
            export_coverage(cases),
            ["concept", "case_count", "accuracy", "under_tested"],
        )
        _write_csv(
            out_dir / "coverage_phase_plan.csv",
            export_coverage_phase_plan(
                cases,
                min_cases=max(1, args.coverage_min_cases),
                sort_mode=args.coverage_phase_sort,
            ),
            [
                "phase_rank",
                "concept_label",
                "case_count",
                "passed",
                "failed",
                "accuracy",
                "under_tested",
                "sort_mode",
                "min_cases_threshold",
            ],
        )
        boost_rows = export_boost_effectiveness(cases)
        if boost_rows:
            _write_csv(
                out_dir / "boost_effectiveness.csv",
                boost_rows,
                [
                    "paired_cases",
                    "boost_triggered_rate",
                    "boost_added_value_rate",
                    "avg_boost_latency_ms",
                    "avg_latency_without_boost_ms",
                    "avg_latency_with_boost_ms",
                ],
            )

    print(f"Wrote analytics CSVs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
