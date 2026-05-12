"""Rule-based eval analytics for pipeline behavior and debugging."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.eval.analytics_common import parse_expected_behavior, parse_json_list
from app.eval.dataset import CAPABILITY_INTENTS, effective_intent
from app.models import EvaluationCaseResult, EvaluationRun
from app.models.content import LectureChunk

PRIMARY_ERROR_TYPES = frozenset(
    {
        "retrieval_miss",
        "retrieval_noise",
        "structure_failure",
        "missing_steps",
        "shallow_explanation",
        "hallucination",
        "template_misuse",
        "critic_pipeline_error",
    }
)

_TEMPLATE_TAGS = frozenset(
    {
        "quiz_not_rendered",
        "summary_wrong_scope",
        "compare_entity_collapse",
        "compare_asymmetry",
        "scaffold_leakage",
    }
)
_STRUCTURE_TAGS = frozenset(
    {
        "expected_section_missing",
        "missing_section",
        "structure_routing_mismatch",
        "structure_chat_missing_expected_section",
        "structure_chat_forbidden",
        "structure_chat_wrong_answer_mode",
        "structure_compare_no_course_answer_header",
        "structure_compare_no_contrast_cue",
        "structure_compare_missing_side_flag",
        "structure_summary_no_header",
        "structure_summary_no_key_section",
        "structure_summary_forbidden_substring",
        "structure_summary_has_direct_answer",
        "structure_summary_wrong_block",
        "structure_quiz_no_header",
        "structure_quiz_no_answer_key",
        "structure_quiz_no_numbered_questions",
        "structure_quiz_forbidden_block",
        "structure_quiz_has_course_answer",
        "structure_clarification_no_followup",
        "structure_clarification_has_direct_answer",
    }
)


@dataclass(frozen=True)
class RetrievalDiagnostic:
    test_id: str
    query_text: str
    concept: str
    retrieved_chunk_ids: list[int]
    top_1_correct: bool
    top_k_contains_correct: bool
    retrieval_noise: bool
    concept_match_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "query_text": self.query_text,
            "concept": self.concept,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "top_1_correct": self.top_1_correct,
            "top_k_contains_correct": self.top_k_contains_correct,
            "retrieval_noise": self.retrieval_noise,
            "concept_match_score": self.concept_match_score,
        }


def _round_rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return round(numer / denom, 6)


def _parse_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _chunk_ids(raw: str | None) -> list[int]:
    ids: list[int] = []
    for value in parse_json_list(raw):
        try:
            ids.append(int(value))
        except ValueError:
            continue
    return ids


def _expected_terms(behavior: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for value in behavior.get("must_include") or []:
        term = str(value or "").strip().lower()
        if term:
            terms.append(term)
    return terms


def _forbidden_terms(behavior: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("must_not_include", "forbidden_sections"):
        for value in behavior.get(key) or []:
            term = str(value or "").strip().lower()
            if term:
                terms.append(term)
    return terms


def _row_intent(row: EvaluationCaseResult) -> str:
    return effective_intent(parse_expected_behavior(row.expected_behavior_json))


def _case_concept(row: EvaluationCaseResult) -> str:
    behavior = parse_expected_behavior(row.expected_behavior_json)
    cc = str(behavior.get("coverage_concept") or "").strip().lower()
    if cc:
        return cc
    terms = _expected_terms(behavior)
    if terms:
        return terms[0]
    category = str(behavior.get("category") or row.expected_mode or "unknown").strip().lower()
    return category or "unknown"


def summarize_coverage_phase_buckets(
    cases: list[EvaluationCaseResult],
    *,
    min_cases: int = 3,
    sort_mode: str = "failure_first",
) -> list[dict[str, Any]]:
    """Per-concept pass/fail buckets with explicit phase ranks for remediation planning.

    ``sort_mode``:
    - ``failure_first`` (default): adequately sampled concepts (``total >= min_cases``)
      rank by descending failures, then ascending accuracy, then label.
    - ``chart_volume``: matches ``coverage_by_concept.png`` — descending total volume.

    Under-sampled concepts (``total < min_cases``) are listed after the main block,
    using the same secondary sort keys.
    """
    by_concept: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0})
    for row in cases:
        concept = _case_concept(row)
        if row.pass_bool:
            by_concept[concept]["passed"] += 1
        else:
            by_concept[concept]["failed"] += 1

    rows: list[dict[str, Any]] = []
    under_tested = sorted(k for k, v in by_concept.items() if v["passed"] + v["failed"] < min_cases)

    for concept, st in sorted(by_concept.items()):
        passed_n = int(st["passed"])
        failed_n = int(st["failed"])
        total = passed_n + failed_n
        acc = _round_rate(passed_n, total) if total else 0.0
        rows.append(
            {
                "concept_label": concept,
                "case_count": total,
                "passed": passed_n,
                "failed": failed_n,
                "accuracy": acc,
                "under_tested": concept in under_tested,
            }
        )

    def sort_key_volume(r: dict[str, Any]) -> tuple[int, str]:
        return (-int(r["case_count"]), str(r["concept_label"]))

    def sort_key_failure(r: dict[str, Any]) -> tuple[int, float, str]:
        return (-int(r["failed"]), float(r["accuracy"]), str(r["concept_label"]))

    tested = [r for r in rows if not r["under_tested"]]
    under = [r for r in rows if r["under_tested"]]
    key = sort_key_volume if sort_mode == "chart_volume" else sort_key_failure
    tested.sort(key=key)
    under.sort(key=key)
    ordered = tested + under

    out: list[dict[str, Any]] = []
    for rank, item in enumerate(ordered, start=1):
        row = dict(item)
        row["phase_rank"] = rank
        row["sort_mode"] = sort_mode
        row["min_cases_threshold"] = min_cases
        out.append(row)
    return out


def _row_validation(row: EvaluationCaseResult) -> dict[str, Any]:
    return _parse_json_dict(row.validation_failures_json)


def _chunk_blob(chunk: LectureChunk | dict[str, Any] | None) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, dict):
        pieces = [
            chunk.get("topic"),
            chunk.get("keywords"),
            chunk.get("source_excerpt"),
            chunk.get("clean_explanation"),
        ]
    else:
        pieces = [
            chunk.topic,
            chunk.keywords,
            chunk.source_excerpt,
            chunk.clean_explanation,
        ]
    out: list[str] = []
    for piece in pieces:
        if isinstance(piece, list):
            out.append(" ".join(str(x) for x in piece))
        elif piece is not None:
            out.append(str(piece))
    return " ".join(out).lower()


def _lecture_chunks_by_id(ids: list[int]) -> dict[int, LectureChunk]:
    if not ids:
        return {}
    rows = LectureChunk.query.filter(LectureChunk.id.in_(ids)).all()
    return {int(row.id): row for row in rows}


def _retrieval_blob_for_ids(ids: list[int]) -> str:
    chunks = _lecture_chunks_by_id(ids)
    return " ".join(_chunk_blob(chunks.get(cid)) for cid in ids)


def _chunk_matches_terms(text: str, terms: list[str]) -> bool:
    if not terms:
        return False
    return all(term in text for term in terms)


def _contains_ordered_steps(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"(?m)^\s*(?:\d+[\).]|step\s+\d+\b|[-*]\s+step\b)", text, re.I))


def _has_contrast_structure(text: str) -> bool:
    return bool(
        re.search(
            r"\b(while|whereas|difference|vs\.?|compared|contrast|unlike|but)\b",
            text or "",
            re.I,
        )
    )


def _boost_metrics(row: EvaluationCaseResult) -> dict[str, Any]:
    return _parse_json_dict(getattr(row, "boost_metrics_json", None))


def derive_primary_error_type(
    case_behavior: dict[str, Any],
    canonical_tags: list[str],
    scoring_errors: list[str],
    validation: dict[str, Any] | None,
    response_text: str,
    *,
    retrieval_text: str = "",
) -> str | None:
    """Assign exactly one brief-defined primary error type to failed cases."""
    tags = set(canonical_tags or [])
    scoring = set(scoring_errors or [])
    validation = validation or {}
    flags = validation.get("flags") if isinstance(validation.get("flags"), dict) else {}
    checks_failed = set(str(x) for x in (validation.get("checks_failed") or []))
    answer_lower = (response_text or "").lower()
    retrieval_lower = (retrieval_text or "").lower()
    required = _expected_terms(case_behavior)
    forbidden = _forbidden_terms(case_behavior)
    intent = effective_intent(case_behavior)

    if required and any(term not in answer_lower and term not in retrieval_lower for term in required):
        return "retrieval_miss"
    if any(term in answer_lower for term in forbidden):
        return "hallucination"
    if "retrieval_leakage" in tags or any(term in retrieval_lower for term in forbidden):
        return "retrieval_noise"
    if tags & _TEMPLATE_TAGS or checks_failed & {
        "must_match_quiz_contract",
        "must_match_summary_contract",
        "must_match_compare_contract",
    }:
        return "template_misuse"
    if scoring & _STRUCTURE_TAGS or any(err.startswith("structure_") for err in scoring):
        return "structure_failure"
    if intent == "step_by_step" and not _contains_ordered_steps(response_text):
        return "missing_steps"
    if flags.get("generic_filler") or len((response_text or "").strip()) < 120:
        return "shallow_explanation"
    if "missing_required_concept" in tags or "must_include_failed" in scoring:
        return "retrieval_miss"
    if "forbidden_topic_leakage" in tags or "forbidden_leak" in scoring:
        return "hallucination"
    return "shallow_explanation"


def primary_error_type_for_row(row: EvaluationCaseResult) -> str | None:
    if row.pass_bool:
        return None
    behavior = parse_expected_behavior(row.expected_behavior_json)
    tags = parse_json_list(row.error_categories_json)
    validation = _row_validation(row)
    retrieval_text = _retrieval_blob_for_ids(_chunk_ids(row.retrieval_chunk_ids_json))
    primary = getattr(row, "primary_error_type", None)
    if primary in PRIMARY_ERROR_TYPES:
        return str(primary)
    return derive_primary_error_type(
        behavior,
        tags,
        tags,
        validation,
        row.actual_response or "",
        retrieval_text=retrieval_text,
    )


def summarize_capability(cases: list[EvaluationCaseResult]) -> dict[str, Any]:
    by_intent: dict[str, dict[str, int]] = {
        intent: {"total_cases": 0, "correct_cases": 0} for intent in sorted(CAPABILITY_INTENTS)
    }
    for row in cases:
        intent = _row_intent(row)
        if intent not in by_intent:
            by_intent[intent] = {"total_cases": 0, "correct_cases": 0}
        by_intent[intent]["total_cases"] += 1
        if row.pass_bool:
            by_intent[intent]["correct_cases"] += 1
    out: dict[str, Any] = {}
    total = 0
    correct = 0
    for intent in sorted(by_intent):
        bucket = by_intent[intent]
        total += bucket["total_cases"]
        correct += bucket["correct_cases"]
        out[intent] = {
            **bucket,
            "accuracy": _round_rate(bucket["correct_cases"], bucket["total_cases"]),
        }
    return {
        "total_cases": total,
        "correct_cases": correct,
        "overall_accuracy": _round_rate(correct, total),
        "by_intent": out,
        "definition_accuracy": out.get("definition", {}).get("accuracy", 0.0),
        "step_by_step_accuracy": out.get("step_by_step", {}).get("accuracy", 0.0),
        "compare_accuracy": out.get("compare", {}).get("accuracy", 0.0),
        "synthesis_accuracy": out.get("synthesis", {}).get("accuracy", 0.0),
        "retrieval_grounded_accuracy": out.get("retrieval_grounded", {}).get("accuracy", 0.0),
    }


def summarize_errors(cases: list[EvaluationCaseResult]) -> dict[str, Any]:
    failed = [row for row in cases if not row.pass_bool]
    counts: Counter[str] = Counter()
    for row in failed:
        primary = primary_error_type_for_row(row)
        if primary:
            counts[primary] += 1
    breakdown = {
        error_type: {
            "count": count,
            "percentage": round(100.0 * count / max(1, len(failed)), 2),
        }
        for error_type, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }
    return {"failed_cases": len(failed), "by_error_type": breakdown}


def retrieval_diagnostics(cases: list[EvaluationCaseResult]) -> list[RetrievalDiagnostic]:
    all_ids: list[int] = []
    for row in cases:
        all_ids.extend(_chunk_ids(row.retrieval_chunk_ids_json))
    chunks = _lecture_chunks_by_id(sorted(set(all_ids)))
    diagnostics: list[RetrievalDiagnostic] = []
    for row in cases:
        behavior = parse_expected_behavior(row.expected_behavior_json)
        terms = _expected_terms(behavior)
        forbidden = _forbidden_terms(behavior)
        ids = _chunk_ids(row.retrieval_chunk_ids_json)
        blobs = [_chunk_blob(chunks.get(cid)) for cid in ids]
        matches = [_chunk_matches_terms(blob, terms) for blob in blobs]
        noise = any(term and term in " ".join(blobs) for term in forbidden)
        matched_terms = 0
        if terms:
            combined = " ".join(blobs)
            matched_terms = sum(1 for term in terms if term in combined)
        diagnostics.append(
            RetrievalDiagnostic(
                test_id=row.test_id,
                query_text=row.query_text,
                concept=_case_concept(row),
                retrieved_chunk_ids=ids,
                top_1_correct=bool(matches[0]) if matches else False,
                top_k_contains_correct=any(matches),
                retrieval_noise=noise or primary_error_type_for_row(row) == "retrieval_noise",
                concept_match_score=None if not terms else round(matched_terms / len(terms), 6),
            )
        )
    return diagnostics


def summarize_retrieval(cases: list[EvaluationCaseResult]) -> dict[str, Any]:
    diagnostics = retrieval_diagnostics(cases)
    evaluable = [d for d in diagnostics if d.concept and d.concept != "unknown"]
    return {
        "top_1_accuracy": _round_rate(sum(1 for d in evaluable if d.top_1_correct), len(evaluable)),
        "top_k_recall": _round_rate(
            sum(1 for d in evaluable if d.top_k_contains_correct), len(evaluable)
        ),
        "retrieval_noise_rate": _round_rate(
            sum(1 for d in diagnostics if d.retrieval_noise), len(diagnostics)
        ),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def _structure_violations(row: EvaluationCaseResult) -> list[str]:
    text = row.actual_response or ""
    intent = _row_intent(row)
    errors = set(parse_json_list(row.error_categories_json))
    violations: list[str] = []
    if intent == "step_by_step" and not _contains_ordered_steps(text):
        violations.append("missing_steps_format")
    if intent == "definition" and "key idea" not in text.lower():
        violations.append("missing_key_idea")
    if intent == "compare" and not _has_contrast_structure(text):
        violations.append("incorrect_template")
    if errors & _TEMPLATE_TAGS or primary_error_type_for_row(row) == "template_misuse":
        violations.append("incorrect_template")
    return sorted(set(violations))


def summarize_structure(cases: list[EvaluationCaseResult]) -> dict[str, Any]:
    by_intent: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "compliant": 0})
    violation_counts: Counter[str] = Counter()
    for row in cases:
        intent = _row_intent(row)
        violations = _structure_violations(row)
        by_intent[intent]["total"] += 1
        if not violations:
            by_intent[intent]["compliant"] += 1
        violation_counts.update(violations)
    return {
        "by_intent": {
            intent: {
                "total_cases": bucket["total"],
                "compliant_cases": bucket["compliant"],
                "compliance_rate": _round_rate(bucket["compliant"], bucket["total"]),
            }
            for intent, bucket in sorted(by_intent.items())
        },
        "violations": dict(sorted(violation_counts.items())),
    }


def summarize_coverage(cases: list[EvaluationCaseResult], *, min_cases: int = 3) -> dict[str, Any]:
    by_concept: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in cases:
        concept = _case_concept(row)
        by_concept[concept]["total"] += 1
        if row.pass_bool:
            by_concept[concept]["correct"] += 1
    return {
        "min_cases": min_cases,
        "count_per_concept": {k: v["total"] for k, v in sorted(by_concept.items())},
        "accuracy_per_concept": {
            k: _round_rate(v["correct"], v["total"]) for k, v in sorted(by_concept.items())
        },
        "under_tested_concepts": sorted(
            k for k, v in by_concept.items() if v["total"] < min_cases
        ),
    }


def summarize_boost(cases: list[EvaluationCaseResult]) -> dict[str, Any] | None:
    rows = [row for row in cases if _boost_metrics(row)]
    if not rows:
        return None
    metrics = [_boost_metrics(row) for row in rows]
    triggered = [m for m in metrics if m.get("boost_triggered")]
    improved = [m for m in metrics if m.get("boost_improved")]
    with_latency = [
        int(m["boost_latency_ms"])
        for m in metrics
        if isinstance(m.get("boost_latency_ms"), int) and m.get("boost_latency_ms") >= 0
    ]
    baseline_lat = [
        int(m["latency_without_boost_ms"])
        for m in metrics
        if isinstance(m.get("latency_without_boost_ms"), int)
    ]
    boosted_lat = [
        int(m["latency_with_boost_ms"])
        for m in metrics
        if isinstance(m.get("latency_with_boost_ms"), int)
    ]
    return {
        "paired_cases": len(rows),
        "boost_triggered_rate": _round_rate(len(triggered), len(rows)),
        "boost_added_value_rate": _round_rate(len(improved), len(rows)),
        "avg_boost_latency_ms": round(sum(with_latency) / len(with_latency), 2)
        if with_latency
        else None,
        "avg_latency_without_boost_ms": round(sum(baseline_lat) / len(baseline_lat), 2)
        if baseline_lat
        else None,
        "avg_latency_with_boost_ms": round(sum(boosted_lat) / len(boosted_lat), 2)
        if boosted_lat
        else None,
    }


def summarize_iteration(
    runs: list[EvaluationRun], cases: list[EvaluationCaseResult]
) -> list[dict[str, Any]]:
    cases_by_run: dict[int, list[EvaluationCaseResult]] = defaultdict(list)
    for row in cases:
        cases_by_run[row.evaluation_run_id].append(row)
    rows: list[dict[str, Any]] = []
    for run in runs:
        cap = summarize_capability(cases_by_run.get(run.id, []))
        rows.append(
            {
                "run_id": run.id,
                "timestamp": run.created_at.isoformat() if run.created_at else "",
                "overall_accuracy": cap["overall_accuracy"],
                "definition_accuracy": cap["definition_accuracy"],
                "step_by_step_accuracy": cap["step_by_step_accuracy"],
                "compare_accuracy": cap["compare_accuracy"],
                "synthesis_accuracy": cap["synthesis_accuracy"],
                "retrieval_grounded_accuracy": cap["retrieval_grounded_accuracy"],
            }
        )
    return rows


def build_analytics_payload(cases: list[EvaluationCaseResult]) -> dict[str, Any]:
    return {
        "summary_metrics": {
            "total_cases": summarize_capability(cases)["total_cases"],
            "correct_cases": summarize_capability(cases)["correct_cases"],
            "overall_accuracy": summarize_capability(cases)["overall_accuracy"],
        },
        "capability_breakdown": summarize_capability(cases),
        "error_breakdown": summarize_errors(cases),
        "retrieval_metrics": summarize_retrieval(cases),
        "structure_metrics": summarize_structure(cases),
        "coverage_metrics": summarize_coverage(cases),
        "boost_metrics": summarize_boost(cases),
    }


def top_three_issues(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    errors = payload.get("error_breakdown", {}).get("by_error_type", {})
    for error_type, data in sorted(
        errors.items(), key=lambda item: (-int(item[1].get("count", 0)), item[0])
    ):
        count = int(data.get("count", 0))
        if count > 0:
            issues.append(f"{error_type}: {count} failed case(s); inspect examples and retrieval logs.")
        if len(issues) == 3:
            return issues
    retrieval = payload.get("retrieval_metrics", {})
    if retrieval.get("top_1_accuracy", 1.0) < 0.8:
        issues.append("retrieval_miss: top-1 retrieval accuracy is below 80%; tune aliases/reranking.")
    structure = payload.get("structure_metrics", {}).get("violations", {})
    for name, count in sorted(structure.items(), key=lambda item: (-int(item[1]), item[0])):
        if count:
            issues.append(f"{name}: {count} structure violation(s); check intent-specific renderers.")
        if len(issues) == 3:
            return issues
    coverage = payload.get("coverage_metrics", {})
    under_tested = coverage.get("under_tested_concepts") or []
    if under_tested:
        issues.append(
            f"coverage: {len(under_tested)} concept(s) have fewer than {coverage.get('min_cases', 3)} cases."
        )
    return (issues or ["No dominant issue surfaced in this run."])[:3]


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def _expected_summary(behavior: dict[str, Any]) -> str:
    pieces: list[str] = []
    if behavior.get("expected_mode"):
        pieces.append(f"mode={behavior['expected_mode']}")
    if behavior.get("intent"):
        pieces.append(f"intent={behavior['intent']}")
    if behavior.get("must_include"):
        pieces.append(f"must_include={behavior['must_include']}")
    if behavior.get("must_not_include"):
        pieces.append(f"must_not_include={behavior['must_not_include']}")
    return "; ".join(pieces) or "(no structured expectation)"


def _short_text(text: str | None, limit: int = 900) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n\n... (truncated)"


def write_capability_report_md(
    path: Path,
    *,
    run: EvaluationRun,
    cases: list[EvaluationCaseResult],
    payload: dict[str, Any],
) -> None:
    cap = payload["capability_breakdown"]["by_intent"]
    errors = payload["error_breakdown"]["by_error_type"]
    retrieval = payload["retrieval_metrics"]
    structure = payload["structure_metrics"]["by_intent"]
    coverage = payload["coverage_metrics"]
    boost = payload.get("boost_metrics")
    lines: list[str] = [
        "# Evaluation analytics report",
        "",
        f"Run id: {run.id}",
        f"Dataset: {run.dataset_name}",
        "",
        "## Summary metrics",
        "",
        *(_markdown_table(
            ["total_cases", "correct_cases", "overall_accuracy"],
            [
                [
                    payload["summary_metrics"]["total_cases"],
                    payload["summary_metrics"]["correct_cases"],
                    payload["summary_metrics"]["overall_accuracy"],
                ]
            ],
        )),
        "",
        "## Accuracy by capability",
        "",
        *(_markdown_table(
            ["intent", "total_cases", "correct_cases", "accuracy"],
            [
                [intent, data["total_cases"], data["correct_cases"], data["accuracy"]]
                for intent, data in sorted(cap.items())
            ],
        )),
        "",
        "## Error breakdown",
        "",
    ]
    if errors:
        lines.extend(
            _markdown_table(
                ["error_type", "count", "percentage"],
                [
                    [name, data["count"], data["percentage"]]
                    for name, data in errors.items()
                ],
            )
        )
    else:
        lines.append("No failed cases in this run.")
    lines.extend(
        [
            "",
            "## Retrieval diagnostics",
            "",
            *(_markdown_table(
                ["top_1_accuracy", "top_k_recall", "retrieval_noise_rate"],
                [
                    [
                        retrieval["top_1_accuracy"],
                        retrieval["top_k_recall"],
                        retrieval["retrieval_noise_rate"],
                    ]
                ],
            )),
            "",
            "## Structure compliance",
            "",
            *(_markdown_table(
                ["intent", "total_cases", "compliant_cases", "compliance_rate"],
                [
                    [
                        intent,
                        data["total_cases"],
                        data["compliant_cases"],
                        data["compliance_rate"],
                    ]
                    for intent, data in sorted(structure.items())
                ],
            )),
            "",
            "## Concept coverage",
            "",
            *(_markdown_table(
                ["concept", "count", "accuracy"],
                [
                    [
                        concept,
                        coverage["count_per_concept"][concept],
                        coverage["accuracy_per_concept"][concept],
                    ]
                    for concept in sorted(coverage["count_per_concept"])
                ],
            )),
            "",
            "Under-tested concepts: "
            + (", ".join(coverage["under_tested_concepts"]) or "(none)"),
            "",
            "## Boost effectiveness",
            "",
        ]
    )
    if boost:
        lines.extend(
            _markdown_table(
                [
                    "paired_cases",
                    "boost_triggered_rate",
                    "boost_added_value_rate",
                    "avg_boost_latency_ms",
                ],
                [
                    [
                        boost["paired_cases"],
                        boost["boost_triggered_rate"],
                        boost["boost_added_value_rate"],
                        boost["avg_boost_latency_ms"],
                    ]
                ],
            )
        )
    else:
        lines.append("Boost metrics unavailable for this run (use `--paired-boost`).")
    passed = [row for row in cases if row.pass_bool][:3]
    failed = [row for row in cases if not row.pass_bool][:3]
    lines.extend(["", "## Correct examples", ""])
    for row in passed:
        behavior = parse_expected_behavior(row.expected_behavior_json)
        lines.extend(
            [
                f"### {row.test_id}",
                "",
                f"Query: {row.query_text}",
                "",
                f"Expected behavior: {_expected_summary(behavior)}",
                "",
                "Model output:",
                "",
                _short_text(row.actual_response),
                "",
            ]
        )
    if not passed:
        lines.append("(none)")
    lines.extend(["", "## Failure examples", ""])
    for row in failed:
        behavior = parse_expected_behavior(row.expected_behavior_json)
        lines.extend(
            [
                f"### {row.test_id}",
                "",
                f"Query: {row.query_text}",
                "",
                f"Expected behavior: {_expected_summary(behavior)}",
                "",
                f"Assigned error_type: {primary_error_type_for_row(row)}",
                "",
                "Model output:",
                "",
                _short_text(row.actual_response),
                "",
            ]
        )
    if not failed:
        lines.append("(none)")
    lines.extend(["", "## Key insights", ""])
    for issue in top_three_issues(payload):
        lines.append(f"- {issue}")
    lines.extend(["", "## Top 3 issues to fix", ""])
    for issue in top_three_issues(payload):
        lines.append(f"- {issue}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
