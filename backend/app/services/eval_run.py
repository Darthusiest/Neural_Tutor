"""Load static eval suites, run :func:`run_reasoning_pipeline`, score, persist rows."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun
from app.services.reasoning_pipeline import PipelineResult, run_reasoning_pipeline

ERROR_CATEGORY_TAGS: frozenset[str] = frozenset(
    {
        "mode_misclassification",
        "mode_routing_failure",
        "retrieval_leakage",
        "forbidden_topic_leakage",
        "missing_required_concept",
        "wrong_direct_answer",
        "compare_entity_collapse",
        "compare_asymmetry",
        "summary_generic",
        "summary_wrong_scope",
        "quiz_not_rendered",
        "clarification_missing",
        "duplicated_content",
        "scaffold_leakage",
        "generic_filler",
        "validation_missed_error",
    }
)

_VALIDATION_CHECK_TO_TAG: dict[str, str] = {
    "must_match_quiz_contract": "quiz_not_rendered",
    "must_match_summary_contract": "summary_wrong_scope",
    "must_stay_in_scope": "summary_wrong_scope",
    "must_respect_lecture_scope": "summary_wrong_scope",
    "must_not_be_boilerplate_summary": "summary_generic",
    "must_cover_main_anchors": "summary_generic",
    "must_match_compare_contract": "compare_entity_collapse",
    "must_have_distinct_compare_evidence": "compare_entity_collapse",
    "must_cover_both_sides": "compare_asymmetry",
    "must_have_each_side_evidence_or_note": "compare_asymmetry",
    "must_include_comparison_axis": "compare_asymmetry",
    "must_direct_answer_match_target": "wrong_direct_answer",
    "must_direct_answer_mention_target_concept": "wrong_direct_answer",
    "must_not_have_section_duplication": "duplicated_content",
    "must_not_leak_forbidden_terms": "forbidden_topic_leakage",
    "must_be_concept_pure": "forbidden_topic_leakage",
}


def load_eval_suite(path: Path) -> dict[str, Any]:
    """Load JSON suite; must have ``name``, ``version``, ``cases`` list."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError(f"eval suite {path} missing 'cases' array")
    return data


def _norm_mode(m: str | None) -> str:
    return (m or "").strip().lower()


def _chunk_text_blob(chunk: dict[str, Any]) -> str:
    parts: list[str] = [
        str(chunk.get("clean_explanation") or ""),
        str(chunk.get("source_excerpt") or ""),
        str(chunk.get("topic") or ""),
    ]
    kw = chunk.get("keywords")
    if isinstance(kw, list):
        parts.append(" ".join(str(x) for x in kw))
    elif kw is not None:
        parts.append(str(kw))
    return " ".join(parts)


def _combined_retrieval_text(pr: PipelineResult) -> str:
    chunks: list[dict[str, Any]] = []
    for c in pr.enhanced_result.chunks or []:
        if isinstance(c, dict):
            chunks.append(c)
    for c in getattr(pr.enhanced_result, "supporting_chunks", None) or []:
        if isinstance(c, dict):
            chunks.append(c)
    return " ".join(_chunk_text_blob(c) for c in chunks)


def failure_tags_for_case(case: dict[str, Any], pr: PipelineResult, *, pass_bool: bool) -> list[str]:
    """
    Canonical failure tags for a suite case. Empty when ``pass_bool`` is True.

    Callers should pass the same ``pass_bool`` returned by :func:`score_eval_case`.
    """
    if pass_bool:
        return []

    tags: set[str] = set()
    answer = pr.course_answer or ""
    answer_lower = answer.lower()
    chunks_lower = _combined_retrieval_text(pr).lower()
    mr = pr.enhanced_result.mode_routing or {}
    val = pr.validation

    exp_mode = case.get("expected_mode")
    if exp_mode:
        exp_n = _norm_mode(str(exp_mode))
        if _norm_mode(mr.get("detected_mode")) != exp_n:
            tags.add("mode_misclassification")
        if _norm_mode(mr.get("effective_mode")) != exp_n:
            tags.add("mode_routing_failure")

    must_in = [s for s in (case.get("must_include") or []) if s]
    if any(s.lower() not in answer_lower for s in must_in):
        tags.add("missing_required_concept")

    for sub in case.get("expected_sections") or []:
        if sub and sub not in answer:
            tags.add("missing_required_concept")

    must_not = [s for s in (case.get("must_not_include") or []) if s]
    for s in must_not:
        sl = s.lower()
        if sl in answer_lower:
            tags.add("forbidden_topic_leakage")
        if sl in chunks_lower:
            tags.add("retrieval_leakage")

    for sub in case.get("forbidden_sections") or []:
        if sub and sub in answer:
            tags.add("scaffold_leakage")

    exp_mode_n = _norm_mode(str(exp_mode)) if exp_mode else ""
    if exp_mode_n == "quiz":
        if "must_match_quiz_contract" in val.checks_failed:
            tags.add("quiz_not_rendered")
        elif any("quiz:" in str(m).lower() for m in must_in) and "quiz:" not in answer_lower:
            tags.add("quiz_not_rendered")
        elif "course answer:" in answer_lower or "### direct answer" in answer_lower:
            tags.add("quiz_not_rendered")

    for name in val.checks_failed:
        mapped = _VALIDATION_CHECK_TO_TAG.get(name)
        if mapped:
            tags.add(mapped)
        else:
            tags.add("validation_missed_error")

    if val.flags.get("generic_filler"):
        tags.add("generic_filler")

    if case.get("category") == "clarification":
        tags.add("clarification_missing")

    if val.severity == "pass" and not val.checks_failed:
        tags.add("validation_missed_error")

    tags = {t for t in tags if t in ERROR_CATEGORY_TAGS}

    if not tags:
        tags.add("validation_missed_error")

    return sorted(tags)


def score_eval_case(case: dict[str, Any], pr: PipelineResult) -> dict[str, Any]:
    """
    Heuristic scores (no LLM). Returns keys: pass_bool, score, error_categories,
    detected_mode, effective_mode.

    ``error_categories`` lists canonical tags for failed cases only; passes get ``[]``.
    """
    answer = pr.course_answer or ""
    lower = answer.lower()
    mr = pr.enhanced_result.mode_routing or {}
    detected = mr.get("detected_mode")
    effective = mr.get("effective_mode")
    val = pr.validation

    errors: list[str] = []
    score_parts: list[float] = []

    exp_mode = case.get("expected_mode")
    if exp_mode:
        if _norm_mode(effective) != _norm_mode(str(exp_mode)):
            errors.append("mode_mismatch")
        score_parts.append(1.0 if _norm_mode(effective) == _norm_mode(str(exp_mode)) else 0.0)

    must_in = [s for s in (case.get("must_include") or []) if s]
    if must_in:
        ratio = sum(1 for s in must_in if s.lower() in lower) / len(must_in)
        score_parts.append(ratio)
        if ratio < 1.0:
            errors.append("missing_phrase")

    must_not = [s for s in (case.get("must_not_include") or []) if s]
    leaked = [s for s in must_not if s.lower() in lower]
    if leaked:
        errors.append("forbidden_leak")
    score_parts.append(0.0 if leaked else 1.0)

    for sub in case.get("expected_sections") or []:
        if sub and sub not in answer:
            errors.append("missing_section")
    if case.get("expected_sections"):
        exp = [s for s in case["expected_sections"] if s]
        if exp:
            ok = all(s in answer for s in exp)
            score_parts.append(1.0 if ok else 0.0)

    for sub in case.get("forbidden_sections") or []:
        if sub and sub in answer:
            errors.append("forbidden_section")
    if case.get("forbidden_sections"):
        fs = [s for s in case["forbidden_sections"] if s]
        if fs:
            bad = any(s in answer for s in fs)
            score_parts.append(0.0 if bad else 1.0)

    if val.severity == "fail":
        errors.append("validation_fail")
    if val.severity == "fail":
        score_parts.append(0.0)
    elif val.severity == "weak":
        score_parts.append(0.7)
    else:
        score_parts.append(1.0)

    pass_bool = len(errors) == 0
    score = sum(score_parts) / max(1, len(score_parts))

    error_categories = failure_tags_for_case(case, pr, pass_bool=pass_bool)

    return {
        "pass_bool": pass_bool,
        "score": round(score, 4),
        "error_categories": error_categories,
        "detected_mode": detected,
        "effective_mode": effective,
    }


def _chunk_ids_from_result(pr: PipelineResult) -> list[Any]:
    ids = []
    for c in pr.enhanced_result.chunks or []:
        if isinstance(c, dict) and c.get("id") is not None:
            ids.append(c.get("id"))
    return ids


def _git_meta() -> tuple[str | None, str | None]:
    """Use repo root (parent of `backend/`) for git so it works from any CWD."""
    commit = None
    branch = None
    here = Path(__file__).resolve()
    # eval_run.py -> services -> app -> backend; repo root is parent of backend
    repo_root = here.parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return commit, branch


def run_eval_suite(
    dataset_path: Path,
    run_name: str,
    *,
    user_mode: str = "auto",
    top_k: int = 8,
    backend: str = "keyword",
    compare_last: bool = False,
) -> EvaluationRun:
    """
    Execute every case in the suite, persist :class:`EvaluationRun` and
    :class:`EvaluationCaseResult` rows, return the run row.
    """
    suite = load_eval_suite(dataset_path)
    dataset_name = f"{suite.get('name', 'eval')}:{suite.get('version', '?')}"
    cases = suite["cases"]

    commit, branch = _git_meta()
    notes: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "user_mode": user_mode,
        "top_k": top_k,
        "backend": backend,
    }

    er = EvaluationRun(
        run_name=run_name,
        git_commit=commit,
        branch_name=branch,
        dataset_name=dataset_name,
        total_cases=len(cases),
        passed_cases=0,
        failed_cases=0,
        overall_score=None,
        notes_json=json.dumps(notes),
    )
    db.session.add(er)
    db.session.flush()

    scores: list[float] = []
    passed = 0

    for case in cases:
        test_id = case["id"]
        query = case["query"]
        t0 = time.perf_counter()
        pr = run_reasoning_pipeline(
            query,
            top_k=top_k,
            backend=backend,
            user_mode=user_mode,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        scored = score_eval_case(case, pr)
        if scored["pass_bool"]:
            passed += 1
        scores.append(float(scored["score"]))

        expected_behavior = {
            k: v
            for k, v in case.items()
            if k not in ("id", "query", "note")
        }

        ecr = EvaluationCaseResult(
            evaluation_run_id=er.id,
            test_id=test_id,
            query_text=query,
            expected_mode=case.get("expected_mode"),
            detected_mode=scored.get("detected_mode"),
            effective_mode=scored.get("effective_mode"),
            expected_behavior_json=json.dumps(expected_behavior),
            actual_response=pr.course_answer,
            pass_bool=bool(scored["pass_bool"]),
            score=scored["score"],
            error_categories_json=json.dumps(scored["error_categories"]),
            validation_failures_json=json.dumps(pr.validation.to_dict()),
            retrieval_chunk_ids_json=json.dumps(_chunk_ids_from_result(pr)),
            latency_ms=latency_ms,
        )
        db.session.add(ecr)

    er.passed_cases = passed
    er.failed_cases = len(cases) - passed
    er.overall_score = round(sum(scores) / max(1, len(scores)), 4)
    db.session.commit()

    if compare_last:
        _print_compare_last(dataset_name, er)

    return er


def _print_compare_last(dataset_name: str, current: EvaluationRun) -> None:
    prev = (
        EvaluationRun.query.filter(
            EvaluationRun.dataset_name == dataset_name,
            EvaluationRun.id != current.id,
        )
        .order_by(EvaluationRun.created_at.desc())
        .first()
    )
    if prev is None:
        print("No previous run for this dataset_name to compare.")
        return
    print(
        f"Compare to previous run id={prev.id} ({prev.run_name} @ {prev.created_at}):\n"
        f"  passed: {prev.passed_cases} -> {current.passed_cases} (delta {current.passed_cases - prev.passed_cases})\n"
        f"  failed: {prev.failed_cases} -> {current.failed_cases} (delta {current.failed_cases - prev.failed_cases})\n"
        f"  overall_score: {prev.overall_score} -> {current.overall_score}"
    )
