"""Rich markdown reports for static eval CLI runs (examples + error analysis)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from sqlalchemy import func

from app.extensions import db
from app.models.content import LectureChunk
from app.models.evaluation import EvaluationCaseResult, EvaluationRun
from app.services.eval_run import _combined_retrieval_text, failure_tags_for_case
from app.services.reasoning_pipeline import PipelineResult
from app.services.retrieval_v2 import EnhancedRetrievalResult
from app.services.answers.answer_validation import ValidationResult

_ACTUAL_MAX_LEN = 3000
_TOP_N_EXAMPLES = 5
_SEP = "=================================================="


_RECOMMENDATIONS: dict[str, list[str]] = {
    "compare_entity_collapse": [
        "Strengthen ConceptEvidenceBundle separation and line-level routing in compare evidence.",
        "Reject or repair compare output where both entities share identical core evidence lines.",
    ],
    "compare_asymmetry": [
        "Ensure each compare side has scoped evidence or an explicit thin-evidence note.",
        "Validate comparison axes are present when the suite expects contrast.",
    ],
    "retrieval_leakage": [
        "Tighten post-retrieval concept constraints and reranking so unrelated chunks drop out.",
        "Raise top-k purity or add hard-drop for obvious forbidden-topic hits on non-relational queries.",
    ],
    "forbidden_topic_leakage": [
        "Extend concept-purity / forbidden-term checks on the final answer path.",
        "Validate `must_not_include` terms against rendered Course Answer before return.",
    ],
    "mode_misclassification": [
        "Review deterministic signals in `query_mode.detect_query_mode` for the failing patterns.",
        "Add or adjust suite cases that pin expected `detected` vs `effective` behavior.",
    ],
    "mode_routing_failure": [
        "Check `mode_override` / legacy `mode` resolution vs `resolve_effective_mode`.",
        "Inspect `retrieval_v2.apply_effective_api_mode` coercion for the query shape.",
    ],
    "missing_required_concept": [
        "Improve retrieval focus or aliases so required phrases appear in the answer.",
        "Verify rule-based renderers emit required headings or entities for the mode.",
    ],
    "wrong_direct_answer": [
        "Tune direct-answer selection and validators (`must_direct_answer_*`) for the topic.",
    ],
    "quiz_not_rendered": [
        "Ensure quiz path uses `quiz_render.format_quiz_markdown` and never four-block Course Answer.",
    ],
    "summary_wrong_scope": [
        "Harden summary renderer lecture/topic filters and summary contract validators.",
    ],
    "summary_generic": [
        "Reduce boilerplate in summary layout; anchor sections to retrieved chunk content.",
    ],
    "scaffold_leakage": [
        "Strip forbidden section headings from summary/quiz/compare outputs.",
    ],
    "clarification_missing": [
        "Preserve clarification prompts for underspecified compare/quiz/summary queries.",
    ],
    "validation_missed_error": [
        "Align `eval.scoring` pass/fail with pipeline validation severity and checks_failed.",
    ],
    "forbidden_leak": [
        "Same as forbidden answer leakage: concept purity and answer-term gating.",
    ],
    "must_include_failed": [
        "Improve retrieval or renderer so required substrings are present.",
    ],
    "mode_mismatch": [
        "See mode_routing_failure / mode_misclassification.",
    ],
}


def _validation_from_diag(val: dict[str, Any] | None) -> ValidationResult:
    if not val:
        return ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={},
            severity="pass",
        )
    return ValidationResult(
        passed=bool(val.get("passed", False)),
        checks_run=list(val.get("checks_run") or []),
        checks_passed=list(val.get("checks_passed") or []),
        checks_failed=list(val.get("checks_failed") or []),
        flags=dict(val.get("flags") or {}),
        severity=str(val.get("severity") or "pass"),
        repair_path=val.get("repair_path"),
    )


def _lecture_rows_to_chunk_dicts(chunk_ids: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cid in chunk_ids:
        row = db.session.get(LectureChunk, cid)
        if row is None:
            continue
        kw_raw = row.keywords
        kw_list: list[str] | str
        try:
            parsed = json.loads(kw_raw) if isinstance(kw_raw, str) else kw_raw
            kw_list = parsed if isinstance(parsed, list) else kw_raw
        except json.JSONDecodeError:
            kw_list = kw_raw
        out.append(
            {
                "id": row.id,
                "clean_explanation": row.clean_explanation or "",
                "source_excerpt": row.source_excerpt or "",
                "topic": row.topic or "",
                "keywords": kw_list,
            }
        )
    return out


def _mode_routing_from_payload(payload: dict[str, Any], pl_diag: dict[str, Any] | None) -> dict[str, Any]:
    mr = payload.get("mode_routing")
    if isinstance(mr, dict) and mr:
        return mr
    if isinstance(pl_diag, dict):
        inner = pl_diag.get("mode_routing")
        if isinstance(inner, dict) and inner:
            return inner
    mm = payload.get("mode") or {}
    if isinstance(mm, dict):
        det = mm.get("detected")
        eff = mm.get("effective")
        if det is not None or eff is not None:
            return {
                "detected_mode": det,
                "effective_mode": eff,
            }
    return {}


def build_pipeline_result_for_tags(
    *,
    course_answer: str,
    payload: dict[str, Any],
    pl_diag: dict[str, Any] | None,
    chunk_ids: list[int],
) -> PipelineResult:
    chunks = _lecture_rows_to_chunk_dicts(chunk_ids)
    mr = _mode_routing_from_payload(payload, pl_diag)
    val = _validation_from_diag(
        pl_diag.get("validation") if isinstance(pl_diag, dict) else None
    )
    enhanced = EnhancedRetrievalResult(
        chunks=chunks,
        confidence=0.5,
        detected_topic=None,
        diagnostics=None,
        mode_routing=mr or None,
    )
    return PipelineResult(
        enhanced_result=enhanced,
        structured_query=MagicMock(),
        answer_plan=MagicMock(),
        course_answer=course_answer or "",
        validation=val,
        used_llm_for_answer=False,
        primary_model="rule_based",
        query_complexity="simple",
        primary_llm_usage={},
    )


def canonical_tags_and_retrieval_blob(
    case_raw: dict[str, Any],
    *,
    cli_pass: bool,
    course_answer: str,
    payload: dict[str, Any],
    pl_diag: dict[str, Any] | None,
    chunk_ids: list[int],
) -> tuple[list[str], str]:
    pr = build_pipeline_result_for_tags(
        course_answer=course_answer,
        payload=payload,
        pl_diag=pl_diag,
        chunk_ids=chunk_ids,
    )
    blob = _combined_retrieval_text(pr).lower()
    if cli_pass:
        return [], blob
    return failure_tags_for_case(case_raw, pr, pass_bool=False), blob


def _truncate_actual(text: str) -> str:
    t = text or ""
    if len(t) <= _ACTUAL_MAX_LEN:
        return t
    return t[: _ACTUAL_MAX_LEN] + "\n\n… (truncated)"


def _expected_bullets(expected: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    em = expected.get("expected_mode")
    if em:
        lines.append(f"mode: {em}")
    for label, key in (
        ("must_include", "must_include"),
        ("must_not_include", "must_not_include"),
        ("expected_sections", "expected_sections"),
        ("forbidden_sections", "forbidden_sections"),
        ("suite error_tags", "error_tags"),
    ):
        v = expected.get(key)
        if v:
            lines.append(f"{label}: {v}")
    cat = expected.get("category")
    if cat:
        lines.append(f"category: {cat}")
    return lines


def _median_id(rows: list[dict[str, Any]], *, reverse: bool = False) -> str | None:
    if not rows:
        return None
    keyf = lambda r: (float(r.get("score", 0)), str(r.get("id", "")))
    sorted_rows = sorted(rows, key=keyf, reverse=reverse)
    return str(sorted_rows[len(sorted_rows) // 2]["id"])


def _pick_best_passing(row_results: list[dict[str, Any]], n: int) -> list[str]:
    passed = [r for r in row_results if r.get("pass") == "true"]
    passed.sort(key=lambda r: (-float(r.get("score", 0)), str(r.get("id", ""))))
    return [str(r["id"]) for r in passed[:n]]


def _pick_worst_failing(row_results: list[dict[str, Any]], n: int) -> list[str]:
    failed = [r for r in row_results if r.get("pass") == "false"]
    failed.sort(key=lambda r: (float(r.get("score", 0)), str(r.get("id", ""))))
    return [str(r["id"]) for r in failed[:n]]


def _representative_per_mode(row_results: list[dict[str, Any]]) -> list[str]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for r in row_results:
        m = str(r.get("expected_mode") or "chat")
        by_mode.setdefault(m, []).append(r)
    ids: list[str] = []
    for mode in sorted(by_mode.keys()):
        bucket = by_mode[mode]
        passed = [r for r in bucket if r.get("pass") == "true"]
        failed = [r for r in bucket if r.get("pass") == "false"]
        mp = _median_id(passed, reverse=True)
        mf = _median_id(failed, reverse=False)
        if mp:
            ids.append(mp)
        if mf and mf != mp:
            ids.append(mf)
    return ids


def _format_example_block(
    test_id: str,
    detail: dict[str, Any],
    row: dict[str, Any] | None,
    *,
    title_prefix: str = "",
) -> list[str]:
    q = detail.get("query") or (row or {}).get("query") or ""
    expected = detail.get("expected_behavior") or {}
    actual = detail.get("actual_response")
    if actual is None and row:
        actual = ""
    actual_s = _truncate_actual(str(actual or ""))
    tags = detail.get("canonical_tags") or []
    scoring = detail.get("scoring_errors") or []

    head = f"## {title_prefix}{test_id}".strip()
    lines = [head, "", "Query:", q, "", "Expected:"]
    eb = _expected_bullets(expected if isinstance(expected, dict) else {})
    if eb:
        for b in eb:
            lines.append(f"- {b}")
    else:
        lines.append("- (no structured expectations in suite)")
    det = detail.get("detected") or (row or {}).get("detected") or ""
    eff = detail.get("effective") or (row or {}).get("effective") or ""
    score = detail.get("score")
    if score is None and row is not None:
        score = row.get("score")
    pass_s = detail.get("pass")
    if pass_s is None and row is not None:
        pass_s = row.get("pass") == "true"
    lines.extend(
        [
            "",
            "Actual:",
            actual_s,
            "",
            f"Detected / Effective: {det} / {eff}",
            f"Score: {score} (pass: {pass_s})",
            "",
            "Failure tags:",
        ]
    )
    if tags:
        for t in tags:
            lines.append(f"- {t}")
    else:
        lines.append("- —")
    if scoring:
        lines.extend(["", "Scoring categories:", *[f"- {x}" for x in scoring]])
    lines.append("")
    lines.append(_SEP)
    lines.append("")
    return lines


def write_examples_md(
    path: Path,
    *,
    row_results: list[dict[str, Any]],
    case_details: dict[str, dict[str, Any]],
    prev_run: EvaluationRun | None,
    prev_by_id: dict[str, dict[str, Any]] | None,
    current_run_id: int,
) -> None:
    rows_by_id = {str(r["id"]): r for r in row_results}
    best = _pick_best_passing(row_results, _TOP_N_EXAMPLES)
    worst = _pick_worst_failing(row_results, _TOP_N_EXAMPLES)
    rep = _representative_per_mode(row_results)

    lines: list[str] = [
        "# Eval examples",
        "",
        f"Run id: {current_run_id}",
        "",
        "## Best passing examples",
        "",
    ]
    for tid in best:
        d = case_details.get(tid, {})
        lines.extend(_format_example_block(tid, d, rows_by_id.get(tid), title_prefix="Passing: "))

    lines.extend(["## Worst failing examples", ""])
    for tid in worst:
        d = case_details.get(tid, {})
        lines.extend(_format_example_block(tid, d, rows_by_id.get(tid), title_prefix="Failed: "))

    lines.extend(["## Representative examples by expected mode", ""])
    for tid in rep:
        d = case_details.get(tid, {})
        lines.extend(_format_example_block(tid, d, rows_by_id.get(tid)))

    lines.extend(["## Before / after (previous run)", ""])
    if prev_run is None or not prev_by_id:
        lines.append("No previous run found for this `dataset_name` in the database.")
        lines.append("")
    else:
        lines.append(
            f"Previous run id: {prev_run.id} ({prev_run.run_name} @ {prev_run.created_at}), "
            f"passed {prev_run.passed_cases}/{prev_run.total_cases}, "
            f"mean score {prev_run.overall_score}."
        )
        lines.append("")
        regressions: list[str] = []
        fixes: list[str] = []
        for r in row_results:
            tid = str(r["id"])
            cur_pass = r.get("pass") == "true"
            cur_sc = float(r.get("score", 0))
            prev = prev_by_id.get(tid)
            if not prev:
                continue
            p_pass = bool(prev.get("pass"))
            p_sc = float(prev.get("score") or 0)
            if p_pass and not cur_pass:
                regressions.append(
                    f"- **{tid}**: pass → fail (score {p_sc:.4f} → {cur_sc:.4f})"
                )
            elif not p_pass and cur_pass:
                fixes.append(f"- **{tid}**: fail → pass (score {p_sc:.4f} → {cur_sc:.4f})")
        lines.append("### Regressions (was pass, now fail)")
        lines.extend(regressions or ["- (none)"])
        lines.append("")
        lines.append("### Fixes (was fail, now pass)")
        lines.extend(fixes or ["- (none)"])
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def aggregate_scoring_categories(
    row_results: list[dict[str, Any]], *, only_failed: bool = True
) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for r in row_results:
        if only_failed and r.get("pass") == "true":
            continue
        err_s = (r.get("errors") or "").strip()
        if not err_s:
            continue
        for part in err_s.split(";"):
            p = part.strip()
            if p:
                c[p] += 1
    return c.most_common(5)


def aggregate_canonical_tags(
    case_details: dict[str, dict[str, Any]], *, only_failed: bool = True
) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for tid, d in case_details.items():
        if only_failed and d.get("pass"):
            continue
        for t in d.get("canonical_tags") or []:
            c[t] += 1
    return c.most_common(5)


def aggregate_forbidden_leakage(
    case_details: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Answer hits vs retrieval-only hits for `must_not_include` strings."""
    ans_c: Counter[str] = Counter()
    ret_c: Counter[str] = Counter()
    for d in case_details.values():
        if d.get("pass"):
            continue
        expected = d.get("expected_behavior") or {}
        must_not = [str(x) for x in (expected.get("must_not_include") or []) if x]
        if not must_not:
            continue
        al = (d.get("actual_response") or "").lower()
        rl = (d.get("retrieval_blob_lower") or "").lower()
        for term in must_not:
            tl = term.lower()
            in_a = tl in al
            in_r = tl in rl
            if in_a:
                ans_c[term] += 1
            elif in_r:
                ret_c[term] += 1
    return ans_c.most_common(15), ret_c.most_common(15)


def mode_score_stats(row_results: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """Return (expected_mode, mean_score, min_score) sorted by mean ascending."""
    by: dict[str, list[float]] = {}
    for r in row_results:
        m = str(r.get("expected_mode") or "chat")
        by.setdefault(m, []).append(float(r.get("score", 0)))
    stats: list[tuple[str, float, float]] = []
    for mode, scores in sorted(by.items()):
        if not scores:
            continue
        stats.append((mode, sum(scores) / len(scores), min(scores)))
    stats.sort(key=lambda x: x[1])
    return stats


def repeated_failure_test_ids(dataset_name: str) -> tuple[int, list[tuple[str, int]]]:
    n_runs = (
        db.session.query(func.count(EvaluationRun.id))
        .filter(EvaluationRun.dataset_name == dataset_name)
        .scalar()
    )
    cnt = func.count(EvaluationCaseResult.id)
    q = (
        db.session.query(EvaluationCaseResult.test_id, cnt)
        .join(EvaluationRun, EvaluationCaseResult.evaluation_run_id == EvaluationRun.id)
        .filter(EvaluationRun.dataset_name == dataset_name)
        .filter(EvaluationCaseResult.pass_bool.is_(False))
        .group_by(EvaluationCaseResult.test_id)
        .having(cnt >= 2)
        .order_by(cnt.desc())
    )
    rows = q.all()
    return int(n_runs or 0), [(str(tid), int(c)) for tid, c in rows]


def fetch_previous_run_map(
    current_run_id: int, dataset_name: str
) -> tuple[EvaluationRun | None, dict[str, dict[str, Any]] | None]:
    prev = (
        EvaluationRun.query.filter(
            EvaluationRun.dataset_name == dataset_name,
            EvaluationRun.id != current_run_id,
        )
        .order_by(EvaluationRun.created_at.desc())
        .first()
    )
    if prev is None:
        return None, None
    ecrs = EvaluationCaseResult.query.filter_by(evaluation_run_id=prev.id).all()
    m = {
        r.test_id: {"pass": r.pass_bool, "score": float(r.score or 0)} for r in ecrs
    }
    return prev, m


def _recommendations(
    top_canonical: list[tuple[str, int]], top_scoring: list[tuple[str, int]]
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name, _ in top_canonical + top_scoring:
        recs = _RECOMMENDATIONS.get(name)
        if not recs:
            continue
        for line in recs:
            key = f"{name}:{line}"
            if key in seen:
                continue
            seen.add(key)
            out.append(f"- **{name}**: {line}")
        if len(out) >= 12:
            break
    return out


def write_error_analysis_md(
    path: Path,
    *,
    row_results: list[dict[str, Any]],
    case_details: dict[str, dict[str, Any]],
    dataset_name: str,
) -> None:
    top_scoring = aggregate_scoring_categories(row_results, only_failed=True)
    top_canonical = aggregate_canonical_tags(case_details, only_failed=True)
    ans_leaks, ret_leaks = aggregate_forbidden_leakage(case_details)
    mode_stats = mode_score_stats(row_results)
    n_runs, repeated = repeated_failure_test_ids(dataset_name)

    lines = [
        "# Error analysis",
        "",
        "## Top scoring error categories (failed cases)",
        "",
    ]
    if top_scoring:
        for name, count in top_scoring:
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- (none)")
    lines.extend(["", "## Top canonical failure tags (failed cases)", ""])
    if top_canonical:
        for name, count in top_canonical:
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Most common forbidden leakage terms", ""])
    lines.append("### In model answer (`must_not_include` hit in `actual_response`)")
    if ans_leaks:
        for term, count in ans_leaks[:10]:
            lines.append(f"- {term!r}: {count}")
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("### In retrieval only (hit in chunk text blob, not in answer)")
    if ret_leaks:
        for term, count in ret_leaks[:10]:
            lines.append(f"- {term!r}: {count}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Modes with lowest mean score", ""])
    for mode, mean_s, min_s in mode_stats:
        lines.append(f"- **{mode}**: mean {mean_s:.4f}, min {min_s:.4f}")

    lines.extend(["", "## Queries failed repeatedly across runs", ""])
    if n_runs < 2:
        lines.append(
            f"Insufficient history: only {n_runs} run(s) in DB for this dataset (need ≥2 for repeats)."
        )
    elif not repeated:
        lines.append("No test id failed in 2+ runs (for this dataset).")
    else:
        for tid, c in repeated[:15]:
            lines.append(f"- {tid}: {c} failing run(s)")

    lines.extend(["", "## Recommended next engineering fixes", ""])
    recs = _recommendations(top_canonical, top_scoring)
    if recs:
        lines.extend(recs)
    else:
        lines.append("- (no mapped recommendations for current top tags)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
