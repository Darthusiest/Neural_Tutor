"""Deterministic 0—1 scoring for eval cases (four 0.25 components)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.eval.dataset import EvalCase

QUARTER = 0.25


@dataclass
class ScoringResult:
    score: float
    pass_ok: bool
    mode_ok: bool
    content_ok: bool
    forbidden_ok: bool
    structure_ok: bool
    error_categories: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)


def _norm(s: str) -> str:
    return (s or "").lower()


def _all_substrings_present(hay: str, needles: list[str]) -> bool:
    if not needles:
        return True
    h = _norm(hay)
    for n in needles:
        if not n:
            continue
        if n.lower() not in h:
            return False
    return True


def _forbidden_hits(text: str, case: EvalCase) -> list[str]:
    bad: list[str] = []
    for x in list(case.must_not_include) + list(case.forbidden_sections):
        if not x:
            continue
        if x.lower() in _norm(text):
            bad.append(x)
    return bad


def _mode_effective_ok(case: EvalCase, mode_meta: dict[str, Any] | None) -> bool:
    if not (case.expected_mode or "").strip():
        return True
    exp = case.expected_mode.strip().lower()
    mm = mode_meta or {}
    eff = str(mm.get("effective") or "chat").strip().lower()
    return eff == exp


def _is_clarification_path(
    case: EvalCase, effective: str, text: str, pipeline_diag: dict | None
) -> bool:
    if case.category == "clarification":
        return True
    if pipeline_diag is not None:
        return False
    e = (effective or "").lower()
    if e not in ("compare", "quiz", "summary"):
        return False
    if "?" not in text:
        return False
    if "tell me which" in _norm(text) or "tell me what" in _norm(text):
        return True
    return len(text.strip()) < 500


def _structure_compare(
    text: str, case: EvalCase, pipeline_diag: dict[str, Any] | None, answer_plan: dict | None
) -> tuple[bool, list[str]]:
    errs: list[str] = []
    t = _norm(text)
    if "course answer:" not in t:
        errs.append("structure_compare_no_course_answer_header")
    if re.search(
        r"\b(while|whereas|difference|vs\.?|compared|contrast|unlike|but)\b", text, re.I
    ) is None:
        errs.append("structure_compare_no_contrast_cue")
    if pipeline_diag and isinstance((pipeline_diag.get("validation") or {}), dict):
        if (pipeline_diag.get("validation") or {}).get("flags", {}).get("missing_comparison_side"):
            errs.append("structure_compare_missing_side_flag")
    return (len(errs) == 0, errs)


def _structure_summary(text: str, case: EvalCase) -> tuple[bool, list[str]]:
    errs: list[str] = []
    if "summary:" not in _norm(text):
        errs.append("structure_summary_no_header")
    if re.search(r"###\s*key\s+(topics|points)", text, re.I) is None:
        errs.append("structure_summary_no_key_section")
    for fs in case.forbidden_sections:
        if fs and fs.lower() in _norm(text):
            errs.append("structure_summary_forbidden_substring")
    if "### direct answer" in _norm(text):
        errs.append("structure_summary_has_direct_answer")
    if "course answer:" in _norm(text) and "summary:" not in _norm(text):
        errs.append("structure_summary_wrong_block")
    return (len(errs) == 0, errs)


def _structure_quiz(text: str, case: EvalCase) -> tuple[bool, list[str]]:
    errs: list[str] = []
    tl = _norm(text)
    if "quiz:" not in tl:
        errs.append("structure_quiz_no_header")
    if "answer key:" not in tl:
        errs.append("structure_quiz_no_answer_key")
    if re.search(r"^\s*Q[1-3]\b", text, re.M) is None and re.search(
        r"^\s*\d+[\).]\s", text, re.M
    ) is None and re.search(r"^\s*\d+\.\s", text, re.M) is None:
        errs.append("structure_quiz_no_numbered_questions")
    for bad in ("### direct answer", "### explanation"):
        if bad in _norm(text):
            errs.append("structure_quiz_forbidden_block")
    if "course answer:" in tl:
        errs.append("structure_quiz_has_course_answer")
    return (len(errs) == 0, errs)


def _structure_clarification(text: str, case: EvalCase) -> tuple[bool, list[str]]:
    errs: list[str] = []
    tn = _norm(text)
    asks_followup = "?" in text or "tell me which" in tn or "for example" in tn
    if not asks_followup:
        errs.append("structure_clarification_no_followup")
    if "### direct answer" in _norm(text):
        errs.append("structure_clarification_has_direct_answer")
    return (len(errs) == 0, errs)


def _structure_chat(
    text: str, case: EvalCase, pipeline_diag: dict[str, Any] | None
) -> tuple[bool, list[str]]:
    errs: list[str] = []
    for sec in case.expected_sections:
        if sec and sec.lower() not in _norm(text):
            errs.append("structure_chat_missing_expected_section")
    for fs in case.forbidden_sections:
        if fs and fs.lower() in _norm(text):
            errs.append("structure_chat_forbidden")
    if pipeline_diag and isinstance(pipeline_diag, dict):
        am = pipeline_diag.get("answer_mode")
        if am in ("lecture_summary", "teaching_plus_check", "compare", "compare_multi"):
            errs.append("structure_chat_wrong_answer_mode")
    return (len(errs) == 0, errs)


def _structure_path(
    case: EvalCase, effective: str, text: str, pipeline_diag: dict[str, Any] | None
) -> str:
    if _is_clarification_path(case, effective, text, pipeline_diag):
        return "clarification"
    e = (effective or "chat").lower()
    if e == "compare":
        return "compare"
    if e == "summary":
        return "summary"
    if e == "quiz":
        return "quiz"
    return "chat"


def _route_expectation_allows(
    case: EvalCase, struct_path: str, effective: str, text: str, pipeline_diag: dict | None
) -> tuple[bool, list[str]]:
    """Ensure effective mode and structure path are consistent with expected_mode (when set)."""
    exp = (case.expected_mode or "").strip().lower()
    if not exp:
        return True, []
    err: list[str] = []
    if exp == "compare" and struct_path not in ("clarification", "compare"):
        err.append("structure_routing_mismatch")
    if exp == "summary" and struct_path not in ("clarification", "summary"):
        err.append("structure_routing_mismatch")
    if exp == "quiz" and struct_path not in ("clarification", "quiz"):
        err.append("structure_routing_mismatch")
    if exp == "chat" and effective not in ("chat",) and not _is_clarification_path(
        case, effective, text, pipeline_diag
    ):
        if exp != (effective or "").lower():
            err.append("structure_routing_mismatch")
    return (len(err) == 0, err)


def score_eval_case(
    case: EvalCase,
    response_text: str,
    mode_meta: dict[str, Any] | None,
    pipeline_diag: dict[str, Any] | None,
) -> ScoringResult:
    """Each of four quarters contributes 0.25 if satisfied. Pass if score is 1.0."""
    err: list[str] = []
    mm = mode_meta or {}
    effective = str(mm.get("effective") or "chat").strip().lower()

    if case.category == "mode_detection":
        det = str(mm.get("detected") or "").strip().lower()
        expm = (case.expected_mode or "").strip().lower()
        mode_ok = det == expm
        if not mode_ok:
            err.append("mode_detected_mismatch")
    else:
        mode_ok = _mode_effective_ok(case, mm)
        if not mode_ok:
            err.append("mode_mismatch")

    content_ok = _all_substrings_present(response_text, case.must_include)
    if not content_ok:
        err.append("must_include_failed")

    forbidden_hits = _forbidden_hits(response_text, case)
    forbidden_ok = len(forbidden_hits) == 0
    if not forbidden_ok:
        err.append("forbidden_leak")

    answer_plan: dict | None = None
    if pipeline_diag and isinstance(pipeline_diag.get("answer_plan"), dict):
        answer_plan = pipeline_diag["answer_plan"]
    if pipeline_diag and answer_plan is None and isinstance(pipeline_diag.get("answer_plan"), str):
        try:
            answer_plan = json.loads(pipeline_diag["answer_plan"])
        except json.JSONDecodeError:
            answer_plan = None

    spath = _structure_path(case, effective, response_text, pipeline_diag)
    structure_ok = True
    s_errs: list[str] = []
    if spath == "clarification":
        structure_ok, s_errs = _structure_clarification(response_text, case)
    elif spath == "compare":
        structure_ok, s_errs = _structure_compare(response_text, case, pipeline_diag, answer_plan)
    elif spath == "summary":
        structure_ok, s_errs = _structure_summary(response_text, case)
    elif spath == "quiz":
        structure_ok, s_errs = _structure_quiz(response_text, case)
    else:
        structure_ok, s_errs = _structure_chat(response_text, case, pipeline_diag)
    if not structure_ok:
        err.extend(s_errs)

    ok_route, r_errs = _route_expectation_allows(
        case, spath, effective, response_text, pipeline_diag
    )
    if not ok_route:
        err.extend(r_errs)
        structure_ok = False

    for sec in case.expected_sections:
        if sec and sec.lower() not in _norm(response_text):
            structure_ok = False
            err.append("expected_section_missing")

    score = (
        QUARTER * float(mode_ok)
        + QUARTER * float(content_ok)
        + QUARTER * float(forbidden_ok)
        + QUARTER * float(structure_ok)
    )
    br = {
        "mode": QUARTER * float(mode_ok),
        "required_content": QUARTER * float(content_ok),
        "forbidden": QUARTER * float(forbidden_ok),
        "structure": QUARTER * float(structure_ok),
    }
    return ScoringResult(
        score=round(score, 4),
        pass_ok=score >= 0.9999,
        mode_ok=mode_ok,
        content_ok=content_ok,
        forbidden_ok=forbidden_ok,
        structure_ok=structure_ok,
        error_categories=sorted(set(e for e in err if e)),
        breakdown=br,
    )
