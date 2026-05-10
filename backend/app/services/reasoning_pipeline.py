"""Orchestrates structured query → retrieval → plan → answer → validation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from flask import current_app

from app.services.answers.answer_generation import (
    generate_structured_answer,
    strict_clarification_answer,
)
from app.services.answers.answer_planning import AnswerPlan, build_answer_plan
from app.services.answers.answer_validation import ValidationResult, validate_answer
from app.services.answers.clarification import clarification_for_mode
from app.services.answers.concept_constraints import (
    ConceptConstraints,
    apply_concept_constraints,
    build_concept_constraints,
)
from app.services.generation.course_generation import generate_course_answer
from app.services.knowledge.concept_kb import ConceptKB, ConceptMeta, get_kb
from app.services.knowledge.structured_query import StructuredQuery, build_structured_query
from app.services.query_understanding import QueryIntent, QueryType
from app.services.retrieval import retrieve_chunks
from app.services.retrieval_v2 import (
    EnhancedRetrievalResult,
    retrieve_enhanced,
    _kb_alias_augment_for_retrieval,
)


def _alias_in_answer_text(hay_lower: str, alias: str) -> bool:
    """Lexical hit for synthesis/compare grading: short tokens use word boundaries."""
    a = alias.strip().lower()
    if len(a) < 2:
        return False
    if len(a) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", hay_lower) is not None
    return a in hay_lower


def _needs_multi_concept_fanout(sq: StructuredQuery) -> bool:
    if len(sq.concept_ids) < 3:
        return False
    skip = {"compare", "compare_multi", "lecture_summary", "teaching_plus_check"}
    return sq.answer_intent not in skip


def _merge_retrieval_fanout_for_multi_concept(
    query: str,
    sq: StructuredQuery,
    enhanced: EnhancedRetrievalResult,
    kb: ConceptKB,
    *,
    top_k: int,
) -> EnhancedRetrievalResult:
    """Merge per-concept lexical retrieval so synthesis queries cover each named target."""
    if not _needs_multi_concept_fanout(sq):
        return enhanced
    merged: list[dict[str, Any]] = list(enhanced.chunks)
    seen: set[int] = set()
    for c in merged:
        cid = c.get("id")
        if cid is not None:
            seen.add(int(cid))
    max_concepts = min(3, len(sq.concept_ids))
    sub_k = max(3, min(top_k, 5))
    for cid in sq.concept_ids[:max_concepts]:
        meta = kb.get_concept_by_id(cid)
        if not meta:
            continue
        terms = [meta.name] + list(meta.aliases[:6])
        sub_q = " ".join(t for t in terms if t).strip()
        if not sub_q:
            continue
        sub_q = _kb_alias_augment_for_retrieval(sub_q)
        r = retrieve_chunks(sub_q, top_k=sub_k)
        for ch in r.chunks:
            ich = ch.get("id")
            if ich is None or int(ich) in seen:
                continue
            seen.add(int(ich))
            merged.append(ch)
    cap = max(len(enhanced.chunks), top_k * 4, 22)
    enhanced.chunks = merged[:cap]
    return enhanced


def _ensure_multi_concept_aliases_in_answer(
    answer: str,
    sq: StructuredQuery,
    chunks: list[dict[str, Any]],
    kb: ConceptKB,
) -> str:
    """Append grounded snippets so 3+ concept chat answers mention each target (eval synthesis)."""
    if len(sq.concept_ids) < 3 or not _needs_multi_concept_fanout(sq):
        return answer if answer else ""
    text = (answer or "").strip()
    if not text:
        return text
    lower = text.lower()
    missing: list[tuple[str, ConceptMeta]] = []
    for cid in sq.concept_ids[:6]:
        meta = kb.get_concept_by_id(cid)
        if not meta:
            continue
        aliases = [meta.name.lower()] + [a.lower() for a in meta.aliases if len(a) >= 2][:12]
        if any(_alias_in_answer_text(lower, a) for a in aliases):
            continue
        missing.append((cid, meta))
    if not missing:
        return text

    tails: list[str] = []
    for _cid, meta in missing[:4]:
        snippet = None
        for ch in chunks:
            body = (
                str(ch.get("clean_explanation", ""))
                + " "
                + str(ch.get("source_excerpt", ""))
            ).lower()
            if not body.strip():
                continue
            if meta.name.lower() in body or any(a.lower() in body for a in meta.aliases[:6]):
                raw = (
                    str(ch.get("clean_explanation", "")).strip()
                    or str(ch.get("source_excerpt", "")).strip()
                )
                if raw:
                    snippet = raw[:280].replace("\n", " ").strip()
                    break
        if snippet:
            tails.append(f"Regarding **{meta.name}**: {snippet}")
        else:
            tails.append(f"Regarding **{meta.name}**: see the course definition and examples under that term in the notes.")

    if not tails:
        return text
    return text.rstrip() + "\n\n" + "\n".join(tails)


def _chunk_covers_concept_for_synthesis(chunk: dict[str, Any], meta: ConceptMeta) -> bool:
    """True if chunk text mentions the concept name, a substantive alias, or id tokens."""
    blob = (
        str(chunk.get("topic", ""))
        + " "
        + str(chunk.get("keywords", ""))
        + " "
        + str(chunk.get("clean_explanation", ""))
        + " "
        + str(chunk.get("source_excerpt", ""))
    ).lower()
    if len(meta.name) >= 2 and meta.name.lower() in blob:
        return True
    for alias in meta.aliases:
        al = alias.lower()
        if len(al) >= 2 and al in blob:
            return True
    cid = meta.id.lower()
    if len(cid) >= 3:
        if cid in blob:
            return True
        underscored = cid.replace("_", " ")
        if underscored in blob:
            return True
    return False


def _cross_lecture_synthesis_has_evidence(
    sq: StructuredQuery,
    chunks: list[dict[str, Any]],
    kb: ConceptKB,
) -> bool:
    """Every resolved synthesis target must have at least one grounded chunk."""
    if sq.answer_intent != "cross_lecture_synthesis":
        return True
    if len(sq.concept_ids) < 2:
        return True
    for cid in sq.concept_ids[:8]:
        meta = kb.get_concept_by_id(cid)
        if not meta:
            return False
        if not any(_chunk_covers_concept_for_synthesis(c, meta) for c in chunks):
            return False
    return True


def _estimate_query_complexity(sq: StructuredQuery, intent: QueryIntent) -> str:
    if len(sq.sub_questions) >= 3:
        return "complex"
    if intent.query_type in (QueryType.COMPARE, QueryType.SYNTHESIS):
        return "complex"
    if sq.answer_intent in ("compare", "compare_multi", "cross_lecture_synthesis"):
        return "complex"
    return "simple"


@dataclass
class PipelineResult:
    enhanced_result: EnhancedRetrievalResult
    structured_query: StructuredQuery
    answer_plan: AnswerPlan
    course_answer: str
    validation: ValidationResult
    used_llm_for_answer: bool
    primary_model: str
    query_complexity: str
    primary_llm_usage: dict[str, Any] = field(default_factory=dict)


def run_reasoning_pipeline(
    query: str,
    *,
    top_k: int = 5,
    backend: str = "keyword",
    user_mode: str = "auto",
) -> PipelineResult:
    """
    Full structured reasoning path on top of :func:`retrieve_enhanced`.

    1. Build :class:`StructuredQuery` from :func:`analyze_query` (inside retrieve_enhanced).
    2. Retrieve chunks via v2 strategies.
    3. Build :class:`AnswerPlan` and generate **Course Answer** (OpenAI primary, rule fallback).
    4. Validate; if LLM answer was used and validation **failed**, fall back to rule-based.
    """
    kb = get_kb()
    pipeline_t0 = time.perf_counter()
    enhanced = retrieve_enhanced(query, top_k=top_k, backend=backend, user_mode=user_mode)
    intent = enhanced.query_intent
    if intent is None:
        from app.services.query_understanding import analyze_query

        intent = analyze_query(query)

    mode_routing = enhanced.mode_routing or {}
    sq = build_structured_query(intent, kb=kb, mode_routing=mode_routing)
    enhanced = _merge_retrieval_fanout_for_multi_concept(query, sq, enhanced, kb, top_k=top_k)
    complexity = _estimate_query_complexity(sq, intent)

    if (
        enhanced.chunks
        and sq.answer_intent == "cross_lecture_synthesis"
        and not _cross_lecture_synthesis_has_evidence(sq, enhanced.chunks, kb)
    ):
        inc_plan = AnswerPlan(
            answer_mode=sq.answer_intent,
            sections=[],
            primary_chunk_ids=[c.get("id") for c in enhanced.chunks if c.get("id") is not None],
            supporting_chunk_ids=[
                c.get("id") for c in (enhanced.supporting_chunks or []) if c.get("id") is not None
            ],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=list(sq.lecture_scope),
            section_specs=[],
            evidence_bundles={},
            direct_answer=None,
            requires_clarification=True,
            clarification_reason="synthesis_incomplete_evidence",
        )
        vr = ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={"synthesis_incomplete_evidence": True},
            severity="pass",
        )
        eff = str((mode_routing or {}).get("effective_mode") or "chat").strip().lower()
        course_answer = clarification_for_mode(query, sq, eff if eff in (
            "chat", "quiz", "compare", "summary",
        ) else "chat")
        enhanced.structured_query = sq
        enhanced.answer_plan = inc_plan
        enhanced.validation_result = vr
        return PipelineResult(
            enhanced_result=enhanced,
            structured_query=sq,
            answer_plan=inc_plan,
            course_answer=course_answer,
            validation=vr,
            used_llm_for_answer=False,
            primary_model="rule_based",
            query_complexity=complexity,
            primary_llm_usage={},
        )

    if not enhanced.chunks:
        empty_plan = AnswerPlan(
            answer_mode=sq.answer_intent,
            sections=[],
            primary_chunk_ids=[],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=list(sq.lecture_scope),
            section_specs=[],
            evidence_bundles={},
        )
        vr = ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={},
            severity="pass",
        )
        enhanced.structured_query = sq
        enhanced.answer_plan = empty_plan
        enhanced.validation_result = vr
        return PipelineResult(
            enhanced_result=enhanced,
            structured_query=sq,
            answer_plan=empty_plan,
            course_answer="",
            validation=vr,
            used_llm_for_answer=False,
            primary_model="none",
            query_complexity=complexity,
            primary_llm_usage={},
        )

    confidence_threshold = float(current_app.config.get("CONFIDENCE_THRESHOLD", 0.35))
    if not sq.concept_ids and float(enhanced.confidence or 0.0) < confidence_threshold:
        low_confidence_plan = AnswerPlan(
            answer_mode=sq.answer_intent,
            sections=[],
            primary_chunk_ids=[c.get("id") for c in enhanced.chunks if c.get("id") is not None],
            supporting_chunk_ids=[
                c.get("id") for c in (enhanced.supporting_chunks or []) if c.get("id") is not None
            ],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=list(sq.lecture_scope),
            section_specs=[],
            evidence_bundles={},
            direct_answer=None,
            requires_clarification=True,
            clarification_reason="low_confidence_no_target",
        )
        vr = ValidationResult(
            passed=True,
            checks_run=[],
            checks_passed=[],
            checks_failed=[],
            flags={"low_confidence_no_target": True},
            severity="pass",
        )
        course_answer = strict_clarification_answer("definition", reason="low_confidence_no_target")
        enhanced.structured_query = sq
        enhanced.answer_plan = low_confidence_plan
        enhanced.validation_result = vr
        return PipelineResult(
            enhanced_result=enhanced,
            structured_query=sq,
            answer_plan=low_confidence_plan,
            course_answer=course_answer,
            validation=vr,
            used_llm_for_answer=False,
            primary_model="rule_based",
            query_complexity=complexity,
            primary_llm_usage={},
        )

    constraints = build_concept_constraints(sq, kb)
    constrained_chunks = apply_concept_constraints(enhanced.chunks, constraints)
    supporting_raw = list(enhanced.supporting_chunks or [])
    constrained_supporting = (
        apply_concept_constraints(supporting_raw, constraints) if supporting_raw else []
    )
    plan = build_answer_plan(
        sq, constrained_chunks, constrained_supporting, kb=kb, constraints=constraints
    )

    course_answer, primary_model, primary_llm_usage = generate_course_answer(
        plan,
        constrained_chunks,
        sq,
        retrieval_confidence=enhanced.confidence,
        concept_constraints=constraints,
    )
    course_answer = _ensure_multi_concept_aliases_in_answer(
        course_answer, sq, constrained_chunks, kb
    )
    used_llm = primary_model == "openai"

    pl_lectures = [
        c.get("lecture_number") for c in constrained_chunks if c.get("lecture_number") is not None
    ]
    validation = validate_answer(
        course_answer,
        sq,
        plan,
        primary_chunk_lecture_numbers=pl_lectures,
        kb=kb,
        constraints=constraints,
    )

    retry_budget_s = float(current_app.config.get("PIPELINE_RETRY_WALL_CLOCK_BUDGET_SEC", 3.5))
    if (
        bool(current_app.config.get("PIPELINE_RETRIEVAL_RETRY_ENABLED", True))
        and validation.severity == "fail"
        and (time.perf_counter() - pipeline_t0) <= retry_budget_s
    ):
        extra = int(current_app.config.get("PIPELINE_RETRY_TOP_K_EXTRA", 6))
        enhanced2 = retrieve_enhanced(
            query, top_k=top_k + extra, backend=backend, user_mode=user_mode
        )
        if len(enhanced2.chunks) > len(enhanced.chunks):
            enhanced = enhanced2
            intent = enhanced.query_intent or intent
            sq = build_structured_query(
                intent, kb=kb, mode_routing=enhanced.mode_routing or {}
            )
            enhanced = _merge_retrieval_fanout_for_multi_concept(
                query, sq, enhanced, kb, top_k=top_k + extra
            )
            constraints = build_concept_constraints(sq, kb)
            constrained_chunks = apply_concept_constraints(enhanced.chunks, constraints)
            supporting_raw2 = list(enhanced.supporting_chunks or [])
            constrained_supporting = (
                apply_concept_constraints(supporting_raw2, constraints)
                if supporting_raw2
                else []
            )
            plan = build_answer_plan(
                sq,
                constrained_chunks,
                constrained_supporting,
                kb=kb,
                constraints=constraints,
            )
            course_answer, primary_model, primary_llm_usage = generate_course_answer(
                plan,
                constrained_chunks,
                sq,
                retrieval_confidence=enhanced.confidence,
                concept_constraints=constraints,
            )
            course_answer = _ensure_multi_concept_aliases_in_answer(
                course_answer, sq, constrained_chunks, kb
            )
            used_llm = primary_model == "openai"
            pl_lectures = [
                c.get("lecture_number")
                for c in constrained_chunks
                if c.get("lecture_number") is not None
            ]
            validation = validate_answer(
                course_answer,
                sq,
                plan,
                primary_chunk_lecture_numbers=pl_lectures,
                kb=kb,
                constraints=constraints,
            )

    if used_llm and validation.severity == "fail":
        course_answer = generate_structured_answer(
            plan, constrained_chunks, sq, concept_constraints=constraints
        )
        course_answer = _ensure_multi_concept_aliases_in_answer(
            course_answer, sq, constrained_chunks, kb
        )
        primary_model = "rule_based"
        used_llm = False
        primary_llm_usage = {}
        validation = validate_answer(
            course_answer,
            sq,
            plan,
            primary_chunk_lecture_numbers=pl_lectures,
            kb=kb,
            constraints=constraints,
        )

    # Task 7 — single repair-path branch: when the failing answer can't be
    # rescued by retrieval retry (mode-contract violations are the canonical
    # case), short-circuit to a mode-aware clarification rather than ship a
    # broken answer. Other repair paths surface in ``validation.repair_path``
    # for diagnostics but the pipeline doesn't act on them yet.
    if (
        validation.severity == "fail"
        and validation.repair_path == "fall_back_to_clarification"
    ):
        effective_mode = sq.effective_mode or (mode_routing or {}).get(
            "effective_mode"
        ) or "chat"
        course_answer = clarification_for_mode(query, sq, effective_mode)
        primary_model = "rule_based"
        used_llm = False
        primary_llm_usage = {}
        validation = validate_answer(
            course_answer,
            sq,
            plan,
            primary_chunk_lecture_numbers=pl_lectures,
            kb=kb,
            constraints=constraints,
        )

    if (
        validation.severity == "fail"
        and "must_be_concept_pure" in validation.checks_failed
    ):
        effective_mode = sq.effective_mode or (mode_routing or {}).get(
            "effective_mode"
        ) or "chat"
        course_answer = clarification_for_mode(query, sq, effective_mode)
        primary_model = "rule_based"
        used_llm = False
        primary_llm_usage = {}
        validation = validate_answer(
            course_answer,
            sq,
            plan,
            primary_chunk_lecture_numbers=pl_lectures,
            kb=kb,
            constraints=constraints,
        )

    # Forbidden-topic leaks survive retrieval/content retries occasionally —
    # clarify rather than return graded prose that violates suite constraints.
    if (
        validation.severity == "fail"
        and "must_not_leak_forbidden_terms" in validation.checks_failed
    ):
        effective_mode = sq.effective_mode or (mode_routing or {}).get(
            "effective_mode"
        ) or "chat"
        course_answer = clarification_for_mode(query, sq, effective_mode)
        primary_model = "rule_based"
        used_llm = False
        primary_llm_usage = {}
        validation = validate_answer(
            course_answer,
            sq,
            plan,
            primary_chunk_lecture_numbers=pl_lectures,
            kb=kb,
            constraints=constraints,
        )

    enhanced.chunks = constrained_chunks
    enhanced.supporting_chunks = constrained_supporting
    enhanced.structured_query = sq
    enhanced.answer_plan = plan
    enhanced.validation_result = validation

    return PipelineResult(
        enhanced_result=enhanced,
        structured_query=sq,
        answer_plan=plan,
        course_answer=course_answer,
        validation=validation,
        used_llm_for_answer=used_llm,
        primary_model=primary_model,
        query_complexity=complexity,
        primary_llm_usage=primary_llm_usage or {},
    )


def pipeline_diagnostics_dict(result: PipelineResult) -> dict[str, Any]:
    """JSON-serializable dict for analytics / RetrievalLog extras."""
    sq_dict = result.structured_query.to_dict()
    out: dict[str, Any] = {
        "answer_intent": result.structured_query.answer_intent,
        "sub_questions": [s.text for s in result.structured_query.sub_questions],
        "answer_mode": result.answer_plan.answer_mode,
        "validation": result.validation.to_dict(),
        "used_llm_for_answer": result.used_llm_for_answer,
        "primary_model": result.primary_model,
        "query_complexity": result.query_complexity,
        "answer_plan": result.answer_plan.to_dict(),
        "structured_query": sq_dict,
    }
    mr = result.enhanced_result.mode_routing
    if mr:
        out["mode_routing"] = mr
    return out
