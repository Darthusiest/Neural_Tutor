"""Orchestrates structured query → retrieval → plan → answer → validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.answer_generation import generate_structured_answer
from app.services.answer_planning import AnswerPlan, build_answer_plan
from app.services.answer_validation import ValidationResult, validate_answer
from app.services.concept_kb import get_kb
from app.services.course_generation import generate_course_answer
from app.services.query_understanding import QueryIntent, QueryType
from app.services.retrieval_v2 import EnhancedRetrievalResult, retrieve_enhanced
from app.services.structured_query import StructuredQuery, build_structured_query


def _estimate_query_complexity(sq: StructuredQuery, intent: QueryIntent) -> str:
    if len(sq.sub_questions) >= 3:
        return "complex"
    if intent.query_type in (QueryType.COMPARE, QueryType.SYNTHESIS):
        return "complex"
    if sq.answer_intent in ("compare", "cross_lecture_synthesis"):
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


def run_reasoning_pipeline(
    query: str,
    *,
    top_k: int = 5,
    backend: str = "keyword",
) -> PipelineResult:
    """
    Full structured reasoning path on top of :func:`retrieve_enhanced`.

    1. Build :class:`StructuredQuery` from :func:`analyze_query` (inside retrieve_enhanced).
    2. Retrieve chunks via v2 strategies.
    3. Build :class:`AnswerPlan` and generate **Course Answer** (OpenAI primary, rule fallback).
    4. Validate; if LLM answer was used and validation **failed**, fall back to rule-based.
    """
    kb = get_kb()
    enhanced = retrieve_enhanced(query, top_k=top_k, backend=backend)
    intent = enhanced.query_intent
    if intent is None:
        from app.services.query_understanding import analyze_query

        intent = analyze_query(query)

    sq = build_structured_query(intent, kb=kb)
    complexity = _estimate_query_complexity(sq, intent)

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
        )

    plan = build_answer_plan(sq, enhanced.chunks, enhanced.supporting_chunks, kb=kb)

    course_answer, primary_model = generate_course_answer(plan, enhanced.chunks, sq)
    used_llm = primary_model == "openai"

    pl_lectures = [c.get("lecture_number") for c in enhanced.chunks if c.get("lecture_number") is not None]
    validation = validate_answer(course_answer, sq, plan, primary_chunk_lecture_numbers=pl_lectures, kb=kb)

    if used_llm and validation.severity == "fail":
        course_answer = generate_structured_answer(plan, enhanced.chunks, sq)
        primary_model = "rule_based"
        used_llm = False
        validation = validate_answer(course_answer, sq, plan, primary_chunk_lecture_numbers=pl_lectures, kb=kb)

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
    )


def pipeline_diagnostics_dict(result: PipelineResult) -> dict[str, Any]:
    """JSON-serializable dict for analytics / RetrievalLog extras."""
    return {
        "answer_intent": result.structured_query.answer_intent,
        "sub_questions": [s.text for s in result.structured_query.sub_questions],
        "answer_mode": result.answer_plan.answer_mode,
        "validation": result.validation.to_dict(),
        "used_llm_for_answer": result.used_llm_for_answer,
        "primary_model": result.primary_model,
        "query_complexity": result.query_complexity,
        "answer_plan": result.answer_plan.to_dict(),
        "structured_query": result.structured_query.to_dict(),
    }
