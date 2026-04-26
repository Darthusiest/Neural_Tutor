"""Primary course answer generation (OpenAI). Rule-based fallback when API unavailable."""

from __future__ import annotations

from typing import Any

from flask import current_app

from app.services.answers.answer_generation import generate_structured_answer
from app.services.answers.answer_planning import AnswerPlan
from app.services.generation.llm import generate_plan_constrained_answer
from app.services.knowledge.structured_query import StructuredQuery


def generate_course_answer(
    plan: AnswerPlan,
    chunks: list[dict[str, Any]],
    sq: StructuredQuery,
) -> tuple[str, str, dict[str, Any]]:
    """
    Generate the main **Course Answer** only.

    Primary path: OpenAI (plan-constrained) when ``PRIMARY_COURSE_ANSWER_OPENAI`` and
    ``OPENAI_API_KEY`` are set. Fallback: rule-based structured template.

    Returns ``(text, primary_model, primary_llm_usage)`` where ``primary_model`` is
    ``"openai"`` or ``"rule_based"``, and ``primary_llm_usage`` is OpenAI metadata (or empty).
    """
    # Compare / quiz / summary modes use deterministic, mode-specific renderers — never an LLM —
    # so the per-mode shape (Quiz: + Answer Key, Summary: + Main idea/Key topics, compare table)
    # cannot be overwritten by a free-form generation path.
    if plan.answer_mode in ("compare", "compare_multi", "lecture_summary", "teaching_plus_check"):
        text = generate_structured_answer(plan, chunks, sq)
        return text.strip(), "rule_based", {}

    use_openai = bool(current_app.config.get("PRIMARY_COURSE_ANSWER_OPENAI")) and bool(
        current_app.config.get("OPENAI_API_KEY")
    )
    if use_openai:
        text, usage_meta = generate_plan_constrained_answer(plan, chunks, sq)
        if text and text.strip():
            return text.strip(), "openai", usage_meta or {}
    return generate_structured_answer(plan, chunks, sq), "rule_based", {}
