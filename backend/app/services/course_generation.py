"""Primary course answer generation (OpenAI). Rule-based fallback when API unavailable."""

from __future__ import annotations

from typing import Any

from flask import current_app

from app.services.answer_generation import generate_structured_answer
from app.services.answer_planning import AnswerPlan
from app.services.llm import generate_plan_constrained_answer
from app.services.structured_query import StructuredQuery


def generate_course_answer(
    plan: AnswerPlan,
    chunks: list[dict[str, Any]],
    sq: StructuredQuery,
) -> tuple[str, str]:
    """
    Generate the main **Course Answer** only.

    Primary path: OpenAI (plan-constrained) when ``PRIMARY_COURSE_ANSWER_OPENAI`` and
    ``OPENAI_API_KEY`` are set. Fallback: rule-based structured template.

    Returns ``(text, primary_model)`` where ``primary_model`` is ``"openai"`` or ``"rule_based"``.
    """
    use_openai = bool(current_app.config.get("PRIMARY_COURSE_ANSWER_OPENAI")) and bool(
        current_app.config.get("OPENAI_API_KEY")
    )
    if use_openai:
        text, _usage = generate_plan_constrained_answer(plan, chunks, sq)
        if text and text.strip():
            return text.strip(), "openai"
    return generate_structured_answer(plan, chunks, sq), "rule_based"
