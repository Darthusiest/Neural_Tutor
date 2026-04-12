"""When to run the secondary boost model (Gemini), never for primary Course Answer."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.services.query_understanding import QueryType

if TYPE_CHECKING:
    from app.services.answer_validation import ValidationResult


# User phrasing that suggests wanting a clearer or richer secondary explanation
_EXPLICIT_BOOST_HINTS = re.compile(
    r"\b(?:explain\s+simpler|simpler|more\s+detail|in\s+detail|more\s+depth|"
    r"give\s+(?:an?\s+)?example|for\s+example|elaborate|expand|break\s+it\s+down)\b",
    re.IGNORECASE,
)


def _is_very_complex_query(
    intent: QueryType | None,
    answer_intent: str | None,
    subquestion_count: int,
) -> bool:
    """Synthesis-style or decomposed-into-many-parts queries — optional boost candidate."""
    if intent == QueryType.SYNTHESIS:
        return True
    if answer_intent == "cross_lecture_synthesis":
        return True
    if subquestion_count >= 5:
        return True
    return False


def should_use_gemini_boost(
    *,
    user_query: str,
    confidence: float,
    validation: "ValidationResult | None",
    confidence_threshold: float,
    boost_toggle: bool,
    mode: str,
    query_type: QueryType | None,
    answer_intent: str | None,
    subquestion_count: int,
) -> tuple[bool, str]:
    """
    Decide whether to call the **secondary** boost model (Gemini).

    Never used for primary Course Answer; only for optional Boosted Explanation.
    ``validation is None`` = legacy (non-structured) path: toggle / confidence / mode / cues only.

    Returns ``(should_boost, reason_code)``.
    """
    if boost_toggle:
        return True, "user_toggle"

    if validation is None:
        if confidence < confidence_threshold:
            return True, "low_confidence"
        if mode in ("compare", "summary"):
            return True, "mode"
        if _EXPLICIT_BOOST_HINTS.search(user_query):
            return True, "user_requested_clarity"
        return False, "none"

    if validation.severity == "fail":
        return True, "validation_fail"

    if validation.severity == "weak":
        return True, "validation_weak"

    if confidence < confidence_threshold:
        return True, "low_confidence"

    if _is_very_complex_query(query_type, answer_intent, subquestion_count):
        return True, "complex_query"

    if _EXPLICIT_BOOST_HINTS.search(user_query):
        return True, "user_requested_clarity"

    return False, "none"


should_use_boost = should_use_gemini_boost
