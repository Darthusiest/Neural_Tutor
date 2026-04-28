"""When to run the secondary boost model (Gemini), never for primary Course Answer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.query_understanding import QueryType

if TYPE_CHECKING:
    from app.services.answers.answer_validation import ValidationResult


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
    Boost is **strictly opt-in**: it runs only when the client passes
    ``boost_toggle=True`` (UI checkbox / explicit API flag). Validation
    severity, confidence, mode, query complexity, and phrasing cues do
    **not** auto-trigger boost on their own.

    The other parameters (``user_query``, ``confidence``, ``validation``,
    ``confidence_threshold``, ``mode``, ``query_type``, ``answer_intent``,
    ``subquestion_count``) are kept on the signature for analytics
    parity and future use; they are intentionally ignored today.

    Returns ``(should_boost, reason_code)``. The reason is ``user_toggle``
    when boost runs and ``boost_disabled`` when the toggle is off.
    """
    del user_query, confidence, validation, confidence_threshold
    del mode, query_type, answer_intent, subquestion_count
    if boost_toggle:
        return True, "user_toggle"
    return False, "boost_disabled"


should_use_boost = should_use_gemini_boost
