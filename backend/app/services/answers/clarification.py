"""Mode-aware clarification helpers (Task 8 / Task 7 clarification fallback).

Two responsibilities:

- :func:`is_underspecified_for_mode` decides whether a routed query carries
  enough signal to actually retrieve and render. The detection is purely
  deterministic and mirrors the signal style used in
  :mod:`app.services.query_mode`.
- :func:`clarification_for_mode` returns a short, mode-tailored ask that
  never invents course content. The same copy is used by both the
  pre-retrieval orchestrator path and the renderer / pipeline fallback
  paths so users see consistent wording.

The four supported modes match the API mode contract: ``compare`` /
``summary`` / ``quiz`` / ``chat``. ``chat`` is treated as never
underspecified by these helpers — chat queries land in the conversational
no-match flow when retrieval returns nothing.
"""

from __future__ import annotations

from typing import Optional

from app.services.knowledge.structured_query import StructuredQuery
from app.services.query_understanding import extract_compare_entities

# Public copy. Templates only — never names a concept that the user didn't.
_CLARIFICATION_COPY: dict[str, str] = {
    "compare": (
        "Tell me which two concepts to compare \u2014 for example, "
        "*Compare CNN and MLP* or *Difference between MFCCs and formants*."
    ),
    "summary": (
        "Tell me which lecture or topic to summarize \u2014 "
        "*Summarize Lecture 10* or *Recap of MFCCs* both work."
    ),
    "quiz": (
        "Tell me which topic or lecture to quiz on \u2014 "
        "*Quiz me on MFCCs* or *Test me on Lecture 11*."
    ),
}

# Default fallback used when the caller passes a mode we don't have a
# template for. Keeps behavior deterministic for the orchestrator.
_DEFAULT_CLARIFICATION = (
    "Tell me which course topic or lecture you'd like help with."
)


def _normalize_mode(mode: str | None) -> str:
    return (mode or "").strip().lower()


def _structured_compare_entities(structured_query: StructuredQuery | None) -> list[str]:
    if structured_query is None:
        return []
    intent = structured_query.intent
    if intent.compare_entities:
        return [e for e in intent.compare_entities if e]
    if intent.compare_concepts:
        return [e for e in intent.compare_concepts if e]
    return []


def is_underspecified_for_mode(
    text: str,
    structured_query: Optional[StructuredQuery],
    effective_mode: str,
) -> bool:
    """Return ``True`` when a query routes to a mode but lacks the required signal.

    Detection rules (all deterministic, no NLP):

    - **compare**: fewer than 2 entities surface from
      :func:`extract_compare_entities`, the structured query carries no
      compare entities/concepts, and ``len(concept_ids) < 2``.
    - **summary**: no lecture numbers and no detected concepts.
    - **quiz**: no lecture numbers and no detected concepts.
    - **chat** (or unknown): always ``False`` — chat lands in the
      conversational no-match path when retrieval returns nothing.
    """
    mode = _normalize_mode(effective_mode)
    body = (text or "").strip()
    if not body or mode in ("", "chat"):
        return False

    if mode == "compare":
        textual = extract_compare_entities(body) or []
        if len(textual) >= 2:
            return False
        struct_ents = _structured_compare_entities(structured_query)
        if len(struct_ents) >= 2:
            return False
        if structured_query is not None and len(structured_query.concept_ids) >= 2:
            return False
        return True

    if mode in ("summary", "quiz"):
        if structured_query is None:
            return True
        intent = structured_query.intent
        if intent.lecture_numbers:
            return False
        if intent.detected_concepts:
            return False
        # Some structured queries carry concept_ids without detected_concepts
        # (KB hit by alias). Treat any KB-resolved concept as enough signal.
        if structured_query.concept_ids:
            return False
        return True

    return False


def clarification_for_mode(
    text: str,
    structured_query: Optional[StructuredQuery],
    effective_mode: str,
) -> str:
    """Return the clarification copy for the given routed mode.

    ``text`` and ``structured_query`` are accepted for symmetry with
    :func:`is_underspecified_for_mode` and to keep the door open for
    future entity-aware suggestions; the current implementation is
    template-only on purpose.
    """
    del text, structured_query
    mode = _normalize_mode(effective_mode)
    return _CLARIFICATION_COPY.get(mode, _DEFAULT_CLARIFICATION)
