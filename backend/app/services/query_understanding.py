"""Classify and expand student queries before retrieval.

Maps raw user text to a :class:`QueryIntent` that controls retrieval strategy
(single-chunk definition vs. multi-chunk compare vs. lecture summary, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.services.domain_knowledge import (
    correct_typos,
    expand_terms_for_query,
    extract_lecture_range,
    get_aliases,
)
from app.services.retrieval import lecture_numbers_mentioned, tokenize_query_terms


# ---------------------------------------------------------------------------
# Query types
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    DEFINITION       = "definition"
    COMPARE          = "compare"
    SUMMARY          = "summary"
    LECTURE_SPECIFIC  = "lecture_specific"
    QUIZ             = "quiz"
    SYNTHESIS         = "cross_lecture_synthesis"
    VAGUE_FOLLOWUP   = "vague_followup"
    GENERAL          = "general"


@dataclass
class QueryIntent:
    query_type: QueryType
    original_query: str
    expanded_query: str
    query_tokens: list[str]
    expanded_tokens: list[str]
    lecture_numbers: list[int]
    detected_concepts: list[str]
    compare_concepts: tuple[str, str] | None = None
    typo_corrections: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

_COMPARE_RE = re.compile(
    r"(?:difference|differ|compare|comparison|contrast|versus|vs\.?)\s+"
    r"(?:between\s+)?(.+?)\s+(?:and|vs\.?|versus|or|&)\s+(.+)",
    re.IGNORECASE,
)
_COMPARE_SIMPLE_RE = re.compile(
    r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", re.IGNORECASE,
)

_SUMMARY_RE = re.compile(
    r"(?:summary|summarize|summarise|overview|recap)\s+(?:of\s+)?(?:lecture|lec\.?|week)",
    re.IGNORECASE,
)

_DEFINITION_STARTS = re.compile(
    r"^(?:what\s+is|what\s+are|define|explain)\b", re.IGNORECASE,
)

_QUIZ_RE = re.compile(
    r"\b(?:quiz|test)\s+(?:me|us)\b", re.IGNORECASE,
)

_VAGUE_TOKENS = frozenset(
    "simpler simple easier again confused unclear rephrase repeat elaborate "
    "what huh".split()
)

_SYNTHESIS_RE = re.compile(
    r"(?:connect|relate|relationship|across|between).+?(?:lecture|lec)", re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Compare concept extraction
# ---------------------------------------------------------------------------

def _extract_compare_concepts(query: str) -> tuple[str, str] | None:
    for pat in (_COMPARE_RE, _COMPARE_SIMPLE_RE):
        m = pat.search(query)
        if m:
            a = m.group(1).strip().rstrip("?.,")
            b = m.group(2).strip().rstrip("?.,")
            if a and b:
                return (a, b)
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(query: str, tokens: list[str], lec_nums: list[int]) -> QueryType:
    ql = query.lower()

    if _QUIZ_RE.search(ql):
        return QueryType.QUIZ

    if _extract_compare_concepts(query) is not None:
        return QueryType.COMPARE
    if "vs" in tokens or "versus" in ql or "difference" in ql or "compare" in ql:
        return QueryType.COMPARE

    if _SUMMARY_RE.search(ql):
        return QueryType.SUMMARY
    if ("summary" in ql or "summarize" in ql or "overview" in ql) and lec_nums:
        return QueryType.SUMMARY

    if extract_lecture_range(query):
        return QueryType.SYNTHESIS
    if _SYNTHESIS_RE.search(ql):
        return QueryType.SYNTHESIS
    if len(lec_nums) >= 2:
        return QueryType.SYNTHESIS

    if len(tokens) <= 3 and all(t in _VAGUE_TOKENS for t in tokens):
        return QueryType.VAGUE_FOLLOWUP

    if lec_nums and len(lec_nums) == 1:
        return QueryType.LECTURE_SPECIFIC

    if _DEFINITION_STARTS.match(ql):
        return QueryType.DEFINITION

    return QueryType.GENERAL


# ---------------------------------------------------------------------------
# Concept detection (match query tokens to known domain aliases)
# ---------------------------------------------------------------------------

def _detect_concepts(tokens: list[str]) -> list[str]:
    """Return canonical concept names recognized in the query tokens."""
    from app.services.domain_knowledge import get_canonical

    seen: set[str] = set()
    out: list[str] = []
    full_text = " ".join(tokens)
    for tok in tokens:
        c = get_canonical(tok)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    for alias_phrase, _ in sorted(
        ((a, a) for a in _get_all_multi_word_aliases()),
        key=lambda x: -len(x[0]),
    ):
        if alias_phrase in full_text:
            c = get_canonical(alias_phrase)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
    return out


def _get_all_multi_word_aliases() -> list[str]:
    from app.services.domain_knowledge import _ALIAS_GROUPS

    return [
        alias
        for group in _ALIAS_GROUPS
        for alias in group
        if " " in alias
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_query(query: str) -> QueryIntent:
    """Full query understanding pipeline: classify → expand → typo-correct."""
    raw_tokens = tokenize_query_terms(query)
    lec_nums_set = lecture_numbers_mentioned(query)
    lec_range = extract_lecture_range(query)
    all_lec = sorted(lec_nums_set | set(lec_range))

    qtype = _classify(query, raw_tokens, all_lec)

    typo_map = correct_typos(raw_tokens)
    corrected = [typo_map.get(t, t) for t in raw_tokens]

    alias_extra = expand_terms_for_query(corrected)
    expanded_tokens = corrected + alias_extra

    expanded_query = query
    if alias_extra:
        expanded_query = query + " " + " ".join(alias_extra)
    for orig, fixed in typo_map.items():
        if fixed not in query.lower():
            expanded_query = expanded_query + " " + fixed

    detected = _detect_concepts(corrected)
    compare_pair = _extract_compare_concepts(query) if qtype == QueryType.COMPARE else None

    return QueryIntent(
        query_type=qtype,
        original_query=query,
        expanded_query=expanded_query,
        query_tokens=raw_tokens,
        expanded_tokens=expanded_tokens,
        lecture_numbers=all_lec,
        detected_concepts=detected,
        compare_concepts=compare_pair,
        typo_corrections=typo_map,
    )
