"""Multi-strategy retrieval orchestrator (v2).

Wraps the existing lexical engine (:mod:`app.services.retrieval`) with:
- query understanding (classification + alias expansion + typo correction)
- strategy selection per query type
- multi-chunk diversification for compare / summary / synthesis queries
- supporting-chunk gathering via the concept graph

All callers that previously used ``retrieve_chunks`` can switch to
``retrieve_enhanced`` and get a backward-compatible superset of
:class:`~app.services.retrieval.RetrievalResult`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.domain_knowledge import (
    get_concept_family_for_lecture,
    get_lectures_in_family,
    get_related_lectures,
    infer_chunk_type,
)
from app.services.query_understanding import QueryIntent, QueryType, analyze_query
from app.services.retrieval import (
    RetrievalDiagnostics,
    RetrievalResult,
    _row_cache,
    format_course_answer,
    retrieve_chunks,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enhanced result (backward-compatible superset of RetrievalResult)
# ---------------------------------------------------------------------------

@dataclass
class EnhancedRetrievalResult(RetrievalResult):
    """Extends RetrievalResult with v2 fields; existing callers use .chunks / .confidence unchanged."""

    supporting_chunks: list[dict[str, Any]] = field(default_factory=list)
    query_intent: QueryIntent | None = None
    related_topics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-query-type strategies
# ---------------------------------------------------------------------------

def _handle_definition(expanded_q: str, intent: QueryIntent, top_k: int) -> EnhancedRetrievalResult:
    """Focused retrieval — top chunks, prefer definition/process chunk_types."""
    base = retrieve_chunks(expanded_q, top_k=top_k)
    chunks = _boost_chunk_type(base.chunks, preferred=("definition", "process"))
    return _wrap(base, chunks, intent)


def _handle_compare(expanded_q: str, intent: QueryIntent, top_k: int) -> EnhancedRetrievalResult:
    """Retrieve chunks for *both* sides of a compare query."""
    if intent.compare_concepts:
        a_q = expanded_q + " " + intent.compare_concepts[0]
        b_q = expanded_q + " " + intent.compare_concepts[1]
        ra = retrieve_chunks(a_q, top_k=max(top_k, 3))
        rb = retrieve_chunks(b_q, top_k=max(top_k, 3))
        merged = _merge_two_sides(ra.chunks, rb.chunks, top_k)
        conf = max(ra.confidence, rb.confidence)
        detected = ra.detected_topic or rb.detected_topic
        diag = ra.diagnostics
    else:
        base = retrieve_chunks(expanded_q, top_k=top_k + 2)
        merged = _diversify_by_lecture(base.chunks, top_k)
        conf = base.confidence
        detected = base.detected_topic
        diag = base.diagnostics
    return EnhancedRetrievalResult(
        chunks=merged,
        confidence=conf,
        detected_topic=detected,
        diagnostics=diag,
        query_intent=intent,
    )


def _handle_summary(expanded_q: str, intent: QueryIntent, top_k: int) -> EnhancedRetrievalResult:
    """Return all chunks from the target lecture (when explicit), else broad retrieval."""
    if intent.lecture_numbers and len(intent.lecture_numbers) == 1:
        lec = intent.lecture_numbers[0]
        lec_chunks = [r for r in _row_cache if r["lecture_number"] == lec]
        if lec_chunks:
            from app.services.retrieval import _row_to_public_dict

            chunks = [_row_to_public_dict(r) for r in lec_chunks]
            detected = (chunks[0].get("topic") or "").split("—")[0].strip()
            return EnhancedRetrievalResult(
                chunks=chunks,
                confidence=0.85,
                detected_topic=detected,
                diagnostics=None,
                query_intent=intent,
            )
    base = retrieve_chunks(expanded_q, top_k=top_k + 3)
    return _wrap(base, base.chunks[:top_k], intent)


def _handle_synthesis(expanded_q: str, intent: QueryIntent, top_k: int) -> EnhancedRetrievalResult:
    """Cross-lecture retrieval: expand related lectures, diversify results."""
    related_lecs: set[int] = set()
    for ln in intent.lecture_numbers:
        related_lecs.update(get_related_lectures(ln))
        related_lecs.add(ln)
    extra_terms: list[str] = []
    for ln in related_lecs:
        fam = get_concept_family_for_lecture(ln)
        if fam:
            from app.services.domain_knowledge import CONCEPT_FAMILIES
            extra_terms.extend(CONCEPT_FAMILIES[fam].get("concepts", [])[:3])
    aug_q = expanded_q
    if extra_terms:
        aug_q += " " + " ".join(dict.fromkeys(extra_terms))
    base = retrieve_chunks(aug_q, top_k=top_k + 4)
    diversified = _diversify_by_lecture(base.chunks, top_k)
    supporting = _gather_supporting(diversified, intent)
    return EnhancedRetrievalResult(
        chunks=diversified,
        confidence=base.confidence,
        detected_topic=base.detected_topic,
        diagnostics=base.diagnostics,
        supporting_chunks=supporting,
        query_intent=intent,
        related_topics=sorted({c.get("topic", "").split("—")[0].strip() for c in supporting} - {""}),
    )


def _handle_lecture_specific(expanded_q: str, intent: QueryIntent, top_k: int) -> EnhancedRetrievalResult:
    base = retrieve_chunks(expanded_q, top_k=top_k)
    return _wrap(base, base.chunks, intent)


def _handle_general(expanded_q: str, intent: QueryIntent, top_k: int) -> EnhancedRetrievalResult:
    base = retrieve_chunks(expanded_q, top_k=top_k)
    supporting = _gather_supporting(base.chunks, intent)
    return EnhancedRetrievalResult(
        chunks=base.chunks,
        confidence=base.confidence,
        detected_topic=base.detected_topic,
        diagnostics=base.diagnostics,
        supporting_chunks=supporting,
        query_intent=intent,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_STRATEGY = {
    QueryType.DEFINITION:      _handle_definition,
    QueryType.COMPARE:         _handle_compare,
    QueryType.SUMMARY:         _handle_summary,
    QueryType.SYNTHESIS:       _handle_synthesis,
    QueryType.LECTURE_SPECIFIC: _handle_lecture_specific,
    QueryType.QUIZ:            _handle_definition,
    QueryType.VAGUE_FOLLOWUP:  _handle_general,
    QueryType.GENERAL:         _handle_general,
}


def retrieve_enhanced(
    query: str,
    *,
    top_k: int = 5,
    backend: str = "keyword",
) -> EnhancedRetrievalResult:
    """
    Full v2 retrieval pipeline.

    1. Analyze query → QueryIntent
    2. Expand with aliases + typo corrections
    3. Route to strategy handler
    4. Return EnhancedRetrievalResult (superset of RetrievalResult)
    """
    if backend not in ("keyword",):
        return _fallback_to_base(query, top_k, backend)

    intent = analyze_query(query)
    handler = _STRATEGY.get(intent.query_type, _handle_general)

    logger.debug(
        "retrieval_v2: type=%s lecs=%s concepts=%s typos=%s expanded_extra=%d",
        intent.query_type.value,
        intent.lecture_numbers,
        intent.detected_concepts[:5],
        intent.typo_corrections,
        len(intent.expanded_tokens) - len(intent.query_tokens),
    )

    return handler(intent.expanded_query, intent, top_k)


def _fallback_to_base(query: str, top_k: int, backend: str) -> EnhancedRetrievalResult:
    """Non-keyword backends fall through to base retrieval (preserves NotImplementedError)."""
    base = retrieve_chunks(query, top_k=top_k, backend=backend)  # type: ignore[arg-type]
    return EnhancedRetrievalResult(
        chunks=base.chunks,
        confidence=base.confidence,
        detected_topic=base.detected_topic,
        diagnostics=base.diagnostics,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(base: RetrievalResult, chunks: list[dict], intent: QueryIntent) -> EnhancedRetrievalResult:
    return EnhancedRetrievalResult(
        chunks=chunks,
        confidence=base.confidence,
        detected_topic=base.detected_topic,
        diagnostics=base.diagnostics,
        query_intent=intent,
    )


def _boost_chunk_type(
    chunks: list[dict[str, Any]], *, preferred: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Mild reorder: preferred chunk_types float up, but don't override strong score gaps."""
    if len(chunks) <= 1:
        return chunks
    def _sort_key(c: dict) -> tuple[int, int]:
        heading = (c.get("topic") or "").split("—")[-1].strip()
        ctype = infer_chunk_type(heading)
        is_preferred = 0 if ctype in preferred else 1
        original_idx = chunks.index(c)
        return (is_preferred, original_idx)
    return sorted(chunks, key=_sort_key)


def _merge_two_sides(
    a_chunks: list[dict[str, Any]],
    b_chunks: list[dict[str, Any]],
    n: int,
) -> list[dict[str, Any]]:
    """Interleave top results from two sub-queries, deduped by chunk id."""
    seen: set[int] = set()
    merged: list[dict[str, Any]] = []
    for pair in zip(a_chunks, b_chunks):
        for c in pair:
            cid = c.get("id")
            if cid not in seen:
                seen.add(cid)
                merged.append(c)
            if len(merged) >= n:
                return merged
    for c in a_chunks + b_chunks:
        cid = c.get("id")
        if cid not in seen:
            seen.add(cid)
            merged.append(c)
        if len(merged) >= n:
            break
    return merged[:n]


def _diversify_by_lecture(chunks: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Pick up to *n* chunks, preferring unique lectures before repeats."""
    if len(chunks) <= n:
        return chunks
    lec_seen: set[int] = set()
    first_pass: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for c in chunks:
        ln = c.get("lecture_number")
        if ln not in lec_seen:
            lec_seen.add(ln)
            first_pass.append(c)
        else:
            rest.append(c)
    out = first_pass[:n]
    if len(out) < n:
        out.extend(rest[: n - len(out)])
    return out


def _gather_supporting(
    primary: list[dict[str, Any]],
    intent: QueryIntent,
) -> list[dict[str, Any]]:
    """Find related chunks from adjacent lectures (not already in primary)."""
    from app.services.retrieval import _row_to_public_dict

    primary_ids = {c.get("id") for c in primary}
    target_lecs: set[int] = set()
    for c in primary:
        ln = c.get("lecture_number")
        if ln is not None:
            target_lecs.update(get_related_lectures(ln))
    target_lecs -= {c.get("lecture_number") for c in primary}

    supporting: list[dict[str, Any]] = []
    for row in _row_cache:
        if row["id"] in primary_ids:
            continue
        if row["lecture_number"] in target_lecs:
            heading = (row.get("topic") or "").split("—")[-1].strip()
            ctype = infer_chunk_type(heading)
            if ctype in ("definition", "process", "overview"):
                supporting.append(_row_to_public_dict(row))
            if len(supporting) >= 4:
                break
    return supporting
