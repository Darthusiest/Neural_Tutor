"""Concept-purity layer: target/allowed/forbidden terms for retrieval, evidence, and validation.

This module centralizes the "what does the user actually want" signal in one
:class:`ConceptConstraints` value object that flows through:

1. **Retrieval rerank** — :func:`apply_concept_constraints` adjusts chunk
   ordering (and drops obvious off-topic leaks) after :func:`retrieve_enhanced`
   has produced its top-K. This intentionally does *not* edit the lexical /
   embedding scorers; it sits on top so leaks that survive scoring still get
   penalized.
2. **Evidence-line selection** — :func:`is_line_concept_pure` is consumed by
   the chat / direct-answer renderers (and indirectly by
   :func:`compare_evidence.scoped_lines_from_chunks` when the optional
   ``constraints`` kwarg is passed) so a single sentence that drifts to a
   forbidden topic is dropped before it reaches the user.
3. **Validation** — :func:`answer_validation.must_be_concept_pure` reads the
   same constraints object so the validator agrees with what retrieval and
   evidence selection were already enforcing.

Forbidden terms come from the existing
:func:`entity_retrieval.forbidden_terms_for_concept` helper (a hardcoded peer
map plus dynamic peer-name aggregation). Target aliases come from
:class:`ConceptKB.get_concept_by_id`. The constraints object itself does *not*
introduce a new KB schema or migrations — it's a thin derivation layer.

Relational queries (``compare`` / ``cross_lecture_synthesis`` / queries with
≥ 2 target concepts that aren't part of a compare pair, e.g. *"How does
dynamic programming relate to backpropagation?"*) deliberately loosen the
forbidden gate via the :attr:`ConceptConstraints.is_relational` flag — the
spec calls for purity *without* over-filtering shared lecture context, and
relational queries are exactly the case where shared context is the point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.services.answers.entity_retrieval import (
    _term_hits,
    forbidden_terms_for_concept,
)
from app.services.knowledge.concept_kb import ConceptKB, get_kb
from app.services.knowledge.structured_query import StructuredQuery
from app.services.query_understanding import QueryType


# Score deltas applied on top of the (negative) original rank index so a
# higher delta moves a chunk *up* in the constrained order. Tuned so a single
# strong target alias hit in topic / keywords (+1.5) outweighs an off-lecture
# penalty (-0.6), but two distinct forbidden hits in topic / keywords
# (-2.4 + cap) do drop a clearly off-topic chunk past the top of the list.
_ALIAS_HIT_TOPIC_WEIGHT = 1.5
_ALIAS_HIT_BODY_WEIGHT = 0.6
_ALIAS_HIT_BODY_CAP = 3.0
_FORBIDDEN_HIT_TOPIC_WEIGHT = 1.2
_FORBIDDEN_HIT_BODY_WEIGHT = 0.4
_FORBIDDEN_HIT_BODY_CAP = 1.6
_LECTURE_MATCH_BONUS = 0.8
_OFF_LECTURE_PENALTY = 0.6

# Hard-drop threshold: a chunk is removed (not just demoted) only when it
# scores this badly *and* has no allowed-term hit *and* the query is not
# relational. Anything above the threshold stays in the pool — we still
# prefer "demote then trust the renderer" over "filter aggressively."
_HARD_DROP_DELTA = -2.0


@dataclass
class ConceptConstraints:
    """Centralized purity signal derived from a :class:`StructuredQuery`.

    Attributes:
        target_concepts: Canonical KB concept ids the user is asking about.
            Mirrors ``sq.concept_ids`` (often length 1 for chat / definition,
            length 2 for compare, length 3+ for multi-compare).
        target_aliases: Lower-cased KB aliases for every id in
            :attr:`target_concepts`. Used both for "boost" scoring and for the
            line-purity check.
        allowed_terms: Currently equal to :attr:`target_aliases`. Carved out
            as a separate field so future iterations can widen it (e.g. add
            topic-shared lecture vocabulary) without touching call sites.
        forbidden_terms: Peer-concept aliases plus the static
            ``_DEFAULT_FORBIDDEN_BY_CONCEPT`` list, with anything already in
            :attr:`target_aliases` removed (we never want to forbid our own
            term).
        target_lectures: Lecture numbers the user explicitly scoped to (via
            ``Lecture N`` or KB ``lecture_scope``). Empty when the user
            didn't anchor to a lecture.
        is_relational: True when the user is asking how two concepts relate
            (compare, synthesis, or any query with two distinct target
            concepts). The validator skips the topic-drift check in this case
            and the rerank loosens its hard-drop rule.
    """

    target_concepts: list[str]
    target_aliases: set[str]
    allowed_terms: set[str]
    forbidden_terms: set[str]
    target_lectures: list[int] = field(default_factory=list)
    is_relational: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_concepts": list(self.target_concepts),
            "target_aliases": sorted(self.target_aliases),
            "allowed_terms": sorted(self.allowed_terms),
            "forbidden_terms": sorted(self.forbidden_terms),
            "target_lectures": list(self.target_lectures),
            "is_relational": self.is_relational,
        }


# ---------------------------------------------------------------------------
# Term resolution
# ---------------------------------------------------------------------------


def _aliases_for_concept(concept_id: str, kb: ConceptKB) -> list[str]:
    """Lower-cased KB aliases for a single concept id, including the bare id.

    The bare id is included so a chunk mentioning ``hardmax`` matches even
    when the canonical KB name is ``"hardmax / winner-take-all"`` (mirrors
    :func:`entity_retrieval._entity_terms_for_aliases`).
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        if not raw:
            return
        term = str(raw).strip().lower()
        if len(term) < 2 or term in seen:
            return
        seen.add(term)
        out.append(term)

    _add(concept_id)
    meta = kb.get_concept_by_id(concept_id)
    if meta:
        _add(meta.name)
        for alias in meta.aliases:
            _add(alias)
    return out


def build_concept_constraints(sq: StructuredQuery, kb: ConceptKB | None = None) -> ConceptConstraints:
    """Derive a :class:`ConceptConstraints` from a structured query.

    - ``target_concepts`` mirrors ``sq.concept_ids``.
    - ``target_aliases`` is the union of aliases for every target id.
    - ``forbidden_terms`` is the union of
      :func:`forbidden_terms_for_concept` for each target (with the *other*
      targets passed as the ``peer_ids`` argument so the existing peer-name
      aggregation kicks in), minus anything already in ``target_aliases``.
    - ``target_lectures`` mirrors ``sq.lecture_scope``.
    - ``is_relational`` is true for compare / synthesis intents and for any
      query whose KB resolution produced ≥ 2 target concepts.
    """
    kb = kb or get_kb()
    targets = list(sq.concept_ids or [])

    target_aliases: set[str] = set()
    for cid in targets:
        target_aliases.update(_aliases_for_concept(cid, kb))

    forbidden: set[str] = set()
    for cid in targets:
        peers = [other for other in targets if other != cid]
        for term in forbidden_terms_for_concept(cid, peers, kb):
            t = term.strip().lower()
            if t and t not in target_aliases:
                forbidden.add(t)

    is_relational = (
        sq.intent.query_type in {QueryType.COMPARE, QueryType.SYNTHESIS}
        or sq.answer_intent in {"compare", "compare_multi", "cross_lecture_synthesis"}
        or len(targets) >= 2
    )

    return ConceptConstraints(
        target_concepts=targets,
        target_aliases=target_aliases,
        allowed_terms=set(target_aliases),
        forbidden_terms=forbidden,
        target_lectures=list(sq.lecture_scope or []),
        is_relational=is_relational,
    )


# ---------------------------------------------------------------------------
# Chunk-level scoring (post-retrieval rerank)
# ---------------------------------------------------------------------------


def _hit_count(blob: str, terms: set[str] | list[str]) -> int:
    """Distinct count of whole-word / phrase hits in ``blob`` for any term in ``terms``."""
    seen: set[str] = set()
    for raw in terms:
        term = raw.strip().lower()
        if not term or term in seen:
            continue
        if _term_hits(blob, term) > 0:
            seen.add(term)
    return len(seen)


def _topic_blob(chunk: dict[str, Any]) -> str:
    """Lower-cased ``topic`` + ``keywords`` text; high-signal fields only."""
    parts = [str(chunk.get("topic", "")), str(chunk.get("keywords", ""))]
    return " ".join(parts).lower()


def _body_blob(chunk: dict[str, Any]) -> str:
    """Lower-cased ``clean_explanation`` + ``source_excerpt`` body text."""
    parts = [
        str(chunk.get("clean_explanation", "")),
        str(chunk.get("source_excerpt", "")),
    ]
    return " ".join(parts).lower()


def score_chunk_against_constraints(
    chunk: dict[str, Any], constraints: ConceptConstraints
) -> float:
    """Additive delta applied on top of the chunk's original rank index.

    Positive delta moves a chunk up; negative delta moves it down. The exact
    weights are tuned so the most common cases land sensibly:

    - A target alias in the chunk's ``topic`` / ``keywords`` (+1.5 each) wins
      against an off-lecture penalty (-0.6).
    - One stray forbidden mention in the body (-0.4) doesn't bury an
      otherwise on-topic chunk.
    - Two distinct forbidden hits in ``topic`` (-2.4) push an off-topic chunk
      past the hard-drop threshold for non-relational queries.
    """
    if not constraints.target_aliases and not constraints.forbidden_terms:
        return 0.0

    topic = _topic_blob(chunk)
    body = _body_blob(chunk)

    alias_topic_hits = _hit_count(topic, constraints.target_aliases)
    alias_body_hits = _hit_count(body, constraints.target_aliases)
    forbidden_topic_hits = _hit_count(topic, constraints.forbidden_terms)
    forbidden_body_hits = _hit_count(body, constraints.forbidden_terms)

    delta = 0.0
    delta += _ALIAS_HIT_TOPIC_WEIGHT * alias_topic_hits
    delta += min(_ALIAS_HIT_BODY_WEIGHT * alias_body_hits, _ALIAS_HIT_BODY_CAP)
    delta -= _FORBIDDEN_HIT_TOPIC_WEIGHT * forbidden_topic_hits
    delta -= min(_FORBIDDEN_HIT_BODY_WEIGHT * forbidden_body_hits, _FORBIDDEN_HIT_BODY_CAP)

    lecture_number = chunk.get("lecture_number")
    if constraints.target_lectures:
        if isinstance(lecture_number, int) and lecture_number in constraints.target_lectures:
            delta += _LECTURE_MATCH_BONUS
        elif lecture_number is not None and (alias_topic_hits + alias_body_hits) == 0:
            delta -= _OFF_LECTURE_PENALTY

    return delta


def apply_concept_constraints(
    chunks: list[dict[str, Any]],
    constraints: ConceptConstraints,
) -> list[dict[str, Any]]:
    """Rerank (and optionally hard-drop) chunks using ``constraints``.

    Reordering is stable for chunks with identical deltas — we sort by
    ``(-delta, original_index)``. Hard-drops only fire when the spec's
    "obvious leak" condition is met:

    - Delta is at or below ``_HARD_DROP_DELTA``,
    - No target alias hit anywhere on the chunk, and
    - The query is *not* relational.

    Anything else stays in the pool, just demoted.
    """
    if not chunks or not constraints.target_concepts:
        return list(chunks)

    scored: list[tuple[float, int, dict[str, Any], int, int]] = []
    for idx, chunk in enumerate(chunks):
        delta = score_chunk_against_constraints(chunk, constraints)
        topic = _topic_blob(chunk)
        body = _body_blob(chunk)
        alias_total = _hit_count(topic, constraints.target_aliases) + _hit_count(
            body, constraints.target_aliases
        )
        forbidden_total = _hit_count(topic, constraints.forbidden_terms) + _hit_count(
            body, constraints.forbidden_terms
        )
        scored.append((delta, idx, chunk, alias_total, forbidden_total))

    survivors: list[tuple[float, int, dict[str, Any]]] = []
    for delta, idx, chunk, alias_total, _forbidden_total in scored:
        if (
            delta <= _HARD_DROP_DELTA
            and alias_total == 0
            and not constraints.is_relational
        ):
            continue
        survivors.append((delta, idx, chunk))

    survivors.sort(key=lambda t: (-t[0], t[1]))
    return [c for _delta, _idx, c in survivors]


# ---------------------------------------------------------------------------
# Line-level helpers (evidence selection, validators, direct-answer)
# ---------------------------------------------------------------------------


def _line_lower(line: str) -> str:
    return (line or "").strip().lower()


def line_has_target(line: str, constraints: ConceptConstraints) -> bool:
    """True when ``line`` mentions at least one target alias (whole-word match)."""
    if not constraints.target_aliases:
        return False
    text = _line_lower(line)
    if not text:
        return False
    for term in constraints.target_aliases:
        if _term_hits(text, term) > 0:
            return True
    return False


def line_has_forbidden(line: str, constraints: ConceptConstraints) -> bool:
    """True when ``line`` mentions at least one forbidden term."""
    if not constraints.forbidden_terms:
        return False
    text = _line_lower(line)
    if not text:
        return False
    for term in constraints.forbidden_terms:
        if _term_hits(text, term) > 0:
            return True
    return False


def is_line_concept_pure(line: str, constraints: ConceptConstraints) -> bool:
    """Whether a single sentence / bullet should survive the purity gate.

    Pure semantics:

    - Empty / whitespace lines are not pure (skip them).
    - For relational queries, the only requirement is the absence of a
      forbidden term that *isn't* one of the targets. A sentence about
      "how dynamic programming relates to backpropagation" mentions
      ``backpropagation`` legitimately because backpropagation is one of the
      targets — not a forbidden term.
    - For non-relational queries, a forbidden hit is fatal *unless* the line
      also mentions a target alias (mixed sentences are returned to the
      caller; the validator decides whether to flag them as ambiguous).
    """
    text = _line_lower(line)
    if not text:
        return False
    has_target = line_has_target(line, constraints)
    has_forbidden = line_has_forbidden(line, constraints)
    if constraints.is_relational:
        return True
    if has_forbidden and not has_target:
        return False
    return True


# ---------------------------------------------------------------------------
# Tiny utility surfaced for tests / direct-answer ranking
# ---------------------------------------------------------------------------


_DEFINITION_CUE_RE = re.compile(
    r"\b(is\s+(?:a|an|the)\b|are\s+(?:a|the)\b|refers?\s+to\b|"
    r"defined?\s+as\b|denotes?\b|denoted\s+by\b|the\s+goal\s+of\b|"
    r"means?\b|stand[s]?\s+for\b)",
    re.IGNORECASE,
)


def has_definition_cue(line: str) -> bool:
    """Lightweight signal that ``line`` looks like a definition statement."""
    return bool(_DEFINITION_CUE_RE.search(line or ""))
