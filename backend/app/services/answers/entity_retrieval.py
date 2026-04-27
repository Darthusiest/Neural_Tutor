"""Entity-aware chunk scoring, purity, and per-concept evidence bundles.

Used to reduce cross-topic contamination before planning and generation.

This module exposes two evidence-bundle dataclasses:

- :class:`ConceptEvidenceBundle` — legacy, minimal bundle (concept id + label
  + chunk ids + support score + gap flags). Still used by simple call sites
  and existing tests; constructed positionally as
  ``ConceptEvidenceBundle("cnn", "CNN", 1.0, [1])``.
- :class:`ConceptEvidenceBundleV2` — canonical bundle for the second-pass
  compare pipeline. Carries the full chunk dicts (``evidence_chunks``),
  pre-extracted ``core_lines`` (entity-pure after forbidden + cross-entity
  filtering), ``shared_lines`` (lines that scored well for both entities in a
  two-way compare), ``forbidden_hits`` (terms that triggered a drop, useful
  for debugging and gap reporting), ``aliases``, and per-bundle
  ``confidence`` derived from ``support_score``. V2 also exposes legacy
  property aliases (``concept_id`` / ``label`` / ``chunk_ids`` /
  ``gap_flags``) and ``from_legacy_bundle`` / ``to_legacy_bundle`` adapters
  so it can be substituted for the legacy type wherever existing code reads
  those four fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from flask import has_app_context

from app.services.knowledge.concept_kb import ConceptKB, ConceptMeta, get_kb

# Penalty terms when the *primary* concept should not borrow another topic's machinery.
# Keys are concept ids from LING487_STRUCTURED_PIPELINE_KB.json.
_DEFAULT_FORBIDDEN_BY_CONCEPT: dict[str, list[str]] = {
    "mfcc": [
        "softmax",
        "transformer",
        "attention",
        "convolution",
        "gradient descent",
        "backprop",
    ],
    "formants": [
        "cepstr",
        "mfcc",
        "dct",
        "filterbank",
    ],
    "cnn": [
        "transformer",
        "self-attention",
        "multi-head",
        "positional encoding",
        "residual connection",
        "layer norm",
    ],
    "transformer": [
        "convolutional",
        "kernel",
        "local receptive",
    ],
    "dynamic_programming": [
        "neural network",
        "backpropagation",
        "sgd",
        "transformer",
        "convolution",
    ],
    "bias_variance": [
        "dropout",
        "l2 regularization",
    ],
}

# Generic NN boilerplate that often leaks into unrelated answers
_GENERIC_NN_FILLER = [
    "neural network",
    "hidden layer",
    "gradient",
    "backprop",
    "training data",
    "epochs",
]


def forbidden_terms_for_concept(concept_id: str, peer_ids: list[str], kb: ConceptKB) -> list[str]:
    terms = list(_DEFAULT_FORBIDDEN_BY_CONCEPT.get(concept_id, []))
    for pid in peer_ids:
        if pid == concept_id:
            continue
        meta = kb.get_concept_by_id(pid)
        if meta:
            terms.append(meta.name.lower())
            terms.extend(a.lower() for a in meta.aliases[:6] if len(a) > 2)
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        tl = t.lower().strip()
        if tl and tl not in seen:
            seen.add(tl)
            out.append(tl)
    return out


def chunk_text_blob(chunk: dict[str, Any]) -> str:
    parts = [
        str(chunk.get("topic", "")),
        str(chunk.get("keywords", "")),
        str(chunk.get("clean_explanation", "")),
        str(chunk.get("source_excerpt", "")),
    ]
    return " ".join(parts).lower()


def _term_hits(blob: str, term: str) -> float:
    if len(term) < 2:
        return 0.0
    if " " in term:
        return float(term in blob)
    # word-ish match
    return float(len(re.findall(r"\b" + re.escape(term) + r"\b", blob)))


def score_chunk_for_entity(
    chunk: dict[str, Any],
    concept_id: str,
    kb: ConceptKB,
    *,
    peer_concept_ids: list[str],
) -> tuple[float, dict[str, float]]:
    """Returns (fused_score, debug_parts). Higher is better."""
    meta = kb.get_concept_by_id(concept_id)
    blob = chunk_text_blob(chunk)
    if not meta:
        return 0.05, {"entity": 0.0, "cross": 0.0, "neg": 0.0, "purity": 0.0}

    entity_score = 0.0
    for term in [meta.name, *meta.aliases[:10]]:
        t = term.lower().strip()
        if len(t) < 2:
            continue
        entity_score += 1.2 * _term_hits(blob, t)

    cross = 0.0
    for pid in peer_concept_ids:
        if pid == concept_id:
            continue
        om = kb.get_concept_by_id(pid)
        if not om:
            continue
        for term in [om.name, *om.aliases[:6]]:
            t = term.lower().strip()
            if len(t) > 2:
                cross += _term_hits(blob, t)

    forbidden = forbidden_terms_for_concept(concept_id, peer_concept_ids, kb)
    neg = 0.0
    for term in forbidden:
        neg += 2.0 * _term_hits(blob, term)

    eps = 1e-6
    purity = entity_score / (entity_score + cross + neg + eps)
    fused = entity_score - 0.45 * cross - 1.1 * neg + 0.15 * purity
    return fused, {
        "entity": entity_score,
        "cross": cross,
        "neg": neg,
        "purity": purity,
    }


@dataclass
class ConceptEvidenceBundle:
    concept_id: str
    label: str
    support_score: float
    chunk_ids: list[int] = field(default_factory=list)
    gap_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "label": self.label,
            "support_score": self.support_score,
            "chunk_ids": list(self.chunk_ids),
            "gap_flags": list(self.gap_flags),
        }


# Reference support score used to map ``support_score`` -> ``confidence``.
# Tuned so a single strong on-topic chunk (entity_score ~ 4-5 from name +
# aliases) clears 1.0; values above 1.0 get clamped.
_BUNDLE_SUPPORT_REFERENCE = 4.0


@dataclass
class ConceptEvidenceBundleV2:
    """Canonical evidence bundle for the V2 compare pipeline.

    Carries full chunks plus pre-extracted ``core_lines`` / ``shared_lines``
    so the renderer doesn't have to recompute scoping. Exposes the legacy
    field names (``concept_id`` / ``label`` / ``chunk_ids`` / ``gap_flags``)
    as read-only properties so existing call sites that iterate over
    ``AnswerPlan.evidence_bundles`` keep working when the planner upgrades
    to V2.
    """

    concept: str
    aliases: list[str] = field(default_factory=list)
    evidence_chunks: list[dict[str, Any]] = field(default_factory=list)
    core_lines: list[str] = field(default_factory=list)
    support_score: float = 0.0
    forbidden_hits: list[str] = field(default_factory=list)
    shared_lines: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    label_override: str | None = None
    gap_flags_override: list[str] | None = None

    # ------------------------------------------------------------------
    # Legacy compatibility surface
    # ------------------------------------------------------------------

    @property
    def concept_id(self) -> str:
        """Alias for :attr:`concept` — keeps legacy ``bundle.concept_id`` accessors working."""
        return self.concept

    @property
    def label(self) -> str:
        """Display label for the entity (overridden via ``label_override`` for tests / non-KB ids)."""
        if self.label_override:
            return self.label_override
        return self.concept

    @property
    def chunk_ids(self) -> list[int]:
        """Chunk ids drawn from :attr:`evidence_chunks` (preserves bundle order)."""
        out: list[int] = []
        for chunk in self.evidence_chunks:
            cid = chunk.get("id") if isinstance(chunk, dict) else None
            if cid is None:
                continue
            try:
                out.append(int(cid))
            except (TypeError, ValueError):
                continue
        return out

    @property
    def gap_flags(self) -> list[str]:
        """Gap flags derived from support level; explicit override wins when provided."""
        if self.gap_flags_override is not None:
            return list(self.gap_flags_override)
        return ["low_support"] if self.support_score < 0.15 else []

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable snapshot (full V2 surface — useful for diagnostics / payloads)."""
        return {
            "concept": self.concept,
            "concept_id": self.concept_id,
            "label": self.label,
            "aliases": list(self.aliases),
            "evidence_chunk_ids": list(self.chunk_ids),
            "core_lines": list(self.core_lines),
            "support_score": self.support_score,
            "forbidden_hits": list(self.forbidden_hits),
            "shared_lines": list(self.shared_lines),
            "source_metadata": dict(self.source_metadata),
            "confidence": self.confidence,
            "gap_flags": list(self.gap_flags),
        }

    @classmethod
    def from_legacy_bundle(
        cls,
        legacy: ConceptEvidenceBundle,
        *,
        kb: ConceptKB | None = None,
        evidence_chunks: list[dict[str, Any]] | None = None,
        core_lines: list[str] | None = None,
        shared_lines: list[str] | None = None,
        forbidden_hits: list[str] | None = None,
    ) -> "ConceptEvidenceBundleV2":
        """Wrap a legacy bundle in the V2 envelope.

        Aliases are pulled from the KB if available; otherwise an empty list
        (and :attr:`label_override` ensures the bundle still renders the
        legacy human label even when ``concept`` is a non-KB string).
        """
        aliases: list[str] = []
        if kb is not None:
            meta = kb.get_concept_by_id(legacy.concept_id)
            if meta:
                aliases = list(meta.aliases)
        confidence = _bundle_confidence(legacy.support_score)
        return cls(
            concept=legacy.concept_id,
            aliases=aliases,
            evidence_chunks=list(evidence_chunks or []),
            core_lines=list(core_lines or []),
            support_score=legacy.support_score,
            forbidden_hits=list(forbidden_hits or []),
            shared_lines=list(shared_lines or []),
            source_metadata={},
            confidence=confidence,
            label_override=legacy.label if legacy.label != legacy.concept_id else None,
            gap_flags_override=list(legacy.gap_flags) if legacy.gap_flags else None,
        )

    def to_legacy_bundle(self) -> ConceptEvidenceBundle:
        """Return the legacy 4-field bundle equivalent for back-compat call sites."""
        return ConceptEvidenceBundle(
            concept_id=self.concept_id,
            label=self.label,
            support_score=self.support_score,
            chunk_ids=self.chunk_ids,
            gap_flags=self.gap_flags,
        )


# Bundle types that expose ``concept_id`` / ``label`` / ``chunk_ids`` /
# ``gap_flags``. Functions and dataclasses that accept *either* type use this
# alias for type hints.
EvidenceBundleLike = ConceptEvidenceBundle | ConceptEvidenceBundleV2


def _bundle_confidence(support_score: float) -> float:
    """Map an absolute support score onto ``[0.0, 1.0]`` for display."""
    if support_score <= 0:
        return 0.0
    return min(1.0, support_score / _BUNDLE_SUPPORT_REFERENCE)


def _entity_enabled() -> bool:
    if not has_app_context():
        return True
    from flask import current_app

    return bool(current_app.config.get("ENTITY_EVIDENCE_SCORING_ENABLED", True))


def rerank_chunks_for_concepts(
    chunks: list[dict[str, Any]],
    concept_ids: list[str],
    kb: ConceptKB | None = None,
) -> list[dict[str, Any]]:
    """Order chunks by fused entity score for the first concept (definition / primary)."""
    if not _entity_enabled() or not chunks or not concept_ids:
        return chunks
    kb = kb or get_kb()
    primary = concept_ids[0]
    peers = concept_ids[1:]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for c in chunks:
        cid = c.get("id")
        if cid is None:
            continue
        s, _ = score_chunk_for_entity(c, primary, kb, peer_concept_ids=peers)
        # No weak fallback: zero or negative scores still sort to bottom
        scored.append((s, int(cid), c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t[2] for t in scored]


def build_bundles_for_compare(
    chunks: list[dict[str, Any]],
    ca_id: str,
    cb_id: str,
    kb: ConceptKB | None = None,
    *,
    top_per_side: int = 4,
    min_support: float = 0.15,
) -> tuple[ConceptEvidenceBundle, ConceptEvidenceBundle]:
    """Split merged retrieval into two ranked pools with gap flags."""
    kb = kb or get_kb()
    a_meta = kb.get_concept_by_id(ca_id)
    b_meta = kb.get_concept_by_id(cb_id)
    label_a = a_meta.name if a_meta else ca_id
    label_b = b_meta.name if b_meta else cb_id

    scored_a: list[tuple[float, dict[str, Any]]] = []
    scored_b: list[tuple[float, dict[str, Any]]] = []
    for c in chunks:
        sa, _ = score_chunk_for_entity(c, ca_id, kb, peer_concept_ids=[cb_id])
        sb, _ = score_chunk_for_entity(c, cb_id, kb, peer_concept_ids=[ca_id])
        scored_a.append((sa, c))
        scored_b.append((sb, c))

    scored_a.sort(key=lambda x: -x[0])
    scored_b.sort(key=lambda x: -x[0])

    def _take_ids(pairs: list[tuple[float, dict[str, Any]]]) -> tuple[list[int], float]:
        seen: set[int] = set()
        ids: list[int] = []
        best = 0.0
        for s, ch in pairs:
            cid = ch.get("id")
            if cid is None or cid in seen:
                continue
            seen.add(int(cid))
            ids.append(int(cid))
            best = max(best, s)
            if len(ids) >= top_per_side:
                break
        return ids, best

    ids_a, sup_a = _take_ids(scored_a)
    ids_b, sup_b = _take_ids(scored_b)

    gap_a: list[str] = []
    gap_b: list[str] = []
    if sup_a < min_support:
        gap_a.append("low_support")
    if sup_b < min_support:
        gap_b.append("low_support")

    return (
        ConceptEvidenceBundle(
            concept_id=ca_id,
            label=label_a,
            support_score=sup_a,
            chunk_ids=ids_a,
            gap_flags=gap_a,
        ),
        ConceptEvidenceBundle(
            concept_id=cb_id,
            label=label_b,
            support_score=sup_b,
            chunk_ids=ids_b,
            gap_flags=gap_b,
        ),
    )


def build_bundles_multi(
    chunks: list[dict[str, Any]],
    concept_ids: list[str],
    kb: ConceptKB | None = None,
    *,
    top_per_entity: int = 3,
) -> list[ConceptEvidenceBundle]:
    """One bundle per entity; peers are all other ids in the compare set."""
    kb = kb or get_kb()
    out: list[ConceptEvidenceBundle] = []
    for eid in concept_ids:
        peers = [x for x in concept_ids if x != eid]
        scored: list[tuple[float, dict[str, Any]]] = []
        for c in chunks:
            s, _ = score_chunk_for_entity(c, eid, kb, peer_concept_ids=peers)
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        meta = kb.get_concept_by_id(eid)
        label = meta.name if meta else eid
        ids: list[int] = []
        best = 0.0
        seen: set[int] = set()
        for s, ch in scored:
            cid = ch.get("id")
            if cid is None or cid in seen:
                continue
            seen.add(int(cid))
            ids.append(int(cid))
            best = max(best, s)
            if len(ids) >= top_per_entity:
                break
        gap: list[str] = []
        if best < 0.15:
            gap.append("low_support")
        out.append(
            ConceptEvidenceBundle(
                concept_id=eid,
                label=label,
                support_score=best,
                chunk_ids=ids,
                gap_flags=gap,
            )
        )
    return out


def generic_nn_filler_score(text: str) -> float:
    """Ratio of generic NN filler hits to length (rough heuristic)."""
    tl = text.lower()
    hits = sum(1 for g in _GENERIC_NN_FILLER if g in tl)
    words = max(len(tl.split()), 1)
    return hits / max(words / 20.0, 1.0)


# ---------------------------------------------------------------------------
# V2 bundle builders (entity-separated, per-line cross-entity filtering)
# ---------------------------------------------------------------------------

# Default thresholds for V2 line classification. ``shared_min_ratio`` keeps a
# line in the shared bucket only when the smaller side's term-hit count is at
# least 60% of the dominant side — i.e. both entities are clearly present, not
# just one with the other glanced incidentally.
_V2_SHARED_MIN_RATIO = 0.6


def _entity_terms_for_aliases(
    label: str, aliases: list[str], *, concept_id: str | None = None
) -> list[str]:
    """Lowercased term set for cross-entity matching.

    Includes (in priority order):

    - ``concept_id`` itself (so a line containing the bare KB id like
      ``hardmax`` matches even when the canonical KB ``label`` is something
      like ``"hardmax / winner-take-all"``)
    - the canonical ``label``
    - each alias

    Multi-word phrases are also split into their content words so a line
    talking about ``probability distribution`` still matches the alias
    ``probability distribution from logits``.
    """
    seen: set[str] = set()
    out: list[str] = []

    def _add(raw: str | None) -> None:
        if not raw:
            return
        term = str(raw).strip().lower()
        if len(term) < 2 or term in seen:
            return
        seen.add(term)
        out.append(term)

    _add(concept_id)
    _add(label)
    for alias in aliases:
        _add(alias)

    # Split punctuation-heavy aliases (e.g. "hardmax / winner-take-all") so
    # individual content words still match. Tokens shorter than 3 chars are
    # dropped to avoid spurious matches like "of" / "to".
    extra_tokens: list[str] = []
    for raw in [label, *aliases]:
        if not raw:
            continue
        for tok in re.split(r"[^a-z0-9-]+", str(raw).lower()):
            tok = tok.strip("-_/.")
            if len(tok) >= 3 and tok not in seen:
                extra_tokens.append(tok)
                seen.add(tok)
    out.extend(extra_tokens)
    return out


def _entity_terms_from_kb(concept_id: str, kb: ConceptKB | None) -> tuple[str, list[str]]:
    if kb is None:
        return concept_id, [concept_id]
    meta = kb.get_concept_by_id(concept_id)
    if not meta:
        return concept_id, [concept_id]
    return meta.name, _entity_terms_for_aliases(meta.name, list(meta.aliases))


def classify_line_for_compare(
    line: str,
    *,
    entity_a_terms: list[str],
    entity_b_terms: list[str],
    forbidden_a: list[str] = (),
    forbidden_b: list[str] = (),
    shared_min_ratio: float = _V2_SHARED_MIN_RATIO,
) -> tuple[str, dict[str, float]]:
    """Classify a single line against the term sets for two compare-side entities.

    Returns ``(label, debug)`` where ``label`` is one of:

    - ``"a"`` — line belongs to side A's core_lines (drops forbidden / B-dominant)
    - ``"b"`` — line belongs to side B's core_lines
    - ``"shared"`` — line scored well for both A and B; goes into shared_lines
    - ``"forbidden_a"`` — A-side forbidden term hit (recorded in forbidden_hits)
    - ``"forbidden_b"`` — B-side forbidden term hit
    - ``"skip"`` — neither side scored above threshold (no signal)
    """
    line_lower = (line or "").lower()
    if not line_lower.strip():
        return "skip", {"a": 0.0, "b": 0.0}

    a_score = sum(_term_hits(line_lower, t) for t in entity_a_terms)
    b_score = sum(_term_hits(line_lower, t) for t in entity_b_terms)
    a_forbidden_hit = any(_term_hits(line_lower, t) > 0 for t in forbidden_a if t)
    b_forbidden_hit = any(_term_hits(line_lower, t) > 0 for t in forbidden_b if t)

    debug = {"a": a_score, "b": b_score}

    # Shared evidence beats individual sides when both terms are present in
    # similar measure (and the line isn't blocked by a forbidden term on
    # either side — a shared line has to be safe for both bundles).
    if (
        a_score >= 1.0
        and b_score >= 1.0
        and not a_forbidden_hit
        and not b_forbidden_hit
    ):
        small, large = sorted([a_score, b_score])
        if large > 0 and (small / large) >= shared_min_ratio:
            return "shared", debug

    if a_score >= 1.0 and a_score > b_score and not a_forbidden_hit:
        return "a", debug
    if b_score >= 1.0 and b_score > a_score and not b_forbidden_hit:
        return "b", debug

    if a_forbidden_hit and a_score > 0:
        return "forbidden_a", debug
    if b_forbidden_hit and b_score > 0:
        return "forbidden_b", debug

    return "skip", debug


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it.strip())
    return out


def _source_metadata_for_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    lectures: set[int] = set()
    topic_heads: list[str] = []
    seen_topics: set[str] = set()
    for c in chunks:
        ln = c.get("lecture_number")
        if isinstance(ln, int):
            lectures.add(ln)
        topic = str(c.get("topic", "")).strip()
        if topic:
            head = re.split(r"\s*[—\-:|]\s*", topic, maxsplit=1)[0].strip()
            if head and head.lower() not in seen_topics:
                seen_topics.add(head.lower())
                topic_heads.append(head)
    return {"lectures": sorted(lectures), "topic_heads": topic_heads}


def _take_top_chunks(
    scored: list[tuple[float, dict[str, Any]]], top_k: int
) -> tuple[list[dict[str, Any]], float]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    best = 0.0
    for s, ch in scored:
        cid = ch.get("id")
        if cid is None:
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        if cid_int in seen:
            continue
        seen.add(cid_int)
        out.append(ch)
        if s > best:
            best = s
        if len(out) >= top_k:
            break
    return out, best


def _build_v2_lines(
    candidate_chunks: list[dict[str, Any]],
    *,
    entity_a_terms: list[str],
    entity_b_terms: list[str],
    forbidden_a: list[str],
    forbidden_b: list[str],
    side: str,
    shared_min_ratio: float,
    max_lines: int,
) -> tuple[list[str], list[str], list[str]]:
    """Walk units from one side's chunks and classify each line.

    Returns ``(core_lines, shared_lines, forbidden_hits)`` where ``core_lines``
    are pure-side lines (label matches ``side``), ``shared_lines`` are lines
    that scored well for both entities (recorded against both bundles by the
    caller), and ``forbidden_hits`` are the literal forbidden terms that
    fired for this side (debug / gap diagnostics).
    """
    from app.services.answers.compare_evidence import chunks_to_raw_units

    units = chunks_to_raw_units(candidate_chunks)
    core: list[str] = []
    shared: list[str] = []
    forbidden_hits: list[str] = []

    side_forbidden = forbidden_a if side == "a" else forbidden_b

    for unit in units:
        label, _ = classify_line_for_compare(
            unit,
            entity_a_terms=entity_a_terms,
            entity_b_terms=entity_b_terms,
            forbidden_a=forbidden_a,
            forbidden_b=forbidden_b,
            shared_min_ratio=shared_min_ratio,
        )
        if label == side:
            core.append(unit)
        elif label == "shared":
            shared.append(unit)
        elif label == f"forbidden_{side}":
            unit_lower = unit.lower()
            for term in side_forbidden:
                if term and _term_hits(unit_lower, term) > 0:
                    forbidden_hits.append(term)
                    break
    return (
        _dedupe_strings(core)[:max_lines],
        _dedupe_strings(shared),
        _dedupe_strings(forbidden_hits),
    )


def build_bundles_for_compare_v2(
    chunks: list[dict[str, Any]],
    ca_id: str,
    cb_id: str,
    kb: ConceptKB | None = None,
    *,
    top_per_side: int = 4,
    min_support: float = 0.15,
    aliases_override: dict[str, list[str]] | None = None,
    label_override: dict[str, str] | None = None,
    shared_min_ratio: float = _V2_SHARED_MIN_RATIO,
    max_core_lines: int = 8,
) -> tuple[ConceptEvidenceBundleV2, ConceptEvidenceBundleV2]:
    """Two-way V2 evidence bundles with per-line cross-entity filtering.

    The pipeline matches the spec:

    1. Score every chunk twice — once for A (with B as peer), once for B
       (with A as peer) — using the existing :func:`score_chunk_for_entity`.
    2. Pick the top ``top_per_side`` chunks per side; promote the actual
       chunk dicts onto :attr:`ConceptEvidenceBundleV2.evidence_chunks`.
    3. Walk the units from each side's chunks and classify each line with
       :func:`classify_line_for_compare`. Lines tagged with the *other*
       side's label are dropped from this side; lines tagged ``shared`` end
       up in both bundles' :attr:`shared_lines`.

    ``aliases_override`` / ``label_override`` let tests construct bundles
    for entities that don't exist in the KB (e.g. ``bias`` and ``variance``
    as separate entities) without polluting the production KB.
    """
    kb = kb or get_kb()
    aliases_override = aliases_override or {}
    label_override = label_override or {}

    # Resolve aliases / labels for both sides — KB first, override when given.
    a_meta = kb.get_concept_by_id(ca_id)
    b_meta = kb.get_concept_by_id(cb_id)
    a_label = label_override.get(ca_id) or (a_meta.name if a_meta else ca_id)
    b_label = label_override.get(cb_id) or (b_meta.name if b_meta else cb_id)

    a_aliases = aliases_override.get(ca_id, list(a_meta.aliases) if a_meta else [])
    b_aliases = aliases_override.get(cb_id, list(b_meta.aliases) if b_meta else [])

    entity_a_terms = _entity_terms_for_aliases(a_label, a_aliases, concept_id=ca_id)
    entity_b_terms = _entity_terms_for_aliases(b_label, b_aliases, concept_id=cb_id)

    # Forbidden terms come from the KB-aware list when available; for
    # off-KB entities, fall back to "the other side's terms" so a line
    # purely about B is still rejected from A.
    if a_meta:
        forbidden_a = forbidden_terms_for_concept(ca_id, [cb_id], kb)
    else:
        forbidden_a = list(entity_b_terms)
    if b_meta:
        forbidden_b = forbidden_terms_for_concept(cb_id, [ca_id], kb)
    else:
        forbidden_b = list(entity_a_terms)

    # Per-chunk scoring (identical to legacy builder so existing tuning
    # carries through).
    scored_a: list[tuple[float, dict[str, Any]]] = []
    scored_b: list[tuple[float, dict[str, Any]]] = []
    for c in chunks:
        if a_meta:
            sa, _ = score_chunk_for_entity(c, ca_id, kb, peer_concept_ids=[cb_id])
        else:
            blob = chunk_text_blob(c)
            sa = sum(_term_hits(blob, t) for t in entity_a_terms) - 0.45 * sum(
                _term_hits(blob, t) for t in entity_b_terms
            )
        if b_meta:
            sb, _ = score_chunk_for_entity(c, cb_id, kb, peer_concept_ids=[ca_id])
        else:
            blob = chunk_text_blob(c)
            sb = sum(_term_hits(blob, t) for t in entity_b_terms) - 0.45 * sum(
                _term_hits(blob, t) for t in entity_a_terms
            )
        scored_a.append((sa, c))
        scored_b.append((sb, c))

    scored_a.sort(key=lambda x: -x[0])
    scored_b.sort(key=lambda x: -x[0])

    chunks_a, support_a = _take_top_chunks(scored_a, top_per_side)
    chunks_b, support_b = _take_top_chunks(scored_b, top_per_side)

    # Per-line classification on each side's chunks. Shared lines from both
    # sides are merged so the renderer sees a single shared bucket.
    core_a, shared_a, forbidden_hits_a = _build_v2_lines(
        chunks_a,
        entity_a_terms=entity_a_terms,
        entity_b_terms=entity_b_terms,
        forbidden_a=forbidden_a,
        forbidden_b=forbidden_b,
        side="a",
        shared_min_ratio=shared_min_ratio,
        max_lines=max_core_lines,
    )
    core_b, shared_b, forbidden_hits_b = _build_v2_lines(
        chunks_b,
        entity_a_terms=entity_a_terms,
        entity_b_terms=entity_b_terms,
        forbidden_a=forbidden_a,
        forbidden_b=forbidden_b,
        side="b",
        shared_min_ratio=shared_min_ratio,
        max_lines=max_core_lines,
    )
    shared_combined = _dedupe_strings([*shared_a, *shared_b])

    gap_a = ["low_support"] if support_a < min_support else None
    gap_b = ["low_support"] if support_b < min_support else None

    bundle_a = ConceptEvidenceBundleV2(
        concept=ca_id,
        aliases=a_aliases,
        evidence_chunks=chunks_a,
        core_lines=core_a,
        support_score=support_a,
        forbidden_hits=forbidden_hits_a,
        shared_lines=list(shared_combined),
        source_metadata=_source_metadata_for_chunks(chunks_a),
        confidence=_bundle_confidence(support_a),
        label_override=a_label if a_label != ca_id else None,
        gap_flags_override=gap_a,
    )
    bundle_b = ConceptEvidenceBundleV2(
        concept=cb_id,
        aliases=b_aliases,
        evidence_chunks=chunks_b,
        core_lines=core_b,
        support_score=support_b,
        forbidden_hits=forbidden_hits_b,
        shared_lines=list(shared_combined),
        source_metadata=_source_metadata_for_chunks(chunks_b),
        confidence=_bundle_confidence(support_b),
        label_override=b_label if b_label != cb_id else None,
        gap_flags_override=gap_b,
    )
    return bundle_a, bundle_b


def build_bundles_multi_v2(
    chunks: list[dict[str, Any]],
    concept_ids: list[str],
    kb: ConceptKB | None = None,
    *,
    top_per_entity: int = 3,
    min_support: float = 0.15,
) -> list[ConceptEvidenceBundleV2]:
    """Multi-entity V2 bundles. Each entity uses every other entity as its peer set.

    Per-line shared classification only makes sense for two-way compare, so
    multi-entity bundles get :attr:`shared_lines` set to ``[]`` and use the
    same forbidden-term filter as the legacy builder. Cross-entity peer
    pressure still drops lines that look stronger on a peer's side.
    """
    kb = kb or get_kb()
    if len(concept_ids) == 2:
        a_id, b_id = concept_ids
        a, b = build_bundles_for_compare_v2(
            chunks,
            a_id,
            b_id,
            kb,
            top_per_side=top_per_entity,
            min_support=min_support,
        )
        return [a, b]

    out: list[ConceptEvidenceBundleV2] = []
    for eid in concept_ids:
        peers = [x for x in concept_ids if x != eid]
        scored: list[tuple[float, dict[str, Any]]] = []
        for c in chunks:
            s, _ = score_chunk_for_entity(c, eid, kb, peer_concept_ids=peers)
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        top_chunks, support = _take_top_chunks(scored, top_per_entity)

        meta = kb.get_concept_by_id(eid)
        label = meta.name if meta else eid
        aliases = list(meta.aliases) if meta else []
        entity_terms = _entity_terms_for_aliases(label, aliases, concept_id=eid)
        forbidden = forbidden_terms_for_concept(eid, peers, kb)
        peer_terms: list[str] = []
        for pid in peers:
            peer_meta = kb.get_concept_by_id(pid)
            if peer_meta:
                peer_terms.extend(
                    _entity_terms_for_aliases(peer_meta.name, peer_meta.aliases, concept_id=pid)
                )

        from app.services.answers.compare_evidence import chunks_to_raw_units

        core: list[str] = []
        forbidden_hits: list[str] = []
        for unit in chunks_to_raw_units(top_chunks):
            unit_lower = unit.lower()
            entity_score = sum(_term_hits(unit_lower, t) for t in entity_terms)
            peer_score = sum(_term_hits(unit_lower, t) for t in peer_terms)
            forbidden_match = next(
                (t for t in forbidden if t and _term_hits(unit_lower, t) > 0),
                None,
            )
            if forbidden_match:
                if entity_score > 0:
                    forbidden_hits.append(forbidden_match)
                continue
            if entity_score < 1.0:
                continue
            if peer_score >= entity_score:
                continue
            core.append(unit)

        gap = ["low_support"] if support < min_support else None
        out.append(
            ConceptEvidenceBundleV2(
                concept=eid,
                aliases=aliases,
                evidence_chunks=top_chunks,
                core_lines=_dedupe_strings(core)[: top_per_entity * 2],
                support_score=support,
                forbidden_hits=_dedupe_strings(forbidden_hits),
                shared_lines=[],
                source_metadata=_source_metadata_for_chunks(top_chunks),
                confidence=_bundle_confidence(support),
                label_override=label if label != eid else None,
                gap_flags_override=gap,
            )
        )
    return out
