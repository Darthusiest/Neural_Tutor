"""Entity-aware chunk scoring, purity, and per-concept evidence bundles.

Used to reduce cross-topic contamination before planning and generation.
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
