"""Structured answer plans: sections, primary vs supporting chunks, comparison axes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from flask import has_app_context

from app.services.answers.concept_constraints import ConceptConstraints
from app.services.answers.entity_retrieval import (
    ConceptEvidenceBundle,
    ConceptEvidenceBundleV2,
    EvidenceBundleLike,
    build_bundles_for_compare_v2,
    build_bundles_multi_v2,
    score_chunk_for_entity,
)
from app.services.knowledge.concept_kb import ConceptKB, get_kb
from app.services.knowledge.structured_query import StructuredQuery


@dataclass
class SectionSpec:
    """Narrow contract for one rendered section (generation / validation)."""

    section_id: str
    purpose: str
    entity_id: str | None
    source_chunk_ids: list[int]
    content_hint: str
    forbidden_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "purpose": self.purpose,
            "entity_id": self.entity_id,
            "source_chunk_ids": list(self.source_chunk_ids),
            "content_hint": self.content_hint,
            "forbidden_terms": list(self.forbidden_terms),
        }


@dataclass
class AnswerSection:
    heading: str
    chunk_ids: list[int]
    content_hint: str

    def to_dict(self) -> dict[str, Any]:
        return {"heading": self.heading, "chunk_ids": list(self.chunk_ids), "content_hint": self.content_hint}


@dataclass
class AnswerPlan:
    answer_mode: str
    sections: list[AnswerSection]
    primary_chunk_ids: list[int]
    supporting_chunk_ids: list[int]
    include_example: bool
    include_analogy: bool
    include_prerequisites: bool
    include_related_concepts: list[str]
    comparison_axes: list[str]
    lecture_scope: list[int]
    section_specs: list[SectionSpec] = field(default_factory=list)
    # Either V1 (legacy) or V2 bundles. V2 is canonical for compare mode going
    # forward; V1 still appears in older test fixtures and external callers.
    # Both expose ``concept_id`` / ``label`` / ``chunk_ids`` / ``gap_flags``.
    evidence_bundles: dict[str, EvidenceBundleLike] = field(default_factory=dict)
    # Deterministic, target-grounded opening sentence selected by
    # :func:`direct_answer.select_direct_answer`. ``None`` for summary / quiz /
    # synthesis paths (where the renderer doesn't open with a single
    # definition sentence). When set, the chat / definition renderers prefer
    # this string over the legacy "first bullet of the first chunk" heuristic.
    direct_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_mode": self.answer_mode,
            "sections": [s.to_dict() for s in self.sections],
            "primary_chunk_ids": list(self.primary_chunk_ids),
            "supporting_chunk_ids": list(self.supporting_chunk_ids),
            "include_example": self.include_example,
            "include_analogy": self.include_analogy,
            "include_prerequisites": self.include_prerequisites,
            "include_related_concepts": list(self.include_related_concepts),
            "comparison_axes": list(self.comparison_axes),
            "lecture_scope": list(self.lecture_scope),
            "section_specs": [s.to_dict() for s in self.section_specs],
            "evidence_bundles": {k: v.to_dict() for k, v in self.evidence_bundles.items()},
            "direct_answer": self.direct_answer,
        }


# Mirrors KB answer_plan_templates keys
ANSWER_PLAN_SECTION_LABELS: dict[str, list[tuple[str, str]]] = {
    "direct_definition": [
        ("Direct answer", "definition"),
        ("Mechanism", "definition"),
        ("Why it matters in this course", "definition"),
    ],
    "compare": [
        ("One-line distinction", "comparison_axis"),
        ("Concept A", "definition"),
        ("Concept B", "definition"),
        ("Direct comparison", "comparison_axis"),
        ("Why the difference matters", "connection"),
    ],
    "lecture_summary": [
        ("Lecture scope", "definition"),
        ("Core ideas", "definition"),
        ("How the ideas fit together", "connection"),
        ("Takeaway", "definition"),
    ],
    "cross_lecture_synthesis": [
        ("Per-lecture anchor", "definition"),
        ("Connecting thread", "connection"),
        ("Progression", "connection"),
        ("Big-picture takeaway", "connection"),
    ],
    "scoped_explanation": [
        ("Scoped answer", "definition"),
        ("Supporting detail", "definition"),
    ],
    "teaching_plus_check": [
        ("Answer", "definition"),
        ("Reasoning", "definition"),
    ],
    "simplified_reteach": [
        ("Plain-language answer", "definition"),
        ("Analogy or example", "example"),
    ],
    "multi_step_explanation": [
        ("Direct answer", "definition"),
        ("Step-by-step mechanism", "definition"),
        ("Purpose or motivation", "definition"),
        ("Example", "example"),
    ],
    "compare_multi": [
        ("Compared items", "definition"),
        ("Contrasts", "comparison_axis"),
        ("Why contrasts matter", "connection"),
    ],
}

# One primary chunk per section (ordered by relevance) — avoids repeating the same excerpt
# under every ### heading for definition-style plans.
_DISTINCT_CHUNK_PER_SECTION_MODES = frozenset({"direct_definition", "multi_step_explanation"})


def _chunk_blob(c: dict[str, Any]) -> str:
    parts = [
        str(c.get("topic", "")),
        str(c.get("keywords", "")),
        str(c.get("clean_explanation", ""))[:400],
    ]
    return " ".join(parts).lower()


def _score_chunk_for_concept(chunk: dict[str, Any], concept_id: str, kb: ConceptKB) -> float:
    cmeta = kb.get_concept_by_id(concept_id)
    if not cmeta:
        return 0.0
    blob = _chunk_blob(chunk)
    score = 0.0
    for term in [cmeta.name, *cmeta.aliases[:8]]:
        t = term.lower()
        if len(t) >= 2 and t in blob:
            score += 1.0
    return score


def _entity_scoring_enabled() -> bool:
    if not has_app_context():
        return True
    from flask import current_app

    return bool(current_app.config.get("ENTITY_EVIDENCE_SCORING_ENABLED", True))


def _ranked_chunk_ids(
    chunks: list[dict[str, Any]],
    concept_ids: list[str],
    kb: ConceptKB,
    max_n: int = 12,
) -> list[int]:
    """All chunk ids for ``chunks``, ordered by relevance to ``concept_ids`` (deduplicated)."""
    scored: list[tuple[float, int]] = []
    use_entity = _entity_scoring_enabled() and len(concept_ids) >= 1
    primary_c = concept_ids[0] if concept_ids else None
    peers = concept_ids[1:] if len(concept_ids) > 1 else []

    for c in chunks:
        cid = c.get("id")
        if cid is None:
            continue
        if use_entity and primary_c:
            s, _ = score_chunk_for_entity(c, primary_c, kb, peer_concept_ids=peers)
        else:
            s = 0.0
            for k in concept_ids:
                s += _score_chunk_for_concept(c, k, kb)
            if s == 0.0 and concept_ids:
                s = 0.1  # weak fallback only when entity scoring off
        scored.append((s, int(cid)))
    scored.sort(key=lambda x: -x[0])
    out: list[int] = []
    for _, i in scored:
        if i not in out:
            out.append(i)
        if len(out) >= max_n:
            break
    return out


def _pick_chunks_for_section(
    chunks: list[dict[str, Any]],
    concept_ids: list[str],
    kb: ConceptKB,
    max_n: int = 2,
) -> list[int]:
    return _ranked_chunk_ids(chunks, concept_ids, kb, max_n=max_n)


def _has_non_empty_sample_question(c: dict[str, Any]) -> bool:
    if (c.get("sample_answer") or "").strip():
        return True
    raw = c.get("sample_questions")
    if not raw:
        return False
    s = str(raw).strip()
    if not s or s in ("[]", "null"):
        return False
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return any(str(x).strip() for x in arr)
    except json.JSONDecodeError:
        pass
    return bool(s)


def _forbidden_for_entity(entity_id: str, peer_ids: list[str], kb: ConceptKB) -> list[str]:
    from app.services.answers.entity_retrieval import forbidden_terms_for_concept

    return forbidden_terms_for_concept(entity_id, peer_ids, kb)[:12]


def build_answer_plan(
    sq: StructuredQuery,
    chunks: list[dict[str, Any]],
    supporting: list[dict[str, Any]],
    kb: ConceptKB | None = None,
    *,
    constraints: ConceptConstraints | None = None,
) -> AnswerPlan:
    """Build the :class:`AnswerPlan` for a structured query.

    ``constraints`` is the optional :class:`ConceptConstraints` produced by
    :func:`build_concept_constraints` upstream. When provided, the planner
    forwards it to :func:`select_direct_answer` so the deterministic opening
    sentence is grounded in the same purity signal that retrieval rerank
    used. Callers that don't have the structured query plumbing yet (e.g.
    legacy tests) can keep calling ``build_answer_plan`` with positional
    args.
    """
    from app.services.answers.direct_answer import select_direct_answer

    kb = kb or get_kb()
    mode = sq.answer_intent
    labels = ANSWER_PLAN_SECTION_LABELS.get(
        mode,
        ANSWER_PLAN_SECTION_LABELS["multi_step_explanation"],
    )

    all_primary_ids = [c.get("id") for c in chunks if c.get("id") is not None]
    primary_ids = [int(x) for x in all_primary_ids]
    sup_ids = [int(c.get("id")) for c in supporting if c.get("id") is not None]

    comparison_axes: list[str] = []
    if mode == "compare" and len(sq.concept_ids) >= 2:
        comparison_axes = kb.get_comparison_axes(sq.concept_ids[0], sq.concept_ids[1])
    elif mode == "compare" and sq.intent.compare_concepts:
        ca = kb.get_concept(sq.intent.compare_concepts[0])
        cb = kb.get_concept(sq.intent.compare_concepts[1])
        if ca and cb:
            comparison_axes = kb.get_comparison_axes(ca.id, cb.id)

    rc = sq.response_constraints
    include_example = any(_has_non_empty_sample_question(c) for c in chunks)
    if rc.no_examples:
        include_example = False
    include_prereq = False
    if sq.concept_ids:
        c0 = kb.get_concept_by_id(sq.concept_ids[0])
        if c0 and c0.prerequisites:
            include_prereq = True

    related_names: list[str] = []
    if sq.concept_ids:
        c0 = kb.get_concept_by_id(sq.concept_ids[0])
        if c0:
            for rid in c0.related[:5]:
                rc_meta = kb.get_concept_by_id(rid)
                if rc_meta:
                    related_names.append(rc_meta.name)

    section_specs: list[SectionSpec] = []
    evidence_bundles: dict[str, EvidenceBundleLike] = {}

    # --- Multi-entity compare (3+) ---
    if mode == "compare_multi" and len(sq.concept_ids) >= 2:
        bundles = build_bundles_multi_v2(chunks, sq.concept_ids[:8], kb, top_per_entity=3)
        for b in bundles:
            evidence_bundles[b.concept_id] = b
        sections: list[AnswerSection] = []
        for b in bundles:
            sections.append(
                AnswerSection(heading=b.label, chunk_ids=b.chunk_ids[:3], content_hint="definition")
            )
        primary_union: list[int] = []
        seen_u: set[int] = set()
        for b in bundles:
            for i in b.chunk_ids:
                if i not in seen_u:
                    seen_u.add(i)
                    primary_union.append(i)
        if not primary_union:
            primary_union = primary_ids[:6]
        direct_answer_text = select_direct_answer(
            sq, chunks=chunks, bundles=list(bundles), constraints=constraints, kb=kb
        )
        return AnswerPlan(
            answer_mode=mode,
            sections=sections or [
                AnswerSection(heading="Course material", chunk_ids=primary_ids[:3], content_hint="definition")
            ],
            primary_chunk_ids=primary_union or primary_ids,
            supporting_chunk_ids=sup_ids,
            include_example=False,
            include_analogy=False,
            include_prerequisites=include_prereq,
            include_related_concepts=related_names,
            comparison_axes=comparison_axes or ["role", "computation", "typical use"],
            lecture_scope=list(sq.lecture_scope),
            section_specs=section_specs,
            evidence_bundles=evidence_bundles,
            direct_answer=direct_answer_text,
        )

    # --- Two-way compare with isolated evidence pools ---
    if mode == "compare" and len(sq.concept_ids) >= 2:
        ca_id, cb_id = sq.concept_ids[0], sq.concept_ids[1]
        bundle_a, bundle_b = build_bundles_for_compare_v2(chunks, ca_id, cb_id, kb)
        evidence_bundles[ca_id] = bundle_a
        evidence_bundles[cb_id] = bundle_b

        def _safe(ids: list[int]) -> list[int]:
            return ids if ids else primary_ids[:1]

        a_ids = _safe(bundle_a.chunk_ids)
        b_ids = _safe(bundle_b.chunk_ids)
        intro_ids: list[int] = []
        if a_ids and b_ids:
            intro_ids = [a_ids[0], b_ids[0]]
        elif a_ids:
            intro_ids = list(a_ids[:2])
        elif b_ids:
            intro_ids = list(b_ids[:2])
        else:
            intro_ids = primary_ids[:2]

        contrast_ids = [a_ids[0], b_ids[0]] if a_ids and b_ids else intro_ids
        sections = [
            AnswerSection(heading="One-line distinction", chunk_ids=intro_ids, content_hint="comparison_axis"),
            AnswerSection(heading=f"Concept A — {bundle_a.label}", chunk_ids=a_ids[:2], content_hint="definition"),
            AnswerSection(heading=f"Concept B — {bundle_b.label}", chunk_ids=b_ids[:2], content_hint="definition"),
            AnswerSection(heading="Direct comparison", chunk_ids=contrast_ids, content_hint="comparison_axis"),
            AnswerSection(heading="Why the difference matters", chunk_ids=contrast_ids, content_hint="connection"),
        ]
        section_specs = [
            SectionSpec(
                section_id="intro",
                purpose="one_line_each",
                entity_id=None,
                source_chunk_ids=intro_ids,
                content_hint="comparison_axis",
                forbidden_terms=[],
            ),
            SectionSpec(
                section_id="side_a",
                purpose="definition",
                entity_id=ca_id,
                source_chunk_ids=a_ids[:2],
                content_hint="definition",
                forbidden_terms=_forbidden_for_entity(ca_id, [cb_id], kb),
            ),
            SectionSpec(
                section_id="side_b",
                purpose="definition",
                entity_id=cb_id,
                source_chunk_ids=b_ids[:2],
                content_hint="definition",
                forbidden_terms=_forbidden_for_entity(cb_id, [ca_id], kb),
            ),
            SectionSpec(
                section_id="contrast",
                purpose="comparison_axis",
                entity_id=None,
                source_chunk_ids=contrast_ids,
                content_hint="comparison_axis",
                forbidden_terms=[],
            ),
            SectionSpec(
                section_id="why",
                purpose="connection",
                entity_id=None,
                source_chunk_ids=contrast_ids,
                content_hint="connection",
                forbidden_terms=[],
            ),
        ]
        merged_primary: list[int] = []
        seen_m: set[int] = set()
        for i in a_ids + b_ids + intro_ids:
            if i not in seen_m:
                seen_m.add(i)
                merged_primary.append(i)
        direct_answer_text = select_direct_answer(
            sq,
            chunks=chunks,
            bundles=[bundle_a, bundle_b],
            constraints=constraints,
            kb=kb,
        )
        return AnswerPlan(
            answer_mode=mode,
            sections=sections,
            primary_chunk_ids=merged_primary or primary_ids,
            supporting_chunk_ids=sup_ids,
            include_example=include_example,
            include_analogy=False,
            include_prerequisites=include_prereq,
            include_related_concepts=related_names,
            comparison_axes=comparison_axes,
            lecture_scope=list(sq.lecture_scope),
            section_specs=section_specs,
            evidence_bundles=evidence_bundles,
            direct_answer=direct_answer_text,
        )

    sections = []
    n_labels = len(labels)
    cids = sq.concept_ids[:2] if sq.concept_ids else []
    ranked_distinct: list[int] | None = None
    if mode in _DISTINCT_CHUNK_PER_SECTION_MODES:
        ranked_distinct = _ranked_chunk_ids(chunks, cids[:1] if cids else [], kb, max_n=max(n_labels, 8))

    for idx, (heading, hint) in enumerate(labels):
        if ranked_distinct is not None:
            pick_ids = [ranked_distinct[idx]] if idx < len(ranked_distinct) else []
        else:
            pick_ids = _pick_chunks_for_section(chunks, cids, kb, max_n=2)

        if not pick_ids and primary_ids:
            if ranked_distinct is None:
                pick_ids = primary_ids[: min(2, len(primary_ids))]
            elif idx == 0:
                pick_ids = primary_ids[: min(2, len(primary_ids))]

        sections.append(AnswerSection(heading=heading, chunk_ids=pick_ids, content_hint=hint))
        peer = cids[1:] if len(cids) > 1 else []
        ec = cids[0] if cids else None
        forb: list[str] = _forbidden_for_entity(ec, peer, kb) if ec else []
        section_specs.append(
            SectionSpec(
                section_id=f"sec_{idx}",
                purpose=hint,
                entity_id=ec,
                source_chunk_ids=pick_ids,
                content_hint=hint,
                forbidden_terms=forb,
            )
        )

    sections = [s for s in sections if s.chunk_ids]
    if not sections and primary_ids:
        sections = [
            AnswerSection(heading="Course material", chunk_ids=primary_ids[:3], content_hint="definition")
        ]

    # Build a target-scoped chunk pool for direct-answer ranking. We re-rank
    # by entity score so the candidate sentences are pulled from the chunks
    # the planner actually grounds the answer in (not arbitrary retrieval
    # order). Falls back to the planner's primary list if entity scoring is
    # disabled.
    direct_answer_pool: list[dict[str, Any]] = list(chunks)
    if cids and chunks:
        primary_cid = cids[0]
        peer_cids = cids[1:]
        scored_pool: list[tuple[float, int, dict[str, Any]]] = []
        for idx, c in enumerate(chunks):
            score, _ = score_chunk_for_entity(
                c, primary_cid, kb, peer_concept_ids=peer_cids
            )
            scored_pool.append((score, idx, c))
        scored_pool.sort(key=lambda t: (-t[0], t[1]))
        direct_answer_pool = [c for _s, _i, c in scored_pool]

    direct_answer_text = select_direct_answer(
        sq,
        chunks=direct_answer_pool,
        bundles=None,
        constraints=constraints,
        kb=kb,
    )

    return AnswerPlan(
        answer_mode=mode,
        sections=sections,
        primary_chunk_ids=primary_ids,
        supporting_chunk_ids=sup_ids,
        include_example=include_example,
        include_analogy=False,
        include_prerequisites=include_prereq,
        include_related_concepts=related_names,
        comparison_axes=comparison_axes,
        lecture_scope=list(sq.lecture_scope),
        section_specs=section_specs,
        evidence_bundles=evidence_bundles,
        direct_answer=direct_answer_text,
    )


def chunks_by_ids(chunks: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
    by_id = {c.get("id"): c for c in chunks if c.get("id") is not None}
    return [by_id[i] for i in ids if i in by_id]
