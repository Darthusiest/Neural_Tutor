"""Structured answer plans: sections, primary vs supporting chunks, comparison axes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.services.knowledge.concept_kb import ConceptKB, get_kb
from app.services.knowledge.structured_query import StructuredQuery


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


def _ranked_chunk_ids(
    chunks: list[dict[str, Any]],
    concept_ids: list[str],
    kb: ConceptKB,
    max_n: int = 12,
) -> list[int]:
    """All chunk ids for ``chunks``, ordered by relevance to ``concept_ids`` (deduplicated)."""
    scored: list[tuple[float, int]] = []
    for c in chunks:
        cid = c.get("id")
        if cid is None:
            continue
        s = 0.0
        for k in concept_ids:
            s += _score_chunk_for_concept(c, k, kb)
        if s == 0.0 and concept_ids:
            s = 0.1  # weak fallback
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


def build_answer_plan(
    sq: StructuredQuery,
    chunks: list[dict[str, Any]],
    supporting: list[dict[str, Any]],
    kb: ConceptKB | None = None,
) -> AnswerPlan:
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

    include_example = any(_has_non_empty_sample_question(c) for c in chunks)
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
                rc = kb.get_concept_by_id(rid)
                if rc:
                    related_names.append(rc.name)

    sections: list[AnswerSection] = []
    n_labels = len(labels)
    cids = sq.concept_ids[:2] if sq.concept_ids else []
    ranked_distinct: list[int] | None = None
    if mode in _DISTINCT_CHUNK_PER_SECTION_MODES:
        ranked_distinct = _ranked_chunk_ids(chunks, cids, kb, max_n=max(n_labels, 8))

    for idx, (heading, hint) in enumerate(labels):
        if mode == "compare" and idx in (1, 2) and sq.intent.compare_concepts:
            side = sq.intent.compare_concepts[0] if idx == 1 else sq.intent.compare_concepts[1]
            cc = kb.get_concept(side)
            pick_ids = _pick_chunks_for_section(chunks, [cc.id] if cc else [], kb, max_n=2)
        elif ranked_distinct is not None:
            pick_ids = [ranked_distinct[idx]] if idx < len(ranked_distinct) else []
        else:
            pick_ids = _pick_chunks_for_section(chunks, cids, kb, max_n=2)

        if not pick_ids and primary_ids:
            if ranked_distinct is None:
                pick_ids = primary_ids[: min(2, len(primary_ids))]
            elif idx == 0:
                pick_ids = primary_ids[: min(2, len(primary_ids))]

        sections.append(AnswerSection(heading=heading, chunk_ids=pick_ids, content_hint=hint))

    # Drop sections with no chunk assignment (after distinct split, later headings may be empty).
    sections = [s for s in sections if s.chunk_ids]
    if not sections and primary_ids:
        sections = [
            AnswerSection(heading="Course material", chunk_ids=primary_ids[:3], content_hint="definition")
        ]

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
    )


def chunks_by_ids(chunks: list[dict[str, Any]], ids: list[int]) -> list[dict[str, Any]]:
    by_id = {c.get("id"): c for c in chunks if c.get("id") is not None}
    return [by_id[i] for i in ids if i in by_id]
