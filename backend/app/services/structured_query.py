"""Structured query representation and decomposition for the reasoning pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from app.services.concept_kb import ConceptKB, ConceptMeta, get_kb
from app.services.query_understanding import QueryIntent, QueryType


@dataclass
class SubQuestion:
    text: str
    target_concept_id: str | None
    purpose: str  # define, compare_side, mechanism, connection, prerequisite, lecture_anchor
    lecture_scope: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredQuery:
    """Rich query object built on top of :class:`QueryIntent`."""

    intent: QueryIntent
    concept_ids: list[str]
    answer_intent: str
    sub_questions: list[SubQuestion]
    retrieval_hints: list[str]
    lecture_scope: list[int]
    answer_style: str
    decomposition_template: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.intent.query_type.value,
            "answer_intent": self.answer_intent,
            "concept_ids": list(self.concept_ids),
            "sub_questions": [s.to_dict() for s in self.sub_questions],
            "retrieval_hints": list(self.retrieval_hints),
            "lecture_scope": list(self.lecture_scope),
            "answer_style": self.answer_style,
            "decomposition_template": list(self.decomposition_template),
        }


# Maps QueryType -> answer_intent string (aligned with KB query_patterns / answer_plan_templates)
_QUERY_TYPE_TO_ANSWER_INTENT: dict[QueryType, str] = {
    QueryType.DEFINITION: "direct_definition",
    QueryType.COMPARE: "compare",
    QueryType.SUMMARY: "lecture_summary",
    QueryType.SYNTHESIS: "cross_lecture_synthesis",
    QueryType.LECTURE_SPECIFIC: "scoped_explanation",
    QueryType.QUIZ: "teaching_plus_check",
    QueryType.VAGUE_FOLLOWUP: "simplified_reteach",
    QueryType.GENERAL: "multi_step_explanation",
}

# Decomposition templates by answer_intent (from KB query_patterns)
_DECOMPOSITION_TEMPLATES: dict[str, list[str]] = {
    "direct_definition": [
        "Give a direct definition",
        "Explain how it works in the course context",
        "Explain why it matters",
    ],
    "compare": [
        "Define concept A",
        "Define concept B",
        "Compare along key axes",
        "State why the distinction matters",
    ],
    "lecture_summary": [
        "Identify requested lecture/topic scope",
        "List main concepts",
        "Explain connections among them",
    ],
    "cross_lecture_synthesis": [
        "Summarize each lecture or concept briefly",
        "Identify shared thread",
        "Explain progression or dependency",
    ],
    "scoped_explanation": [
        "Respect requested lecture scope",
        "Answer with scoped concepts unless prerequisites are needed",
    ],
    "teaching_plus_check": [
        "Answer directly",
        "Teach the reasoning",
        "Mention common confusion if helpful",
    ],
    "simplified_reteach": [
        "Restate answer in plainer language",
        "Use one analogy",
        "Use one concrete example",
    ],
    "multi_step_explanation": [
        "Define concept",
        "Explain mechanism",
        "Explain purpose or motivation",
        "Optional example",
    ],
}


def _resolve_kb_concepts(intent: QueryIntent, kb: ConceptKB) -> list[ConceptMeta]:
    found = kb.find_concepts_in_text(intent.query_tokens + intent.expanded_tokens[:20])
    # Also try detected canonical names as surface strings
    for dc in intent.detected_concepts:
        c = kb.get_concept(dc)
        if c and c not in found:
            found.append(c)
    seen: set[str] = set()
    out: list[ConceptMeta] = []
    for c in found:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


def _lecture_scope_union(intent: QueryIntent, concepts: list[ConceptMeta]) -> list[int]:
    nums: set[int] = set(intent.lecture_numbers)
    for c in concepts:
        nums.update(c.lecture_scope)
    return sorted(nums)


def _retrieval_hints(concepts: list[ConceptMeta]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for c in concepts:
        for h in c.retrieval_hints[:4]:
            if h not in seen:
                seen.add(h)
                hints.append(h)
        if len(hints) >= 12:
            break
    return hints


def decompose_query(intent: QueryIntent, kb: ConceptKB, concepts: list[ConceptMeta]) -> list[SubQuestion]:
    """Produce atomic sub-questions grounded in course concepts."""
    qt = intent.query_type
    subs: list[SubQuestion] = []

    if qt == QueryType.VAGUE_FOLLOWUP:
        return [
            SubQuestion(
                text="Re-explain the previous topic in simpler terms.",
                target_concept_id=concepts[0].id if concepts else None,
                purpose="define",
            )
        ]

    if qt == QueryType.COMPARE and intent.compare_concepts:
        a, b = intent.compare_concepts
        ca = kb.get_concept(a) or kb.get_concept(a.split()[0])
        cb = kb.get_concept(b) or kb.get_concept(b.split()[0])
        subs.extend(
            [
                SubQuestion(
                    text=f"What is {a.strip()}?",
                    target_concept_id=ca.id if ca else None,
                    purpose="compare_side",
                    lecture_scope=ca.lecture_scope if ca else [],
                ),
                SubQuestion(
                    text=f"What is {b.strip()}?",
                    target_concept_id=cb.id if cb else None,
                    purpose="compare_side",
                    lecture_scope=cb.lecture_scope if cb else [],
                ),
                SubQuestion(
                    text=f"How do {a.strip()} and {b.strip()} differ in purpose and use?",
                    target_concept_id=None,
                    purpose="mechanism",
                ),
                SubQuestion(
                    text="Why does the distinction matter for speech / ML in this course?",
                    target_concept_id=None,
                    purpose="connection",
                ),
            ]
        )
        return subs

    if qt == QueryType.SUMMARY and intent.lecture_numbers:
        for n in intent.lecture_numbers:
            lm = kb.get_lecture(n)
            title = lm.title if lm else f"Lecture {n}"
            subs.append(
                SubQuestion(
                    text=f"What is {title} mainly about?",
                    target_concept_id=None,
                    purpose="lecture_anchor",
                    lecture_scope=[n],
                )
            )
        subs.append(
            SubQuestion(
                text="What concepts tie these sections together?",
                target_concept_id=None,
                purpose="connection",
            )
        )
        return subs

    if qt == QueryType.SYNTHESIS:
        lecs = sorted(set(intent.lecture_numbers))
        if not lecs:
            lecs = _lecture_scope_union(intent, concepts)[:6]
        for n in lecs:
            lm = kb.get_lecture(n)
            title = lm.title if lm else f"Lecture {n}"
            subs.append(
                SubQuestion(
                    text=f"Core ideas from lecture {n}: {title}",
                    target_concept_id=None,
                    purpose="lecture_anchor",
                    lecture_scope=[n],
                )
            )
        subs.append(
            SubQuestion(
                text="What progression or dependency connects these lectures?",
                target_concept_id=None,
                purpose="connection",
            )
        )
        return subs

    # Definition / general / lecture_specific / quiz
    if concepts:
        c0 = concepts[0]
        for sq_text in c0.common_subquestions[:3]:
            subs.append(
                SubQuestion(
                    text=sq_text,
                    target_concept_id=c0.id,
                    purpose="define",
                    lecture_scope=c0.lecture_scope,
                )
            )
        if not subs:
            subs.append(
                SubQuestion(
                    text=f"What is {c0.name}?",
                    target_concept_id=c0.id,
                    purpose="define",
                    lecture_scope=c0.lecture_scope,
                )
            )
    else:
        subs.append(
            SubQuestion(
                text="Answer the student's question using course definitions and mechanisms.",
                target_concept_id=None,
                purpose="define",
            )
        )

    if len(concepts) > 1:
        subs.append(
            SubQuestion(
                text=f"How do {concepts[0].name} and {concepts[1].name} relate?",
                target_concept_id=None,
                purpose="connection",
            )
        )

    return subs


def build_structured_query(intent: QueryIntent, kb: ConceptKB | None = None) -> StructuredQuery:
    kb = kb or get_kb()
    concepts = _resolve_kb_concepts(intent, kb)
    if intent.compare_concepts:
        for part in intent.compare_concepts:
            c = kb.get_concept(part.strip())
            if c and all(c.id != x.id for x in concepts):
                concepts.append(c)
    concept_ids = [c.id for c in concepts]

    answer_intent = _QUERY_TYPE_TO_ANSWER_INTENT.get(intent.query_type, "multi_step_explanation")
    lecture_scope = _lecture_scope_union(intent, concepts)
    hints = _retrieval_hints(concepts)

    template = list(_DECOMPOSITION_TEMPLATES.get(answer_intent, _DECOMPOSITION_TEMPLATES["multi_step_explanation"]))

    answer_style = "teaching"
    if intent.query_type == QueryType.VAGUE_FOLLOWUP:
        answer_style = "simplified"
    elif intent.query_type == QueryType.QUIZ:
        answer_style = "quiz"

    sub_questions = decompose_query(intent, kb, concepts)

    return StructuredQuery(
        intent=intent,
        concept_ids=concept_ids,
        answer_intent=answer_intent,
        sub_questions=sub_questions,
        retrieval_hints=hints,
        lecture_scope=lecture_scope,
        answer_style=answer_style,
        decomposition_template=template,
    )
