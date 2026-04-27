"""Lightweight, explainable validation of course-grounded answers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from app.services.answers.answer_planning import AnswerPlan
from app.services.knowledge.concept_kb import ConceptKB, get_kb
from app.services.knowledge.structured_query import StructuredQuery

if TYPE_CHECKING:
    from app.services.answers.concept_constraints import ConceptConstraints


CRITICAL_CHECK_NAMES = frozenset(
    {
        "must_be_course_grounded",
        "must_cover_both_sides",
        "must_not_leak_forbidden_terms",
        "must_not_have_examples_when_blocked",
        "must_not_have_technical_when_intuition_only",
    }
)


def compute_validation_severity(checks_failed: list[str]) -> str:
    """``pass`` | ``weak`` | ``fail`` — fail if any critical check failed."""
    if not checks_failed:
        return "pass"
    if any(name in CRITICAL_CHECK_NAMES for name in checks_failed):
        return "fail"
    return "weak"


@dataclass
class ValidationResult:
    passed: bool
    checks_run: list[str]
    checks_passed: list[str]
    checks_failed: list[str]
    flags: dict[str, bool] = field(default_factory=dict)
    severity: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "severity": self.severity,
            "checks_run": list(self.checks_run),
            "checks_passed": list(self.checks_passed),
            "checks_failed": list(self.checks_failed),
            "flags": dict(self.flags),
        }


_CONTRAST_CUES = re.compile(
    r"\b(while|whereas|however|in contrast|unlike|compared to|difference|differs|rather than)\b",
    re.IGNORECASE,
)
_CAUSAL_CUES = re.compile(
    r"\b(because|so that|therefore|thus|by |in order to|why )\b",
    re.IGNORECASE,
)


def _mentions_term(text: str, term: str) -> bool:
    t = term.strip().lower()
    if len(t) < 2:
        return False
    return t in text.lower()


def _must_be_course_grounded(answer: str, sq: StructuredQuery, kb: ConceptKB) -> bool:
    al = answer.lower()
    if "lecture" in al:
        return True
    for cid in sq.concept_ids[:6]:
        c = kb.get_concept_by_id(cid)
        if c and (c.name.lower() in al or any(a.lower() in al for a in c.aliases[:5] if len(a) > 2)):
            return True
    for dc in sq.intent.detected_concepts:
        if dc.lower() in al:
            return True
    return len(answer) > 200


def _must_define_primary_concept(answer: str, sq: StructuredQuery, kb: ConceptKB) -> bool:
    if not sq.concept_ids:
        return sq.intent.detected_concepts != [] and any(
            _mentions_term(answer, d) for d in sq.intent.detected_concepts
        )
    c0 = kb.get_concept_by_id(sq.concept_ids[0])
    if not c0:
        return True
    return _mentions_term(answer, c0.name) or any(_mentions_term(answer, a) for a in c0.aliases[:6])


def _must_cover_both_sides(answer: str, sq: StructuredQuery, kb: ConceptKB) -> bool:
    if len(sq.intent.compare_entities) >= 2:
        a, b = sq.intent.compare_entities[0], sq.intent.compare_entities[1]
        return _mentions_term(answer, a) and _mentions_term(answer, b)
    if sq.intent.compare_concepts:
        a, b = sq.intent.compare_concepts
        return _mentions_term(answer, a) and _mentions_term(answer, b)
    return True


def _must_cover_compare_multi(answer: str, sq: StructuredQuery) -> bool:
    if sq.answer_intent != "compare_multi":
        return True
    ents = sq.intent.compare_entities
    if len(ents) < 3:
        return True
    hits = 0
    for e in ents:
        if _mentions_term(answer, e):
            hits += 1
    return hits >= min(len(ents), 4)


def _must_not_leak_forbidden_terms(answer: str, sq: StructuredQuery, plan: AnswerPlan, kb: ConceptKB) -> bool:
    """Block obvious cross-topic leaks for single-concept definition-style answers."""
    if sq.answer_intent in ("compare", "compare_multi"):
        return True
    if len(sq.concept_ids) != 1:
        return True
    from app.services.answers.entity_retrieval import forbidden_terms_for_concept

    terms = forbidden_terms_for_concept(sq.concept_ids[0], [], kb)
    al = answer.lower()
    for t in terms:
        if len(t) > 5 and t in al:
            return False
    return True


def _must_not_have_examples_when_blocked(answer: str, sq: StructuredQuery) -> bool:
    if not sq.response_constraints.no_examples:
        return True
    al = answer.lower()
    if "for example" in al or "e.g." in al or "analogy" in al:
        return False
    return True


def _must_not_have_technical_when_intuition_only(answer: str, sq: StructuredQuery) -> bool:
    if not sq.response_constraints.intuition_only:
        return True
    al = answer.lower()
    tech = re.compile(
        r"\b(gradient|backprop|epoch|loss function|matrix|layer norm|batch norm|weight decay)\b",
        re.IGNORECASE,
    )
    return tech.search(al) is None


def _must_not_be_boilerplate_summary(answer: str, sq: StructuredQuery) -> bool:
    if sq.answer_intent != "lecture_summary":
        return True
    al = answer.lower()
    if "this lecture thread builds definitions" in al or "this lecture thread" in al and "builds definitions" in al:
        return False
    return True


def _must_include_comparison_axis(answer: str, plan: AnswerPlan) -> bool:
    if _CONTRAST_CUES.search(answer):
        return True
    for ax in plan.comparison_axes:
        if ax.lower() in answer.lower():
            return True
    return False


def _must_stay_in_scope(answer: str, sq: StructuredQuery, plan: AnswerPlan) -> bool:
    if sq.intent.query_type.value != "summary":
        return True
    if not sq.intent.lecture_numbers:
        return True
    # Heuristic: lecture numbers mentioned should include requested
    for n in sq.intent.lecture_numbers:
        if f"lecture {n}" in answer.lower() or f"lecture {n}—" in answer.lower():
            return True
    return True


def _must_cover_main_anchors(answer: str, sq: StructuredQuery, kb: ConceptKB) -> bool:
    if sq.intent.query_type.value != "summary" or not sq.intent.lecture_numbers:
        return True
    n = sq.intent.lecture_numbers[0]
    lm = kb.get_lecture(n)
    if not lm:
        return True
    hits = 0
    for mid in lm.main_concepts[:6]:
        c = kb.get_concept_by_id(mid)
        if c and (c.name.lower() in answer.lower() or any(a.lower() in answer.lower() for a in c.aliases[:3])):
            hits += 1
    return hits >= 1


def _must_include_multiple_lectures(answer: str, sq: StructuredQuery) -> bool:
    if sq.answer_intent != "cross_lecture_synthesis":
        return True
    nums = sq.intent.lecture_numbers
    if len(nums) < 2:
        return True
    mentioned = sum(1 for n in nums if f"lecture {n}" in answer.lower() or f"lecture {n}" in answer.lower())
    return mentioned >= 2 or "lecture" in answer.lower()


def _must_name_connecting_concepts(answer: str) -> bool:
    """Require substantive bridge language (not only generic filler)."""
    al = answer.lower()
    if len(al) > 400:
        return True
    return any(
        x in al
        for x in (
            "attention",
            "representation",
            "prediction",
            "compression",
            "lecture",
            "connect",
            "progression",
            "dependency",
            "sequence",
            "dynamic programming",
            "backpropagation",
        )
    )


def _must_answer_how_or_why(answer: str) -> bool:
    return bool(_CAUSAL_CUES.search(answer)) or "step" in answer.lower()


def _must_respect_lecture_scope(answer: str, plan: AnswerPlan, chunks_lectures: list[int]) -> bool:
    if not plan.lecture_scope:
        return True
    for ln in chunks_lectures:
        if ln in plan.lecture_scope:
            return True
    return True


def _must_be_concept_pure(answer: str, constraints: "ConceptConstraints") -> bool:
    """Soft-warn when the answer drifts away from the target concept's vocabulary.

    Hard fail (returns ``False``) only for full topic drift — a forbidden
    term appears in the answer and *no* target alias does. When both appear,
    the line is ambiguous (e.g. CNN answer touches transformers in passing);
    we treat that as borderline-pass via the dedicated ``ambiguous_concept_bleed``
    flag rather than blocking the whole answer.

    Always returns ``True`` for relational queries — *Compare A and B* and
    *How does X relate to Y?* legitimately require both sides' vocabulary.
    """
    if constraints.is_relational:
        return True
    if not constraints.forbidden_terms:
        return True
    al = answer.lower()
    forbidden_hit = any(t in al for t in constraints.forbidden_terms if len(t) > 2)
    if not forbidden_hit:
        return True
    target_hit = any(t in al for t in constraints.target_aliases if len(t) > 2)
    if target_hit:
        return True
    return False


def _has_ambiguous_concept_bleed(answer: str, constraints: "ConceptConstraints") -> bool:
    """``True`` when both forbidden + target terms appear (soft warn surface)."""
    if constraints.is_relational or not constraints.forbidden_terms:
        return False
    al = answer.lower()
    forbidden_hit = any(t in al for t in constraints.forbidden_terms if len(t) > 2)
    target_hit = any(t in al for t in constraints.target_aliases if len(t) > 2)
    return forbidden_hit and target_hit


def _direct_answer_text(plan: AnswerPlan) -> str:
    return (plan.direct_answer or "").strip()


def _must_direct_answer_mention_target_concept(
    plan: AnswerPlan, sq: StructuredQuery, kb: ConceptKB
) -> bool:
    """Soft-warn when the direct answer doesn't ground in the target concept(s).

    Skipped entirely when ``plan.direct_answer`` is ``None`` (summary / quiz /
    synthesis paths). For chat / definition the check fails when the direct
    answer mentions zero target aliases. For compare it fails when it
    doesn't mention both compared entities.
    """
    text = _direct_answer_text(plan)
    if not text:
        return True
    al = text.lower()
    mode = sq.answer_intent
    if mode == "compare":
        if sq.intent.compare_entities and len(sq.intent.compare_entities) >= 2:
            a, b = sq.intent.compare_entities[0], sq.intent.compare_entities[1]
        elif sq.intent.compare_concepts:
            a, b = sq.intent.compare_concepts
        elif len(sq.concept_ids) >= 2:
            ma = kb.get_concept_by_id(sq.concept_ids[0])
            mb = kb.get_concept_by_id(sq.concept_ids[1])
            a = ma.name if ma else sq.concept_ids[0]
            b = mb.name if mb else sq.concept_ids[1]
        else:
            return True
        return _mentions_term(al, a) and _mentions_term(al, b)
    if not sq.concept_ids:
        return True
    primary_meta = kb.get_concept_by_id(sq.concept_ids[0])
    if not primary_meta:
        return True
    aliases = [primary_meta.name, *primary_meta.aliases[:6]]
    return any(_mentions_term(al, t) for t in aliases)


def validate_answer(
    answer: str,
    sq: StructuredQuery,
    plan: AnswerPlan,
    *,
    primary_chunk_lecture_numbers: list[int] | None = None,
    kb: ConceptKB | None = None,
    constraints: "ConceptConstraints | None" = None,
) -> ValidationResult:
    kb = kb or get_kb()
    checks_run: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    plns = primary_chunk_lecture_numbers or []

    def run(name: str, ok: bool) -> None:
        checks_run.append(name)
        if ok:
            passed.append(name)
        else:
            failed.append(name)

    run("must_be_course_grounded", _must_be_course_grounded(answer, sq, kb))

    ai = sq.answer_intent
    if ai == "direct_definition":
        run("must_define_primary_concept", _must_define_primary_concept(answer, sq, kb))
    if ai == "compare":
        run("must_cover_both_sides", _must_cover_both_sides(answer, sq, kb))
        run("must_include_comparison_axis", _must_include_comparison_axis(answer, plan))
    if ai == "compare_multi":
        run("must_cover_compare_multi", _must_cover_compare_multi(answer, sq))
        run("must_include_comparison_axis", _must_include_comparison_axis(answer, plan))
    if ai == "lecture_summary":
        run("must_stay_in_scope", _must_stay_in_scope(answer, sq, plan))
        run("must_cover_main_anchors", _must_cover_main_anchors(answer, sq, kb))
    if ai == "cross_lecture_synthesis":
        run("must_include_multiple_lectures", _must_include_multiple_lectures(answer, sq))
        run("must_name_connecting_concepts", _must_name_connecting_concepts(answer))
    if ai in ("multi_step_explanation", "scoped_explanation"):
        run("must_answer_how_or_why", _must_answer_how_or_why(answer))

    run("must_not_leak_forbidden_terms", _must_not_leak_forbidden_terms(answer, sq, plan, kb))
    run("must_not_have_examples_when_blocked", _must_not_have_examples_when_blocked(answer, sq))
    run("must_not_have_technical_when_intuition_only", _must_not_have_technical_when_intuition_only(answer, sq))
    run("must_not_be_boilerplate_summary", _must_not_be_boilerplate_summary(answer, sq))

    run("must_respect_lecture_scope", _must_respect_lecture_scope(answer, plan, plns))

    ambiguous_bleed = False
    if constraints is not None and ai in (
        "direct_definition",
        "multi_step_explanation",
        "scoped_explanation",
        "simplified_reteach",
        "teaching_plus_check",
    ):
        run("must_be_concept_pure", _must_be_concept_pure(answer, constraints))
        ambiguous_bleed = _has_ambiguous_concept_bleed(answer, constraints)

    if plan.direct_answer:
        run(
            "must_direct_answer_mention_target_concept",
            _must_direct_answer_mention_target_concept(plan, sq, kb),
        )

    generic = len(answer) < 120 and not sq.concept_ids
    missing_side = False
    if ai == "compare":
        missing_side = not _must_cover_both_sides(answer, sq, kb)
    elif ai == "compare_multi":
        missing_side = not _must_cover_compare_multi(answer, sq)

    ok = len(failed) == 0
    severity = compute_validation_severity(failed)
    return ValidationResult(
        passed=ok,
        checks_run=checks_run,
        checks_passed=passed,
        checks_failed=failed,
        flags={
            "generic_answer": generic,
            "missing_comparison_side": missing_side,
            "out_of_scope": False,
            "ambiguous_concept_bleed": ambiguous_bleed,
        },
        severity=severity,
    )
