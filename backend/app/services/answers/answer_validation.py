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
        # Task 7 hardened checks
        "must_match_quiz_contract",
        "must_match_summary_contract",
        "must_match_compare_contract",
        "must_have_distinct_compare_evidence",
        "must_have_each_side_evidence_or_note",
        "must_direct_answer_match_target",
        "must_not_have_section_duplication",
    }
)


# Maps a critical-check name onto one of four named repair paths consumed by
# the reasoning pipeline + analytics. The pipeline currently branches only on
# ``fall_back_to_clarification``; the others surface in logs / diagnostics
# for operators and a future repair loop.
_REPAIR_PATHS_BY_CHECK: dict[str, str] = {
    "must_match_quiz_contract": "fall_back_to_clarification",
    "must_match_summary_contract": "fall_back_to_clarification",
    "must_match_compare_contract": "rebuild_evidence_bundles",
    "must_have_distinct_compare_evidence": "rebuild_evidence_bundles",
    "must_have_each_side_evidence_or_note": "render_limitation_message",
    "must_direct_answer_match_target": "retry_retrieval_with_stricter_constraints",
    "must_not_have_section_duplication": "rebuild_evidence_bundles",
    "must_not_leak_forbidden_terms": "retry_retrieval_with_stricter_constraints",
    # Existing critical names (kept for completeness in diagnostics)
    "must_be_course_grounded": "retry_retrieval_with_stricter_constraints",
    "must_cover_both_sides": "rebuild_evidence_bundles",
    "must_not_have_examples_when_blocked": "render_limitation_message",
    "must_not_have_technical_when_intuition_only": "render_limitation_message",
    "must_be_concept_pure": "retry_retrieval_with_stricter_constraints",
}


def compute_validation_severity(checks_failed: list[str]) -> str:
    """``pass`` | ``weak`` | ``fail`` — fail if any critical check failed."""
    if not checks_failed:
        return "pass"
    if any(name in CRITICAL_CHECK_NAMES for name in checks_failed):
        return "fail"
    return "weak"


def _select_repair_path(checks_failed: list[str]) -> str | None:
    """Return the first repair path triggered by a critical failure, or ``None``."""
    for name in checks_failed:
        if name in CRITICAL_CHECK_NAMES and name in _REPAIR_PATHS_BY_CHECK:
            return _REPAIR_PATHS_BY_CHECK[name]
    return None


@dataclass
class ValidationResult:
    passed: bool
    checks_run: list[str]
    checks_passed: list[str]
    checks_failed: list[str]
    flags: dict[str, bool] = field(default_factory=dict)
    severity: str = "pass"
    # Suggested recovery path when ``severity == "fail"``. One of:
    # ``retry_retrieval_with_stricter_constraints`` |
    # ``rebuild_evidence_bundles`` | ``render_limitation_message`` |
    # ``fall_back_to_clarification``. ``None`` for soft / passing results.
    repair_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "severity": self.severity,
            "checks_run": list(self.checks_run),
            "checks_passed": list(self.checks_passed),
            "checks_failed": list(self.checks_failed),
            "flags": dict(self.flags),
            "repair_path": self.repair_path,
        }


_CONTRAST_CUES = re.compile(
    r"\b(while|whereas|however|in contrast|unlike|compared to|difference|differs|rather than)\b",
    re.IGNORECASE,
)
_CAUSAL_CUES = re.compile(
    r"\b(because|so that|therefore|thus|by |in order to|why )\b",
    re.IGNORECASE,
)


# Section / heading markers that belong to the four-block Course Answer
# layout and must NOT appear inside quiz / summary output.
_QUIZ_FORBIDDEN_HEADINGS: tuple[str, ...] = (
    "### Direct Answer",
    "### Explanation",
    "### Example / Intuition",
    "### Why it matters",
    "Course Answer:",
)
_SUMMARY_FORBIDDEN_HEADINGS: tuple[str, ...] = (
    "### Direct Answer",
    "### Explanation",
    "Course Answer:",
)


# Filler patterns the renderer is supposed to suppress. Mirrors
# ``answer_generation._GENERIC_FILLER_PATTERNS`` but kept inline here so the
# validator doesn't import the renderer module.
_GENERIC_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"you[\u2019']ll keep running into", re.IGNORECASE),
    re.compile(r"you will keep running into", re.IGNORECASE),
    re.compile(r"this topic connects to", re.IGNORECASE),
    re.compile(r"solid intuition here makes the next topics", re.IGNORECASE),
    re.compile(r"notation and vocabulary pay off later", re.IGNORECASE),
)


# Phrases the compare renderer emits when one bundle has thin / no evidence.
# Used by ``must_have_each_side_evidence_or_note`` to confirm the renderer
# already surfaced the gap before we hard-fail.
_COMPARE_LIMITATION_PHRASES: tuple[str, ...] = (
    "limited direct material",
    "evidence support is thin",
    "evidence notes",
    "no scoped line in retrieved notes",
    "limited evidence in retrieved chunks",
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


def _compare_entity_labels(sq: StructuredQuery, kb: ConceptKB) -> list[str]:
    """Best-effort list of compared entity labels for compare / compare_multi.

    Pulls from ``intent.compare_entities`` first (preserves original
    casing), then falls back to ``intent.compare_concepts`` and KB lookups
    by ``concept_ids``. Empty list when nothing nameable is available.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        if not raw:
            return
        clean = str(raw).strip()
        if not clean:
            return
        key = clean.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(clean)

    for ent in sq.intent.compare_entities or []:
        _add(ent)
    if sq.intent.compare_concepts:
        for ent in sq.intent.compare_concepts:
            _add(ent)
    for cid in sq.concept_ids or []:
        meta = kb.get_concept_by_id(cid)
        if meta:
            _add(meta.name)
        else:
            _add(cid)
    return out


def _must_match_quiz_contract(answer: str, plan: AnswerPlan) -> bool:
    """Quiz output must use quiz layout (no Course Answer headings).

    Skipped when ``plan.answer_mode != "teaching_plus_check"``. The header
    check is lenient — early empty-evidence fallback messages also start
    with ``Quiz:`` (the renderer emits the header even when no questions
    were built), so we only require the absence of Course-Answer scaffolding.
    """
    if plan.answer_mode != "teaching_plus_check":
        return True
    body = (answer or "").strip()
    if not body:
        return True
    for marker in _QUIZ_FORBIDDEN_HEADINGS:
        if marker in body:
            return False
    return True


def _must_match_summary_contract(answer: str, plan: AnswerPlan) -> bool:
    """Summary output must use summary layout (no Course Answer scaffolding)."""
    if plan.answer_mode != "lecture_summary":
        return True
    body = (answer or "").strip()
    if not body:
        return True
    for marker in _SUMMARY_FORBIDDEN_HEADINGS:
        if marker in body:
            return False
    return True


def _must_match_compare_contract(answer: str, sq: StructuredQuery, kb: ConceptKB) -> bool:
    """Compare answers must mention every compared entity label.

    Promotes the existing soft ``must_cover_both_sides`` rule into a
    critical contract for compare / compare_multi. ``compare_multi``
    requires hits on at least 2 of the named entities (matches the
    existing ``must_cover_compare_multi`` behaviour) so a missing fourth
    entity is a soft warn rather than a hard fail.
    """
    if sq.answer_intent not in ("compare", "compare_multi"):
        return True
    labels = _compare_entity_labels(sq, kb)
    if len(labels) < 2:
        # Nothing nameable to enforce — let upstream contract handle it.
        return True
    al = (answer or "").lower()
    if sq.answer_intent == "compare":
        return _mentions_term(al, labels[0]) and _mentions_term(al, labels[1])
    hits = sum(1 for label in labels if _mentions_term(al, label))
    return hits >= 2


def _normalize_line_for_dup(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "")).strip().lower()


def _must_have_distinct_compare_evidence(plan: AnswerPlan, sq: StructuredQuery) -> bool:
    """Two-way compare bundles must not share their leading core lines verbatim.

    Checks the first three core lines of each side after whitespace + case
    normalization. Identical sets indicate the V2 builder didn't separate
    Concept A from Concept B (a common symptom of retrieval contamination
    or a bundle assembly bug).
    """
    if sq.answer_intent != "compare":
        return True
    bundles = list(plan.evidence_bundles.values())
    if len(bundles) < 2:
        return True
    bundle_a, bundle_b = bundles[0], bundles[1]
    lines_a = list(getattr(bundle_a, "core_lines", []) or [])
    lines_b = list(getattr(bundle_b, "core_lines", []) or [])
    if not lines_a or not lines_b:
        # Empty side handled by ``must_have_each_side_evidence_or_note``.
        return True
    norm_a = {_normalize_line_for_dup(line) for line in lines_a[:3] if line}
    norm_b = {_normalize_line_for_dup(line) for line in lines_b[:3] if line}
    if not norm_a or not norm_b:
        return True
    return norm_a != norm_b


def _must_have_each_side_evidence_or_note(
    answer: str, plan: AnswerPlan, sq: StructuredQuery
) -> bool:
    """Either both compare bundles carry core lines, or the answer flags the gap.

    The compare renderer already inserts ``Limited direct material …`` and
    ``Evidence notes`` blocks when a bundle is empty. This validator just
    enforces that the renderer actually surfaced the gap when one occurred.
    """
    if sq.answer_intent != "compare":
        return True
    bundles = list(plan.evidence_bundles.values())
    if len(bundles) < 2:
        return True
    bundle_a, bundle_b = bundles[0], bundles[1]
    lines_a = list(getattr(bundle_a, "core_lines", []) or [])
    lines_b = list(getattr(bundle_b, "core_lines", []) or [])
    if lines_a and lines_b:
        return True
    al = (answer or "").lower()
    return any(phrase in al for phrase in _COMPARE_LIMITATION_PHRASES)


def _bullet_lines_in_answer(answer: str) -> list[str]:
    """Bullet rows (``- ``, ``* ``, ``\u2022 ``) that look like content lines."""
    out: list[str] = []
    for raw in (answer or "").split("\n"):
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ", "\u2022 ")):
            content = stripped[2:].strip()
            # Skip ultra-short bullets (e.g. "- A)") that aren't real content.
            if len(content) >= 6:
                out.append(content)
    return out


def _heading_lines_in_answer(answer: str) -> list[str]:
    out: list[str] = []
    for raw in (answer or "").split("\n"):
        stripped = raw.strip()
        if stripped.startswith("### "):
            out.append(stripped)
    return out


def _must_not_have_section_duplication(answer: str) -> bool:
    """No identical ``###`` heading lines and no identical content bullets.

    Repeated headings indicate the renderer accidentally emitted the same
    section twice; repeated bullet lines indicate the explanation collator
    didn't dedupe (regression for the legacy compare scaffold).
    """
    body = (answer or "").strip()
    if not body:
        return True
    headings = _heading_lines_in_answer(body)
    seen_h: set[str] = set()
    for h in headings:
        key = _normalize_line_for_dup(h)
        if not key:
            continue
        if key in seen_h:
            return False
        seen_h.add(key)
    bullets = _bullet_lines_in_answer(body)
    seen_b: set[str] = set()
    for line in bullets:
        key = _normalize_line_for_dup(line)
        if not key:
            continue
        if key in seen_b:
            return False
        seen_b.add(key)
    return True


def _has_generic_filler(answer: str, plan: AnswerPlan) -> bool:
    """Soft signal: legacy filler phrasing in the closer / why-it-matters block.

    Flips the new ``flags["generic_filler"]`` warn surface; never appears
    in ``checks_failed``. We only flag when the planner *also* lacks
    related-concept context — the renderer's grounded-closer path needs
    related concepts to avoid the legacy generic copy. Empty
    ``include_related_concepts`` AND a filler hit means the generic copy
    leaked through.
    """
    if plan.include_related_concepts:
        return False
    body = (answer or "")
    if not body:
        return False
    return any(p.search(body) for p in _GENERIC_FILLER_PATTERNS)


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

    # ------------------------------------------------------------------
    # Task 7 hardened contracts
    # ------------------------------------------------------------------
    run("must_match_quiz_contract", _must_match_quiz_contract(answer, plan))
    run("must_match_summary_contract", _must_match_summary_contract(answer, plan))
    if ai in ("compare", "compare_multi"):
        run("must_match_compare_contract", _must_match_compare_contract(answer, sq, kb))
    if ai == "compare":
        run(
            "must_have_distinct_compare_evidence",
            _must_have_distinct_compare_evidence(plan, sq),
        )
        run(
            "must_have_each_side_evidence_or_note",
            _must_have_each_side_evidence_or_note(answer, plan, sq),
        )
    run("must_not_have_section_duplication", _must_not_have_section_duplication(answer))

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
        soft_ok = _must_direct_answer_mention_target_concept(plan, sq, kb)
        run("must_direct_answer_mention_target_concept", soft_ok)
        # Promote to a critical contract for direct_definition /
        # multi_step_explanation only — these modes own a clearly target-
        # bound direct answer. Other intents keep the legacy soft warn.
        if ai in ("direct_definition", "multi_step_explanation"):
            run("must_direct_answer_match_target", soft_ok)

    generic = len(answer) < 120 and not sq.concept_ids
    missing_side = False
    if ai == "compare":
        missing_side = not _must_cover_both_sides(answer, sq, kb)
    elif ai == "compare_multi":
        missing_side = not _must_cover_compare_multi(answer, sq)

    generic_filler = _has_generic_filler(answer, plan)

    ok = len(failed) == 0
    severity = compute_validation_severity(failed)
    repair_path = _select_repair_path(failed) if severity == "fail" else None
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
            "generic_filler": generic_filler,
        },
        severity=severity,
        repair_path=repair_path,
    )
