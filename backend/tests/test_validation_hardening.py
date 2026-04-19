"""Stricter validator rules (forbidden lemmas, constraints, boilerplate)."""

from app.services.answers.answer_planning import AnswerPlan, AnswerSection
from app.services.knowledge.structured_query import StructuredQuery, build_structured_query
from app.services.query_understanding import QueryIntent, QueryType, analyze_query
from app.services.knowledge.concept_kb import get_kb
from app.services.answers.answer_validation import validate_answer


def _sq(intent: QueryIntent) -> StructuredQuery:
    return build_structured_query(intent, kb=get_kb())


def test_no_examples_validator_flags_for_example():
    intent = analyze_query("What is softmax with no examples")
    sq = _sq(intent)
    plan = AnswerPlan(
        answer_mode="multi_step_explanation",
        sections=[AnswerSection("Direct", [1], "definition")],
        primary_chunk_ids=[1],
        supporting_chunk_ids=[],
        include_example=False,
        include_analogy=False,
        include_prerequisites=False,
        include_related_concepts=[],
        comparison_axes=[],
        lecture_scope=[],
        section_specs=[],
        evidence_bundles={},
    )
    bad = "Course Answer:\n\n### Direct Answer\nsoftmax\n\n### Explanation\nfor example, consider..."
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[10], kb=get_kb())
    assert "must_not_have_examples_when_blocked" in vr.checks_failed


def test_intuition_only_flags_gradient():
    intent = analyze_query("Explain CNN with intuition only")
    sq = _sq(intent)
    plan = AnswerPlan(
        answer_mode="multi_step_explanation",
        sections=[],
        primary_chunk_ids=[1],
        supporting_chunk_ids=[],
        include_example=False,
        include_analogy=False,
        include_prerequisites=False,
        include_related_concepts=[],
        comparison_axes=[],
        lecture_scope=[],
        section_specs=[],
        evidence_bundles={},
    )
    bad = "Course Answer:\n\n### Direct Answer\n\nWe use gradient descent to train layers."
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[10], kb=get_kb())
    assert "must_not_have_technical_when_intuition_only" in vr.checks_failed


def test_boilerplate_summary_detected():
    intent = QueryIntent(
        query_type=QueryType.SUMMARY,
        original_query="Summarize lecture 5",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[5],
        detected_concepts=[],
        compare_concepts=None,
        compare_entities=[],
    )
    sq = _sq(intent)
    plan = AnswerPlan(
        answer_mode="lecture_summary",
        sections=[],
        primary_chunk_ids=[1],
        supporting_chunk_ids=[],
        include_example=False,
        include_analogy=False,
        include_prerequisites=False,
        include_related_concepts=[],
        comparison_axes=[],
        lecture_scope=[5],
        section_specs=[],
        evidence_bundles={},
    )
    bad = "Lecture 5: this lecture thread builds definitions"
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[5], kb=get_kb())
    assert "must_not_be_boilerplate_summary" in vr.checks_failed
