"""Stricter validator rules (forbidden lemmas, constraints, boilerplate)."""

from app.services.answers.answer_planning import AnswerPlan, AnswerSection
from app.services.answers.entity_retrieval import ConceptEvidenceBundleV2
from app.services.knowledge.structured_query import StructuredQuery, build_structured_query
from app.services.query_understanding import QueryIntent, QueryType, analyze_query
from app.services.knowledge.concept_kb import get_kb
from app.services.answers.answer_validation import validate_answer


def _sq(intent: QueryIntent) -> StructuredQuery:
    return build_structured_query(intent, kb=get_kb())


def _empty_plan(answer_mode: str = "multi_step_explanation", **overrides) -> AnswerPlan:
    """Tiny ``AnswerPlan`` factory for hardening tests.

    All non-default fields are zero/empty by design — individual tests
    only override what they care about (mode, evidence_bundles, etc.).
    """
    base = dict(
        answer_mode=answer_mode,
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
    base.update(overrides)
    return AnswerPlan(**base)


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


# ---------------------------------------------------------------------------
# Task 7 — hardened mode-contract / compare / duplication / direct-answer tests
# ---------------------------------------------------------------------------


def test_quiz_contract_rejects_course_answer_headings():
    intent = QueryIntent(
        query_type=QueryType.QUIZ,
        original_query="Quiz me on softmax",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[],
        detected_concepts=["softmax"],
        compare_concepts=None,
        compare_entities=[],
    )
    sq = _sq(intent)
    plan = _empty_plan(answer_mode="teaching_plus_check")
    bad = (
        "Quiz: Softmax\n\n"
        "### Direct Answer\n"
        "softmax normalizes logits.\n"
    )
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_match_quiz_contract" in vr.checks_failed
    assert vr.severity == "fail"
    assert vr.repair_path == "fall_back_to_clarification"


def test_summary_contract_rejects_course_answer_marker():
    intent = QueryIntent(
        query_type=QueryType.SUMMARY,
        original_query="Summarize Lecture 10",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[10],
        detected_concepts=[],
        compare_concepts=None,
        compare_entities=[],
    )
    sq = _sq(intent)
    plan = _empty_plan(answer_mode="lecture_summary", lecture_scope=[10])
    bad = (
        "Summary: Lecture 10\n\n"
        "Course Answer:\n\n"
        "Lecture 10 covers softmax and friends."
    )
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[10], kb=get_kb())
    assert "must_match_summary_contract" in vr.checks_failed
    assert vr.severity == "fail"
    assert vr.repair_path == "fall_back_to_clarification"


def test_compare_contract_requires_both_entities():
    intent = QueryIntent(
        query_type=QueryType.COMPARE,
        original_query="Compare CNN and MLP",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[],
        detected_concepts=[],
        compare_concepts=("CNN", "MLP"),
        compare_entities=["CNN", "MLP"],
    )
    sq = _sq(intent)
    plan = _empty_plan(answer_mode="compare")
    bad = (
        "Course Answer:\n\n"
        "### Direct Answer\n"
        "CNNs use convolutional kernels to process spatial data.\n"
    )
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_match_compare_contract" in vr.checks_failed
    assert vr.severity == "fail"
    assert vr.repair_path == "rebuild_evidence_bundles"


def test_compare_asymmetry_rejects_identical_evidence():
    intent = QueryIntent(
        query_type=QueryType.COMPARE,
        original_query="Compare CNN and MLP",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[],
        detected_concepts=[],
        compare_concepts=("CNN", "MLP"),
        compare_entities=["CNN", "MLP"],
    )
    sq = _sq(intent)
    shared_lines = [
        "Both architectures train via gradient descent.",
        "They use weighted sums of inputs.",
        "They learn parameters from data.",
    ]
    bundle_a = ConceptEvidenceBundleV2(
        concept="cnn",
        core_lines=list(shared_lines),
        support_score=0.5,
        label_override="CNN",
    )
    bundle_b = ConceptEvidenceBundleV2(
        concept="mlp",
        core_lines=list(shared_lines),
        support_score=0.5,
        label_override="MLP",
    )
    plan = _empty_plan(
        answer_mode="compare",
        evidence_bundles={"cnn": bundle_a, "mlp": bundle_b},
    )
    answer = (
        "Course Answer:\n\n"
        "### Direct Answer\n"
        "CNN and MLP both train via gradient descent.\n"
    )
    vr = validate_answer(answer, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_have_distinct_compare_evidence" in vr.checks_failed
    assert vr.repair_path in (
        "rebuild_evidence_bundles",
        # In case the compare-contract failure surfaces first the repair
        # path still maps to bundle rebuild for both cases.
    )


def test_compare_empty_side_without_limitation_phrase_fails():
    intent = QueryIntent(
        query_type=QueryType.COMPARE,
        original_query="Compare CNN and MLP",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[],
        detected_concepts=[],
        compare_concepts=("CNN", "MLP"),
        compare_entities=["CNN", "MLP"],
    )
    sq = _sq(intent)
    bundle_a = ConceptEvidenceBundleV2(
        concept="cnn",
        core_lines=[
            "CNNs slide kernels across inputs to extract spatial features.",
            "They share weights to keep parameter counts manageable.",
        ],
        support_score=0.5,
        label_override="CNN",
    )
    bundle_b = ConceptEvidenceBundleV2(
        concept="mlp",
        core_lines=[],
        support_score=0.0,
        label_override="MLP",
    )
    plan = _empty_plan(
        answer_mode="compare",
        evidence_bundles={"cnn": bundle_a, "mlp": bundle_b},
    )
    answer = (
        "Course Answer:\n\n"
        "### Direct Answer\n"
        "CNNs and MLPs both learn from data.\n"
        "CNNs slide kernels across inputs."
    )
    vr = validate_answer(answer, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_have_each_side_evidence_or_note" in vr.checks_failed
    assert vr.repair_path == "render_limitation_message"


def test_compare_empty_side_with_limitation_phrase_passes():
    intent = QueryIntent(
        query_type=QueryType.COMPARE,
        original_query="Compare CNN and MLP",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[],
        detected_concepts=[],
        compare_concepts=("CNN", "MLP"),
        compare_entities=["CNN", "MLP"],
    )
    sq = _sq(intent)
    bundle_a = ConceptEvidenceBundleV2(
        concept="cnn",
        core_lines=[
            "CNNs slide kernels across inputs to extract spatial features.",
        ],
        support_score=0.5,
        label_override="CNN",
    )
    bundle_b = ConceptEvidenceBundleV2(
        concept="mlp",
        core_lines=[],
        support_score=0.0,
        label_override="MLP",
    )
    plan = _empty_plan(
        answer_mode="compare",
        evidence_bundles={"cnn": bundle_a, "mlp": bundle_b},
    )
    answer = (
        "CNNs slide kernels across inputs.\n"
        "Limited direct material for MLP in retrieved notes; treat claims as provisional."
    )
    vr = validate_answer(answer, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_have_each_side_evidence_or_note" not in vr.checks_failed


def test_direct_answer_mismatch_fails_for_definition():
    intent = analyze_query("What is MFCC?")
    sq = _sq(intent)
    plan = _empty_plan(
        answer_mode="direct_definition",
        direct_answer="Transformers use self-attention to model sequences.",
    )
    answer = (
        "Course Answer:\n\n"
        "### Direct Answer\n"
        "Transformers use self-attention to model sequences."
    )
    vr = validate_answer(answer, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_direct_answer_match_target" in vr.checks_failed
    assert vr.repair_path == "retry_retrieval_with_stricter_constraints"


def test_section_duplication_rejects_repeated_heading():
    intent = analyze_query("What is softmax?")
    sq = _sq(intent)
    plan = _empty_plan(answer_mode="direct_definition")
    bad = (
        "Course Answer:\n\n"
        "### Direct Answer\n"
        "softmax converts logits to probabilities.\n\n"
        "### Direct Answer\n"
        "softmax is the same as the previous direct answer."
    )
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_not_have_section_duplication" in vr.checks_failed


def test_section_duplication_rejects_repeated_bullet():
    intent = analyze_query("What is softmax?")
    sq = _sq(intent)
    plan = _empty_plan(answer_mode="direct_definition")
    bad = (
        "Course Answer:\n\n"
        "### Explanation\n"
        "- softmax normalizes a vector of logits into a probability distribution.\n"
        "- softmax normalizes a vector of logits into a probability distribution.\n"
    )
    vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_not_have_section_duplication" in vr.checks_failed


def test_generic_filler_warns_without_failing():
    intent = analyze_query("What is softmax?")
    sq = _sq(intent)
    plan = _empty_plan(
        answer_mode="direct_definition",
        include_related_concepts=[],
    )
    answer = (
        "Course Answer:\n\n"
        "### Direct Answer\n"
        "softmax converts logits to probabilities.\n\n"
        "### Why it matters\n"
        "Solid intuition here makes the next topics easier to absorb."
    )
    vr = validate_answer(answer, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert vr.flags.get("generic_filler") is True
    assert "must_not_have_generic_filler" not in vr.checks_failed


def test_passing_quiz_answer_clears_contract():
    intent = QueryIntent(
        query_type=QueryType.QUIZ,
        original_query="Quiz me on softmax",
        expanded_query="",
        query_tokens=[],
        expanded_tokens=[],
        lecture_numbers=[],
        detected_concepts=["softmax"],
        compare_concepts=None,
        compare_entities=[],
    )
    sq = _sq(intent)
    plan = _empty_plan(answer_mode="teaching_plus_check")
    good = (
        "Quiz: Softmax\n\n"
        "1. What does softmax do?\n\n"
        "Answer Key:\n"
        "1. It converts logits into a probability distribution."
    )
    vr = validate_answer(good, sq, plan, primary_chunk_lecture_numbers=[], kb=get_kb())
    assert "must_match_quiz_contract" not in vr.checks_failed
    assert vr.repair_path is None or vr.severity != "fail"
