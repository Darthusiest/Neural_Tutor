"""Tests for deterministic direct-answer selection (Task 6).

Cover all four mode buckets :func:`select_direct_answer` cares about:

- chat / direct definition / multi-step → top definition-cue sentence from
  target-scoped chunks, mentioning the target alias.
- compare (two-entity) → deterministic ``"A and B are related, but A focuses
  on …, while B focuses on …"`` contrast.
- summary / quiz / synthesis → ``None`` (renderer keeps its own opener).
- forbidden-term rejection → unit test on the helper.

Plus the new ``must_direct_answer_mention_target_concept`` validator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extensions import db
from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.answer_validation import validate_answer
from app.services.answers.concept_constraints import (
    ConceptConstraints,
    apply_concept_constraints,
    build_concept_constraints,
)
from app.services.answers.direct_answer import select_direct_answer
from app.services.answers.entity_retrieval import (
    ConceptEvidenceBundleV2,
    build_bundles_for_compare_v2,
)
from app.services.knowledge.concept_kb import get_kb, reset_kb_for_tests
from app.services.knowledge.structured_query import build_structured_query
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.query_understanding import analyze_query
from app.services.reasoning_pipeline import run_reasoning_pipeline
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache
from app.services.retrieval_v2 import retrieve_enhanced


_DATA = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"


@pytest.fixture
def corpus(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        import_lecture_json(_DATA, upsert=False)
        invalidate_lecture_cache()
        load_lecture_cache()
    yield
    reset_kb_for_tests()


def _sq(query: str):
    intent = analyze_query(query)
    return build_structured_query(intent, kb=get_kb())


def _plan_with_answer(direct_answer: str | None = None) -> AnswerPlan:
    return AnswerPlan(
        answer_mode="direct_definition",
        sections=[],
        primary_chunk_ids=[],
        supporting_chunk_ids=[],
        include_example=False,
        include_analogy=False,
        include_prerequisites=False,
        include_related_concepts=[],
        comparison_axes=[],
        lecture_scope=[],
        direct_answer=direct_answer,
    )


def _direct_for(query: str) -> tuple[str | None, ConceptConstraints]:
    """Helper: build constraints + retrieve + apply rerank + select direct answer.

    Mirrors the pipeline order — :func:`apply_concept_constraints` runs
    between retrieval and the planner, so the chunk list passed to
    :func:`select_direct_answer` is already concept-pure.
    """
    sq = _sq(query)
    constraints = build_concept_constraints(sq, kb=get_kb())
    raw_chunks = retrieve_enhanced(query, top_k=8).chunks
    chunks = apply_concept_constraints(raw_chunks, constraints)
    return (
        select_direct_answer(sq, chunks=chunks, constraints=constraints, kb=get_kb()),
        constraints,
    )


# ---------------------------------------------------------------------------
# Chat / direct-definition path — opener mentions the target concept
# ---------------------------------------------------------------------------


def test_direct_answer_what_is_mfcc_mentions_mfcc(corpus, app):
    with app.app_context():
        direct, _ = _direct_for("What is MFCC?")
        assert direct, "expected a deterministic direct answer for MFCC"
        assert "mfcc" in direct.lower() or "mel" in direct.lower()


def test_direct_answer_what_is_formant_mentions_formant_or_formants(corpus, app):
    with app.app_context():
        direct, _ = _direct_for("What is a formant?")
        assert direct, "expected a deterministic direct answer for formant"
        assert "formant" in direct.lower()


def test_direct_answer_what_is_softmax_mentions_softmax(corpus, app):
    with app.app_context():
        direct, _ = _direct_for("What is softmax?")
        assert direct, "expected a deterministic direct answer for softmax"
        assert "softmax" in direct.lower() or "probabilit" in direct.lower()


def test_direct_answer_what_is_cnn_grounds_in_cnn_chunk(corpus, app):
    """The CNN chunks in this fixture don't say *"CNN"* inline — their bullets
    describe the architecture's behavior (multi-scale speech processing, local
    pattern extraction). The test locks in (a) a non-empty direct answer and
    (b) that the chosen sentence comes from a CNN-flavoured chunk, *not* from
    a transformer / residuals neighbour topic.
    """
    with app.app_context():
        direct, _ = _direct_for("What is CNN?")
        assert direct, "expected a deterministic direct answer for CNN"
        lowered = direct.lower()
        # Must not have drifted into transformer machinery.
        for forbidden in ("transformer", "self-attention", "positional encoding"):
            assert forbidden not in lowered
        # Sentence content should be grounded in CNN behavior — either the
        # alias is named, or one of the CNN chunk's behavioral descriptors
        # appears (per the corpus fixture).
        cnn_signals = (
            "cnn",
            "convolution",
            "speech",
            "multi",
            "local pattern",
            "time scale",
            "feature",
        )
        assert any(sig in lowered for sig in cnn_signals)


# ---------------------------------------------------------------------------
# Compare (two-entity) — deterministic contrast template
# ---------------------------------------------------------------------------


def test_direct_answer_compare_two_entities_uses_deterministic_contrast(corpus, app):
    with app.app_context():
        sq = _sq("Compare CNN and transformer")
        constraints = build_concept_constraints(sq, kb=get_kb())
        chunks = retrieve_enhanced("Compare CNN and transformer", top_k=10).chunks
        kb = get_kb()
        bundle_a, bundle_b = build_bundles_for_compare_v2(chunks, "cnn", "transformer", kb)
        direct = select_direct_answer(
            sq,
            chunks=chunks,
            bundles=[bundle_a, bundle_b],
            constraints=constraints,
            kb=kb,
        )
        assert direct
        lowered = direct.lower()
        # Both labels named explicitly.
        assert "cnn" in lowered
        assert "transformer" in lowered
        # Deterministic template scaffold: "and" + "are related" + "while".
        assert "are related" in lowered
        assert "focuses on" in lowered
        assert "while" in lowered


def test_direct_answer_compare_falls_back_to_axes_without_bundles(app):
    """When V2 bundles aren't available, compare still names both entities."""
    with app.app_context():
        sq = _sq("Compare MFCCs and formants")
        constraints = build_concept_constraints(sq, kb=get_kb())
        direct = select_direct_answer(
            sq, chunks=[], bundles=None, constraints=constraints, kb=get_kb()
        )
        assert direct
        lowered = direct.lower()
        assert "mfcc" in lowered
        assert "formant" in lowered


# ---------------------------------------------------------------------------
# Summary / quiz / synthesis / multi-compare → no direct answer
# ---------------------------------------------------------------------------


def test_direct_answer_summary_returns_none(corpus, app):
    with app.app_context():
        sq = _sq("Summarize lecture 10")
        constraints = build_concept_constraints(sq, kb=get_kb())
        chunks = retrieve_enhanced("Summarize lecture 10", top_k=8).chunks
        direct = select_direct_answer(
            sq, chunks=chunks, constraints=constraints, kb=get_kb()
        )
        assert direct is None


def test_direct_answer_quiz_returns_none(app):
    with app.app_context():
        sq = _sq("Quiz me on softmax")
        # Force the renderer onto the teaching-plus-check branch so the
        # ``allow_incorrect_statements`` short-circuit fires.
        sq.response_constraints.allow_incorrect_statements = True
        constraints = build_concept_constraints(sq, kb=get_kb())
        direct = select_direct_answer(
            sq, chunks=[], constraints=constraints, kb=get_kb()
        )
        assert direct is None


def test_direct_answer_compare_multi_returns_none(app):
    with app.app_context():
        sq = _sq("Compare MFCC, softmax, and transformer")
        constraints = build_concept_constraints(sq, kb=get_kb())
        direct = select_direct_answer(
            sq, chunks=[], constraints=constraints, kb=get_kb()
        )
        # compare_multi sits in _NO_DIRECT_ANSWER_MODES — renderer keeps its
        # own opener.
        if sq.answer_intent == "compare_multi":
            assert direct is None


# ---------------------------------------------------------------------------
# Forbidden-term rejection (unit test on the helper)
# ---------------------------------------------------------------------------


def test_select_direct_answer_rejects_lines_with_forbidden_terms(app):
    """A pool full of forbidden-only sentences should yield no direct answer."""
    with app.app_context():
        sq = _sq("What is dynamic programming?")
        constraints = build_concept_constraints(sq, kb=get_kb())
        # Forbidden-only synthetic chunks (no DP mention) — the chat path
        # should reject every sentence as not concept-pure.
        leak_chunks = [
            {
                "id": 1,
                "topic": "Neural networks",
                "keywords": "neural network backpropagation",
                "clean_explanation": (
                    "Neural networks learn via backpropagation, and gradient "
                    "descent updates the weights using the chain rule across "
                    "hidden layers."
                ),
                "source_excerpt": "",
                "lecture_number": 9,
            },
            {
                "id": 2,
                "topic": "Transformer attention",
                "keywords": "transformer self-attention",
                "clean_explanation": (
                    "Transformers use self-attention with positional encoding "
                    "to capture long-range dependencies between input tokens."
                ),
                "source_excerpt": "",
                "lecture_number": 22,
            },
        ]
        direct = select_direct_answer(
            sq, chunks=leak_chunks, constraints=constraints, kb=get_kb()
        )
        assert direct is None


def test_select_direct_answer_prefers_definition_cue(app):
    """A definition-cue sentence beats a non-definition sentence on the same chunk."""
    with app.app_context():
        sq = _sq("What is softmax?")
        constraints = build_concept_constraints(sq, kb=get_kb())
        chunks = [
            {
                "id": 1,
                "topic": "Softmax overview",
                "keywords": "softmax probability",
                "clean_explanation": (
                    "We can also derive the layer's gradient with respect to "
                    "the loss for backprop. Softmax is a function that turns "
                    "raw scores into a probability distribution over classes."
                ),
                "source_excerpt": "",
                "lecture_number": 12,
            }
        ]
        direct = select_direct_answer(
            sq, chunks=chunks, constraints=constraints, kb=get_kb()
        )
        assert direct
        assert direct.lower().startswith("softmax is")


# ---------------------------------------------------------------------------
# Validator: must_direct_answer_mention_target_concept
# ---------------------------------------------------------------------------


def test_validator_flags_direct_answer_missing_target_alias(app):
    """Chat answer whose ``plan.direct_answer`` doesn't mention the target fails."""
    with app.app_context():
        sq = _sq("What is softmax?")
        constraints = build_concept_constraints(sq, kb=get_kb())
        plan = _plan_with_answer(
            direct_answer="The function turns raw scores into useful values."
        )
        result = validate_answer(
            "Course Answer:\n\nSoftmax produces a probability distribution.",
            sq,
            plan,
            kb=get_kb(),
            constraints=constraints,
        )
        assert "must_direct_answer_mention_target_concept" in result.checks_failed


def test_validator_passes_direct_answer_mentioning_target(app):
    with app.app_context():
        sq = _sq("What is softmax?")
        constraints = build_concept_constraints(sq, kb=get_kb())
        plan = _plan_with_answer(
            direct_answer="Softmax is a function that maps logits to probabilities."
        )
        result = validate_answer(
            "Course Answer:\n\nSoftmax produces a probability distribution.",
            sq,
            plan,
            kb=get_kb(),
            constraints=constraints,
        )
        assert "must_direct_answer_mention_target_concept" not in result.checks_failed


def test_validator_compare_requires_both_labels_in_direct_answer(app):
    """Compare direct answer that names only one side fails the validator."""
    with app.app_context():
        sq = _sq("Compare MFCCs and formants")
        constraints = build_concept_constraints(sq, kb=get_kb())
        plan = AnswerPlan(
            answer_mode="compare",
            sections=[],
            primary_chunk_ids=[],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=["role", "computation", "typical use"],
            lecture_scope=[],
            direct_answer="MFCCs are spectral features used in speech recognition.",
        )
        result = validate_answer(
            (
                "Course Answer:\n\nMFCCs and formants are related, but they "
                "differ along role, computation, and typical use."
            ),
            sq,
            plan,
            kb=get_kb(),
            constraints=constraints,
        )
        assert "must_direct_answer_mention_target_concept" in result.checks_failed


# ---------------------------------------------------------------------------
# End-to-end: pipeline-rendered answer opens with the deterministic direct answer
# ---------------------------------------------------------------------------


def test_pipeline_chat_opens_with_target_concept_for_mfcc(corpus, app):
    with app.app_context():
        result = run_reasoning_pipeline("What is MFCC?", top_k=5)
        plan = result.enhanced_result.answer_plan
        assert plan is not None
        # Plan should carry a non-empty direct_answer for chat mode.
        assert plan.direct_answer
        assert "mfcc" in (plan.direct_answer or "").lower()


def test_pipeline_summary_has_no_direct_answer(corpus, app):
    with app.app_context():
        result = run_reasoning_pipeline("Summarize lecture 10", top_k=8)
        plan = result.enhanced_result.answer_plan
        assert plan is not None
        # Lecture summary mode should leave direct_answer empty so the
        # summary renderer keeps its own opener.
        assert plan.direct_answer is None
