"""Tests for :mod:`app.services.answers.concept_answer_composer`."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extensions import db
from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.concept_answer_composer import (
    DEFINITION,
    EXAMPLE,
    KEY_IDEA,
    MECHANISM,
    RELEVANCE,
    classify_line,
    collect_role_buckets,
    compose_concept_answer,
)
from app.services.answers.concept_constraints import build_concept_constraints
from app.services.knowledge.concept_kb import get_kb, reset_kb_for_tests
from app.services.knowledge.structured_query import build_structured_query
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.query_understanding import analyze_query
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

_DATA = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"
_KB = Path(__file__).resolve().parent.parent / "data" / "LING487_STRUCTURED_PIPELINE_KB.json"


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


def test_classify_line_roles():
    assert classify_line("Softmax is a function that maps logits to probabilities.") == DEFINITION
    assert classify_line("It computes exp over each value and normalizes.") == MECHANISM
    assert classify_line("For example, [2, 5] becomes [0.05, 0.95].") == EXAMPLE
    assert classify_line("The key idea: turn scores into probabilities.") == KEY_IDEA
    assert classify_line("Useful for classification because outputs sum to one.") == RELEVANCE


def test_collect_role_buckets_dedupes_and_caps_per_role(app):
    with app.app_context():
        kb = get_kb(_KB)
        sq = build_structured_query(analyze_query("What is softmax?"), kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        dup_chunk = {
            "id": 101,
            "topic": "Softmax — overview",
            "keywords": "softmax",
            "clean_explanation": (
                "Softmax is a function that maps logits to probabilities.\n"
                "Softmax is a function that maps logits to probabilities.\n"
                "It computes exponentials and normalizes rows.\n"
                "It computes exponentials and normalizes rows.\n"
                "It computes exponentials and normalizes rows.\n"
                "The key idea: probabilities sum to one.\n"
                "The main idea: probabilities sum to one."
            ),
            "source_excerpt": "",
            "sample_answer": "",
        }
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[101],
            supporting_chunk_ids=[],
            include_example=True,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        out = collect_role_buckets(plan, [dup_chunk], constraints=c)
        buckets = out["buckets"]
        assert len(buckets.get(DEFINITION, [])) <= 2
        assert len(buckets.get(MECHANISM, [])) <= 2
        assert len(buckets[DEFINITION]) == 1


def test_collect_role_buckets_drops_off_topic_forbidden_line(app):
    """Pure CNN chunk survives; a transformer-only sentence must not enter buckets."""
    with app.app_context():
        kb = get_kb(_KB)
        sq = build_structured_query(analyze_query("What is CNN?"), kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        chunks = [
            {
                "id": 201,
                "topic": "CNNs — CNN",
                "keywords": "cnn convolution",
                "clean_explanation": (
                    "A convolutional neural network (CNN) is a model that applies convolution "
                    "to extract spatial structure from inputs."
                ),
                "source_excerpt": "",
                "sample_answer": "",
            },
            {
                "id": 202,
                "topic": "Transformers",
                "keywords": "attention",
                "clean_explanation": (
                    "Transformers use self-attention layers to relate tokens without recurrence."
                ),
                "source_excerpt": "",
                "sample_answer": "",
            },
        ]
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[201, 202],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        out = collect_role_buckets(plan, chunks, constraints=c)
        blob = " ".join(
            s.lower()
            for xs in out["buckets"].values()
            for s in xs
        )
        assert "transformer" not in blob


def test_compose_respects_no_example_when_example_bucket_empty(app, corpus):
    """When EXAMPLE bucket is empty, omit ``Think of it this way:`` (fixture stripped)."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("What is softmax?")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        from app.services.retrieval_v2 import retrieve_enhanced

        r = retrieve_enhanced("What is softmax?", top_k=5)
        stripped = []
        for chunk in r.chunks:
            copy = dict(chunk)
            copy["sample_answer"] = ""
            copy["sample_questions"] = ""
            copy["source_excerpt"] = ""
            stripped.append(copy)
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[x["id"] for x in stripped if x.get("id") is not None][:8],
            supporting_chunk_ids=[],
            include_example=True,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        text = compose_concept_answer(plan, stripped, sq, constraints=c)
        assert "Think of it this way:" not in text


def test_compose_prefers_plan_direct_answer(app, corpus):
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("What is softmax?")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        from app.services.retrieval_v2 import retrieve_enhanced

        r = retrieve_enhanced("What is softmax?", top_k=5)
        primary_ids = [x["id"] for x in r.chunks if x.get("id") is not None][:8]
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=primary_ids,
            supporting_chunk_ids=[],
            include_example=True,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer="Softmax is deliberately injected for this unit test.",
        )
        text = compose_concept_answer(plan, r.chunks, sq, constraints=c)
        assert "deliberately injected" in text


def test_compose_narrative_contract(app, corpus):
    """Course Answer narrative shape + no markdown bullets/section headings."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("What is softmax?")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        from app.services.retrieval_v2 import retrieve_enhanced

        r = retrieve_enhanced("What is softmax?", top_k=5)
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[x["id"] for x in r.chunks if x.get("id") is not None][:8],
            supporting_chunk_ids=[x["id"] for x in (r.supporting_chunks or []) if x.get("id")][:3],
            include_example=True,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        text = compose_concept_answer(plan, r.chunks + list(r.supporting_chunks or []), sq, constraints=c)
        assert text.startswith("Course Answer:")
        assert "The key idea:" in text
        assert "matters because" in text.lower()
        assert "### Direct Answer" not in text
        for ln in text.split("\n"):
            if ln.strip():
                assert not ln.lstrip().startswith(("- ", "* ", "• "))


def test_softmax_answer_does_not_bleed_mfcc(app, corpus):
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("What is softmax?")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        from app.services.retrieval_v2 import retrieve_enhanced

        r = retrieve_enhanced("What is softmax?", top_k=8)
        primary_ids = [x["id"] for x in r.chunks if x.get("id") is not None][:8]
        sup_ids = [x["id"] for x in (r.supporting_chunks or []) if x.get("id")][:3]
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=primary_ids,
            supporting_chunk_ids=sup_ids,
            include_example=True,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        pool = r.chunks + list(r.supporting_chunks or [])
        text = compose_concept_answer(plan, pool, sq, constraints=c).lower()
        assert "mfcc" not in text


def test_grammar_no_double_subject_after_works_by(app):
    """Standalone ``Softmax converts …`` must not become ``It works by softmax converts …``."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("How does softmax work?")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        chunk = {
            "id": 991,
            "topic": "Softmax — overview",
            "keywords": "softmax",
            "clean_explanation": (
                "Softmax converts logits to probabilities using exponentials and normalization."
            ),
            "source_excerpt": "",
            "sample_answer": "",
        }
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[991],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        text = compose_concept_answer(plan, [chunk], sq, constraints=c).lower()
        assert "it works by softmax converts" not in text


def test_grammar_no_pronoun_after_transition(app):
    """Bridge clauses must not surface ``This means that They …`` after stitching."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("Contrast softmax versus hardmax.")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        chunk = {
            "id": 992,
            "topic": "Softmax — contrast",
            "keywords": "softmax hardmax",
            "clean_explanation": (
                "Softmax spreads probability mass smoothly across logits instead of picking one winner.\n"
                "They emphasize larger logits without zeroing everything else.\n"
                "Hardmax assigns probability one to the largest logit only."
            ),
            "source_excerpt": "",
            "sample_answer": "",
        }
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[992],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        text = compose_concept_answer(plan, [chunk], sq, constraints=c)
        tl = text.lower()
        assert "this means that they " not in tl
        assert "as a result, captures" not in tl


def test_what_is_query_short_layout(app):
    """Definitional ``what is`` depth: opening + one mechanism sentence before key idea."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("What is softmax?")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        chunk = {
            "id": 993,
            "topic": "Softmax — overview",
            "keywords": "softmax",
            "clean_explanation": (
                "Softmax is a function that maps logits to probabilities.\n"
                "Softmax applies a smooth exponential tilt toward larger logits without collapsing smaller ones entirely."
            ),
            "source_excerpt": "",
            "sample_answer": "",
        }
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[993],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer="Softmax is a function that maps logits to probabilities.",
        )
        text = compose_concept_answer(plan, [chunk], sq, constraints=c)
        head = text.split("The key idea:")[0]
        parts = [p.strip() for p in head.split("\n\n") if p.strip()]
        parts = [p for p in parts if not p.startswith("Course Answer")]
        parts = [p for p in parts if not p.startswith("Think of it this way")]
        assert len(parts) == 2


def test_step_by_step_inline_sequence(app):
    """Process queries emit ``First, …; then, …`` inline markers."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("Walk me through MFCCs step by step.")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        chunk = {
            "id": 994,
            "topic": "MFCC — pipeline",
            "keywords": "mfcc mel fft",
            "clean_explanation": (
                "MFCC applies a window to each frame of speech.\n"
                "MFCC computes an FFT for local spectra.\n"
                "MFCC maps power onto mel-scaled frequency bins.\n"
                "MFCC compresses coefficients using the discrete cosine transform."
            ),
            "source_excerpt": "",
            "sample_answer": "",
        }
        plan = AnswerPlan(
            answer_mode="multi_step_explanation",
            sections=[],
            primary_chunk_ids=[994],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        text = compose_concept_answer(plan, [chunk], sq, constraints=c).lower()
        markers = ("first,", "then,", "next,", "finally,")
        assert sum(1 for m in markers if m in text) >= 2


def test_strong_why_it_matters_canonical_pattern(app):
    """Relevance lines feed canonical ``This matters because it allows the system to …``."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("Why softmax?")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        chunk = {
            "id": 995,
            "topic": "Softmax — relevance",
            "keywords": "softmax",
            "clean_explanation": (
                "It is used to map logits for classification decisions in neural networks."
            ),
            "source_excerpt": "",
            "sample_answer": "",
        }
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[995],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        text = compose_concept_answer(plan, [chunk], sq, constraints=c)
        assert "This matters because it allows the system to " in text
        assert "which is important for " in text


def test_failsafe_drops_broken_segments(app):
    """Never emit stitched ``As a result, Captures`` / ``This means that They`` fragments."""
    with app.app_context():
        kb = get_kb(_KB)
        intent = analyze_query("Explain MFCC briefly.")
        sq = build_structured_query(intent, kb=kb)
        c = build_concept_constraints(sq, kb=kb)
        chunk = {
            "id": 996,
            "topic": "MFCC — overview",
            "keywords": "mfcc",
            "clean_explanation": (
                "Captures spectral envelope compactly.\n"
                "They align with human pitch perception.\n"
                "Used for speech recognition pipelines."
            ),
            "source_excerpt": "",
            "sample_answer": "",
        }
        plan = AnswerPlan(
            answer_mode="direct_definition",
            sections=[],
            primary_chunk_ids=[996],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=[],
            lecture_scope=[],
            direct_answer=None,
        )
        text = compose_concept_answer(plan, [chunk], sq, constraints=c)
        tl = text.lower()
        assert text.startswith("Course Answer:")
        assert "as a result, captures" not in tl
        assert "this means that they " not in tl
        assert tl.strip().endswith(".") or "matters because" in tl
