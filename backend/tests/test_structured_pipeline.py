"""Tests for structured reasoning pipeline (concept KB, query, plan, validation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extensions import db
from app.services.knowledge.concept_kb import get_kb, load_concept_kb, reset_kb_for_tests
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.query_understanding import analyze_query
from app.services.reasoning_pipeline import run_reasoning_pipeline
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache
from app.services.knowledge.structured_query import build_structured_query, decompose_query
from app.services.answers.answer_generation import generate_structured_answer
from app.services.answers.answer_planning import build_answer_plan
from app.services.answers.answer_validation import validate_answer

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


class TestConceptKB:
    def test_load_kb(self):
        kb = load_concept_kb(_KB)
        assert kb.get_concept_by_id("softmax") is not None
        assert kb.get_comparison_axes("mfcc", "formants")

    def test_find_concepts_in_tokens(self):
        kb = load_concept_kb(_KB)
        found = kb.find_concepts_in_text(["what", "is", "softmax"])
        ids = {c.id for c in found}
        assert "softmax" in ids


class TestStructuredQuery:
    def test_softmax_definition(self, corpus, app):
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax?")
            sq = build_structured_query(intent, kb=kb)
            assert sq.answer_intent == "direct_definition"
            assert sq.sub_questions

    def test_compare_mfcc_formants(self, corpus, app):
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("difference between MFCCs and formants")
            sq = build_structured_query(intent, kb=kb)
            assert sq.answer_intent == "compare"
            subs = decompose_query(intent, kb, kb.find_concepts_in_text(intent.query_tokens))
            assert len(subs) >= 2


class TestAnswerPlanning:
    def test_plan_has_sections(self, corpus, app):
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            assert plan.answer_mode == "direct_definition"
            assert plan.sections

    def test_direct_definition_distinct_chunk_per_section(self, corpus, app):
        """Avoid assigning the same top chunks to every ### section (duplicated excerpts)."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=8)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            assert plan.answer_mode == "direct_definition"
            seen: set[int] = set()
            for sec in plan.sections:
                for cid in sec.chunk_ids:
                    assert cid not in seen, f"chunk {cid} reused across sections"
                    seen.add(cid)


class TestValidation:
    def test_compare_missing_side_fails(self, corpus, app):
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("difference between MFCCs and formants")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced(intent.original_query, top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            bad = "Course Answer:\n\nMFCCs are features."
            vr = validate_answer(bad, sq, plan, primary_chunk_lecture_numbers=[10], kb=kb)
            assert "must_cover_both_sides" in vr.checks_failed
            assert vr.severity == "fail"


class TestRuleBasedTutorFormat:
    def test_generate_structured_answer_four_sections(self, corpus, app):
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, r.chunks, sq)
            assert "### Direct Answer" in text
            assert "### Explanation" in text
            assert "### Example / Intuition" in text
            assert "### Why it matters" in text

    def test_compare_answer_no_per_line_scaffold_spam(self, corpus, app):
        """Regression: compare mode must not repeat 'First idea' / 'In one line' on every bullet."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("Compare MFCCs and formants")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced(intent.original_query, top_k=8)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            assert plan.answer_mode == "compare"
            text = generate_structured_answer(plan, r.chunks, sq)
            assert text.count("**First idea:**") <= 1
            assert text.count("**In one line:**") <= 1
            assert text.count("**Second idea:**") <= 1


class TestEndToEndPipeline:
    def test_pipeline_returns_answer(self, corpus, app):
        with app.app_context():
            pr = run_reasoning_pipeline("What is softmax?", top_k=5)
            assert pr.enhanced_result.chunks
            assert "Course Answer" in pr.course_answer
            assert pr.validation is not None

    def test_summary_query(self, corpus, app):
        with app.app_context():
            pr = run_reasoning_pipeline("summary of lecture 10", top_k=8)
            assert pr.structured_query.answer_intent == "lecture_summary"
