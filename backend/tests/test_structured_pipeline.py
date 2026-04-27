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
    def test_chat_mode_uses_tutor_narrative_format(self, corpus, app):
        """Chat-mode answers flow as a tutor narrative (no '###' section headings)."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, r.chunks, sq)
            assert text.startswith("Course Answer:")
            assert "### Direct Answer" not in text
            assert "### Explanation" not in text
            assert "### Example / Intuition" not in text
            assert "### Why it matters" not in text
            assert "The key idea:" in text
            assert "That matters because" in text

    def test_chat_mode_no_examples_uses_tutor_narrative_without_example_block(
        self, corpus, app
    ):
        """Under ``no_examples`` chat-mode keeps the tutor narrative but drops the example."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax? Please don't use examples.")
            sq = build_structured_query(intent, kb=kb)
            sq.response_constraints.no_examples = True
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, r.chunks, sq)
            assert "### Direct Answer" not in text
            assert "### Why it matters" not in text
            assert "The key idea:" in text
            assert "Think of it this way:" not in text

    def test_chat_mode_drops_legacy_section_labels(self, corpus, app):
        """Output must not contain any of the legacy section labels (Task 8)."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("How does backpropagation work?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced(intent.original_query, top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, r.chunks, sq)
            for forbidden in (
                "### Direct Answer",
                "### Explanation",
                "### Example / Intuition",
                "### Why it matters",
                "Direct Answer",
                "Example / Intuition",
                "Why it matters",
            ):
                assert forbidden not in text, f"chat output unexpectedly contains '{forbidden}'"
            # The only explicit label kept in chat output is "The key idea:".
            assert "The key idea:" in text

    def test_chat_mode_natural_paragraph_flow(self, corpus, app):
        """Chat output reads as short paragraphs, not bullet spam or dense blocks (Task 4 + 8)."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("How does backpropagation work?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced(intent.original_query, top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, r.chunks, sq)
            # No bullet markers in chat-mode tutor output.
            content_lines = [ln for ln in text.split("\n") if ln.strip() and ln.strip() != "Course Answer:"]
            for line in content_lines:
                assert not line.lstrip().startswith(("- ", "* ", "• ")), (
                    f"chat output unexpectedly contains a bullet: {line!r}"
                )
            # Paragraphs are visually separated: at least one blank line in the body.
            assert "\n\n" in text
            # Closer flows with causal language so it doesn't read like a data dump.
            assert "That matters because" in text

    def test_chat_mode_softmax_example_and_no_repeated_lines(self, corpus, app):
        """Softmax-specific: bracketed example, probability framing, key idea, no duplicates (Task 8)."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, r.chunks, sq)
            assert "The key idea:" in text
            # Probability framing (Softmax → probabilities) — case-insensitive.
            lowered = text.lower()
            assert "probabilit" in lowered, (
                "softmax answer must mention probabilities to ground the explanation"
            )
            # Bracketed numeric example like "[2,5]" or "[2, 5]" should appear when
            # the corpus carries it. The fixture corpus does, so this is locked in.
            assert "[2," in text, "expected a bracketed numeric example like [2,5]"
            # No repeated content lines (paragraph dedupe + sentence dedupe in renderer).
            normalized = [
                " ".join(ln.lower().split()).rstrip(".!?:—-")
                for ln in text.split("\n")
                if ln.strip()
            ]
            assert len(normalized) == len(set(normalized)), (
                "tutor output should not repeat any non-empty line verbatim"
            )

    def test_chat_mode_skips_example_block_when_no_concrete_example(self, corpus, app):
        """Task 6: when no good example exists the example block is skipped, not forced."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("What is softmax?")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=5)
            # Strip example data off every chunk so _example_intuition_block falls back.
            stripped = []
            for chunk in r.chunks:
                copy = dict(chunk)
                copy["sample_answer"] = ""
                copy["sample_questions"] = ""
                copy["source_excerpt"] = ""
                stripped.append(copy)
            plan = build_answer_plan(sq, stripped, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, stripped, sq)
            assert "Think of it this way:" not in text, (
                "renderer should not force an example block when no concrete example is available"
            )
            assert "Think of the explanation above as the core picture" not in text
            assert "The key idea:" in text

    def test_chat_mode_repeat_explanation_keeps_legacy_layout(self, corpus, app):
        """Structured-explanation constraints keep the legacy section layout."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("Explain softmax twice please.")
            sq = build_structured_query(intent, kb=kb)
            sq.response_constraints.repeat_explanation_times = 2
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced("What is softmax?", top_k=5)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            text = generate_structured_answer(plan, r.chunks, sq)
            assert "### Direct Answer" in text
            assert "### Repeated explanation (as requested)" in text

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

    def test_compare_contrast_section_not_placeholder(self, corpus, app):
        """Contrast must use paired evidence, not generic 'contrast the two using the definitions' copy."""
        with app.app_context():
            kb = get_kb(_KB)
            intent = analyze_query("Compare MFCCs and formants")
            sq = build_structured_query(intent, kb=kb)
            from app.services.retrieval_v2 import retrieve_enhanced

            r = retrieve_enhanced(intent.original_query, top_k=8)
            plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
            assert plan.answer_mode == "compare"
            text = generate_structured_answer(plan, r.chunks, sq)
            assert "contrast the two using the definitions above" not in text.lower()
            assert "### Contrast along course axes" in text


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
