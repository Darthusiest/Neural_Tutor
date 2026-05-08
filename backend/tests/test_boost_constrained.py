"""Constrained boost payload shape (no live HTTP)."""

from __future__ import annotations

import json

from app.services.answers.concept_constraints import ConceptConstraints
from app.services.boost_deferred import (
    _cap_marker_prefixed_sentences,
    _evidence_fallback_from_course_answer,
    _validate_boost_output,
    course_answer_is_shallow,
)
from app.services.generation.llm import (
    _OPENAI_BOOST_CONSTRAINED_SYSTEM,
    _OPENAI_BOOST_EXTENDED_SYSTEM,
)


def _empty_constraints() -> ConceptConstraints:
    return ConceptConstraints(
        target_concepts=[],
        target_aliases=set(),
        allowed_terms=set(),
        forbidden_terms=set(),
    )


def test_validate_boost_keeps_output_when_forbids_clear():
    c = _empty_constraints()
    valid, reason = _validate_boost_output(
        "Boosted Explanation:\n\nNormalization exponentials stabilize logits numerically.",
        c,
    )
    assert valid is not None
    assert reason is None


def test_constrained_boost_system_mentions_clarity_only():
    assert "improving clarity ONLY" in _OPENAI_BOOST_CONSTRAINED_SYSTEM
    assert "allowed_evidence_lines" in _OPENAI_BOOST_CONSTRAINED_SYSTEM
    assert "forbidden_terms" in _OPENAI_BOOST_CONSTRAINED_SYSTEM


def test_boost_payload_size_caps():
    payload = {
        "target_concept": "cnn",
        "allowed_evidence_lines": ["x" * 500 for _ in range(10)],
        "forbidden_terms": ["a"] * 50,
        "draft_answer": "y" * 20_000,
        "mode": "chat",
    }
    lines = [ln[:400] for ln in (payload["allowed_evidence_lines"] or [])[:5]]
    assert len(lines) == 5
    assert all(len(l) <= 400 for l in lines)
    slim = {
        "target_concept": payload["target_concept"],
        "allowed_evidence_lines": lines,
        "forbidden_terms": list(payload["forbidden_terms"])[:40],
        "draft_answer": (payload["draft_answer"] or "")[:8000],
        "mode": payload["mode"],
    }
    assert len(json.dumps(slim)) < 100_000


def test_extended_system_prompt_documents_marker_phrases():
    low = _OPENAI_BOOST_EXTENDED_SYSTEM.lower()
    assert "a useful clarification is" in low
    assert "in standard speech processing terms" in low
    assert "in standard machine learning terms" in low
    assert "more generally" in low
    assert "one new technical term" in low


def test_validate_rejects_unmarked_external_sentence():
    c = _empty_constraints()
    allowed = ["Softmax maps logits into probabilities."]
    boost = (
        "Boosted Explanation:\n\n"
        "Softmax maps logits.\n"
        "Rocket staging uses hydrolox fuels unrelated to lecture classification pipelines."
    )
    valid, reason = _validate_boost_output(
        boost,
        c,
        allowed_evidence_lines=allowed,
        allow_external_clarification=True,
    )
    assert valid is None
    assert reason == "unmarked_external"


def test_validate_caps_marker_sentences_at_two():
    c = _empty_constraints()
    allowed = ["Softmax maps logits into probabilities."]
    boost = (
        "Boosted Explanation:\n\n"
        "Softmax maps logits into probabilities.\n"
        "More generally softmax behaves smoothly.\n"
        "More generally softmax preserves uncertainty.\n"
        "More generally softmax drops excess wording.\n"
    )
    valid, reason = _validate_boost_output(
        boost,
        c,
        allowed_evidence_lines=allowed,
        allow_external_clarification=True,
    )
    assert reason is None
    assert valid is not None
    assert "drops excess wording" not in (valid or "").lower()


def test_validate_rejects_when_two_new_terms_introduced():
    c = _empty_constraints()
    allowed = ["Softmax maps logits into probabilities."]
    boost = (
        "Boosted Explanation:\n\n"
        "Softmax maps logits.\n"
        "More generally Melbins correlate spectra.\n"
        "More generally Cryptoz diverge oddly.\n"
    )
    valid, reason = _validate_boost_output(
        boost,
        c,
        allowed_evidence_lines=allowed,
        allow_external_clarification=True,
    )
    assert valid is None
    assert reason == "too_many_new_terms"


def test_cap_marker_prefixed_sentences_keeps_two_markers_only():
    raw = (
        "Boosted Explanation:\n\n"
        "Intro grounding sentence.\n"
        "More generally first marker.\n"
        "More generally second marker.\n"
        "More generally third marker dropped.\n"
    )
    capped = _cap_marker_prefixed_sentences(raw, max_markers=2)
    assert "third marker dropped" not in capped.lower()


def test_course_answer_is_shallow_thresholds():
    long_body = "x" * 601
    ev_many = ["line one about softmax here.", "line two softmax.", "line three softmax."]
    assert course_answer_is_shallow("short", ev_many) is True
    assert course_answer_is_shallow(long_body, ev_many) is False

    ev_few = ["only one softmax evidence line here."]
    assert course_answer_is_shallow(long_body, ev_few) is True


def test_evidence_fallback_from_course_answer_strips_heading_and_splits():
    ca = (
        "Course Answer:\n\n"
        "This is a long enough first line about CNNs for unit test.\n"
        "Second line continues the course material for grounding."
    )
    lines = _evidence_fallback_from_course_answer(ca)
    assert len(lines) == 2
    assert "cnn" in lines[0].lower()

    tiny = "Course Answer:\n\nok"
    assert _evidence_fallback_from_course_answer(tiny) == ["ok"]
