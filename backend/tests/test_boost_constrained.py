"""Constrained boost payload shape (no live HTTP)."""

from __future__ import annotations

import json

from app.services.answers.concept_constraints import ConceptConstraints
from app.services.boost_deferred import _validate_boost_output
from app.services.generation.llm import _OPENAI_BOOST_CONSTRAINED_SYSTEM


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
