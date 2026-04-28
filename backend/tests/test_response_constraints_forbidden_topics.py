"""Parsing user 'do not mention X' constraints from free text."""

from __future__ import annotations

from app.services.answers.response_constraints import parse_response_constraints


def test_do_not_mention_phrase():
    rc = parse_response_constraints(
        "What is CNN? Do not mention transformers or residuals."
    )
    low = [t.lower() for t in rc.forbidden_topics]
    assert "transformers" in low and "residuals" in low


def test_without_mentioning():
    rc = parse_response_constraints(
        "What is MFCC without mentioning softmax."
    )
    topics = [t.lower() for t in rc.forbidden_topics]
    assert any("softmax" in t for t in topics)


def test_exclude_clause():
    rc = parse_response_constraints("Explain DP. Exclude neural networks.")
    assert rc.forbidden_topics


def test_do_not_mention_neural_networks():
    rc = parse_response_constraints(
        "What is dynamic programming? Do not mention neural networks."
    )
    blob = " ".join(rc.forbidden_topics).lower()
    assert "neural" in blob or "network" in blob
