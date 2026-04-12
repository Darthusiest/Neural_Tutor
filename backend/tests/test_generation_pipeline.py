"""Tests for clean generation input and output post-processing."""

from __future__ import annotations

from app.services.generation.generation_input import format_generation_prompt_user_message
from app.services.generation.output_cleanup import clean_output, enforce_structure


def test_format_generation_prompt_user_message_shape():
    body = format_generation_prompt_user_message(
        {
            "question": "What is softmax?",
            "concepts": ["softmax"],
            "answer_mode": "direct_definition",
            "primary_content": [
                "Softmax converts outputs into probabilities.",
                "It ensures outputs sum to 1.",
            ],
            "supporting_content": [
                "It transforms arbitrary scores into a distribution.",
            ],
        }
    )
    assert "Question:\nWhat is softmax?" in body
    assert "Concepts:\nsoftmax" in body
    assert "Primary Content:" in body
    assert "Softmax converts outputs into probabilities." in body
    assert "Supporting Content:" in body
    assert "1. Softmax" not in body  # primary/supporting are joined paragraphs, not numbered lists


def test_clean_output_strips_leaky_lines():
    raw = """Course Answer:

### Direct Answer
Test.

keywords: foo bar
### Explanation
More.
"""
    out = clean_output(raw)
    assert "keywords" not in out.lower()


def test_enforce_structure_appends_missing():
    incomplete = """Course Answer:

### Direct Answer
Only this.
"""
    out = enforce_structure(incomplete)
    assert "### Explanation" in out
    assert "### Example / Intuition" in out or "### Example" in out
    assert "### Why it matters" in out
