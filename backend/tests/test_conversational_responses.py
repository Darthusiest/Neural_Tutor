"""Tests for no-chunk conversational replies (classification + template variety)."""

from __future__ import annotations

import pytest

from app.services.conversational_responses import (
    classify_no_match_query,
    varied_no_chunk_course_answer,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hi", "greeting"),
        ("Hello there!", "greeting"),
        ("HEY", "greeting"),
        ("good morning", "greeting"),
        ("what's up", "greeting"),
        ("thanks", "short_ack"),
        ("ok", "short_ack"),
        ("got it", "short_ack"),
        ("wsg twin", "off_topic"),
        ("why is the sky blue", "off_topic"),
        ("", "off_topic"),
    ],
)
def test_classify_no_match_query(text: str, expected: str) -> None:
    assert classify_no_match_query(text) == expected


def test_varied_no_chunk_course_answer_prefix_and_body() -> None:
    out = varied_no_chunk_course_answer("greeting")
    assert out.startswith("Course Answer:\n")
    assert len(out) > len("Course Answer:\n")


def test_varied_no_chunk_course_answer_has_variety_per_kind() -> None:
    for kind in ("greeting", "short_ack", "off_topic"):
        samples = {varied_no_chunk_course_answer(kind) for _ in range(80)}
        assert len(samples) > 1, f"expected multiple templates for {kind}"
