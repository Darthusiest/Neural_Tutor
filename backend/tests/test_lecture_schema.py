"""Lecture corpus JSON validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.lecture_corpus import validate_lecture_corpus_dict


def test_validate_minimal_corpus():
    data = {
        "lectures": [
            {
                "lecture_number": 1,
                "title": "Test",
                "sections": [{"heading": "A", "content": ["line"]}],
            }
        ]
    }
    c = validate_lecture_corpus_dict(data)
    assert len(c.lectures) == 1
    assert c.lectures[0].sections[0].heading == "A"


def test_rejects_empty_lectures():
    with pytest.raises(ValidationError):
        validate_lecture_corpus_dict({"lectures": []})


def test_rejects_section_without_body():
    with pytest.raises(ValidationError):
        validate_lecture_corpus_dict(
            {
                "lectures": [
                    {
                        "lecture_number": 1,
                        "title": "T",
                        "sections": [{"heading": "H", "content": []}],
                    }
                ]
            }
        )


def test_invalid_chunk_key_format():
    with pytest.raises(ValidationError):
        validate_lecture_corpus_dict(
            {
                "lectures": [
                    {
                        "lecture_number": 1,
                        "title": "T",
                        "sections": [
                            {
                                "heading": "H",
                                "content": ["x"],
                                "chunk_key": "Bad Key With Spaces",
                            }
                        ],
                    }
                ]
            }
        )
