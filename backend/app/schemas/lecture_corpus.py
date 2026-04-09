"""Strict structure for lecture JSON used by ``import_lecture_json``."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


_CHUNK_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class LectureSection(BaseModel):
    """One retrievable section (maps to one ``LectureChunk`` row)."""

    heading: str
    content: list[str] = Field(default_factory=list)
    source_excerpt: str | None = None
    source_text: str | None = None
    clean_explanation: str | None = None
    clean: str | None = None
    sample_questions: list[str] | str | None = None
    sample_question: str | None = None
    sample_answer: str | None = None
    # Stable identity for upserts (optional; otherwise derived at import).
    chunk_key: str | None = None
    section_id: str | None = None
    # Curated lexical hints (merged with auto keywords, capped by config).
    keywords: list[str] | None = None
    keywords_extra: list[str] | None = None

    @field_validator("heading")
    @classmethod
    def heading_non_empty(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("section.heading must be non-empty")
        return s

    @field_validator("chunk_key", "section_id")
    @classmethod
    def optional_key_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        if not _CHUNK_KEY_RE.match(s.lower()):
            raise ValueError(
                "chunk_key / section_id must match ^[a-z0-9][a-z0-9._-]{0,127}$ (case-insensitive)"
            )
        return s.lower()

    @field_validator("content", mode="before")
    @classmethod
    def content_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("content must be a list of strings")
        return [str(x).strip() for x in v if str(x).strip()]

    @field_validator("keywords", "keywords_extra", mode="before")
    @classmethod
    def keyword_lists(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("keywords must be a list of strings")
        return [str(x).strip() for x in v if str(x).strip()]

    @model_validator(mode="after")
    def has_some_body(self) -> LectureSection:
        has_lines = bool(self.content)
        has_src = bool(
            (self.source_excerpt and str(self.source_excerpt).strip())
            or (self.source_text and str(self.source_text).strip())
        )
        if not has_lines and not has_src:
            raise ValueError(
                f"section '{self.heading!r}' needs non-empty content[] or source_excerpt/source_text"
            )
        return self


class Lecture(BaseModel):
    lecture_number: int
    title: str
    sections: list[LectureSection]

    @field_validator("lecture_number")
    @classmethod
    def lecture_num_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("lecture_number must be >= 0")
        return v

    @field_validator("title")
    @classmethod
    def title_strip(cls, v: str) -> str:
        return str(v).strip()

    @field_validator("sections")
    @classmethod
    def non_empty_sections(cls, v: list[LectureSection]) -> list[LectureSection]:
        if not v:
            raise ValueError("each lecture must have at least one section")
        return v


class LectureCorpus(BaseModel):
    """Root JSON object for course lecture files."""

    title: str | None = None
    source_file: str | None = None
    lectures: list[Lecture]

    @field_validator("lectures")
    @classmethod
    def non_empty_lectures(cls, v: list[Lecture]) -> list[Lecture]:
        if not v:
            raise ValueError("lectures must be non-empty")
        return v


def validate_lecture_corpus_dict(data: dict[str, Any]) -> LectureCorpus:
    """Parse and validate; raises ``pydantic.ValidationError`` on failure."""
    return LectureCorpus.model_validate(data)


def validate_chunk_keys_unique(keys: list[str]) -> None:
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            raise ValueError(f"duplicate chunk_key in import batch: {k!r}")
        seen.add(k)
