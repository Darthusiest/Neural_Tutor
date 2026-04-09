"""Stable ``chunk_key`` generation for lecture sections (import / upsert identity)."""

from __future__ import annotations

import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_heading(heading: str, max_len: int = 48) -> str:
    s = _SLUG_RE.sub("-", heading.lower()).strip("-")
    return (s[:max_len] if s else "section").strip("-") or "section"


def derive_chunk_key(
    lecture_number: int,
    heading: str,
    section_index: int,
    explicit: str | None,
) -> str:
    """
    Prefer explicit ``chunk_key`` / ``section_id`` from JSON (already normalized).
    Otherwise ``{lecture_number}:{index:02d}:{heading_slug}``.
    """
    if explicit:
        return explicit.lower()
    return f"{lecture_number}:{section_index:02d}:{slugify_heading(heading)}"
