"""Load and query the LING 487 structured concept knowledge pack (JSON).

Used by the structured reasoning pipeline for concept linking, comparison axes,
and retrieval hints. Lazy-loads from :data:`KB_JSON_PATH` (see :mod:`app.config`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default path next to lecture corpus (…/app/services/knowledge → backend root)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_KB_PATH = _BACKEND_ROOT / "data" / "LING487_STRUCTURED_PIPELINE_KB.json"


@dataclass
class ConceptMeta:
    id: str
    name: str
    aliases: list[str]
    lecture_scope: list[int]
    summary: str
    prerequisites: list[str] = field(default_factory=list)
    builds_on: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    compare_with: list[str] = field(default_factory=list)
    retrieval_hints: list[str] = field(default_factory=list)
    common_subquestions: list[str] = field(default_factory=list)


@dataclass
class LectureMeta:
    lecture_number: int
    title: str
    main_concepts: list[str]
    builds_on: list[int]
    connects_to: list[int]
    summary: str


class ConceptKB:
    """In-memory index over the structured pipeline KB JSON."""

    def __init__(
        self,
        *,
        concepts_by_id: dict[str, ConceptMeta],
        concepts_by_alias: dict[str, str],
        lectures_by_number: dict[int, LectureMeta],
        comparison_axes: dict[tuple[str, str], list[str]],
        comparison_axes_by_key: dict[str, list[str]],
        raw: dict[str, Any],
    ) -> None:
        self.concepts_by_id = concepts_by_id
        self.concepts_by_alias = concepts_by_alias
        self.lectures_by_number = lectures_by_number
        self.comparison_axes = comparison_axes
        self.comparison_axes_by_key = comparison_axes_by_key
        self.raw = raw

    def get_concept(self, name_or_alias: str) -> ConceptMeta | None:
        """Resolve a surface string to a concept (canonical name or alias)."""
        key = name_or_alias.strip().lower()
        if not key:
            return None
        cid = self.concepts_by_alias.get(key)
        if cid:
            return self.concepts_by_id.get(cid)
        # Longest alias match: try tokenizing
        for tok in re.split(r"[^\w]+", key):
            if len(tok) < 2:
                continue
            cid = self.concepts_by_alias.get(tok.lower())
            if cid:
                return self.concepts_by_id.get(cid)
        return None

    def get_concept_by_id(self, concept_id: str) -> ConceptMeta | None:
        return self.concepts_by_id.get(concept_id)

    def get_prerequisites(self, concept_id: str) -> list[ConceptMeta]:
        c = self.concepts_by_id.get(concept_id)
        if not c:
            return []
        return [self.concepts_by_id[p] for p in c.prerequisites if p in self.concepts_by_id]

    def get_related(self, concept_id: str) -> list[ConceptMeta]:
        c = self.concepts_by_id.get(concept_id)
        if not c:
            return []
        return [self.concepts_by_id[r] for r in c.related if r in self.concepts_by_id]

    def get_comparison_axes(self, a_id: str, b_id: str) -> list[str]:
        """Axes for comparing two concepts (order-independent)."""
        key = _pair_key(a_id, b_id)
        if key in self.comparison_axes:
            return list(self.comparison_axes[key])
        # Fallback: raw JSON keys use slug pairs like mfcc__formants
        for raw_k, axes in self.comparison_axes_by_key.items():
            ids = _parse_comparison_key(raw_k, self)
            if ids and set(ids) == {a_id, b_id}:
                return list(axes)
        return []

    def get_lecture(self, n: int) -> LectureMeta | None:
        return self.lectures_by_number.get(n)

    def find_concepts_in_text(self, tokens: list[str]) -> list[ConceptMeta]:
        """Match query tokens / phrases to KB concepts (deduped, stable iteration order)."""
        text = " ".join(t.lower() for t in tokens if t)
        word_set = set(text.split())
        seen: set[str] = set()
        out: list[ConceptMeta] = []
        for cid, c in self.concepts_by_id.items():
            if cid in seen:
                continue
            for al in [c.name, *c.aliases]:
                s = al.strip().lower()
                if len(s) < 2:
                    continue
                if " " in s:
                    if s in text:
                        seen.add(cid)
                        out.append(c)
                        break
                elif s in word_set:
                    seen.add(cid)
                    out.append(c)
                    break
        return out


_kb: ConceptKB | None = None


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _parse_comparison_key(key: str, kb: ConceptKB) -> tuple[str, str] | None:
    """Map keys like ``mfcc__formants`` to concept ids (JSON uses concept ids as slugs)."""
    parts = key.split("__")
    if len(parts) != 2:
        return None
    a, b = parts[0].strip(), parts[1].strip()
    if a in kb.concepts_by_id and b in kb.concepts_by_id and a != b:
        return (a, b)
    return None


def load_concept_kb(path: Path | str | None = None) -> ConceptKB:
    """Parse KB JSON and build indices."""
    p = Path(path) if path else _DEFAULT_KB_PATH
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    concepts_by_id: dict[str, ConceptMeta] = {}
    concepts_by_alias: dict[str, str] = {}

    for row in raw.get("concepts", []):
        cid = row["id"]
        cm = ConceptMeta(
            id=cid,
            name=row.get("name", cid),
            aliases=list(row.get("aliases", [])),
            lecture_scope=list(row.get("lecture_scope", [])),
            summary=row.get("summary", ""),
            prerequisites=list(row.get("prerequisites", [])),
            builds_on=list(row.get("builds_on", [])),
            related=list(row.get("related", [])),
            compare_with=list(row.get("compare_with", [])),
            retrieval_hints=list(row.get("retrieval_hints", [])),
            common_subquestions=list(row.get("common_subquestions", [])),
        )
        concepts_by_id[cid] = cm
        concepts_by_alias[cid.lower()] = cid
        concepts_by_alias[cm.name.lower()] = cid
        for al in cm.aliases:
            concepts_by_alias[al.strip().lower()] = cid

    lectures_by_number: dict[int, LectureMeta] = {}
    for row in raw.get("lectures", []):
        n = int(row["lecture_number"])
        lectures_by_number[n] = LectureMeta(
            lecture_number=n,
            title=row.get("title", ""),
            main_concepts=list(row.get("main_concepts", [])),
            builds_on=list(row.get("builds_on", [])),
            connects_to=list(row.get("connects_to", [])),
            summary=row.get("summary", ""),
        )

    comparison_axes_by_key: dict[str, list[str]] = {}
    ca_raw = raw.get("comparison_axes") or {}
    if isinstance(ca_raw, dict):
        comparison_axes_by_key = {str(k): list(v) for k, v in ca_raw.items()}

    # Build normalized pair -> axes using a temporary KB shell for _parse_comparison_key
    temp = ConceptKB(
        concepts_by_id=concepts_by_id,
        concepts_by_alias=concepts_by_alias,
        lectures_by_number=lectures_by_number,
        comparison_axes={},
        comparison_axes_by_key=comparison_axes_by_key,
        raw=raw,
    )
    comparison_axes: dict[tuple[str, str], list[str]] = {}
    for k, axes in comparison_axes_by_key.items():
        ids = _parse_comparison_key(k, temp)
        if ids:
            comparison_axes[_pair_key(ids[0], ids[1])] = list(axes)

    return ConceptKB(
        concepts_by_id=concepts_by_id,
        concepts_by_alias=concepts_by_alias,
        lectures_by_number=lectures_by_number,
        comparison_axes=comparison_axes,
        comparison_axes_by_key=comparison_axes_by_key,
        raw=raw,
    )


def get_kb(path: Path | str | None = None) -> ConceptKB:
    """Singleton KB (lazy). Pass ``path`` only for tests or alternate corpus."""
    global _kb
    if path is not None:
        return load_concept_kb(path)
    if _kb is None:
        try:
            from flask import has_app_context, current_app

            if has_app_context():
                p = current_app.config.get("KB_JSON_PATH")
                if p:
                    _kb = load_concept_kb(Path(p))
                    return _kb
        except Exception:
            pass
        _kb = load_concept_kb(_DEFAULT_KB_PATH)
    return _kb


def reset_kb_for_tests() -> None:
    """Clear singleton (pytest)."""
    global _kb
    _kb = None
