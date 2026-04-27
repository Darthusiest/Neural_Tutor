"""Deterministic extraction of compare-safe lines from chunks (entity scope, forbidden terms)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.services.answers.entity_retrieval import (
    forbidden_terms_for_concept,
    generic_nn_filler_score,
)
from app.services.knowledge.concept_kb import ConceptKB

if TYPE_CHECKING:
    from app.services.answers.concept_constraints import ConceptConstraints


def _term_hits(text_normalized: str, phrase: str) -> float:
    """Count phrase matches in already-lowercased text (whole words unless phrase contains spaces)."""
    if len(phrase) < 2:
        return 0.0
    if " " in phrase:
        return float(phrase in text_normalized)
    return float(len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_normalized)))


def line_has_forbidden(line: str, forbidden: list[str]) -> bool:
    line_lower = line.lower().strip()
    for t in forbidden:
        tl = t.lower().strip()
        if len(tl) < 2:
            continue
        if _term_hits(line_lower, tl) > 0:
            return True
    return False


def line_peer_pressure(line: str, peer_terms: list[str]) -> float:
    line_lower = line.lower().strip()
    weight = 0.0
    for t in peer_terms:
        tl = t.lower().strip()
        if len(tl) < 2:
            continue
        weight += _term_hits(line_lower, tl)
    return weight


def entity_terms_for_concept(concept_id: str, kb: ConceptKB) -> list[str]:
    meta = kb.get_concept_by_id(concept_id)
    if not meta:
        return []
    out: list[str] = [meta.name.lower()]
    out.extend(a.lower() for a in meta.aliases[:10] if len(a) > 2)
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t and t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def peer_highlight_terms(peer_ids: list[str], kb: ConceptKB) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for pid in peer_ids:
        for t in entity_terms_for_concept(pid, kb):
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def split_text_units(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    lines = [p.strip() for p in text.split("\n") if p.strip()]
    if len(lines) <= 1 and lines and len(lines[0]) > 200:
        parts = re.split(r"(?<=[.!?])\s+", lines[0])
        return [p.strip() for p in parts if p.strip()]
    return lines


def chunks_to_raw_units(chunks: list[dict[str, Any]]) -> list[str]:
    units: list[str] = []
    for c in chunks:
        expl = (c.get("clean_explanation") or c.get("source_excerpt") or "").strip()
        for u in split_text_units(expl):
            if u:
                units.append(u)
    return units


def _rank_units(units: list[str], entity_terms: list[str]) -> list[str]:
    def score(unit: str) -> tuple[float, float]:
        unit_lower = unit.lower()
        entity_match = sum(_term_hits(unit_lower, t) for t in entity_terms if len(t) >= 2)
        filler_pen = generic_nn_filler_score(unit)
        return (entity_match - 0.35 * filler_pen, -len(unit))

    return sorted(units, key=lambda u: score(u), reverse=True)


def _collect_tier(
    units: list[str],
    *,
    forbidden: list[str],
    peer_terms: list[str],
    entity_terms: list[str],
    filter_peer: bool,
) -> list[str]:
    out: list[str] = []
    for u in units:
        if line_has_forbidden(u, forbidden):
            continue
        if filter_peer and peer_terms and line_peer_pressure(u, peer_terms) >= 1.0:
            continue
        out.append(u)
    return _rank_units(out, entity_terms)


def scoped_lines_from_chunks(
    chunks: list[dict[str, Any]],
    concept_id: str,
    peer_concept_ids: list[str],
    kb: ConceptKB,
    forbidden_override: list[str] | None = None,
    *,
    max_lines: int = 8,
    constraints: "ConceptConstraints | None" = None,
) -> tuple[list[str], bool]:
    """Return (lines, provisional) — provisional True if tier-3 raw fallback used.

    When ``constraints`` is provided, the entity-term and forbidden-term sets
    are taken from ``constraints.target_aliases`` / ``constraints.forbidden_terms``
    instead of the per-concept KB lookup. This is the seam used by chat /
    direct-answer renderers that need the same purity gate the validator
    will run later.
    """
    if constraints is not None and constraints.target_aliases:
        forbidden = list(constraints.forbidden_terms)
        entity_terms = list(constraints.target_aliases)
        peer_terms = list(constraints.forbidden_terms)
    else:
        if forbidden_override is None:
            forbidden = forbidden_terms_for_concept(concept_id, peer_concept_ids, kb)
        elif forbidden_override:
            forbidden = list(forbidden_override)
        else:
            forbidden = forbidden_terms_for_concept(concept_id, peer_concept_ids, kb)
        peer_terms = peer_highlight_terms(peer_concept_ids, kb)
        entity_terms = entity_terms_for_concept(concept_id, kb)
    units = chunks_to_raw_units(chunks)

    tier1 = _collect_tier(units, forbidden=forbidden, peer_terms=peer_terms, entity_terms=entity_terms, filter_peer=True)
    if tier1:
        return _dedupe_cap(tier1, max_lines), False

    tier2 = _collect_tier(units, forbidden=forbidden, peer_terms=peer_terms, entity_terms=entity_terms, filter_peer=False)
    if tier2:
        return _dedupe_cap(tier2, max_lines), False

    raw_ranked = _rank_units(list(units), entity_terms)
    return _dedupe_cap(raw_ranked[: max(max_lines * 2, 8)], max_lines), True


def _dedupe_cap(lines: list[str], max_lines: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in lines:
        key = u.strip().lower()[:240]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(u.strip())
        if len(out) >= max_lines:
            break
    return out


def axis_token_overlap(axis: str, line: str) -> float:
    axis_words = [w for w in re.findall(r"[a-zA-Z]+", axis.lower()) if len(w) > 2]
    if not axis_words:
        return 0.0
    line_lower = line.lower()
    hits = sum(1 for w in axis_words if _term_hits(line_lower, w) > 0)
    return hits / len(axis_words)


def pick_line_for_axis(lines: list[str], axis: str) -> str | None:
    if not lines:
        return None
    best: str | None = None
    best_s = -1.0
    for ln in lines:
        s = axis_token_overlap(axis, ln)
        if s > best_s:
            best_s = s
            best = ln
    if best is not None and best_s > 0:
        return best
    return lines[0]


def shorten_for_compare_cell(text: str, max_len: int = 220) -> str:
    t = text.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def sanitize_table_cell(text: str, max_len: int = 200) -> str:
    t = shorten_for_compare_cell(text, max_len=max_len)
    return t.replace("|", "/").replace("\n", " ")
