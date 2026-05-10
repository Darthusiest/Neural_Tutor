"""Heuristic lecture-chunk coverage per KB concept (alias overlap).

Used by ``scripts/audit_kb_chunk_coverage.py`` and tests to ensure every
structured-pipeline concept appears in at least ``min_chunks`` indexed chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from app.models import LectureChunk
    from app.services.knowledge.concept_kb import ConceptKB


def lecture_chunk_blob(row: LectureChunk) -> str:
    """Lowercased text blob used for substring overlap checks."""
    parts = [
        row.topic or "",
        row.keywords or "",
        row.clean_explanation or "",
        row.source_excerpt or "",
    ]
    return " ".join(parts).lower()


def terms_for_concept(meta) -> set[str]:
    """Search terms for one KB concept (id, name, aliases; length > 1)."""
    terms: set[str] = {
        meta.id.lower(),
        meta.name.lower(),
        *[a.strip().lower() for a in meta.aliases if len(a.strip()) > 1],
    }
    return {t for t in terms if len(t) > 1}


def count_chunks_per_concept(
    kb: ConceptKB,
    blobs: Iterable[str],
) -> dict[str, int]:
    """Map concept_id -> count of chunks whose blob matches any concept term."""
    blob_list = list(blobs)
    out: dict[str, int] = {}
    for cid, meta in sorted(kb.concepts_by_id.items()):
        terms = terms_for_concept(meta)
        n = sum(1 for b in blob_list if any(t in b for t in terms))
        out[cid] = n
    return out


@dataclass(frozen=True)
class ConceptChunkAudit:
    counts: dict[str, int]
    below_threshold: list[tuple[str, int]]

    def ok(self) -> bool:
        return not self.below_threshold


def audit_kb_chunk_coverage(
    kb: ConceptKB,
    chunks: list[LectureChunk],
    *,
    min_chunks: int = 2,
) -> ConceptChunkAudit:
    blobs = [lecture_chunk_blob(r) for r in chunks]
    counts = count_chunks_per_concept(kb, blobs)
    low = sorted((cid, n) for cid, n in counts.items() if n < min_chunks)
    return ConceptChunkAudit(counts=counts, below_threshold=low)
