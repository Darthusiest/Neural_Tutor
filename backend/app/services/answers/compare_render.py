"""Deterministic compare layouts (two-way and multi-entity) from evidence bundles."""

from __future__ import annotations

from typing import Any

from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.entity_retrieval import ConceptEvidenceBundle
from app.services.knowledge.structured_query import StructuredQuery


def _lines_from_chunks(chunks: list[dict[str, Any]], *, max_lines: int = 4) -> list[str]:
    lines: list[str] = []
    for c in chunks:
        expl = (c.get("clean_explanation") or c.get("source_excerpt") or "").strip()
        for part in expl.split("\n"):
            p = part.strip()
            if p:
                lines.append(p)
            if len(lines) >= max_lines:
                return lines
    return lines


def render_compare_two_markdown(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    sq: StructuredQuery,
    bundle_a: ConceptEvidenceBundle,
    bundle_b: ConceptEvidenceBundle,
) -> str:
    """Axis-oriented compare without repeating scaffold labels per line."""
    by_id = {c.get("id"): c for c in all_chunks if c.get("id") is not None}
    ca_chunks = [by_id[i] for i in bundle_a.chunk_ids if i in by_id]
    cb_chunks = [by_id[i] for i in bundle_b.chunk_ids if i in by_id]

    a_lines = _lines_from_chunks(ca_chunks)
    b_lines = _lines_from_chunks(cb_chunks)
    a_one = a_lines[0] if a_lines else f"(Limited direct material for **{bundle_a.label}** in retrieved notes.)"
    b_one = b_lines[0] if b_lines else f"(Limited direct material for **{bundle_b.label}** in retrieved notes.)"

    axes = plan.comparison_axes[:6] if plan.comparison_axes else ["purpose", "computation", "typical use in the course"]
    axis_block = "\n".join(f"- **{ax}:** contrast the two using the definitions above." for ax in axes)

    gap_notes: list[str] = []
    if bundle_a.gap_flags:
        gap_notes.append(f"- **{bundle_a.label}:** evidence support is thin; treat claims as provisional.")
    if bundle_b.gap_flags:
        gap_notes.append(f"- **{bundle_b.label}:** evidence support is thin; treat claims as provisional.")
    gap_txt = "\n".join(gap_notes)

    parts = [
        "Course Answer:",
        "",
        "### Direct Answer",
        "",
        f"**{bundle_a.label}** in one line: {a_one}",
        "",
        f"**{bundle_b.label}** in one line: {b_one}",
        "",
        "### Explanation",
        "",
        f"**{bundle_a.label} (from course text):**",
        "",
    ]
    for ln in a_lines[1:5]:
        parts.append(f"- {ln}")
    parts.extend(
        [
            "",
            f"**{bundle_b.label} (from course text):**",
            "",
        ]
    )
    for ln in b_lines[1:5]:
        parts.append(f"- {ln}")

    parts.extend(
        [
            "",
            "### Contrast along course axes",
            "",
            axis_block,
            "",
            "### Why the difference matters",
            "",
            "Getting the contrast right matters when you interpret model behavior, read plots, "
            "or choose an architecture for speech or language tasks in this course.",
        ]
    )
    if gap_txt:
        parts.extend(["", "### Evidence notes", "", gap_txt])
    return "\n".join(parts).rstrip()


def render_compare_multi_markdown(
    bundles: list[ConceptEvidenceBundle],
    all_chunks: list[dict[str, Any]],
    sq: StructuredQuery,
) -> str:
    """Markdown table + per-entity notes for 3+ concepts."""
    by_id = {c.get("id"): c for c in all_chunks if c.get("id") is not None}

    header = "| Architecture | Evidence snapshot (retrieved notes) |"
    sep = "| --- | --- |"
    rows: list[str] = []
    for b in bundles:
        ca = [by_id[i] for i in b.chunk_ids[:1] if i in by_id]
        summary = _lines_from_chunks(ca, max_lines=1)
        cell0 = summary[0][:320] + ("…" if summary and len(summary[0]) > 320 else "") if summary else (
            "Limited material in retrieved chunks."
        )
        row = f"| **{b.label}** | {cell0} |"
        rows.append(row)

    parts = [
        "Course Answer:",
        "",
        "### Compared architectures",
        "",
        "This answer uses **separate evidence pools** per architecture to reduce cross-talk.",
        "",
        header,
        sep,
        *rows,
        "",
        "### Entity notes (course-grounded)",
        "",
    ]
    for b in bundles:
        ca = [by_id[i] for i in b.chunk_ids if i in by_id]
        lines = _lines_from_chunks(ca, max_lines=5)
        parts.append(f"#### {b.label}")
        parts.append("")
        if b.gap_flags:
            parts.append("*Limited direct evidence in retrieved chunks—claims are provisional.*")
            parts.append("")
        if lines:
            for ln in lines:
                parts.append(f"- {ln}")
        else:
            parts.append("- (No matching chunk text—try a narrower term from the syllabus.)")
        parts.append("")

    parts.extend(
        [
            "### Why contrasts matter",
            "",
            "These architectures differ in inductive biases and compute patterns; the table highlights "
            "where each one shows up in the course storyline.",
        ]
    )
    return "\n".join(parts).rstrip()
