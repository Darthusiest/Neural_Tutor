"""Structured course answer text from an :class:`AnswerPlan` (rule-based by default)."""

from __future__ import annotations

from typing import Any

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.knowledge.structured_query import StructuredQuery


def _bullet_lines_from_chunk(c: dict[str, Any]) -> list[str]:
    expl = (c.get("clean_explanation") or "").strip()
    if not expl:
        expl = (c.get("source_excerpt") or "").strip()
    lines: list[str] = []
    for part in expl.split("\n"):
        p = part.strip()
        if p:
            lines.append(p)
    return lines[:12]


def generate_structured_answer(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    sq: StructuredQuery,
) -> str:
    """Build **Course Answer:** with section headings from the plan (grounded in chunks)."""
    lines: list[str] = ["Course Answer:", ""]

    for sec in plan.sections:
        section_chunks = chunks_by_ids(all_chunks, sec.chunk_ids)
        if not section_chunks:
            continue
        if len(plan.sections) > 1 or sec.heading:
            lines.append(f"### {sec.heading}")
            lines.append("")
        for c in section_chunks:
            num = c.get("lecture_number")
            topic = c.get("topic", "")
            lines.append(f"Lecture {num} — {topic}")
            for bl in _bullet_lines_from_chunk(c):
                lines.append(f"- {bl}")
            lines.append("")

        if plan.answer_mode == "compare" and plan.comparison_axes and sec.content_hint == "comparison_axis":
            lines.append(
                "- **Comparison focus:** " + "; ".join(plan.comparison_axes[:4])
            )
            lines.append("")

    if plan.include_related_concepts and sq.concept_ids:
        rel = ", ".join(plan.include_related_concepts[:6])
        if rel:
            lines.append(f"### Related course concepts")
            lines.append("")
            lines.append(f"- {rel}")
            lines.append("")

    if plan.include_example:
        for c in all_chunks[:2]:
            sq_text = (c.get("sample_questions") or "").strip()
            if sq_text:
                lines.append("### Example question (from materials)")
                lines.append("")
                lines.append(f"- {sq_text[:500]}")
                lines.append("")
                break

    return "\n".join(lines).rstrip()
