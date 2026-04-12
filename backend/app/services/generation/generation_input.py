"""Build clean, teaching-only inputs for Course Answer generation (no retrieval metadata)."""

from __future__ import annotations

from typing import Any

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import StructuredQuery


def _primary_concept_names(sq: StructuredQuery) -> list[str]:
    """Human-readable concept names only (no IDs, no graph structure)."""
    kb = get_kb()
    names: list[str] = []
    for cid in sq.concept_ids[:12]:
        meta = kb.get_concept_by_id(cid)
        if meta and meta.name not in names:
            names.append(meta.name)
    for d in (sq.intent.detected_concepts or [])[:8]:
        if d and d not in names:
            names.append(d)
    return names


def _compress_text_list(text_list: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for t in text_list:
        t = t.strip()
        if t and t not in seen:
            result.append(t)
            seen.add(t)
    return result


def _chunk_teaching_text(c: dict[str, Any]) -> str:
    """ONLY explanation prose—no keywords, lecture numbers, or other fields."""
    ex = (c.get("clean_explanation") or "").strip()
    if not ex:
        ex = (c.get("source_excerpt") or "").strip()
    return ex[:8000]


def build_generation_input(
    structured_query: StructuredQuery,
    answer_plan: AnswerPlan,
    retrieved_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Convert raw retrieval rows into clean teaching input for the LLM.

    Does not include: keywords, lecture numbers, concept_family, related_topics, chunk ids,
    or any debug fields—only ``clean_explanation`` / ``source_excerpt`` text split into
    primary vs supporting by the answer plan.
    """
    primary_ids = list(answer_plan.primary_chunk_ids)
    sup_ids = list(answer_plan.supporting_chunk_ids)

    primary_chunks = chunks_by_ids(retrieved_chunks, primary_ids)
    supporting_chunks = chunks_by_ids(retrieved_chunks, sup_ids)

    if not primary_chunks and retrieved_chunks:
        primary_chunks = list(retrieved_chunks)[:12]

    primary_text: list[str] = []
    for c in primary_chunks:
        t = _chunk_teaching_text(c)
        if t:
            primary_text.append(t)

    supporting_text: list[str] = []
    for c in supporting_chunks:
        t = _chunk_teaching_text(c)
        if t:
            supporting_text.append(t)

    primary_text = _compress_text_list(primary_text)
    supporting_text = _compress_text_list(supporting_text)

    return {
        "question": structured_query.intent.original_query,
        "concepts": _primary_concept_names(structured_query),
        "answer_mode": answer_plan.answer_mode,
        "primary_content": primary_text,
        "supporting_content": supporting_text,
    }


def format_generation_prompt_user_message(clean_input: dict[str, Any]) -> str:
    """Plain-text user block for the tutor (no JSON metadata dump)."""
    question = (clean_input.get("question") or "").strip()
    concepts = clean_input.get("concepts") or []
    concept_line = ", ".join(c for c in concepts if c) if concepts else "(none listed)"
    mode = clean_input.get("answer_mode", "general")
    primary = clean_input.get("primary_content") or []
    supporting = clean_input.get("supporting_content") or []

    def _joined(items: list[str]) -> str:
        lines = [s.strip() for s in items if s and str(s).strip()]
        return "\n".join(lines) if lines else "(none)"

    return (
        "Answer the following question using the provided course material.\n\n"
        f"Question:\n{question}\n\n"
        f"Concepts:\n{concept_line}\n\n"
        f"Teaching style (hint only; do not echo this label as jargon):\n{mode}\n\n"
        f"Primary Content:\n{_joined(primary)}\n\n"
        f"Supporting Content:\n{_joined(supporting)}"
    )
