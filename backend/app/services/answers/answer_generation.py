"""Structured course answer text from an :class:`AnswerPlan` (rule-based by default).

Uses the same four-section tutor layout as the OpenAI primary path:
Direct Answer, Explanation, Example / Intuition, Why it matters.
Student-facing text only—no lecture IDs, keyword dumps, or retrieval jargon.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.knowledge.structured_query import StructuredQuery
from app.services.retrieval import _sample_questions_as_text


def _bullet_lines_from_chunk(c: dict[str, Any]) -> list[str]:
    expl = (c.get("clean_explanation") or "").strip()
    if not expl:
        expl = (c.get("source_excerpt") or "").strip()
    lines: list[str] = []
    for part in expl.split("\n"):
        p = part.strip()
        if p:
            lines.append(p)
    return lines[:16]


def _strip_bullet_prefix(line: str) -> str:
    return re.sub(r"^[-•*]\s*", "", line.strip()).strip()


def _compose_direct_answer_with_count(lines: list[str]) -> tuple[str, int]:
    """Direct answer text and how many leading lines of ``lines`` it consumed (for explanation dedup)."""
    if not lines:
        return "", 0
    first = _strip_bullet_prefix(lines[0])
    if not first:
        return "", 0
    has_terminal = bool(re.search(r"[.!?]\s*$", first))
    if len(first) >= 100 and has_terminal:
        return first[:420], 1
    if len(lines) >= 2:
        second = _strip_bullet_prefix(lines[1])
        if second and (len(first) < 100 or not has_terminal):
            merged = f"{first.rstrip('.')} — {second}"
            if len(merged) <= 450:
                return merged, 2
    return (_first_sentence_or_line(first) or first[:420]), 1


def _first_sentence_or_line(text: str, max_len: int = 420) -> str:
    """First sentence if clear; else first line; capped for a short 'Direct Answer'."""
    t = text.strip()
    if not t:
        return ""
    # Prefer sentence boundary in first segment
    m = re.match(r"([^.!?]+[.!?])(\s|$)", t[:800])
    if m:
        return m.group(1).strip()
    line = t.split("\n")[0].strip()
    return line[:max_len] + ("…" if len(line) > max_len else "")


def _dedupe_lines(lines: list[str], cap: int = 16) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for L in lines:
        key = L.strip().lower()[:240]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(L.strip())
        if len(out) >= cap:
            break
    return out


def _primary_chunks_ordered(plan: AnswerPlan, all_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = plan.primary_chunk_ids or []
    out = chunks_by_ids(all_chunks, ids)
    if out:
        return out
    return list(all_chunks)[:8]


def _example_intuition_block(primary: list[dict[str, Any]]) -> str:
    for c in primary[:3]:
        sa = (c.get("sample_answer") or "").strip()
        if sa and sa not in ("[]", "null"):
            return sa[:600]
    for c in primary[:3]:
        sq = _sample_questions_as_text(c).strip()
        if sq and sq not in ("[]", "null", "None"):
            return (
                f"A question the materials pair with this topic: {sq[:500]}"
                if len(sq) < 400
                else sq[:600]
            )
    if primary:
        ex = (primary[0].get("source_excerpt") or "").strip()
        if len(ex) > 40:
            return _first_sentence_or_line(ex[:500]) or ex[:280]
    return (
        "Think of the explanation above as the core picture—ask if you want a different angle "
        "or a walkthrough with numbers."
    )


def _why_matters_block(plan: AnswerPlan, sq: StructuredQuery, primary: list[dict[str, Any]]) -> str:
    """Tutor-style closing—no lecture IDs, scope lists, or 'graph' jargon."""
    parts: list[str] = []
    if plan.include_related_concepts:
        rel = ", ".join(plan.include_related_concepts[:6])
        if rel:
            parts.append(
                f"You’ll keep running into related ideas such as {rel} as you move through "
                "models, data, and evaluation in the course."
            )
    if plan.comparison_axes and plan.answer_mode == "compare":
        parts.append(
            "Getting the contrast right matters when you interpret model behavior or compare architectures."
        )
    if not parts and sq.concept_ids:
        parts.append(
            "This topic connects to what you’re building toward in the rest of the syllabus—notation and vocabulary pay off later."
        )
    if not parts:
        parts.append(
            "Solid intuition here makes the next topics—layers, objectives, and decisions—much easier to follow."
        )
    return " ".join(parts)


def _build_explanation_bullets(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    skip_first_chunk_lines: int = 0,
) -> list[str]:
    """Bullets for ### Explanation (compare vs default). Skips lines already used in Direct Answer."""
    if plan.answer_mode == "compare":
        # One heading per plan section; at most a few lines under each (no per-line
        # "First idea / In one line" scaffolding—that produced hundreds of repeated labels).
        lines: list[str] = []
        for sec in plan.sections:
            if not sec.chunk_ids:
                continue
            heading = sec.heading
            blob: list[str] = []
            for c in chunks_by_ids(all_chunks, sec.chunk_ids):
                for bl in _bullet_lines_from_chunk(c)[:4]:
                    blob.append(bl.strip())
                    if len(blob) >= 4:
                        break
                if len(blob) >= 4:
                    break
            blob = _dedupe_lines(blob, cap=4)
            if not blob:
                continue
            lines.append(f"**{heading}:** {blob[0]}")
            for extra in blob[1:]:
                lines.append(extra)
        if plan.comparison_axes:
            lines.append("**Contrast to keep in mind:** " + "; ".join(plan.comparison_axes[:4]))
        return _dedupe_lines(lines, cap=22)

    expl: list[str] = []
    if not primary:
        return ["Add more detail by asking a follow-up with a specific term from class."]
    first_lines = _bullet_lines_from_chunk(primary[0])
    if not first_lines:
        raw = (primary[0].get("clean_explanation") or primary[0].get("source_excerpt") or "").strip()
        first_lines = [p.strip() for p in raw.split("\n") if p.strip()] if raw else []
    if first_lines:
        expl.extend(first_lines[skip_first_chunk_lines:])
    for c in primary[1:]:
        expl.extend(_bullet_lines_from_chunk(c))
    # Cap supporting material to reduce retrieval contamination (unrelated chunks).
    for cid in plan.supporting_chunk_ids[:3]:
        c = next((x for x in all_chunks if x.get("id") == cid), None)
        if c:
            expl.extend(_bullet_lines_from_chunk(c)[:2])
    return _dedupe_lines(expl, cap=16)


def _direct_answer_and_skip(
    plan: AnswerPlan, primary: list[dict[str, Any]]
) -> tuple[str, int]:
    """Direct answer text and number of first-chunk lines consumed (non-compare)."""
    if plan.answer_mode == "compare" and plan.comparison_axes:
        return "; ".join(plan.comparison_axes[:3]), 0
    if not primary:
        return "I don’t have a short direct line for that phrasing—see the bullets below.", 0
    first_lines = _bullet_lines_from_chunk(primary[0])
    if not first_lines:
        raw = (primary[0].get("clean_explanation") or primary[0].get("source_excerpt") or "").strip()
        if raw:
            pseudo = [p.strip() for p in raw.split("\n") if p.strip()]
            if pseudo:
                return _compose_direct_answer_with_count(pseudo)
            return raw[:400], 0
        return "See the explanation below for how the notes develop this idea.", 0
    return _compose_direct_answer_with_count(first_lines)


def generate_structured_answer(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    sq: StructuredQuery,
) -> str:
    """Build **Course Answer:** with the tutor four-section layout (aligned with OpenAI primary path)."""
    primary = _primary_chunks_ordered(plan, all_chunks)
    if not primary:
        return (
            "Course Answer:\n\n"
            "### Direct Answer\n"
            "I couldn’t tie that question to specific notes yet.\n\n"
            "### Explanation\n"
            "- Ask again using a vocabulary term from class (e.g. softmax, attention, MFCC).\n\n"
            "### Example / Intuition\n"
            "A sharper question usually unlocks a concrete example on the next try.\n\n"
            "### Why it matters\n"
            "Staying close to the course vocabulary keeps answers aligned with what you’re graded on."
        )

    direct, skip_lines = _direct_answer_and_skip(plan, primary)
    expl_bullets = _build_explanation_bullets(
        plan, all_chunks, primary, skip_first_chunk_lines=skip_lines
    )
    example = _example_intuition_block(primary)
    why = _why_matters_block(plan, sq, primary)

    lines: list[str] = [
        "Course Answer:",
        "",
        "### Direct Answer",
        "",
        direct,
        "",
        "### Explanation",
        "",
    ]
    for b in expl_bullets:
        lines.append(f"- {b}")
    if len(expl_bullets) == 0:
        lines.append(
            "- The notes may pack the idea into a short block—say if you want it slower or with a diagram."
        )

    lines.extend(
        [
            "",
            "### Example / Intuition",
            "",
            example,
            "",
            "### Why it matters",
            "",
            why,
        ]
    )
    return "\n".join(lines).rstrip()
