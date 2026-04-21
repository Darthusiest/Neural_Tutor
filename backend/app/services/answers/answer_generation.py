"""Structured course answer text from an :class:`AnswerPlan` (rule-based by default).

Uses the same four-section tutor layout as the OpenAI primary path:
Direct Answer, Explanation, Example / Intuition, Why it matters.
Student-facing text only—no lecture IDs, keyword dumps, or retrieval jargon.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.answers.compare_render import (
    format_multi_entity_compare_markdown,
    format_two_entity_compare_markdown,
)
from app.services.knowledge.structured_query import StructuredQuery
from app.services.retrieval import _sample_questions_as_text


def _bullet_lines_from_chunk(lecture_chunk: dict[str, Any]) -> list[str]:
    expl = (lecture_chunk.get("clean_explanation") or "").strip()
    if not expl:
        expl = (lecture_chunk.get("source_excerpt") or "").strip()
    lines: list[str] = []
    for raw_line in expl.split("\n"):
        line = raw_line.strip()
        if line:
            lines.append(line)
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
    trimmed = text.strip()
    if not trimmed:
        return ""
    # Prefer sentence boundary in first segment
    sentence_match = re.match(r"([^.!?]+[.!?])(\s|$)", trimmed[:800])
    if sentence_match:
        return sentence_match.group(1).strip()
    first_line = trimmed.split("\n")[0].strip()
    return first_line[:max_len] + ("…" if len(first_line) > max_len else "")


def _dedupe_lines(lines: list[str], cap: int = 16) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for candidate in lines:
        key = candidate.strip().lower()[:240]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(candidate.strip())
        if len(out) >= cap:
            break
    return out


def _primary_chunks_ordered(plan: AnswerPlan, all_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_chunk_ids = plan.primary_chunk_ids or []
    ordered = chunks_by_ids(all_chunks, primary_chunk_ids)
    if ordered:
        return ordered
    return list(all_chunks)[:8]


def _example_intuition_block(primary: list[dict[str, Any]]) -> str:
    for chunk in primary[:3]:
        sample_answer = (chunk.get("sample_answer") or "").strip()
        if sample_answer and sample_answer not in ("[]", "null"):
            return sample_answer[:600]
    for chunk in primary[:3]:
        paired_question = _sample_questions_as_text(chunk).strip()
        if paired_question and paired_question not in ("[]", "null", "None"):
            return (
                f"A question the materials pair with this topic: {paired_question[:500]}"
                if len(paired_question) < 400
                else paired_question[:600]
            )
    if primary:
        excerpt = (primary[0].get("source_excerpt") or "").strip()
        if len(excerpt) > 40:
            return _first_sentence_or_line(excerpt[:500]) or excerpt[:280]
    return (
        "Think of the explanation above as the core picture—ask if you want a different angle "
        "or a walkthrough with numbers."
    )


def _why_matters_block(plan: AnswerPlan, structured_query: StructuredQuery, primary: list[dict[str, Any]]) -> str:
    """Tutor-style closing—no lecture IDs, scope lists, or 'graph' jargon."""
    parts: list[str] = []
    if plan.include_related_concepts:
        related_names = ", ".join(plan.include_related_concepts[:6])
        if related_names:
            parts.append(
                f"You’ll keep running into related ideas such as {related_names} as you move through "
                "models, data, and evaluation in the course."
            )
    if plan.comparison_axes and plan.answer_mode == "compare":
        parts.append(
            "Getting the contrast right matters when you interpret model behavior or compare architectures."
        )
    if not parts and structured_query.concept_ids:
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
        compare_bullets: list[str] = []
        for section in plan.sections:
            if not section.chunk_ids:
                continue
            heading = section.heading
            excerpt_lines: list[str] = []
            for lecture_chunk in chunks_by_ids(all_chunks, section.chunk_ids):
                for bullet_line in _bullet_lines_from_chunk(lecture_chunk)[:4]:
                    excerpt_lines.append(bullet_line.strip())
                    if len(excerpt_lines) >= 4:
                        break
                if len(excerpt_lines) >= 4:
                    break
            excerpt_lines = _dedupe_lines(excerpt_lines, cap=4)
            if not excerpt_lines:
                continue
            compare_bullets.append(f"**{heading}:** {excerpt_lines[0]}")
            for extra_line in excerpt_lines[1:]:
                compare_bullets.append(extra_line)
        if plan.comparison_axes:
            compare_bullets.append(
                "**Contrast to keep in mind:** " + "; ".join(plan.comparison_axes[:4])
            )
        return _dedupe_lines(compare_bullets, cap=22)

    explanation_bullets: list[str] = []
    if not primary:
        return ["Add more detail by asking a follow-up with a specific term from class."]
    first_lines = _bullet_lines_from_chunk(primary[0])
    if not first_lines:
        raw = (primary[0].get("clean_explanation") or primary[0].get("source_excerpt") or "").strip()
        first_lines = [raw_line.strip() for raw_line in raw.split("\n") if raw_line.strip()] if raw else []
    if first_lines:
        explanation_bullets.extend(first_lines[skip_first_chunk_lines:])
    for lecture_chunk in primary[1:]:
        explanation_bullets.extend(_bullet_lines_from_chunk(lecture_chunk))
    # Cap supporting material to reduce retrieval contamination (unrelated chunks).
    for supporting_id in plan.supporting_chunk_ids[:3]:
        supporting_chunk = next((x for x in all_chunks if x.get("id") == supporting_id), None)
        if supporting_chunk:
            explanation_bullets.extend(_bullet_lines_from_chunk(supporting_chunk)[:2])
    return _dedupe_lines(explanation_bullets, cap=16)


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
            paragraph_lines = [raw_line.strip() for raw_line in raw.split("\n") if raw_line.strip()]
            if paragraph_lines:
                return _compose_direct_answer_with_count(paragraph_lines)
            return raw[:400], 0
        return "See the explanation below for how the notes develop this idea.", 0
    return _compose_direct_answer_with_count(first_lines)


def generate_structured_answer(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
) -> str:
    """Build **Course Answer:** with the tutor four-section layout (aligned with OpenAI primary path)."""
    constraints = structured_query.response_constraints
    if constraints.allow_incorrect_statements:
        refusal_message = (
            "Course Answer:\n\n"
            "### Direct Answer\n"
            "I can’t mix deliberately false statements with true ones in a tutor response.\n\n"
            "### Explanation\n"
            "- If you want practice, ask for a short quiz with separate options, or ask for common "
            "misconceptions explained *as* misconceptions.\n\n"
            "### Example / Intuition\n"
            "Try: “What is a typical mistake people make about softmax vs hardmax?”\n\n"
            "### Why it matters\n"
            "Clear, correct explanations are safer for learning than blended true/false prompts."
        )
        return refusal_message

    if plan.answer_mode == "compare_multi" and plan.evidence_bundles:
        entity_bundles = list(plan.evidence_bundles.values())
        return format_multi_entity_compare_markdown(
            entity_bundles, all_chunks, structured_query, plan=plan
        )

    if plan.answer_mode == "compare" and len(plan.evidence_bundles) >= 2:
        bundle_concept_ids = list(plan.evidence_bundles.keys())
        left_bundle = plan.evidence_bundles[bundle_concept_ids[0]]
        right_bundle = plan.evidence_bundles[bundle_concept_ids[1]]
        return format_two_entity_compare_markdown(
            plan, all_chunks, structured_query, left_bundle, right_bundle
        )

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

    direct_answer_text, lines_consumed_by_direct_answer = _direct_answer_and_skip(plan, primary)
    explanation_bullets = _build_explanation_bullets(
        plan, all_chunks, primary, skip_first_chunk_lines=lines_consumed_by_direct_answer
    )
    example_intuition_text = _example_intuition_block(primary)
    why_it_matters_text = _why_matters_block(plan, structured_query, primary)

    requested_distinct_explanations = constraints.exact_explanation_count
    wants_numbered_explanation_subsections = (
        requested_distinct_explanations is not None and requested_distinct_explanations >= 2
    )
    if wants_numbered_explanation_subsections:
        while len(explanation_bullets) < requested_distinct_explanations:
            explanation_bullets.append(
                "Another angle on the same idea from the notes (distinct wording): see the preceding bullets."
            )
        explanation_bullets = explanation_bullets[:requested_distinct_explanations]

    course_answer_lines: list[str] = [
        "Course Answer:",
        "",
        "### Direct Answer",
        "",
        direct_answer_text,
        "",
        "### Explanation",
        "",
    ]
    max_numbered_explanation_sections = 12
    if wants_numbered_explanation_subsections:
        for section_index in range(
            min(requested_distinct_explanations, max_numbered_explanation_sections)
        ):
            if section_index < len(explanation_bullets):
                subsection_body = explanation_bullets[section_index]
            elif explanation_bullets:
                subsection_body = explanation_bullets[-1]
            else:
                subsection_body = ""
            course_answer_lines.append(f"#### Explanation {section_index + 1}")
            course_answer_lines.append("")
            course_answer_lines.append(subsection_body or "(See course text.)")
            course_answer_lines.append("")
    else:
        for bullet_text in explanation_bullets:
            course_answer_lines.append(f"- {bullet_text}")
        if len(explanation_bullets) == 0:
            course_answer_lines.append(
                "- The notes may pack the idea into a short block—say if you want it slower or with a diagram."
            )

    repeat_explanation_count = constraints.repeat_explanation_times
    if repeat_explanation_count is not None and repeat_explanation_count >= 2:
        repeated_explanation_markdown = (
            "\n".join(f"- {bullet_text}" for bullet_text in explanation_bullets)
            if explanation_bullets
            else direct_answer_text
        )
        course_answer_lines.extend(
            [
                "",
                "### Repeated explanation (as requested)",
                "",
                repeated_explanation_markdown,
            ]
        )

    if constraints.intuition_only:
        course_answer_lines.extend(
            [
                "",
                "### Example / Intuition",
                "",
                "(Technical training details omitted for intuition-only request.)",
                "",
                "### Why it matters",
                "",
                why_it_matters_text,
            ]
        )
        return "\n".join(course_answer_lines).rstrip()

    if constraints.no_examples:
        course_answer_lines.extend(
            [
                "",
                "### Why it matters",
                "",
                why_it_matters_text,
            ]
        )
        return "\n".join(course_answer_lines).rstrip()

    course_answer_lines.extend(
        [
            "",
            "### Example / Intuition",
            "",
            example_intuition_text,
            "",
            "### Why it matters",
            "",
            why_it_matters_text,
        ]
    )
    return "\n".join(course_answer_lines).rstrip()
