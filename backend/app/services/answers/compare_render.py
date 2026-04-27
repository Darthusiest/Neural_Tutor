"""Deterministic Course Answer markdown for compare queries.

Two entry points:
- ``format_two_entity_compare_markdown`` — exactly two concepts (e.g. MFCC vs formants).
- ``format_multi_entity_compare_markdown`` — three or more concepts; uses a table plus notes.

Both use per-entity chunk pools and :mod:`compare_evidence` so lines are scoped and
cross-topic leakage is reduced before text is shown to the student. When the
caller passes a V2 bundle (``ConceptEvidenceBundleV2``) the renderer reads
``core_lines`` / ``shared_lines`` directly off the bundle instead of
recomputing scoping. ``shared_lines`` (lines that scored well for both
entities in the two-way compare) drive the optional ``### What they share``
section.
"""

from __future__ import annotations

from typing import Any

from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.compare_evidence import (
    pick_line_for_axis,
    sanitize_table_cell,
    scoped_lines_from_chunks,
    shorten_for_compare_cell,
)
from app.services.answers.entity_retrieval import (
    ConceptEvidenceBundle,
    ConceptEvidenceBundleV2,
    EvidenceBundleLike,
    forbidden_terms_for_concept,
)
from app.services.knowledge.concept_kb import ConceptKB, get_kb
from app.services.knowledge.structured_query import StructuredQuery


# Minimum support score the *weaker* bundle must reach before the rendered
# output emits "### What they share". Keeps the section out when one side is
# almost entirely lacking direct evidence (the shared note would be confusing).
COMPARE_SHARED_MIN_SUPPORT = 0.25


def _lookup_forbidden_terms_from_plan(
    plan: AnswerPlan,
    section_id: str,
    entity_concept_id: str,
) -> list[str] | None:
    """Return planner ``SectionSpec.forbidden_terms`` for this side, or ``None`` if no spec matches.

    Compare plans tag ``side_a`` / ``side_b`` with the entity id so renderers can reuse
    the same forbidden-word lists the planner computed (peer names + static blocklist).
    """
    for spec in plan.section_specs:
        if spec.section_id == section_id and spec.entity_id == entity_concept_id:
            return list(spec.forbidden_terms)
    return None


def _resolve_forbidden_terms_for_entity(
    plan: AnswerPlan,
    section_id: str,
    entity_concept_id: str,
    other_entity_concept_id: str,
    kb: ConceptKB,
) -> list[str]:
    """Planner list if present and non-empty; otherwise compute from KB (peer + defaults)."""
    from_plan = _lookup_forbidden_terms_from_plan(plan, section_id, entity_concept_id)
    if from_plan:
        return from_plan
    return forbidden_terms_for_concept(entity_concept_id, [other_entity_concept_id], kb)


def _scoped_lines_for_bundle(
    bundle: EvidenceBundleLike,
    *,
    bundle_chunks: list[dict[str, Any]],
    peer_concept_ids: list[str],
    forbidden_override: list[str] | None,
    kb: ConceptKB,
    max_lines: int = 8,
) -> tuple[list[str], bool]:
    """Read ``core_lines`` from a V2 bundle, or fall back to recomputing for legacy bundles.

    V2 bundles already carry the entity-pure ``core_lines`` produced by
    :func:`build_bundles_for_compare_v2`, so the renderer doesn't need to
    rerun the scoping pipeline. Legacy bundles (``ConceptEvidenceBundle``)
    don't carry pre-extracted lines, so we still call
    :func:`scoped_lines_from_chunks` for them.

    Returns ``(lines, used_provisional_fallback)`` matching the existing
    contract from :func:`scoped_lines_from_chunks`. V2 bundles never report
    "provisional fallback" — line classification was deterministic.
    """
    if isinstance(bundle, ConceptEvidenceBundleV2) and bundle.core_lines:
        return list(bundle.core_lines)[:max_lines], False
    return scoped_lines_from_chunks(
        bundle_chunks,
        bundle.concept_id,
        peer_concept_ids,
        kb,
        forbidden_override,
        max_lines=max_lines,
    )


def _shared_lines_from_bundles(
    left_bundle: EvidenceBundleLike,
    right_bundle: EvidenceBundleLike,
    *,
    min_support: float = COMPARE_SHARED_MIN_SUPPORT,
    max_lines: int = 5,
) -> list[str]:
    """Pull ``shared_lines`` off the V2 bundles when both sides clear ``min_support``.

    Returns ``[]`` for legacy bundles (no shared bucket) or when either side
    falls below the support threshold — that prevents a confident-sounding
    shared section when one side has almost no on-topic evidence to back it.
    """
    if not isinstance(left_bundle, ConceptEvidenceBundleV2):
        return []
    if not isinstance(right_bundle, ConceptEvidenceBundleV2):
        return []
    if left_bundle.support_score < min_support or right_bundle.support_score < min_support:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for line in [*left_bundle.shared_lines, *right_bundle.shared_lines]:
        key = line.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line.strip())
        if len(out) >= max_lines:
            break
    return out


def _compare_clarification_fallback(
    structured_query: StructuredQuery,
    left_bundle: ConceptEvidenceBundle | ConceptEvidenceBundleV2,
    right_bundle: ConceptEvidenceBundle | ConceptEvidenceBundleV2,
) -> str:
    """Renderer-level fallback for the residual zero-chunk compare case.

    Used when retrieval returned chunks but neither bundle carries usable
    chunk ids or core lines. Emits the same mode-aware clarification copy
    as the orchestrator pre-check so users see consistent wording, with a
    one-line preface naming the two requested entities.
    """
    from app.services.answers.clarification import clarification_for_mode

    clarify = clarification_for_mode(
        structured_query.intent.original_query or "",
        structured_query,
        "compare",
    )
    return (
        f"Compare: {left_bundle.label} vs {right_bundle.label}\n\n"
        "I couldn't pull enough course material to compare those two side-by-side.\n\n"
        f"{clarify}"
    )


def format_two_entity_compare_markdown(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
    left_bundle: ConceptEvidenceBundle | ConceptEvidenceBundleV2,
    right_bundle: ConceptEvidenceBundle | ConceptEvidenceBundleV2,
    kb: ConceptKB | None = None,
) -> str:
    """Build markdown for a two-way compare: direct lines, per-side bullets, axis contrast, closing.

    ``structured_query`` is kept for API consistency with the pipeline; the body is driven
    by ``plan`` (comparison axes, section specs) and the two evidence bundles.

    Steps:
    1. Pull ``core_lines`` directly off each V2 bundle (or recompute scoped
       lines for legacy bundles via :func:`scoped_lines_from_chunks`).
    2. **Direct answer** — first scoped line per side (with provisional note if extraction fell back).
    3. **Explanation** — extra bullets per side, only from that side's chunks.
    4. **Contrast** — for each KB axis, pair a short line from the left pool with one from the right.
    5. **What they share** — shared bullets (only when both bundles carry
       :attr:`ConceptEvidenceBundleV2.shared_lines` clearing
       :data:`COMPARE_SHARED_MIN_SUPPORT`).
    """
    kb = kb or get_kb()
    chunks_by_id = {c.get("id"): c for c in all_chunks if c.get("id") is not None}
    left_chunks = [chunks_by_id[i] for i in left_bundle.chunk_ids if i in chunks_by_id]
    right_chunks = [chunks_by_id[i] for i in right_bundle.chunk_ids if i in chunks_by_id]

    # Residual zero-chunk compare case: retrieval returned chunks but neither
    # bundle carries scope-able evidence. Emit a clarification rather than
    # the empty Course Answer scaffold.
    left_core = list(getattr(left_bundle, "core_lines", []) or [])
    right_core = list(getattr(right_bundle, "core_lines", []) or [])
    if not left_chunks and not right_chunks and not left_core and not right_core:
        return _compare_clarification_fallback(
            structured_query, left_bundle, right_bundle
        )

    forbidden_for_left = _resolve_forbidden_terms_for_entity(
        plan, "side_a", left_bundle.concept_id, right_bundle.concept_id, kb
    )
    forbidden_for_right = _resolve_forbidden_terms_for_entity(
        plan, "side_b", right_bundle.concept_id, left_bundle.concept_id, kb
    )

    left_lines, left_is_provisional = _scoped_lines_for_bundle(
        left_bundle,
        bundle_chunks=left_chunks,
        peer_concept_ids=[right_bundle.concept_id],
        forbidden_override=forbidden_for_left,
        kb=kb,
    )
    right_lines, right_is_provisional = _scoped_lines_for_bundle(
        right_bundle,
        bundle_chunks=right_chunks,
        peer_concept_ids=[left_bundle.concept_id],
        forbidden_override=forbidden_for_right,
        kb=kb,
    )

    shared_lines = _shared_lines_from_bundles(left_bundle, right_bundle)

    left_summary_line = (
        shorten_for_compare_cell(left_lines[0], max_len=380)
        if left_lines
        else f"(Limited direct material for **{left_bundle.label}** in retrieved notes.)"
    )
    right_summary_line = (
        shorten_for_compare_cell(right_lines[0], max_len=380)
        if right_lines
        else f"(Limited direct material for **{right_bundle.label}** in retrieved notes.)"
    )

    if left_is_provisional and left_lines:
        left_summary_line = (
            f"{left_summary_line} *(provisional wording—notes mix topics; prefer a follow-up "
            f"scoped to {left_bundle.label}.)*"
        )
    if right_is_provisional and right_lines:
        right_summary_line = (
            f"{right_summary_line} *(provisional wording—notes mix topics; prefer a follow-up "
            f"scoped to {right_bundle.label}.)*"
        )

    axis_labels = plan.comparison_axes[:6] if plan.comparison_axes else [
        "purpose",
        "computation",
        "typical use in the course",
    ]

    contrast_bullets: list[str] = []
    for axis_label in axis_labels:
        left_pick = pick_line_for_axis(left_lines, axis_label)
        right_pick = pick_line_for_axis(right_lines, axis_label)
        left_snippet = (
            shorten_for_compare_cell(left_pick, max_len=200)
            if left_pick
            else "*(No scoped line in retrieved notes for this axis.)*"
        )
        right_snippet = (
            shorten_for_compare_cell(right_pick, max_len=200)
            if right_pick
            else "*(No scoped line in retrieved notes for this axis.)*"
        )
        contrast_bullets.append(
            f"- **{axis_label}:** **{left_bundle.label}:** {left_snippet} **{right_bundle.label}:** {right_snippet}"
        )

    evidence_gap_bullets: list[str] = []
    if left_bundle.gap_flags:
        evidence_gap_bullets.append(
            f"- **{left_bundle.label}:** evidence support is thin; treat claims as provisional."
        )
    if right_bundle.gap_flags:
        evidence_gap_bullets.append(
            f"- **{right_bundle.label}:** evidence support is thin; treat claims as provisional."
        )
    evidence_gaps_block = "\n".join(evidence_gap_bullets)

    markdown_parts = [
        "Course Answer:",
        "",
        "### Direct Answer",
        "",
        f"**{left_bundle.label}** in one line: {left_summary_line}",
        "",
        f"**{right_bundle.label}** in one line: {right_summary_line}",
        "",
        "### Explanation",
        "",
        f"**{left_bundle.label} (from course text):**",
        "",
    ]
    for line in left_lines[1:5]:
        markdown_parts.append(f"- {line}")
    markdown_parts.extend(
        [
            "",
            f"**{right_bundle.label} (from course text):**",
            "",
        ]
    )
    for line in right_lines[1:5]:
        markdown_parts.append(f"- {line}")

    markdown_parts.extend(
        [
            "",
            "### Contrast along course axes",
            "",
            *contrast_bullets,
            "",
            "### Why the difference matters",
            "",
            "Getting the contrast right matters when you interpret model behavior, read plots, "
            "or choose an architecture for speech or language tasks in this course.",
        ]
    )
    if shared_lines:
        markdown_parts.extend(["", "### What they share", ""])
        for line in shared_lines:
            markdown_parts.append(f"- {shorten_for_compare_cell(line, max_len=240)}")
    if evidence_gaps_block:
        markdown_parts.extend(["", "### Evidence notes", "", evidence_gaps_block])
    return "\n".join(markdown_parts).rstrip()


def format_multi_entity_compare_markdown(
    entity_bundles: list[ConceptEvidenceBundle | ConceptEvidenceBundleV2],
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
    plan: AnswerPlan | None = None,
    kb: ConceptKB | None = None,
) -> str:
    """Build markdown for 3+ entities: comparison table (entity × axis) plus per-entity bullet notes.

    Each table row uses only that entity's retrieved chunks, passed through the same scoping
    pipeline as two-entity compare, so cells are not filled from a shared contaminated pool.
    Accepts a mix of V1 (legacy) and V2 bundles — V2 ``core_lines`` are
    consumed directly; V1 bundles trigger :func:`scoped_lines_from_chunks`
    just like the previous behaviour.
    """
    _ = structured_query
    kb = kb or get_kb()
    chunks_by_id = {c.get("id"): c for c in all_chunks if c.get("id") is not None}

    concept_ids_in_query = [b.concept_id for b in entity_bundles]
    axis_labels = (
        plan.comparison_axes[:5]
        if plan and plan.comparison_axes
        else ["role", "computation", "typical use in the course"]
    )

    # Map each concept id → (scoped lines, used_relaxed_fallback)
    scoped_lines_by_concept: dict[str, tuple[list[str], bool]] = {}
    for bundle in entity_bundles:
        peer_concept_ids = [cid for cid in concept_ids_in_query if cid != bundle.concept_id]
        bundle_chunks = [chunks_by_id[i] for i in bundle.chunk_ids if i in chunks_by_id]
        lines, used_provisional_fallback = _scoped_lines_for_bundle(
            bundle,
            bundle_chunks=bundle_chunks,
            peer_concept_ids=peer_concept_ids,
            forbidden_override=None,
            kb=kb,
            max_lines=10,
        )
        scoped_lines_by_concept[bundle.concept_id] = (lines, used_provisional_fallback)

    table_header_cells = ["Architecture", *[label[:48] for label in axis_labels]]
    table_header_row = "| " + " | ".join(table_header_cells) + " |"
    table_separator_row = "| " + " | ".join(["---"] * len(table_header_cells)) + " |"
    table_body_rows: list[str] = []
    for bundle in entity_bundles:
        row_lines, _ = scoped_lines_by_concept.get(bundle.concept_id, ([], False))
        row_cells = [f"**{bundle.label}**"]
        for axis_label in axis_labels:
            best_line = pick_line_for_axis(row_lines, axis_label) if row_lines else None
            if best_line:
                row_cells.append(sanitize_table_cell(best_line, max_len=180))
            elif row_lines:
                row_cells.append(sanitize_table_cell(row_lines[0], max_len=180))
            else:
                row_cells.append("*Limited evidence in retrieved chunks.*")
        table_body_rows.append("| " + " | ".join(row_cells) + " |")

    markdown_parts = [
        "Course Answer:",
        "",
        "### Compared architectures",
        "",
        "This answer uses **separate evidence pools** per entity; each table cell is drawn only "
        "from that row’s retrieved chunks after topic filtering.",
        "",
        table_header_row,
        table_separator_row,
        *table_body_rows,
        "",
        "### Entity notes (course-grounded)",
        "",
    ]
    for bundle in entity_bundles:
        note_lines, used_provisional_fallback = scoped_lines_by_concept.get(
            bundle.concept_id, ([], False)
        )
        markdown_parts.append(f"#### {bundle.label}")
        markdown_parts.append("")
        if bundle.gap_flags:
            markdown_parts.append(
                "*Limited direct evidence in retrieved chunks—claims are provisional.*"
            )
            markdown_parts.append("")
        if used_provisional_fallback and note_lines:
            markdown_parts.append(
                "*Some lines are shown with relaxed filtering where notes mix multiple topics.*"
            )
            markdown_parts.append("")
        if note_lines:
            for line in note_lines[:6]:
                markdown_parts.append(f"- {line}")
        else:
            markdown_parts.append(
                "- (No matching chunk text after scoping—try a narrower term from the syllabus.)"
            )
        markdown_parts.append("")

    markdown_parts.extend(
        [
            "### Why contrasts matter",
            "",
            "These ideas differ in how they summarize signals, share parameters, or compose across layers; "
            "the table ties each entity to course-sized descriptions along the same axes.",
        ]
    )
    return "\n".join(markdown_parts).rstrip()
