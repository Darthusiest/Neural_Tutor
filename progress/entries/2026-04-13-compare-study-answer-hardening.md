# Compare / study answer hardening (2026-04-13)

## Problem

Stress tests on chat and study **compare** surfaced: repeated outline labels (“First idea”, “In one line”) on every explanation line, useless **Similarities** keyword lists (generic overlap tokens), **summary** “Topics to cross-link” repeating the same title, and occasional LLM echo of outline scaffolding.

## Changes

- **`answer_generation._build_explanation_bullets` (compare):** One `**Section heading:**` line per plan section, then a few deduped content lines (no per-line scaffold labels). **Contrast** line uses plan comparison axes.
- **`study.format_compare_answer`:** Stopword filter on keyword sets; similarities/differences copy revised; shorter “when each matters”. **`_format_summary_recap`:** Dedupe topic titles for cross-link list.
- **`output_cleanup.clean_output`:** Regex removal of common outline-only lines (e.g. “First idea:”, “In one line:”) for LLM primary path.
- **Docs / README:** Current status updated for study compare/summary, chat rule-based compare, Postgres + schema doc title, admin insights PostgreSQL note.

## Follow-ups

- Multi-entity compare (e.g. four architectures) still needs a dedicated plan + renderer.
- User constraints (“do not mention X”) need retrieval/validation allowlists, not templates alone.
