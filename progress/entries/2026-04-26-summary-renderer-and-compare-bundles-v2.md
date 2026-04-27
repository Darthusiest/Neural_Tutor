# 2026-04-26 — Real summary renderer + compare evidence bundles V2

## Context

Two follow-on tasks landed in the same pass — both touch deterministic answer
rendering, both leave retrieval / chat flow alone:

1. **Summary renderer (Task 3).** `summary_render.format_summary_markdown`
   already had a lecture-scoped path and a topic-scoped path, but the topic
   path didn't actually filter the candidate chunks down to the requested
   topic. A query like *"Give me a recap of MFCCs"* could surface formant or
   softmax chunks if retrieval brought them in. The lecture-scoped path also
   had no notion of in-lecture ordering metadata, so we couldn't honour
   `chunk_order` / `position` once those columns are added.
2. **Compare evidence bundles (Task 4).** `ConceptEvidenceBundle` was a
   minimal four-field shape (`concept_id` / `label` / `support_score` /
   `chunk_ids` / `gap_flags`). The compare renderer recomputed scoped lines
   from chunks every time, and there was no first-class way to mark a
   sentence as "shared between A and B" — so a query like *"Bias versus
   variance"* would either drop the shared bias-variance tradeoff sentence
   from both sides or assign it to one side arbitrarily.

## Decisions confirmed before coding

- **Header format:** keep `Summary: Lecture N` / `Summary: <topic>` (colon,
  matches `Quiz:` / `Course Answer:`). The original ask used an em-dash, but
  every existing mode-routing test asserts the colon form, so switching would
  break a dozen tests for cosmetic gain.
- **Bundle shape:** ship `ConceptEvidenceBundleV2` as a *new* dataclass
  (compatibility-layer approach). Keep `ConceptEvidenceBundle` (legacy)
  intact, and have V2 expose the legacy field names as read-only properties
  plus `from_legacy_bundle` / `to_legacy_bundle` adapters. That gives a clean
  future architecture without churning every existing call site or test
  fixture.
- **Shared evidence:** render a new `### What they share` section under
  compare output, only when both bundles have at least one shared line and
  both clear a `min_support` threshold (default 0.25).

## What changed

### Task 3 — `summary_render.py`

- **Lecture-scoped path** (`_format_lecture_summary`): kept the hard
  `lecture_number` filter, added `_lecture_chunk_sort_key` so chunks are
  ordered by `chunk_order` / `position` / `order` metadata when present
  (forward-compatible — `LectureChunk` doesn't carry the column today) and
  fall back to `id` ascending. Primary chunks are kept first so that the
  "Main idea" anchor stays close to whatever retrieval ranked top. Topic
  heads are deduped via `_dedupe_preserve_order` on `_topic_head` outputs
  (already in place). Layout is unchanged: `Summary: Lecture N` +
  `### Main idea` / `### Key topics` / `### How the topics connect` /
  `### Study focus`.
- **Topic-scoped path** (`_format_topic_summary`): rewrote so that the
  candidate chunks are filtered against the canonical topic *plus* its
  `ConceptKB` aliases before the renderer reads them. `_topic_term_set`
  pulls the term set from `intent.detected_concepts` + KB lookup +
  `structured_query.concept_ids`. `_topic_scoped_chunks` drops any chunk
  whose `topic` / `keywords` / `clean_explanation` don't mention at least
  one term, so a "Recap of MFCCs" never lets a softmax chunk through.
  Layout is `Summary: <topic>` + `### Core idea` / `### Key points` /
  `### Study focus` (note: `Core idea`, not `Main idea` — the
  topic-scope spec calls for `Core idea`).
- Empty-after-filter both paths fall back to a short clarification message
  (still `Summary: …` header so the mode-renderer contract holds).

### Task 4 — `entity_retrieval.py` + `compare_render.py`

- **`ConceptEvidenceBundleV2`** dataclass with full surface (`concept`,
  `aliases`, `evidence_chunks`, `core_lines`, `support_score`,
  `forbidden_hits`, `shared_lines`, `source_metadata`, `confidence`) plus
  `label_override` / `gap_flags_override` knobs for tests / non-KB ids.
  Legacy compatibility:
  - `concept_id`, `label`, `chunk_ids`, `gap_flags` are read-only
    properties so existing call sites that iterate over
    `AnswerPlan.evidence_bundles` keep working.
  - `from_legacy_bundle(legacy, kb=..., evidence_chunks=..., core_lines=...,
    shared_lines=..., forbidden_hits=...)` upgrades a V1 bundle into the V2
    envelope (aliases pulled from KB when available).
  - `to_legacy_bundle()` round-trips back to the four-field shape.
  - `EvidenceBundleLike = ConceptEvidenceBundle | ConceptEvidenceBundleV2`
    type alias is what `AnswerPlan.evidence_bundles` is now typed as.
- **`build_bundles_for_compare_v2`** runs the existing per-side scoring
  (`score_chunk_for_entity` with the *other* concept as peer), promotes the
  full chunk dicts onto `evidence_chunks`, and then walks every unit through
  `classify_line_for_compare`:
  1. A line whose A-score is strictly higher than B and isn't blocked by an
     A-side forbidden term goes to side A's `core_lines`.
  2. A line whose B-score is strictly higher and is forbidden-clean for B
     goes to side B's `core_lines`.
  3. A line that scores >= 1 for both entities at a min-ratio threshold
     (default 0.6) and is forbidden-clean for both becomes a `shared_lines`
     entry on **both** bundles.
  4. Anything else is dropped.

  Aliases / forbidden terms come from the KB by default; tests can override
  via `aliases_override` / `label_override` for non-KB entities (e.g.
  *bias* and *variance* as separate entities — the LING487 KB packages them
  as one `bias_variance` concept).
- **`build_bundles_multi_v2`** uses the same V2 envelope. Two-entity multi
  delegates to `build_bundles_for_compare_v2`. For 3+ entities, the
  shared bucket isn't well-defined, so `shared_lines` stays empty and lines
  are filtered against the union of peer terms.
- **`_entity_terms_for_aliases`** (helper used by the V2 builders) now
  includes the bare `concept_id` token in the term set and tokenises
  multi-word aliases on punctuation. Required for KB concepts whose
  canonical name is something like `"hardmax / winner-take-all"` — without
  this, the term `hardmax` from the chunk explanation wouldn't match
  anything and the bundle's `core_lines` would come out empty.
- **`compare_render.format_two_entity_compare_markdown`** — added
  `_scoped_lines_for_bundle` and `_shared_lines_from_bundles`. When the
  bundle is V2 and has non-empty `core_lines`, the renderer reads them
  directly off the bundle (no `scoped_lines_from_chunks` recompute). When
  both bundles' `support_score` clears `COMPARE_SHARED_MIN_SUPPORT` (0.25)
  and shared lines exist, the renderer emits a new `### What they share`
  section after `### Why the difference matters`. Otherwise the section is
  omitted entirely (no placeholder text).
- **`compare_render.format_multi_entity_compare_markdown`** — same source
  switch (V2 `core_lines` when available); shared section is omitted (3+
  entity semantics).
- **`answer_planning.build_answer_plan`** — wired to `build_bundles_for_compare_v2`
  / `build_bundles_multi_v2`. The V2 properties make this transparent to the
  call sites in `answer_generation.py` that iterate over
  `plan.evidence_bundles.values()` and read `concept_id` / `label` /
  `chunk_ids` / `gap_flags`.

## Tests

- **Summary** ([`test_answers_summary_render.py`](../../backend/tests/test_answers_summary_render.py)):
  - `test_summary_render_summarize_lecture_10`
  - `test_summary_render_main_takeaways_lecture_10`
  - `test_summary_render_main_ideas_lecture_16_drops_other_lectures`
  - `test_summary_render_lecture_dedupes_topic_heads`
  - `test_summary_render_lecture_uses_chunk_order_metadata`
  - `test_summary_render_lecture_with_no_evidence_falls_back`
  - `test_summary_render_recap_mfccs_topic_only` — softmax chunk filtered
    out before rendering.
  - `test_summary_render_topic_uses_topic_layout` — switched assertion from
    `### Main idea` to `### Core idea`.
- **Compare** ([`test_compare_render.py`](../../backend/tests/test_compare_render.py)):
  - `test_compare_cnn_and_mlp_disjoint_evidence` — `core_lines` disjoint;
    no `convolution` in MLP, no `fully connected` in CNN.
  - `test_compare_cnn_vs_transformer_no_shared_attention_leak` — no
    `self-attention` in CNN `core_lines`; no `convolution` in transformer
    `core_lines`.
  - `test_compare_mfccs_and_formants` — cepstrum / filterbank stay on
    MFCC; vocal tract stays on formants.
  - `test_compare_softmax_and_hardmax_separate_bundles` — disjoint
    `core_lines`; probability-distribution language stays on softmax,
    argmax / one-hot stays on hardmax.
  - `test_compare_bias_versus_variance_shared_section` — shared sentence
    surfaces under `### What they share`.
  - `test_compare_two_entity_no_shared_section_when_disjoint` — section
    omitted when bundles don't share any line.
  - `test_concept_evidence_bundle_v2_legacy_adapter_roundtrip` —
    `from_legacy_bundle` then `to_legacy_bundle` preserves the four legacy
    fields; KB aliases are pulled in.

Existing tests stay green: full backend suite reports **255 passed**
(248 → 255, +7 new tests; no regressions).

```bash
cd backend && source .venv/bin/activate
python -m pytest -q
# 255 passed, 16 warnings in 14.99s
```

## Files changed

- [`backend/app/services/answers/summary_render.py`](../../backend/app/services/answers/summary_render.py)
  — rewritten module docstring, lecture-scoped + topic-scoped paths
  redesigned per spec.
- [`backend/app/services/answers/entity_retrieval.py`](../../backend/app/services/answers/entity_retrieval.py)
  — added `ConceptEvidenceBundleV2`, `EvidenceBundleLike`,
  `_bundle_confidence`, `_entity_terms_for_aliases`, `_take_top_chunks`,
  `_source_metadata_for_chunks`, `classify_line_for_compare`,
  `_build_v2_lines`, `build_bundles_for_compare_v2`, `build_bundles_multi_v2`.
- [`backend/app/services/answers/answer_planning.py`](../../backend/app/services/answers/answer_planning.py)
  — V2 imports, `evidence_bundles` typed as `dict[str, EvidenceBundleLike]`,
  compare branches now call the V2 builders.
- [`backend/app/services/answers/compare_render.py`](../../backend/app/services/answers/compare_render.py)
  — `_scoped_lines_for_bundle`, `_shared_lines_from_bundles`,
  `COMPARE_SHARED_MIN_SUPPORT`, new `### What they share` block.
- [`backend/tests/test_answers_summary_render.py`](../../backend/tests/test_answers_summary_render.py)
  — new spec-query tests + topic-layout assertion update.
- [`backend/tests/test_compare_render.py`](../../backend/tests/test_compare_render.py)
  — V2-bundle test coverage, shared section, adapter round-trip.
- [`CHANGELOG.md`](../../CHANGELOG.md) — Unreleased / Added entries.
- [`progress/context-for-new-chat.md`](../context-for-new-chat.md) — recent
  product / API notes; "Compare evidence isolation" deferred-debt note now
  marked as addressed for evidence assembly (retrieval purity is still
  open).

## Out of scope

- No changes to retrieval (`retrieval_v2.py`), `chat_orchestrator.py`, or
  `query_mode.py`.
- No new DB columns or migrations.
- No frontend changes.
- LLM primary-path prompt for compare stays rule-based-only (already
  enforced in `course_generation.py`).

## Follow-ups

- Concept-purity scoring on retrieval would let the compare bundles start
  from a cleaner candidate pool. Today, even with line-level filtering, a
  bundle can still be limited by what retrieval surfaced.
- A `chunk_order` / `position` column on `LectureChunk` would let the
  lecture-scoped summary path use real curriculum order instead of falling
  back to `id` ascending. The renderer is already metadata-aware, so this
  is purely a DB change away.
- Multi-entity (3+) compare currently doesn't compute shared lines because
  the semantics get fuzzy. If we want a tri-shared section ("what all three
  share") later, `build_bundles_multi_v2` would need an extension —
  scoped here to keep the change small.
