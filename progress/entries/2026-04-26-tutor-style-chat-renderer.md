# 2026-04-26 — Tutor-style chat-mode renderer

## Context

Chat-mode (i.e. non-compare / non-summary) answers were rendered with a fixed
four-block markdown layout — `### Direct Answer`, `### Explanation`,
`### Example / Intuition`, `### Why it matters` — built by
`generate_structured_answer` in
[`backend/app/services/answers/answer_generation.py`](../../backend/app/services/answers/answer_generation.py).

That layout reads more like a data dump than a tutor explaining something:
each block is independent, the explanation block repeats material from the
direct answer, and the “why it matters” closer falls back to a small set of
generic templates. We wanted chat replies to flow like teaching prose while
keeping retrieval, planning, and validation untouched (and adding zero new
LLM dependencies).

## Change

Added `render_tutor_style_answer(plan, evidence) -> str` — a narrative
renderer for chat-mode replies — and routed chat-mode intents through it from
`generate_structured_answer` whenever no exotic response constraint is in
play.

The new layout per response:

1. **Opening sentence** — pulled from the existing `direct_answer` line,
   redundancy-trimmed (drops awkward `"Direct Answer:"` / `"Definition:"`
   prefixes when present).
2. **Optional contrast / clarification** — scans `explanation_lines` for
   cues like `vs`, `instead`, `whereas`, `unlike`, `however`, `hardmax`,
   `differs`. Renders the matched line, plus a follow-up line that mentions
   the focal concept when one is available.
3. **Concrete example block** — uses the same `_example_intuition_block`
   source as before. Numeric arrays / tuples (`[2, 5]`, `0.12, 0.88`) are
   lifted onto their own line so the example reads visually like the
   tutor-style target.
4. **Key idea** — explicit `"The key idea:\n<sentence>"` highlight. The
   sentence is the shortest concept-mentioning candidate from the explanation
   lines (12–160 chars), with a fallback to the direct answer or the concept
   label.
5. **Why it matters** — concept- and topic-grounded closer that always begins
   with `"That matters because"`. Uses `plan.include_related_concepts`,
   primary chunk topic, and the cleaned concept label (lecture prefixes and
   trailing dash-section qualifiers stripped). The legacy generic templates
   (`"You'll keep running into related ideas such as …"`,
   `"Solid intuition here makes the next topics …"`) are no longer reused.
   Capped at two sentences.

## Content cleanup (Task 5)

A small content-cleanup layer runs before the renderer reads its sources:

- `_is_generic_filler` strips lines containing the legacy filler phrasings
  (`"you'll keep running into"`, `"this topic connects to"`,
  `"solid intuition here makes the next topics"`,
  `"notation and vocabulary pay off later"`,
  `"think of the explanation above as the core picture"`,
  `"see the explanation below for how the notes develop"`).
- `_clean_explanation_lines` deduplicates explanation lines and drops any
  line that is the same sentence as the direct answer (after whitespace /
  punctuation normalization), so contrast / key-idea cannot repeat the
  opening.
- `_dedupe_paragraphs` is applied to the assembled paragraphs as a final
  pass so no rendered paragraph repeats earlier content.
- `_truncate_to_first_sentences` keeps each paragraph to ≤ 2 sentences (≤ 1
  for the key-idea sentence) — that gives the spacing rules from Task 4
  (short paragraphs, blank line between sections, no dense text blocks). The
  renderer never emits bullet markers (`-` / `*`); explanation lines are
  selected as full sentences instead.

## Scope (what stays untouched)

- **Compare** (`compare`) and **multi-entity compare** (`compare_multi`):
  unchanged — `format_two_entity_compare_markdown` /
  `format_multi_entity_compare_markdown` still own those paths.
- **Lecture summary** and **cross-lecture synthesis**: unchanged — they fall
  through to the legacy four-section layout below the new chat-mode branch.
- **Quiz / summary renderers** in `quiz_render.py` / `summary_render.py`: not
  touched.
- **Retrieval, planning, validation**: not touched.
- **Constraint-driven legacy layout**: the safety refusal
  (`allow_incorrect_statements`) and the explicit structured-explanation
  constraints (`exact_explanation_count`, `repeat_explanation_times`) keep
  the legacy `###`-section layout because they explicitly request the
  numbered subsections / repeated block / safety copy.
- **`no_examples` and `intuition_only`** flow through the tutor renderer but
  suppress the example block via a call-site
  `dataclasses.replace(plan, include_example=False)` — the planner instance
  itself is never mutated.

## Validation impact

Validators in `answer_validation.py` operate on the answer string, not on
section headings, so removing `### Direct Answer / …` does not regress them.
Specifically:

- `must_be_course_grounded` — opening sentence preserves the concept name
  via `direct_answer`; the closer mentions the concept again. Outputs are
  comfortably > 200 chars.
- `must_define_primary_concept` (`direct_definition`) — opening sentence
  contains the primary concept (it comes from the chunk that defines it).
- `must_answer_how_or_why` (`multi_step_explanation`, `scoped_explanation`)
  — the closer always begins with `"That matters because"`, satisfying the
  causal-cue requirement.
- `must_not_have_examples_when_blocked` / `must_not_have_technical_when_intuition_only`
  — exotic constraints route around the new renderer, so existing copy still
  applies verbatim.

Confirmed by running the full backend test suite (240 passed,
`tests/test_e2e_chat_flow.py` skipped because it needs a live network).

## Files changed

- [`backend/app/services/answers/answer_generation.py`](../../backend/app/services/answers/answer_generation.py)
  — module docstring updated, new helpers (`_primary_concept_label`,
  `_natural_opening_sentence`, `_format_example_block`,
  `_contrast_lines_block`, `_key_idea_sentence`, `_grounded_why_it_matters`),
  new public `render_tutor_style_answer`, and a chat-mode dispatch added to
  `generate_structured_answer`.
- [`backend/tests/test_structured_pipeline.py`](../../backend/tests/test_structured_pipeline.py)
  — replaced `test_generate_structured_answer_four_sections` with
  `test_chat_mode_uses_tutor_narrative_format` (asserts the new flow markers:
  `Course Answer:`, `The key idea:`, `That matters because`, no `###`
  section headings),
  `test_chat_mode_no_examples_uses_tutor_narrative_without_example_block`
  (locks in the new behavior: `no_examples` keeps the tutor narrative but
  drops the `Think of it this way:` block), and
  `test_chat_mode_repeat_explanation_keeps_legacy_layout` (locks in the
  legacy structured layout for `repeat_explanation_times`).
- [`CHANGELOG.md`](../../CHANGELOG.md) — Unreleased / Added entry.

## Verification

```bash
cd backend && source .venv/bin/activate
python -c "from app import create_app; create_app()"        # OK
python -m pytest tests/ -q --ignore=tests/test_e2e_chat_flow.py
# 240 passed, 16 warnings
```

End-to-end smoke test on `What is softmax?`, `How does backpropagation work?`,
`Explain attention briefly` — all return tutor-narrative outputs, validation
severity `pass`, and concept labels are clean (e.g. `Softmax`, not
`Softmax — Core Idea`).

## Follow-ups

- The OpenAI primary-path prompt in
  [`generation_input.py`](../../backend/app/services/generation/generation_input.py)
  still nudges the model toward the legacy four-section layout. If we want
  the LLM Course Answer to match the new tutor narrative, that prompt and
  `enforce_structure` in `output_cleanup.py` will need a paired update — out
  of scope for this task.
- The example block currently only renders when `plan.include_example` is
  true (i.e. at least one chunk carries a non-empty `sample_answer` /
  `sample_questions`). Authoring more example data on chunks would let the
  tutor-style block fire more often.
