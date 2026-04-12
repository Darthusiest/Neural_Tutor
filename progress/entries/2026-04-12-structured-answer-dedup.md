# Structured Course Answer: section dedup (2026-04-12)

**Problem:** For queries like “explain attention,” `direct_definition` used the same top-ranked chunk IDs for every plan section (“Direct answer,” “Mechanism,” “Why it matters”), and `generate_structured_answer` printed each chunk again under each `###` heading. A blanket fallback also filled empty sections with `primary_ids[:2]`, amplifying duplication. Empty `sample_questions` JSON (`[]`) was treated as present, producing `- []` under “Example question.”

**Changes:** `build_answer_plan` assigns **one distinct chunk per section** (in relevance order) for `direct_definition` and `multi_step_explanation`; sections with no chunk after splitting are removed. `generate_structured_answer` skips chunks already emitted and uses `_sample_questions_as_text` for examples. OpenAI plan-constrained system prompt asks not to repeat the same excerpt under multiple headings.

**Tests:** `test_direct_definition_distinct_chunk_per_section` in `test_structured_pipeline.py`.
