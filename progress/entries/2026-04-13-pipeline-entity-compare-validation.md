# 2026-04-13 — Entity evidence, multi-compare, validation hardening

## What shipped

- **Entity retrieval:** Fused scoring (entity hits − cross-entity − forbidden lemmas + purity) replaces the old `0.1` weak fallback when entity scoring is on. Per-concept **evidence bundles** for two-way and multi-entity compare.
- **Query parsing:** `extract_compare_entities` handles `vs` chains (3+) and comma lists after *compare*; `QueryIntent.compare_entities` carries the full ordered list.
- **Structured query:** `compare_multi` answer intent when more than two entities; `ResponseConstraints` from deterministic regex parsing on the raw question.
- **Planning:** `SectionSpec` + `evidence_bundles` on `AnswerPlan`; two-way compare assigns **isolated** chunk pools per section (intro/contrast use paired top IDs from each side).
- **Rendering:** `compare_render.py` — axis-oriented two-way Markdown; table + per-entity notes for multi-compare. **Course generation** skips OpenAI for `compare` / `compare_multi` so contamination isn’t reintroduced by the LLM.
- **Retrieval v2:** For ≥3 compare entities, merges side-specific `retrieve_chunks` calls (deduped).
- **Pipeline:** Optional retrieval retry with larger `top_k` when validation severity is `fail` (before LLM→rule fallback).
- **LLM path:** When `SECTION_CONTRACTS_ENABLED`, user prompt includes section-scoped evidence blocks and hard constraints (no examples, intuition-only, N explanations, repeat).
- **Validation:** New critical checks — forbidden-topic leakage for single-concept non-compare intents, no-examples / intuition-only violations, boilerplate summary phrase; `compare_multi` coverage heuristic; stricter synthesis bridge vocabulary list.

## Follow-ups

- Tune forbidden lemma lists per concept as corpus evolves; consider lecture-conditional DP↔NN links.
- Section-boundary validation currently operates on full answer text for single-concept forbidden terms; per-`###` slicing would reduce false positives if needed.
