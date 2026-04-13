# Admin insights Phases 2–4 (2026-04-12)

## Shipped

- **Phase 2:** Paged **`/api/admin/insights/low-confidence`**, CSV **`/api/admin/insights/low-confidence.csv`** (no user emails; question text only).
- **Phase 3:** **`/api/admin/insights/chunks`** — top lecture chunks in low-confidence retrievals vs overall (JOIN `retrieval_chunk_hits` + `retrieval_logs`, enrich from `lecture_chunks`).
- **Phase 4:** **`models_and_tokens`** on main insights payload; **`response_variants.token_usage_json`** stores `primary` + `boost` blobs; **`retrieval_logs.token_usage_json`** stores primary only; OpenAI meta from **`_openai_chat`**; Gemini **`usageMetadata`** when returned.

## Follow-ups

- Paging UI for low-confidence (offset/next).
- Daily token time series (would need date bucketing or new materialized logic).
