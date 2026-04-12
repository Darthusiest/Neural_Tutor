# Retrieval v2 hardening — documentation sync

**Summary:** Aligned root [`README.md`](../../README.md), [`backend/docs/schema.md`](../../backend/docs/schema.md), and [`backend/.env.example`](../../backend/.env.example) with the hardened pipeline: chat uses `retrieve_enhanced`, lexical `lecture_filter` / `summary_rank`, `SUMMARY_MAX_CHUNKS`, and assistant `payload_json` optional `query_type`.

**Changes:** See [`CHANGELOG.md`](../../CHANGELOG.md) under `[Unreleased]` for behavior; this entry records the doc touchpoints only.
