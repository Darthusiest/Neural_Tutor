# Analytics layer redesign

## Summary

Redesigned the analytics schema from 3 shallow tables to 5 optimization-oriented tables. The previous schema stored `retrieved_chunk_ids` as a JSON blob, had no score decomposition, and tracked only thumbs for feedback. The new schema captures the full decision chain from query features through chunk-level scoring to response metadata to user outcome signals.

## What changed

### Models (`backend/app/models/analytics.py`)

**RetrievalLog** (enriched) — now one-to-one with assistant messages (unique `message_id`). Added: `normalized_query`, `query_tokens_json`, `lecture_numbers_detected_json`, `retrieval_backend`, `top_k_requested`, `num_chunks_scored`, `num_chunks_hit`, `top_score`, `second_score`, `score_margin`, `query_coverage`, `is_low_confidence`, `is_off_topic`. Legacy `retrieved_chunk_ids` kept but deprecated (nullable, not written by new code).

**RetrievalChunkHit** (new) — one row per selected chunk per retrieval event. Stores `rank`, `score`, score decomposition (`token_score`, `phrase_score`, `lecture_bonus`, `strong_field_token_score`), `matched_query_terms`, `phrase_events`, and `field_scores_json` (per-field token contribution breakdown). Composite index on `(retrieval_log_id, rank)`.

**ResponseVariant** (enriched) — now links to `retrieval_log_id`. Added boost decomposition: `boost_used`, `boost_auto_triggered`, `boost_toggle_user_selected`. Added prompt versioning: `course_answer_prompt_version`, `boost_prompt_version`. Added `provider_name`, `course_answer_length`, `boosted_answer_length`, and `response_fingerprint` (SHA-256 prefix for duplicate detection).

**Feedback** (enriched) — added `helpfulness_rating` (1-5), `resolved`, `follow_up_required`, `follow_up_type`, `explicit_confusion_flag`, `feedback_note`, `preference_strength`. All nullable, backward-compatible with existing thumb-only payloads.

**MessageOutcome** (new) — retroactively populated when the next user message arrives. Stores `had_follow_up`, `follow_up_count`, `follow_up_type`, `was_rephrased`, `user_changed_topic_after`, `answer_resolved`. Uses heuristic detection: token-overlap ratio > 0.6 for rephrase, keyword matching for follow-up type classification.

### Retrieval diagnostics (`backend/app/services/retrieval.py`)

Added `ChunkHitDiag` and `RetrievalDiagnostics` public dataclasses. `RetrievalResult` gains an optional `diagnostics` field (backward-compatible — callers that don't read it are unaffected). Added `field_scores` tracking to `_ScoreParts` and the scoring loop. `score_chunks_keyword` now builds full diagnostics on every call.

### Orchestrator (`backend/app/services/chat_orchestrator.py`)

`handle_chat_turn` now: writes enriched `RetrievalLog` columns from `r.diagnostics`; creates `RetrievalChunkHit` rows per selected chunk; computes `response_fingerprint` via SHA-256; decomposes boost trigger into three booleans; retroactively populates `MessageOutcome` for the previous assistant message before processing the new turn.

### Migration infrastructure

Set up Flask-Migrate (Alembic). Initial migration: `backend/migrations/versions/001_analytics_layer_redesign.py`. Adds nullable columns to existing tables and creates new tables. Uses `batch_alter_table` for SQLite compatibility.

## Decisions

- **Chunk hits limited to selected (top_k) chunks** rather than all scored chunks. Write volume stays proportional to top_k, not corpus size. Can expand later for deep diagnostics.
- **MessageOutcome heuristics are intentionally conservative.** Token overlap for rephrase detection, keyword matching for follow-up classification. Simple enough to audit; can be replaced with classifiers or embedding similarity when embeddings exist.
- **Response fingerprint is SHA-256, not SimHash.** Catches exact duplicates. Near-duplicate detection needs embeddings or SimHash, which is a future extension.
- **All new columns are nullable** so existing rows survive migration. No data loss.
- **`retrieved_chunk_ids` kept as deprecated** rather than dropped. Existing rows retain data; future migration can remove it.

## Analytics questions this schema now supports

- Which topics are frequently low-confidence? (`retrieval_logs.is_low_confidence` + `detected_topic`)
- Which chunks correlate with bad outcomes? (`retrieval_chunk_hits` JOIN `feedback`/`message_outcomes`)
- What score margin patterns predict bad answers? (`retrieval_logs.score_margin` vs outcomes)
- Does boost help or hurt? (`response_variants.boost_used` vs `feedback.preferred`)
- Which answers trigger rephrases? (`message_outcomes.was_rephrased` by topic/chunk)
- Which chunks need rewriting? (`retrieval_chunk_hits` with low helpfulness correlation)
- Are answers being repeated? (`response_variants.response_fingerprint` grouping)
- What is the latency/cost per retrieval backend? (`retrieval_logs.latency_ms`, `retrieval_backend`)

## Follow-ups

- Build admin insights dashboard aggregating analytics tables
- Tune `FIELD_WEIGHTS` and `CONFIDENCE_THRESHOLD` using score_margin/coverage vs feedback correlation
- Add embedding backend and populate dense scoring columns
- Replace MessageOutcome heuristics with embedding-based similarity when available
- Consider expanding RetrievalChunkHit to store all scored chunks (not just top_k) if deep diagnostics are needed
