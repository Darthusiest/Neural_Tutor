# SQLite schema (SQLAlchemy models)

Tables are created with:

```bash
cd backend && flask --app wsgi init-db
```

**Alembic:** the app registers Flask-Migrate ([`app/__init__.py`](../app/__init__.py)). For an existing database that predates analytics columns, run `flask --app wsgi db upgrade` after pulling migrations (see [`migrations/versions/`](../migrations/versions/)). Fresh clones can use `init-db` alone when starting from an empty file.

This document mirrors [`app/models/`](../app/models/). Types are logical (SQLite may store them differently).

## users

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| email | String(255) UNIQUE, indexed | |
| password_hash | String(256) | Werkzeug hash |
| created_at | DateTime | server default now |
| is_admin | Boolean | default false |

## password_reset_tokens

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| user_id | FK → users.id, indexed | |
| token_hash | String(128), indexed | SHA-256 hex of raw token |
| expires_at | DateTime | UTC |
| used_at | DateTime nullable | set when consumed |

## chat_sessions

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| user_id | FK → users.id, indexed | |
| title | String(512) | |
| mode | String(32) | chat / quiz / compare / summary |
| created_at | DateTime | |
| updated_at | DateTime | |

## messages

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| session_id | FK → chat_sessions.id, indexed | |
| role | String(32) | user / assistant / system |
| content_text | Text nullable | user plain text |
| payload_json | Text nullable | assistant metadata JSON |
| created_at | DateTime | |

## retrieval_logs

One row per assistant turn that runs retrieval. Stores query features, aggregate scores, and flags for analytics (tuning `CONFIDENCE_THRESHOLD`, comparing backends). Chunk IDs live in **`retrieval_chunk_hits`**; `retrieved_chunk_ids` is legacy JSON, nullable, not written by new code.

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| session_id | FK → chat_sessions.id nullable, indexed | |
| message_id | FK → messages.id nullable, **unique**, indexed | |
| user_question | Text | |
| normalized_query | Text nullable | |
| query_tokens_json | Text nullable | JSON array |
| detected_topic | String(512) nullable | |
| lecture_numbers_detected_json | Text nullable | JSON array of ints |
| retrieval_backend | String(32) nullable | e.g. keyword |
| top_k_requested | SmallInteger nullable | |
| num_chunks_scored | Integer nullable | |
| num_chunks_hit | Integer nullable | |
| confidence | Float nullable | |
| top_score | Float nullable | |
| second_score | Float nullable | |
| score_margin | Float nullable | |
| query_coverage | Float nullable | |
| is_low_confidence | Boolean nullable | |
| is_off_topic | Boolean nullable | |
| latency_ms | Integer nullable | |
| token_usage_json | Text nullable | |
| retrieved_chunk_ids | Text nullable | **deprecated**; use `retrieval_chunk_hits` |
| created_at | DateTime | server default now |

## retrieval_chunk_hits

One row per ranked chunk for a retrieval event. Enables chunk-level analytics (scores, field breakdown, lecture bonus).

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| retrieval_log_id | FK → retrieval_logs.id ON DELETE CASCADE, indexed | |
| lecture_chunk_id | FK → lecture_chunks.id, indexed | |
| rank | SmallInteger | 1-based order |
| score | Float | |
| selected_for_answer | Boolean | default true |
| token_score | Float nullable | |
| phrase_score | Float nullable | |
| lecture_bonus | Float nullable | |
| strong_field_token_score | Float nullable | |
| matched_query_terms | SmallInteger nullable | |
| phrase_events | SmallInteger nullable | |
| field_scores_json | Text nullable | JSON dict field → float |
| created_at | DateTime | server default now |

Index: `(retrieval_log_id, rank)`.

## response_variants

One row per assistant message: course answer, optional boost, and generation metadata.

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| message_id | FK → messages.id **unique**, nullable false | |
| retrieval_log_id | FK → retrieval_logs.id nullable, indexed | |
| course_answer | Text | |
| boosted_explanation | Text nullable | |
| boost_used | Boolean nullable | |
| boost_reason | String(64) nullable | |
| boost_auto_triggered | Boolean nullable | |
| boost_toggle_user_selected | Boolean nullable | |
| model_name | String(128) nullable | |
| provider_name | String(64) nullable | |
| course_answer_prompt_version | String(32) nullable | |
| boost_prompt_version | String(32) nullable | |
| token_usage_json | Text nullable | |
| course_answer_length | Integer nullable | |
| boosted_answer_length | Integer nullable | |
| response_fingerprint | String(40) nullable, indexed | SHA-256 prefix (32 hex chars) |
| created_at | DateTime | server default now |

## feedback

User feedback on an assistant message (thumbs, preference, optional enriched fields).

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| message_id | FK → messages.id **unique**, nullable false | |
| course_thumb | String(8) nullable | |
| boost_thumb | String(8) nullable | |
| preferred | String(16) nullable | |
| helpfulness_rating | SmallInteger nullable | |
| resolved | Boolean nullable | |
| follow_up_required | Boolean nullable | |
| follow_up_type | String(32) nullable | |
| explicit_confusion_flag | Boolean nullable | |
| feedback_note | Text nullable | |
| preference_strength | String(16) nullable | |
| created_at | DateTime | server default now |

## message_outcomes

Heuristic outcome for the **previous** assistant message when the user sends a follow-up (rephrase, topic change, resolution). Populated by chat orchestration, not a classifier.

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| message_id | FK → messages.id **unique**, indexed | assistant message |
| had_follow_up | Boolean nullable | |
| follow_up_count | SmallInteger nullable | |
| follow_up_type | String(32) nullable | |
| was_rephrased | Boolean nullable | |
| user_changed_topic_after | Boolean nullable | |
| answer_resolved | Boolean nullable | |
| created_at | DateTime | server default now |

## lecture_chunks (course corpus)

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| lecture_number | Integer, indexed | |
| topic | String(512), indexed | e.g. `{lecture title} — {section heading}` |
| keywords | Text | JSON array of strings |
| source_excerpt | Text | Raw course material (joined bullets / slides text) |
| clean_explanation | Text | Pedagogical text |
| sample_questions | Text nullable | JSON array of strings |
| sample_answer | Text nullable | Single exemplar answer |

Seed JSON may use `source_text` or `content`; the loader maps both to `source_excerpt` (see [`lecture_loader.py`](../app/services/lecture_loader.py)).
