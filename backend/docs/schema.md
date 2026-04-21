# Database schema (SQLAlchemy models)

The app supports **SQLite** (default local file) and **PostgreSQL** (set **`DATABASE_URL`**, e.g. `postgresql+psycopg2://…`; see root [`README.md`](../../README.md)). Table definitions are the same; column types below are **logical** (SQLite and PostgreSQL may represent them slightly differently).

Tables are created with:

```bash
cd backend && flask --app wsgi init-db
```

**Alembic:** the app registers Flask-Migrate ([`app/__init__.py`](../app/__init__.py)). For an existing database that predates analytics columns, run `flask --app wsgi db upgrade` after pulling migrations (see [`migrations/versions/`](../migrations/versions/)). Fresh clones can use `init-db` alone when starting from an empty file. **Production** databases should use **`db upgrade`**, not only `init-db`, so migrations stay aligned with code.

This document mirrors [`app/models/`](../app/models/).

**Structured pipeline JSON** (e.g. `validation_checks_json`) is produced by code under [`app/services/answers/`](../app/services/answers/) (see `answer_validation.ValidationResult`).

**Admin insights** (read-only reporting): aggregates and drill-downs are computed in [`app/services/admin_insights.py`](../app/services/admin_insights.py) over the tables below; HTTP API reference in [`admin_insights.md`](admin_insights.md).

## users

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| email | String(255) UNIQUE, indexed | |
| password_hash | String(256) | Werkzeug hash |
| created_at | DateTime | server default now |
| is_admin | Boolean | default false |
| email_verified_at | DateTime nullable | set when email is verified |
| failed_login_attempts | Integer | lockout counter |
| locked_until | DateTime nullable | temporary lockout end |

## email_verification_tokens

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| user_id | FK → users.id, CASCADE | |
| token_hash | String(64), unique | |
| expires_at | DateTime | |
| consumed_at | DateTime nullable | |

## audit_events

Append-only **security event** log (no secrets). ORM: [`SecurityLogEntry`](../app/models/security_log.py) (table name remains `audit_events` from migration **005**); insert helper: [`log_security_event`](../app/services/security_logging.py).

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| created_at | DateTime, indexed | |
| actor_user_id | FK → users.id nullable | User who triggered the event (legacy column name from migration **005**) |
| actor_email | String(255) nullable | Same user’s email, denormalized for pre-login events (legacy column name) |
| event_type | String(64), indexed | e.g. `login_success`, `login_failed`, `register` |
| severity | String(16) nullable | |
| ip | String(64) nullable | |
| user_agent | String(512) nullable | |
| metadata_json | Text nullable | small JSON |

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
| payload_json | Text nullable | Assistant metadata JSON. Typical keys: `course_answer`, `confidence`, optional `query_type` (retrieval v2), optional `boosted_explanation`, optional `structured_pipeline`, optional `pipeline_diagnostics` (includes `answer_intent`, `validation` with `severity` pass/weak/fail, `primary_model` `openai`\|`rule_based`\|`none`, `query_complexity`, `answer_plan`, etc.), optional `primary_model`, `validation_severity`, `boost_provider` (`gemini`\|`openai` when a boost ran), `boost_reason` (e.g. `user_toggle`, `validation_weak`, `low_confidence`, `complex_query`, `mode`, `none`), `query_complexity` (`simple`\|`complex`), optional `no_match_kind` (`greeting`\|`short_ack`\|`off_topic` when no chunks matched; see [`conversational_responses.py`](../app/services/conversational_responses.py) for classification + rotating **Course Answer** text; boost is not applied when there are no chunks) |
| created_at | DateTime | |

## retrieval_logs

One row per assistant turn that runs retrieval. Stores query features, aggregate scores, and flags for analytics (tuning `CONFIDENCE_THRESHOLD`, comparing backends). Chunk IDs live in **`retrieval_chunk_hits`**; `retrieved_chunk_ids` is legacy JSON, nullable, not written by new code. When chat uses **retrieval v2** compare mode, aggregate scores in this row reflect the **merged** compare diagnostics (e.g. conservative confidence); per-chunk rows in **`retrieval_chunk_hits`** still describe the chunks shown in the Course Answer.

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
| token_usage_json | Text nullable | When structured pipeline uses OpenAI for the Course Answer, primary usage metadata JSON (same shape as `primary` in `response_variants.token_usage_json`) |
| retrieved_chunk_ids | Text nullable | **deprecated**; use `retrieval_chunk_hits` |
| query_type_v2 | String(64) nullable | Structured pipeline **answer_intent** (e.g. `direct_definition`, `compare`) when enabled |
| sub_questions_json | Text nullable | JSON array of decomposition sub-question strings |
| answer_mode | String(64) nullable | Mirrors answer plan mode |
| validation_passed | Boolean nullable | Structured validation aggregate pass/fail |
| validation_checks_json | Text nullable | JSON from :class:`~app.services.answers.answer_validation.ValidationResult` (includes `passed`, **`severity`** `pass`\|`weak`\|`fail`, `checks_*`, `flags`) |
| generic_answer_flag | Boolean nullable | Heuristic: thin / generic answer |
| missing_comparison_side_flag | Boolean nullable | Compare validation: one side missing |
| answer_plan_json | Text nullable | Serialized answer plan for debugging |
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
| boost_reason | String(64) nullable | Short code (e.g. `user_toggle`, `validation_fail`, `low_confidence`, `complex_query`, `mode`); aligns with chat gating in [`boost_triggers.py`](../app/services/generation/boost_triggers.py) |
| boost_auto_triggered | Boolean nullable | |
| boost_toggle_user_selected | Boolean nullable | |
| model_name | String(128) nullable | |
| provider_name | String(64) nullable | |
| course_answer_prompt_version | String(32) nullable | |
| boost_prompt_version | String(32) nullable | |
| token_usage_json | Text nullable | Optional `{"primary": {...}, "boost": {...}}` — OpenAI chat usage under `usage`, plus `model` / `provider`; boost may include Gemini `usageMetadata` |
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
| chunk_key | String(128) UNIQUE, indexed | stable import identity |
| lecture_number | Integer, indexed | |
| topic | String(512), indexed | e.g. `{lecture title} — {section heading}` |
| keywords | Text | JSON array of strings |
| source_excerpt | Text | Raw course material (joined bullets / slides text) |
| clean_explanation | Text | Pedagogical text |
| sample_questions | Text nullable | JSON array of strings |
| sample_answer | Text nullable | Single exemplar answer |
| chunk_type | String(32) nullable | retrieval v2 |
| concept_family | String(64) nullable | retrieval v2 |
| embedding_model | String(64) nullable | OpenAI embedding model id when present |
| embedding_dim | SmallInteger nullable | vector length |
| embedding_blob | BLOB nullable | float32 little-endian vector (`flask embed-chunks`) |

Seed JSON may use `source_text` or `content`; the loader maps both to `source_excerpt` (see [`lecture_loader.py`](../app/services/lectures/lecture_loader.py)).
