# SQLite schema (SQLAlchemy models)

Tables are created with:

```bash
cd backend && flask --app wsgi init-db
```

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

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| session_id | FK nullable, indexed | |
| message_id | FK nullable, indexed | |
| user_question | Text | |
| detected_topic | String(512) nullable | |
| retrieved_chunk_ids | Text nullable | JSON |
| confidence | Float nullable | |
| latency_ms | Integer nullable | |
| token_usage_json | Text nullable | |
| created_at | DateTime | |

## response_variants

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| message_id | FK → messages.id UNIQUE | one row per assistant message |
| course_answer | Text | |
| boosted_explanation | Text nullable | |
| boost_reason | String(64) nullable | |
| model_name | String(128) nullable | |
| token_usage_json | Text nullable | |
| created_at | DateTime | |

## feedback

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| message_id | FK → messages.id UNIQUE | |
| course_thumb | String(8) nullable | |
| boost_thumb | String(8) nullable | |
| preferred | String(16) nullable | |
| created_at | DateTime | |

## lecture_chunks (course corpus)

| Column | Type | Notes |
|--------|------|--------|
| id | Integer PK | |
| lecture_number | Integer, indexed | |
| topic | String(512), indexed | e.g. `{lecture title} — {section heading}` |
| keywords | Text | JSON array of strings |
| source_excerpt | Text | Raw course material (joined bullets / slides text) |
| clean_explanation | Text | Pedagogical text (may match source until seed adds a separate field) |
| sample_questions | Text nullable | JSON array of strings |
| sample_answer | Text nullable | Single exemplar answer |
