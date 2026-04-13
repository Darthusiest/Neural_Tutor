# Admin insights API

Admin-only analytics over chat **`retrieval_logs`**, **`response_variants`**, **`feedback`**, **`message_outcomes`**, and **`retrieval_chunk_hits`** / **`lecture_chunks`**. Implementation: [`app/services/admin_insights.py`](../app/services/admin_insights.py); routes: [`app/routes/admin.py`](../app/routes/admin.py). Blueprint prefix: **`/api/admin`**.

**Auth:** session cookie + **`users.is_admin`** (`GET /api/auth/me` exposes `is_admin`). Non-admins receive **403**. Promote a user locally:

```bash
sqlite3 backend/ling487.db "UPDATE users SET is_admin = 1 WHERE email = 'you@example.com';"
```

**SPA:** [`/admin`](../../frontend/src/App.jsx) uses [`AdminRoute`](../../frontend/src/components/AdminRoute.jsx) (non-admins redirect to `/chat`).

## Endpoints

| Method | Path | Query params | Notes |
|--------|------|----------------|------|
| GET | `/api/admin/insights` | `days` (1–365, default 7) | Dashboard JSON: volume, retrieval KPIs, pipeline, boost, feedback, outcomes, **`models_and_tokens`**, **`insufficient_data`**. Rate: 120/min. |
| GET | `/api/admin/insights/low-confidence` | `days`, `limit` (≤200), `offset` | Paged **`is_low_confidence`** retrieval logs: IDs, truncated question, confidence, pipeline fields. **No user emails.** 60/min. |
| GET | `/api/admin/insights/low-confidence.csv` | `days` | Same window as JSON drill-down; CSV attachment; capped row count. 30/min. |
| GET | `/api/admin/insights/chunks` | `days`, `limit` (≤100) | Top **`lecture_chunk_id`** by hit count: in low-confidence retrievals vs overall; joined to **`lecture_chunks`** for topic / lecture number. 60/min. |

All filters use **`created_at`** in **UTC** (naive timestamps stored as UTC in typical setups).

## Token usage persistence

When the structured pipeline uses **OpenAI** for the primary Course Answer, [`chat_orchestrator`](../app/services/chat_orchestrator.py) stores primary metadata on **`retrieval_logs.token_usage_json`** and combined **`primary` / `boost`** on **`response_variants.token_usage_json`**, plus **`model_name`** / **`provider_name`** on **`response_variants`** when applicable. See [`schema.md`](schema.md) (`retrieval_logs`, `response_variants`).

**Gemini boost** may include **`usageMetadata`** in the JSON blob when the API returns it.

## Follow-ups

- Paging controls in the SPA for low-confidence (beyond first page).
- Per-day token time series (would require date-bucketed queries or extra columns).
