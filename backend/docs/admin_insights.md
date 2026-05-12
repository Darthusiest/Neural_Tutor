# Admin insights API

Admin-only analytics over chat **`retrieval_logs`**, **`response_variants`**, **`feedback`**, **`message_outcomes`**, and **`retrieval_chunk_hits`** / **`lecture_chunks`**. Implementation: [`app/services/admin_insights.py`](../app/services/admin_insights.py); routes: [`app/routes/admin.py`](../app/routes/admin.py). Blueprint prefix: **`/api/admin`**.

**Auth:** session cookie + **`users.is_admin`** (`GET /api/auth/me` exposes `is_admin`). Non-admins receive **403**. Promote a user locally (SQLite):

```bash
sqlite3 backend/ling487.db "UPDATE users SET is_admin = 1 WHERE email = 'you@example.com';"
```

On **PostgreSQL**, use your SQL client, e.g. `psql` with the same `UPDATE` against the `users` table.

**Implementation note:** Dashboard queries use dialect-aware SQL where needed (e.g. JSON extraction for validation **severity** on SQLite vs PostgreSQL) so analytics work with either database.

**SPA:** [`/admin`](../../frontend/src/App.jsx) uses [`AdminRoute`](../../frontend/src/components/AdminRoute.jsx) (non-admins redirect to `/chat`).

## Endpoints

| Method | Path | Query params | Notes |
|--------|------|----------------|------|
| GET | `/api/admin/insights` | `days` (1–365, default 7) | Dashboard JSON: volume, retrieval KPIs, pipeline, boost, feedback, outcomes, **`models_and_tokens`**, **`insufficient_data`**. Rate: 120/min. |
| GET | `/api/admin/insights/low-confidence` | `days`, `limit` (≤200), `offset` | Paged **`is_low_confidence`** retrieval logs: IDs, truncated question, confidence, pipeline fields. **No user emails.** 60/min. |
| GET | `/api/admin/insights/low-confidence.csv` | `days` | Same window as JSON drill-down; CSV attachment; capped row count. 30/min. |
| GET | `/api/admin/insights/chunks` | `days`, `limit` (≤100) | Top **`lecture_chunk_id`** by hit count: in low-confidence retrievals vs overall; joined to **`lecture_chunks`** for topic / lecture number. 60/min. |
| GET | `/api/admin/insights/tokens-by-day` | `days` (1–365, default 7) | Per **UTC calendar day**: **`response_variants`** count, estimated sum of primary+boost tokens (same rules as dashboard rollups), count of variants with nonzero usage. Oldest → newest. 60/min. |
| GET | `/api/admin/insights/cost-summary` | `days` | Token totals vs optional **`LLM_MONTHLY_TOKEN_CAP`** / warn threshold; optional USD via **`LLM_COST_USD_PER_MTOKENS`**; spike note. 60/min. |
| GET | `/api/admin/insights/content-quality` | `days` | Heuristic **weak chunks** (low-confidence hit counts) + thumbs-down count. 60/min. |
| POST | `/api/admin/eval/runs/<id>/critic` | JSON `{ "force"?: bool, "modes"?: ["chat","compare","summary"] }` — **`modes`** optional (default from **`CRITIC_CASE_MODES`**). Subset batches write **`manifest.json`** next to charts; **`422`** if no cases match. | Run **Gemini critic** on in-scope cases; writes **`evaluation_critic_results`** and **`evaluation_outputs/critic/<batch>/`**. **6/min** (slow). |
| GET | `/api/admin/eval/runs/<id>/critic` | — | Latest critic batch summary (`critic_pass_rate`, `critic_mean_score`, `artifact_urls`, …). **60/min**. |
| GET | `/api/admin/eval/runs/<id>/critic/cases` | `group_by` (`query_type_v2` \| `category` \| `answer_mode`), optional `category` | Grouped + flat case list (chatbot vs critic, disagreement flag). **60/min**. |
| GET | `/api/admin/eval/critic-image/<run_id>/<filename>` | `batch` (optional; defaults to latest critic batch) | Whitelisted PNG/CSV artifacts only. **120/min**. |

All filters use **`created_at`** in **UTC** (naive timestamps stored as UTC in typical setups).

## Token usage persistence

When the structured pipeline uses **OpenAI** for the primary Course Answer, [`chat_orchestrator`](../app/services/chat_orchestrator.py) stores primary metadata on **`retrieval_logs.token_usage_json`** and combined **`primary` / `boost`** on **`response_variants.token_usage_json`**, plus **`model_name`** / **`provider_name`** on **`response_variants`** when applicable. See [`schema.md`](schema.md) (`retrieval_logs`, `response_variants`).

**Gemini boost** may include **`usageMetadata`** in the JSON blob when the API returns it.

## Follow-ups

- Optional chart visualization for **`tokens-by-day`**; budget / cap alerts (not implemented).
