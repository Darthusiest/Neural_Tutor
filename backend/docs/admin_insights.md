# Admin insights API

Admin-only analytics over chat **`retrieval_logs`**, **`response_variants`**, **`feedback`**, **`message_outcomes`**, and **`retrieval_chunk_hits`** / **`lecture_chunks`**. Implementation: [`app/services/admin_insights.py`](../app/services/admin_insights.py); routes: [`app/routes/admin.py`](../app/routes/admin.py). Blueprint prefix: **`/api/admin`**.

**Auth:** session cookie + **`users.is_admin`** (`GET /api/auth/me` exposes `is_admin`). When **`ADMIN_EMAILS`** (comma-separated) is set in **`backend/.env`**, matching accounts receive **`is_admin=True`** on register and on each successful login. **`.env.example`** includes `ADMIN_EMAILS=pajouhes@usc.edu` as a placeholder. To promote manually locally (SQLite):

```bash
sqlite3 backend/ling487.db "UPDATE users SET is_admin = 1 WHERE email = 'you@example.com';"
```

On **PostgreSQL**, use your SQL client, e.g. `psql` with the same `UPDATE` against the `users` table.

**Implementation note:** Dashboard queries use dialect-aware SQL where needed (e.g. JSON extraction for validation **severity** on SQLite vs PostgreSQL) so analytics work with either database.

**SPA:** [`/admin`](../../frontend/src/App.jsx) uses [`AdminRoute`](../../frontend/src/components/AdminRoute.jsx) (non-admins redirect to `/chat`).

**Eval runs + Gemini critic:** `GET /api/admin/eval/runs` lists **`evaluation_runs`** only. **Primary path for the critic:** **`PYTHONPATH=. python -m app.eval.run_eval --dataset data/eval/l487_eval_suite.json …`** (chat-turn runner; sets **`assistant_message_id`**). See **`progress/entries/2026-05-10-gemini-critic.md`**. Structured shortcuts: **`flask run-eval`**, **`seed-demo-eval`** (**default `--suite mini`**) — omit **`handle_chat_turn`** payloads. Judge rubric defaults to **`CRITIC_PROMPT_VERSION=v2`** (generous calibration vs chunks + **`EXPECTED_BEHAVIOR_JSON.error_tags`** carve-out for nonsense/off-topic suite rows); version string is stored per critic row. Large batches: Admin **`POST …/critic`** uses a **daemon thread** — **`flask --debug`** reloads kill mid-batch jobs; prefer **`flask run --no-reload`** or **`PYTHONPATH=. python scripts/run_gemini_critic_batch.py --run-id … --force`** from **`backend/`**.

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
| GET | `/api/admin/eval/runs/<id>/critic` | — | Latest critic batch summary (`critic_pass_rate`, `critic_mean_score`, **`critic_batch_complete`**, **`critic_job_in_progress`**, `artifact_urls`, …). On disk, each batch folder may include **`critic_metrics.json`**. **60/min**. |
| GET | `/api/admin/eval/runs/<id>/critic/cases` | `group_by` (`query_type_v2` \| `category` \| `answer_mode`), optional `category` | Grouped + flat case list (chatbot vs critic, disagreement flag). **60/min**. |
| GET | `/api/admin/eval/critic-image/<run_id>/<filename>` | `batch` (optional; defaults to latest critic batch) | Whitelisted PNG/CSV artifacts only. **120/min**. |

All filters use **`created_at`** in **UTC** (naive timestamps stored as UTC in typical setups).

## Token usage persistence

When the structured pipeline uses **OpenAI** for the primary Course Answer, [`chat_orchestrator`](../app/services/chat_orchestrator.py) stores primary metadata on **`retrieval_logs.token_usage_json`** and combined **`primary` / `boost`** on **`response_variants.token_usage_json`**, plus **`model_name`** / **`provider_name`** on **`response_variants`** when applicable. See [`schema.md`](schema.md) (`retrieval_logs`, `response_variants`).

**Gemini boost** may include **`usageMetadata`** in the JSON blob when the API returns it.

## Follow-ups

- Optional chart visualization for **`tokens-by-day`**; budget / cap alerts (not implemented).
