# Neural Tutor — session handoff context (for new chats)

Paste this (or point the assistant at this file) to restore full project context from work on admin insights, Course Answer prompts, and chat Markdown rendering.

---

## Stack & repo

- **Backend:** Flask + SQLite, session cookie auth, CSRF, rate limits. Main app under `backend/app/`.
- **Frontend:** React 19 + Vite; `vite.config.js` proxies `/api` to Flask. Entry: `frontend/src/App.jsx`.
- **Chat orchestration:** `backend/app/services/chat_orchestrator.py` — `handle_chat_turn` composes retrieval, course answer, optional boost, persists `messages`, `retrieval_logs`, `retrieval_chunk_hits`, `response_variants`.

---

## Structured reasoning pipeline (when `STRUCTURED_PIPELINE_ENABLED` is on)

Flow (high level): **`retrieve_enhanced`** → build **structured query** (concept KB) → **answer plan** → **Course Answer** (OpenAI primary when configured, else rule-based) → **validate_answer** → optional **Boosted Explanation** (Gemini primary, OpenAI fallback if `OPENAI_BOOST_FALLBACK`).

- **Config:** `backend/app/config.py` — e.g. `STRUCTURED_PIPELINE_ENABLED`, `PRIMARY_COURSE_ANSWER_OPENAI` / legacy `LLM_ANSWER_GENERATION`, `OPENAI_API_KEY`, `CONFIDENCE_THRESHOLD`, Gemini keys for boost.
- **Pipeline code:** `reasoning_pipeline.py` (`run_reasoning_pipeline`, `PipelineResult` includes `primary_model`, `primary_llm_usage` when OpenAI path runs).
- **Course Answer generation:** `generation/course_generation.py` → `llm.generate_plan_constrained_answer` or `answers/answer_generation.generate_structured_answer`.
- **Empty retrieval:** no chunks → `conversational_responses` (`classify_no_match_query`, `varied_no_chunk_course_answer`), sets `no_match_kind` in payload, skips boost.

---

## Course Answer — product rules (student-facing)

- Final text should read like a **tutor**, not a retrieval dump.
- **Do not surface:** raw keyword lists, lecture IDs as “scope,” concept-graph jargon, “chunk/retrieved/indexed,” debug strings.
- **Target structure** (OpenAI + rule-based): **Course Answer** with sections: **### Direct Answer**, **### Explanation**, **### Example / Intuition**, **### Why it matters** (prefix `Course Answer:` then body).
- **OpenAI path:** `generation/generation_input.py` — `build_generation_input` (question, concepts, `answer_mode`, primary/supporting teaching text from chunk `clean_explanation` / `source_excerpt`, deduped, no chunk metadata). `format_generation_prompt_user_message` — plain Question / Concepts / Teaching style / Primary & Supporting content. Tutor rules live in **`_COURSE_ANSWER_SYSTEM_PROMPT`** in `generation/llm.py` (not duplicated in a long “Remember” user block). `generate_plan_constrained_answer` → `clean_output` + `enforce_structure` in `generation/output_cleanup.py`.
- **Rule-based path:** `answers/answer_generation.py` — same four `###` sections; planning in `answer_planning.py` (distinct chunks per section where possible).

---

## Analytics & persistence

- **Models:** `backend/app/models/analytics.py` — `RetrievalLog`, `RetrievalChunkHit`, `ResponseVariant`, `Feedback`, `MessageOutcome`. Schema: `backend/docs/schema.md`.
- **Token / model fields:** On structured-pipeline turns with OpenAI primary, `chat_orchestrator` fills `retrieval_logs.token_usage_json` (primary usage JSON), `response_variants.token_usage_json` (`primary` + optional `boost`), `model_name` / `provider_name` when applicable. Gemini boost may store `usageMetadata` in the boost blob. `_openai_chat` in `llm.py` returns `{ usage, model, provider }`.

---

## Admin insights (implemented)

- **Service:** `backend/app/services/admin_insights.py` — `compute_insights_summary(days)`, drill-down, CSV, chunk analytics, `models_and_tokens` rollups.
- **Routes:** `backend/app/routes/admin.py` — blueprint `/api/admin`:
  - `GET /insights?days=`
  - `GET /insights/low-confidence?days=&limit=&offset=`
  - `GET /insights/low-confidence.csv?days=`
  - `GET /insights/chunks?days=&limit=`
- **Auth:** `users.is_admin` only; **403** otherwise. SPA: `/admin` with `frontend/src/components/AdminRoute.jsx` (non-admins → `/chat`).
- **Docs:** `backend/docs/admin_insights.md`, README API table + rate limits.

---

## Frontend chat UI (recent)

- **Assistant bubbles:** `frontend/src/components/MarkdownContent.jsx` — `react-markdown` + **`rehype-sanitize`** (XSS-safe) + **`remark-breaks`**. Used in `ChatPanel.jsx` for **Course Answer** and **Boosted Explanation**.
- **User bubbles:** plain text, `pre-wrap` (unchanged).
- **Styles:** `frontend/src/index.css` — `.msg.assistant .markdown-body` typography; theme variables for light/dark.
- Other UX mentioned in-session: `messagesEndRef` scroll to bottom; theme toggle + `localStorage` (`neural-tutor-theme`).

---

## Documentation policy (repo)

- User-visible behavior / API: **`CHANGELOG.md`** `[Unreleased]`, root **`README.md`** (Current status, API).
- **Schema / payload:** `backend/docs/schema.md` when DB or assistant payload columns matter.
- **Narrative / decisions:** `progress/entries/YYYY-MM-DD-slug.md` (this file is an example).

---

## Not implemented / follow-ups (from README & discussion)

- Rich **study-mode**-specific answer copy (compare/summary/quiz) beyond current retrieval + boost behavior.
- **Embedding retrieval** (`backend=embedding`) — still **501** until implemented.
- **Admin:** offset paging UI for low-confidence; per-day **token time series**; deeper “insights product” (alerts, saved reports).
- **Auth hardening:** email verification, account lockout, formal audit pipeline (called out in README Security / Not done).
- **Optional product follow-ups:** stricter LLM retry when sections leak; further alignment of rule-based tone with OpenAI tutor voice.

---

## Key file index

| Area | Path |
|------|------|
| Chat turn | `backend/app/services/chat_orchestrator.py` |
| Structured pipeline | `backend/app/services/reasoning_pipeline.py` |
| OpenAI Course Answer | `backend/app/services/generation/llm.py`, `course_generation.py`, `generation_input.py`, `output_cleanup.py` |
| Rule-based answer | `backend/app/services/answers/answer_generation.py`, `answer_planning.py`, `answer_validation.py` |
| Admin insights | `backend/app/services/admin_insights.py`, `backend/app/routes/admin.py` |
| Chat UI | `frontend/src/components/ChatPanel.jsx`, `MarkdownContent.jsx` |
| Analytics models | `backend/app/models/analytics.py` |

---

*Last updated to match repo state as of session discussing admin insights phases 1–4, token persistence, docs, and chat Markdown rendering.*
