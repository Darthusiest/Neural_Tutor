# Context for a new chat (Neural Tutor)

**Use:** Paste or `@`-reference this file at the start of a Cursor chat when working on this repo.

## Repo & stack

- **Root:** `Neural_Tutor` — **Flask** API (`backend/`), **React + Vite** SPA (`frontend/`), **SQLAlchemy + Alembic**.
- **Default DB:** SQLite file `backend/ling487.db`. **Production / Postgres:** set `DATABASE_URL` (e.g. `postgresql+psycopg2://USER:PASS@HOST:5432/DB`); install deps with `pip install -r requirements.txt` (includes `psycopg2-binary`).
- **Read first:** root [`README.md`](../README.md) (setup, API, Current status), [`CHANGELOG.md`](../CHANGELOG.md), [`backend/docs/schema.md`](../backend/docs/schema.md), [`backend/docs/AUTH_LOCAL.md`](../backend/docs/AUTH_LOCAL.md), [`backend/docs/admin_insights.md`](../backend/docs/admin_insights.md).
- **Doc policy:** [`.cursor/rules/documentation.mdc`](../.cursor/rules/documentation.mdc) — user-visible/API/config/schema/deploy changes → `CHANGELOG.md`, `README.md`, `schema.md` as appropriate; narrative → `progress/entries/YYYY-MM-DD-slug.md`.

## Secrets & env (backend `.env`)

- **`FLASK_SECRET_KEY`** (required), **`DATABASE_URL`** (optional; omit → SQLite).
- **`OPENAI_API_KEY`** — chat, embeddings, `flask embed-chunks`.
- **`RESEND_API_KEY`**, **`RESEND_FROM_EMAIL`** — email; **`PASSWORD_RESET_BASE_URL`**, **`EMAIL_VERIFICATION_BASE_URL`**.
- **`FRONTEND_ORIGIN`**, **`SESSION_COOKIE_SECURE=1`** in prod.
- **Gemini (optional):** `GEMINI_API_KEY` or `GOOGLE_API_KEY` — Boosted Explanation only (see README).

## Feature flags (`backend/app/config.py`)

Examples: `STRUCTURED_PIPELINE_ENABLED`, `PRIMARY_COURSE_ANSWER_OPENAI` / `LLM_ANSWER_GENERATION`, `EMBEDDING_RETRIEVAL_ENABLED`, `RETRIEVAL_HYBRID_ENABLED`, `STRUCTURED_STUDY_PIPELINE_ENABLED`, `EMAIL_VERIFICATION_REQUIRED`, `OPENAI_BOOST_FALLBACK`.

## Database & migrations

- After pulling **new code:** `cd backend && flask --app wsgi db upgrade`.
- **`flask init-db`** only creates tables from models; it does **not** migrate an **existing** `ling487.db` to new columns. If models are ahead of the file, you get errors like `no such column: users.email_verified_at` → run **`db upgrade`**, or fix a half-applied migration (see [`CHANGELOG.md`](../CHANGELOG.md) / migration `005`).
- **Render:** use a **persistent** Postgres `DATABASE_URL`, not ephemeral SQLite.

## Hot paths (code)

| Area | Location |
|------|----------|
| HTTP routes | `backend/app/routes/` (`auth`, `chat`, `lectures`, `study`, `admin`) |
| Chat turn | `backend/app/services/chat_orchestrator.py` |
| Pipeline | `backend/app/services/reasoning_pipeline.py` → `generate_course_answer` → `answers/answer_generation.py` (rule) or `generation/llm.py` (OpenAI) |
| Retrieval | `backend/app/services/retrieval.py`, `retrieval_v2.py`, `lecture_data.py` |
| Study compare/summary | `backend/app/services/study.py` (`format_compare_answer`, `_format_summary_recap`); optional overlay via `STRUCTURED_STUDY_PIPELINE_ENABLED` |
| UI | `frontend/src/components/ChatPanel.jsx` — **chat** uses `/api/chat`; **compare/summary** modes call **`/api/study/...`** |
| Admin | `backend/app/services/admin_insights.py`, `frontend/src/pages/AdminPage.jsx` |

## Answer quality (recent)

- **Compare (rule-based chat):** `compare_render.py` builds two-way and multi-entity answers from **per-side evidence bundles** (`entity_retrieval.py`); OpenAI is **not** used for compare / compare_multi course answers.
- **Compare (rule-based chat, legacy):** `answer_generation.py` groups explanation lines under **one heading per plan section** (no repeated “First idea” / “In one line” on every line).
- **Study compare:** keyword overlap uses **stopwords**; summary dedupes **Topics to cross-link**.
- **`output_cleanup.py`:** strips leaked outline-style lines from LLM output.
- **Validation:** `answer_validation.py` enforces no-examples / intuition-only / boilerplate / single-concept forbidden-topic checks; optional retrieval retry in `reasoning_pipeline.py`.
- **Admin insights on Postgres:** dialect-specific JSON + boolean filters in `admin_insights.py`.

## Tests

```bash
cd backend && source .venv/bin/activate && python -m pytest tests/ -q
```

## Open follow-ups

- Multi-entity compare (e.g. four architectures) needs a dedicated plan/renderer.
- Strong “negative constraints” (e.g. “don’t mention transformers”) need retrieval/validation, not only prompts.
