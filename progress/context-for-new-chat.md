# Context for a new Cursor / AI chat (LING 487 Tutor)

Use this file when you **start a fresh chat** and want the assistant to align quickly with this repo—stack, constraints, and where the important code lives—without re-deriving it from scratch.

---

## How to start a new chat (suggested prompt)

Copy the block below into a new conversation and **fill in the bracketed parts**. Attach or `@`-reference this file so the model can open it.

```text
I'm working on LING 487 Tutor (Neural_Tutor repo). Read progress/context-for-new-chat.md for stack, constraints, and hot paths.

Goal: [what you want done in this session]

Constraints: [e.g. smallest correct change; no full redesign; backend only; match existing patterns in backend/app/]

Optional context: [branch, error message, or feature area]
```

**Recommended attachments**

| Attachment | When |
|------------|------|
| This file (`progress/context-for-new-chat.md`) | Almost always for backend work |
| Root `README.md` | API tables, setup, current behavior |
| `backend/docs/schema.md` | Anything touching DB models or migrations |
| Specific paths (e.g. `backend/app/routes/chat.py`) | Narrow tasks |

**What you should expect from the assistant**

- **Course-grounded product:** This is a **LING 487–scoped tutor**, not a generic chat app. Retrieval and structured pipelines are central; avoid treating it like a blank CRUD app.
- **Small, focused diffs:** Prefer the smallest change that solves the problem; don’t expand scope unless you ask.
- **Follow repo conventions:** Match naming, imports, and patterns already in `backend/app/` (and `frontend/src/` for UI work).
- **Verify backend changes:** Run tests from `backend/` (`pytest`) when behavior changes.
- **Docs policy:** After user-visible or API changes, update what `progress/README.md` requires (`CHANGELOG.md`, `README.md`, `schema.md`, etc.).

---

## Stack (short)

| Layer | Notes |
|-------|--------|
| **Backend** | Flask, SQLAlchemy, SQLite default (PostgreSQL via `DATABASE_URL`). Session cookie auth, CSRF, rate limits. Entry: `backend/app/__init__.py`. |
| **Frontend** | React + Vite; proxy `/api` → Flask (`vite.config.js`). |
| **Config / env** | `backend/app/config.py`; local secrets in `backend/.env` (see `backend/.env.example`). `load_dotenv()` runs at app startup. |

---

## Hot paths (where to look first)

| Concern | Location |
|---------|----------|
| HTTP: chat, sessions, feedback | `backend/app/routes/chat.py` |
| One assistant turn (retrieval → answer → boost → DB) | `backend/app/services/chat_orchestrator.py` |
| Structured pipeline (retrieve → plan → Course Answer → validate) | `backend/app/services/reasoning_pipeline.py` |
| Retrieval + query intent + **mode routing** | `backend/app/services/retrieval_v2.py` (calls `query_mode.py` for API modes: auto / chat / quiz / compare / summary); builds `mode_routing` on `EnhancedRetrievalResult` |
| Deterministic **mode detection** (no LLM) | `backend/app/services/query_mode.py` |
| Compare answers (rule-based): scoped evidence + markdown | `backend/app/services/answers/compare_evidence.py`, `compare_render.py` (`format_two_entity_compare_markdown`, `format_multi_entity_compare_markdown`); wired from `answer_generation.py` |
| Security event rows (auth/admin) | `backend/app/models/security_log.py` (`SecurityLogEntry` → table `audit_events`); `backend/app/services/security_logging.py` (`log_security_event`) |
| Query classification / aliases | `backend/app/services/query_understanding.py` |
| Structured query + decomposition | `backend/app/services/knowledge/structured_query.py` |
| Primary Course Answer (OpenAI / rules) | `backend/app/services/generation/course_generation.py`, `generation/llm.py` |
| Boosted Explanation (Gemini / OpenAI fallback) | `generation/gemini_boost.py`, `generation/llm.py` |
| Study flows (quiz / compare / summary) | `backend/app/routes/study.py`, `backend/app/services/study.py` |
| ORM models (barrel export) | `backend/app/models/__init__.py` → `chat.py`, `analytics.py`, `user.py`, … |
| Admin analytics | `backend/app/routes/admin.py`, `backend/app/services/admin_insights.py` |

---

## Modes and chat API (quick)

- **`POST /api/chat`** requires `session_id` and `message`. **`mode`** (legacy) and **`mode_override`** are optional; default is **auto** (detect from message text). If both are sent, **`mode_override`** wins. Invalid values fall back to **auto**.
- **Response:** **`course_answer`**, **`answer`** (duplicate for clients that expect one field), **`boosted_explanation`**, **`retrieval_confidence`**, **`boost_applied`**, **`assistant_message_id`**, **`mode`** (stable shape below), **`mode_routing`** (internal field names preserved for older clients).
- **`mode` object:** `detected`, `effective`, `confidence`, `signals`, `overridden`, `ambiguous`, optional `candidate_modes`. Built by `mode_metadata_for_api()` in `chat_orchestrator.py` from `mode_routing`.
- **Frontend:** [`ChatPage.jsx`](../frontend/src/pages/ChatPage.jsx) sends **`mode_override`** when the toolbar is not `auto`; **`auto`** omits mode keys. [`ChatPanel.jsx`](../frontend/src/components/ChatPanel.jsx) reads `effective` / `ambiguous` with fallback to `effective_mode` / `mode_ambiguous`.

---

## Recent product / API notes (for handoff)

- **Phase 3 compare renderers:** Entity-scoped lines (`compare_evidence.py`), two-way and multi-entity markdown (`compare_render.py`), tests in `test_compare_render.py` and structured pipeline tests. Compare / compare_multi primary paths stay **rule-based** (`course_generation.py` short-circuit).
- **Phase 4 chat contract:** Optional mode fields; response **`mode`** + **`answer`**; see [`chat.py`](../backend/app/routes/chat.py) `_resolve_user_api_mode`.
- **Naming cleanups:** `answer_generation` uses explicit variable names; compare evidence uses `text_normalized` / `phrase` in `_term_hits`; auth security rows use `security_logging` module (not `audit.py`).

---

## Tunables (env)

See **`backend/.env.example`** and **`backend/app/config.py`**. Examples: `CHAT_RETRIEVAL_TOP_K`, `PIPELINE_RETRY_TOP_K_EXTRA`, `OPENAI_TEMPERATURE_*`, `GEMINI_TEMPERATURE_BOOST`, `CONFIDENCE_THRESHOLD`, `STRUCTURED_PIPELINE_ENABLED`.

---

## Deeper narrative handoffs

- **`progress/entries/2026-04-12-session-handoff-neural-tutor.md`** — Longer session notes (admin insights, Course Answer prompts, Markdown UI, follow-ups).
- **`progress/README.md`** — When to update `CHANGELOG.md`, `README.md`, `entries/`, and schema docs.

---

## Migrations & DB

- Apply migrations from `backend/` with Flask-Migrate / project conventions (see root `README.md`).
- Logical schema reference: **`backend/docs/schema.md`**.
