# LING 487 Tutor

Full-stack app for a course-specific tutor: **React + Vite** frontend, **Flask + SQLite** backend, **session cookie** auth (with **CSRF**, **rate limits**, and strict **password policy**), and **retrieval-first** answers from lecture JSON in **`lecture_chunks`**. Retrieval uses **lexical v1** plus **retrieval v2** (query understanding, aliases, strategies) via [`lecture_data`](backend/app/services/lecture_data.py). The structured pipeline uses **OpenAI** for the primary **Course Answer** when **`PRIMARY_COURSE_ANSWER_OPENAI`** and **`OPENAI_API_KEY`** are set ([`course_generation.py`](backend/app/services/generation/course_generation.py), [`llm.py`](backend/app/services/generation/llm.py)). **Boosted Explanation** (secondary only, never the main answer) prefers **Gemini** when **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`** is set ([`gemini_boost.py`](backend/app/services/generation/gemini_boost.py)), with **OpenAI** as fallback ([`llm.py`](backend/app/services/generation/llm.py)); compare **expansion** in study flows may still use OpenAI where configured.

## Documentation & change log

| Doc | Role |
|-----|------|
| This **`README.md`** | Setup, API overview, current behavior — **update when behavior changes** |
| [`CHANGELOG.md`](CHANGELOG.md) | Shippable deltas (Keep a Changelog style) |
| [`progress/README.md`](progress/README.md) | **Policy:** what to update for each kind of change |
| [`progress/entries/`](progress/entries/) | Dated narrative notes (decisions, tuning, follow-ups) |
| [`backend/docs/schema.md`](backend/docs/schema.md) | SQLite tables / columns |
| [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md) | Auth, CSRF, local testing |

## Repository layout

| Path | Purpose |
|------|--------|
| [`backend/`](backend/) | Flask app: [`app/`](backend/app/), [`wsgi.py`](backend/wsgi.py), [`requirements.txt`](backend/requirements.txt) |
| [`backend/data/LING487_SUPER_TUTOR.json`](backend/data/LING487_SUPER_TUTOR.json) | Lecture corpus (`lectures[]` → `sections` with `heading` + `content`); load into SQLite via **`flask import-lectures`** |
| [`frontend/`](frontend/) | Vite + React: sidebar (sessions), chat panel, header (auth / admin) |
| [`progress/`](progress/) | Track record: policy in [`progress/README.md`](progress/README.md), narratives in [`progress/entries/`](progress/entries/) |
| [`CHANGELOG.md`](CHANGELOG.md) | Version-style history of notable changes |

### Backend code map (quick)

| Area | Location |
|------|----------|
| App factory, CORS, CLI | [`backend/app/__init__.py`](backend/app/__init__.py) |
| Config | [`backend/app/config.py`](backend/app/config.py) |
| `db`, `login_manager`, `csrf`, `limiter` | [`backend/app/extensions.py`](backend/app/extensions.py) |
| Auth + CSRF token route | [`backend/app/routes/auth.py`](backend/app/routes/auth.py) |
| Chat, sessions, feedback | [`backend/app/routes/chat.py`](backend/app/routes/chat.py) |
| JSON input + password/email checks + timing helpers | [`backend/app/utils/security.py`](backend/app/utils/security.py) |
| Lecture import + lexical retrieval + v2 orchestration + cache | [`lectures/lecture_loader.py`](backend/app/services/lectures/lecture_loader.py), [`retrieval.py`](backend/app/services/retrieval.py), [`retrieval_v2.py`](backend/app/services/retrieval_v2.py), [`lecture_data.py`](backend/app/services/lecture_data.py) |
| Structured reasoning (concept KB, plan, validation) | [`knowledge/concept_kb.py`](backend/app/services/knowledge/concept_kb.py), [`knowledge/domain_knowledge.py`](backend/app/services/knowledge/domain_knowledge.py) (lexical aliases / typo helpers), [`knowledge/structured_query.py`](backend/app/services/knowledge/structured_query.py), [`answers/answer_planning.py`](backend/app/services/answers/answer_planning.py), [`answers/answer_generation.py`](backend/app/services/answers/answer_generation.py), [`generation/course_generation.py`](backend/app/services/generation/course_generation.py), [`answers/answer_validation.py`](backend/app/services/answers/answer_validation.py), [`reasoning_pipeline.py`](backend/app/services/reasoning_pipeline.py), [`generation/boost_triggers.py`](backend/app/services/generation/boost_triggers.py), [`generation/gemini_boost.py`](backend/app/services/generation/gemini_boost.py) |
| Study modes (quiz / compare / summary) | [`routes/study.py`](backend/app/routes/study.py), [`study.py`](backend/app/services/study.py) |
| Chat turn orchestration + analytics persistence | [`chat_orchestrator.py`](backend/app/services/chat_orchestrator.py), [`conversational_responses.py`](backend/app/services/conversational_responses.py) (varied no-chunk replies) |
| Lectures API (topics, summary, retrieve) | [`routes/lectures.py`](backend/app/routes/lectures.py) |
| Analytics / feedback / outcomes models | [`models/analytics.py`](backend/app/models/analytics.py) |
| Password reset email (Resend) | [`backend/app/services/reset_email.py`](backend/app/services/reset_email.py) |
| SPA `fetch` + CSRF | [`frontend/src/api/client.js`](frontend/src/api/client.js) |
| Auth + DB testing / schema notes | [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md), [`backend/docs/schema.md`](backend/docs/schema.md) |

**Backend services layout** ([`backend/app/services/`](backend/app/services/)): subpackages group related modules — **`answers/`** (planning, rule-based generation, validation), **`knowledge/`** (structured concept KB JSON, `domain_knowledge` lexical helpers, structured query), **`generation/`** (OpenAI client, Gemini boost, `course_generation`, boost triggers), **`lectures/`** (JSON import, chunk keys). Lexical **retrieval** stays at the package root (`retrieval.py`, `retrieval_v2.py`, `lecture_data.py`) so imports stay `from app.services.retrieval import …` without clashing with a `retrieval/` package. **Example:** `from app.services.answers.answer_planning import build_answer_plan`, `from app.services.generation.llm import generate_plan_constrained_answer`.

## Current status

- **Auth:** Register / login / logout / `GET /api/auth/me`. **CSRF:** mutating requests need **`Content-Type: application/json`** and **`X-CSRFToken`** (see [`client.js`](frontend/src/api/client.js)). **Passwords:** 8+ chars with upper, lower, digit, and special (register + reset). **Password reset:** **`POST /api/auth/forgot-password`** persists a time-limited token and sends mail via **Resend** when **`RESEND_API_KEY`** and **`RESEND_FROM_EMAIL`** are set; link base is **`PASSWORD_RESET_BASE_URL`**. Without Resend, **`FLASK_DEBUG=1`** may include **`dev_reset_token`** in JSON (or set **`DEV_RETURN_RESET_TOKEN=1`** with Resend for manual QA). Details: [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md).
- **Course data:** `lecture_chunks` stores `chunk_key`, `source_excerpt`, `clean_explanation`, optional `chunk_type` / `concept_family`, keywords, and optional sample Q/A ([`content.py`](backend/app/models/content.py)); **`import-lectures`** fills them from [`data/LING487_SUPER_TUTOR.json`](backend/data/LING487_SUPER_TUTOR.json). Use **`--upsert`** to merge by **`chunk_key`**. **`GET /api/lectures/topics`**, **`GET /api/lectures/<n>/summary`**, and **`POST /api/lectures/retrieve`** use [`lecture_data.py`](backend/app/services/lecture_data.py) ([`lectures.py`](backend/app/routes/lectures.py)); retrieval v2 may add fields like `query_type` / `supporting_chunks`; `backend=embedding` returns **501** until implemented.
- **Retrieval:** Lexical **token-aligned** scoring in [`retrieval.py`](backend/app/services/retrieval.py) (field weights, phrases, confidence, diagnostics; optional **`lecture_filter`** + **`summary_rank`** for single-lecture ranked lists). **Retrieval v2** ([`retrieval_v2.py`](backend/app/services/retrieval_v2.py)) adds query classification, alias expansion, typo hints, and strategy-specific chunk lists (compare uses side-only subqueries and merged diagnostics; synthesis uses a primary pass plus light augmentation; single-lecture summary is ranked and capped by **`SUMMARY_MAX_CHUNKS`** in [`config.py`](backend/app/config.py)). **`load_lecture_cache()`** / **`invalidate_lecture_cache()`** on import.
- **Study:** **`POST /api/study/quiz/next`**, **`/quiz/answer`**, **`/compare`**, **`/summary`** — grounded in lecture chunks; optional GPT comparison when OpenAI is configured ([`routes/study.py`](backend/app/routes/study.py)).
- **Chat:** Sessions and messages persist. When **`STRUCTURED_PIPELINE_ENABLED`** (default on in [`config.py`](backend/app/config.py)), [`handle_chat_turn`](backend/app/services/chat_orchestrator.py) runs [`run_reasoning_pipeline`](backend/app/services/reasoning_pipeline.py): **`retrieve_enhanced`** → structured query + decomposition → **answer plan** → primary **Course Answer** via OpenAI when **`PRIMARY_COURSE_ANSWER_OPENAI`** and **`OPENAI_API_KEY`** are set ([`generation/course_generation.py`](backend/app/services/generation/course_generation.py)), else rule-based templates in [`answers/answer_generation.py`](backend/app/services/answers/answer_generation.py) (definition-style plans use **one chunk per section** to avoid repeating the same excerpt under every `###` heading) → **validation** (pass / weak / fail). Otherwise the legacy path uses **`retrieve_enhanced`** + [`format_course_answer`](backend/app/services/retrieval.py). Concept metadata lives in [`data/LING487_STRUCTURED_PIPELINE_KB.json`](backend/data/LING487_STRUCTURED_PIPELINE_KB.json) (config **`KB_JSON_PATH`**). Persists **`retrieval_logs`** (including pipeline diagnostics), **`retrieval_chunk_hits`**, **`response_variants`**, **`message_outcomes`**. Assistant **`payload_json`** includes **`course_answer`**, **`confidence`**, optional **`query_type`**, optional **`pipeline_diagnostics`**, **`primary_model`**, **`validation_severity`**, **`boost_provider`**, **`boost_reason`**, **`query_complexity`**, optional **`no_match_kind`** when retrieval found no chunks (`greeting` / `short_ack` / `off_topic`; [`conversational_responses.py`](backend/app/services/conversational_responses.py) picks multi-paragraph **Course Answer** copy from rotating templates; **boost** does not run without chunks), and optional **`boosted_explanation`**. **Boosted Explanation** is secondary only: gated by [`should_use_boost`](backend/app/services/generation/boost_triggers.py) (alias of `should_use_gemini_boost`); produced by **Gemini** ([`gemini_boost.generate_boosted_explanation`](backend/app/services/generation/gemini_boost.py)) when a Google AI key is set. Optional **`OPENAI_BOOST_FALLBACK=1`** uses OpenAI for boost only if Gemini is unavailable or fails.
- **UI:** ChatGPT-style layout; modes `chat` / `quiz` / `compare` / `summary` with study controls in [`ChatPanel.jsx`](frontend/src/components/ChatPanel.jsx); boost checkbox; the message column **scrolls to the latest message** when new content arrives; **light/dark theme** toggle in the header (auth pages: top-right fixed control); preference persisted in **`localStorage`**; **`ProtectedRoute`** + **`ErrorBoundary`**; auth and reset flows; admin insights stub.
- **Feedback:** `POST /api/feedback` accepts thumbs / preference plus optional enriched fields: `helpfulness_rating` (1-5), `resolved`, `follow_up_required`, `follow_up_type`, `explicit_confusion_flag`, `feedback_note`, `preference_strength`. All enriched fields are nullable; the endpoint is backward-compatible with the original thumb-only payload.

**Not done yet:** rich compare/summary/quiz-specific answer copy (beyond retrieval + boost triggers), admin **insights** aggregates over stored analytics, optional Render manifests, email verification / account lockout / formal audit pipeline.

## Local setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Create backend/.env (see environment variables below). FLASK_SECRET_KEY is required for sessions and CSRF.
```

If `DATABASE_URL` is unset or blank, the default is **`backend/ling487.db`** (absolute path; see [`config.py`](backend/app/config.py)).

#### Environment variables (backend, LLM & structured pipeline)

Set these in **`backend/.env`** (loaded by Flask via `os.getenv` in [`config.py`](backend/app/config.py)). Omit API keys to disable that path (retrieval-only Course Answer and no boost).

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Primary **Course Answer** (chat) when structured pipeline is on and **`PRIMARY_COURSE_ANSWER_OPENAI`** is enabled; also **Boosted Explanation** fallback when Gemini is unset or fails. |
| `PRIMARY_COURSE_ANSWER_OPENAI` | `1` / `0` — use OpenAI for plan-constrained Course Answer. If unset, falls back to legacy **`LLM_ANSWER_GENERATION`** (default `1`). |
| `LLM_ANSWER_GENERATION` | Legacy alias for the primary flag when **`PRIMARY_COURSE_ANSWER_OPENAI`** is not set. |
| `OPENAI_CHAT_MODEL` | Chat model id (default `gpt-4o-mini`). |
| `OPENAI_TIMEOUT_SEC` | OpenAI HTTP timeout seconds. |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | **Boosted Explanation** only (never replaces Course Answer). Either key is accepted. |
| `GEMINI_MODEL` | Gemini model id for boost (default `gemini-1.5-flash`). |
| `GEMINI_TIMEOUT_SEC` | Gemini HTTP timeout seconds. |
| `OPENAI_BOOST_FALLBACK` | `1` / `0` — if boost is triggered but Gemini fails or is unset, optionally generate **Boosted Explanation** with OpenAI (default **off**; product rule is Gemini-only secondary). |
| `STRUCTURED_PIPELINE_ENABLED` | `1` / `0` — structured query → plan → validation path (default on). |
| `KB_JSON_PATH` | Concept KB JSON for the pipeline (default [`LING487_STRUCTURED_PIPELINE_KB.json`](backend/data/LING487_STRUCTURED_PIPELINE_KB.json)). |
| `CONFIDENCE_THRESHOLD` | Used for low-confidence flags and boost gating (default `0.35`). |

```bash
flask --app wsgi init-db
# If upgrading an existing DB after pulling new migrations:
# flask --app wsgi db upgrade
flask --app wsgi import-lectures   # data/LING487_SUPER_TUTOR.json → lecture_chunks
flask --app wsgi run --debug
```

Override import path: `flask --app wsgi import-lectures /path/to.json` or set **`LECTURE_JSON_PATH`** in `.env`.

API: `http://127.0.0.1:5000` by default. **`GET /api/health`**.

If **`lecture_chunks`** is empty, answers use the off-topic / no-match Course Answer until you run **`import-lectures`**.

**Database changes:** Flask-Migrate is enabled. For a **new** database, **`flask --app wsgi init-db`** creates tables from the current SQLAlchemy models, then **`flask --app wsgi db stamp head`** records that the DB matches the latest migration (`004`) without replaying migrations (migrations expect base tables to already exist). Then **`flask --app wsgi import-lectures`**. For an **existing** `ling487.db` from an older revision, run **`flask --app wsgi db upgrade`**. If you see errors like **“table … has no column …”** or **“lecture_chunks schema mismatch”**, back up the file, remove **`ling487.db`**, then **`init-db`**, **`db stamp head`**, **`import-lectures`**, and register again. See [`backend/docs/schema.md`](backend/docs/schema.md).

**Auth, CSRF, curl, Resend:** see [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md) (minimum `.env` vars). **LLM / pipeline vars:** table above. **Table reference:** [`backend/docs/schema.md`](backend/docs/schema.md).

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# Production: set VITE_API_BASE_URL to the public API origin (no trailing slash).

npm run dev
```

Dev URL: `http://127.0.0.1:5173`. [`vite.config.js`](frontend/vite.config.js) proxies **`/api`** to Flask (cookies + CSRF-friendly same-origin requests). The API client attaches **CSRF** automatically for POST/PUT/PATCH/DELETE.

### Admin insights

`GET /api/admin/insights` returns **403** unless `users.is_admin` is true. Example promotion:

```bash
sqlite3 backend/ling487.db "UPDATE users SET is_admin = 1 WHERE email = 'you@example.com';"
```

## HTTP API

All **`POST` / `PUT` / `PATCH` / `DELETE`** API routes expect **`Content-Type: application/json`** (where a body is used) and **`X-CSRFToken`** matching **`GET /api/auth/csrf`**, except you typically call **`/csrf`** from the SPA before the first write.

| Method | Path | Notes |
|--------|------|--------|
| GET | `/api/health` | Liveness |
| GET | `/api/auth/csrf` | `{ "csrf_token": "..." }` (rate-limited) |
| POST | `/api/auth/register` | `email`, `password`; session cookie |
| POST | `/api/auth/login` | Same |
| POST | `/api/auth/logout` | Authenticated |
| GET | `/api/auth/me` | `{ "user": null \| { id, email, is_admin } }` |
| POST | `/api/auth/forgot-password` | `email` |
| POST | `/api/auth/reset-password` | `token`, `password` (new password must meet policy) |
| GET | `/api/sessions` | List sessions |
| POST | `/api/sessions` | `title`, `mode` (optional) |
| GET | `/api/sessions/<id>` | One session |
| GET | `/api/sessions/<id>/messages` | Messages + assistant fields |
| POST | `/api/chat` | `session_id`, `message`, `boost_toggle`, `mode` — assistant `payload_json` may include `pipeline_diagnostics`, `primary_model`, `validation_severity`, `boost_provider`, `boost_reason`, `query_complexity` when the structured pipeline runs; when retrieval returns **no chunks**, `no_match_kind` is set (`greeting` / `short_ack` / `off_topic`) and `course_answer` comes from rotating templates in [`conversational_responses.py`](backend/app/services/conversational_responses.py) (boost is skipped) |
| POST | `/api/feedback` | `message_id`, optional thumb / `preferred` / enriched fields |
| GET | `/api/lectures/topics` | Authenticated; list lectures + chunk counts + section topics |
| GET | `/api/lectures/<n>/summary` | Authenticated; sections for lecture `n` |
| POST | `/api/lectures/retrieve` | `query`, optional `top_k`, optional `backend` (`keyword` / `embedding`; embedding returns 501 until implemented) |
| POST | `/api/study/quiz/next` | `question_type` (`mc` \| `short`), optional `topic`, optional `session_id` |
| POST | `/api/study/quiz/answer` | `chunk_id`, `question_type`, `quiz_token`, optional answer fields, optional `session_id` |
| POST | `/api/study/compare` | `concept_a`, `concept_b`, optional `expand`, optional `session_id` |
| POST | `/api/study/summary` | `kind` (`lecture` \| `topic`), `lecture_number` or `topic`, optional `session_id` |
| GET | `/api/admin/insights` | Admin-only stub |

Lecture routes are rate-limited (e.g. **120/min** for GET catalog endpoints, **90/min** for POST retrieve); see [`lectures.py`](backend/app/routes/lectures.py).

### Default per-IP rate limits (Flask-Limiter)

| Scope | Limit |
|-------|--------|
| `GET /api/auth/csrf` | 60 / minute |
| `POST .../register` | 5 / minute |
| `POST .../login` | 10 / minute |
| `POST .../logout` | 30 / minute |
| `POST .../forgot-password` | 5 / minute |
| `POST .../reset-password` | 10 / minute |
| `POST .../sessions` (create) | 45 / minute |
| `POST .../chat` | 90 / minute |
| `POST .../feedback` | 90 / minute |
| `GET .../lectures/topics`, `GET .../lectures/<n>/summary` | 120 / minute |
| `POST .../lectures/retrieve` | 90 / minute |
| `GET .../admin/insights` | 120 / minute |

Use **`RATELIMIT_STORAGE_URI`** (e.g. **Redis**) when running multiple Gunicorn workers so limits are shared.

## Answer format (product rule)

- **Course Answer** — Always returned; **only** from retrieved lecture sections when there are hits; otherwise a short no-match / off-scope message.
- **Boosted Explanation** — Separate field only when the backend generates it (never merged into Course Answer in the JSON response).

## Security (details)

- **Secrets** live in backend **`.env`** only (`os.getenv` in [`config.py`](backend/app/config.py)); never put **OpenAI** or **Gemini / Google AI** keys in the frontend. Frontend env is for **`VITE_*`** public config only.
- **CSRF:** [Flask-WTF](https://flask-wtf.palletsprojects.com/) validates **`X-CSRFToken`** on unsafe methods. The SPA uses [`frontend/src/api/client.js`](frontend/src/api/client.js) to call **`GET /api/auth/csrf`** and attach the token; CORS allows that header for **`FRONTEND_ORIGIN`**.
- **Rate limits:** [Flask-Limiter](https://flask-limiter.readthedocs.io/) (per-IP defaults in the table above). Set **`RATELIMIT_STORAGE_URI`** to **Redis** when using multiple workers.
- **Passwords:** enforced on register and reset (length + upper / lower / digit / special) in [`app/utils/security.py`](backend/app/utils/security.py).
- **Hardening:** duplicate-register races → **`IntegrityError`** + rollback; strict JSON + **`application/json`** via **`parse_request_json`** on auth and chat writes; password reset uses **`hmac.compare_digest`**, timing padding, and uniform responses where applicable; missing-user login path uses **`burn_auth_timing_budget`**; failures and notable events go to the **`auth.security`** logger.
- **Git:** [`.gitignore`](.gitignore) covers `.env`, `*.db`, `node_modules/`, `dist/`, `build/`, `instance/`, etc.

**Recommended later:** email verification, account lockout, exponential backoff, expanding **`backend/tests/`**, full audit pipeline — see [`progress/entries/2026-04-08-auth-security-hardening.md`](progress/entries/2026-04-08-auth-security-hardening.md). Run **`cd backend && python -m pytest tests/ -v`** for the current suite.

## Deployment (Render or similar)

- **Backend:** e.g. Gunicorn `wsgi:app` from [`backend/`](backend/). Set **`FLASK_SECRET_KEY`**, **`FRONTEND_ORIGIN`**, **`SESSION_COOKIE_SECURE=1`**, and **`RATELIMIT_STORAGE_URI`** (Redis URL recommended). Use a production **`DATABASE_URL`** if you leave SQLite.
- **Frontend:** static or Node host; set **`VITE_API_BASE_URL`** to the API origin; CORS must allow **`credentials`** and the **`X-CSRFToken`** header for that origin.

## Next steps

- **Analytics-driven tuning:** Use `retrieval_logs` (score margin, coverage, low-confidence flags), `retrieval_chunk_hits` (per-chunk field scores), `feedback` (enriched signals), and `message_outcomes` (rephrase/follow-up detection) to tune `FIELD_WEIGHTS`, phrase bonuses, and `CONFIDENCE_THRESHOLD` in [`retrieval.py`](backend/app/services/retrieval.py). Build admin insights aggregates over these tables.
- **Boost evaluation:** Compare `response_variants.boost_used` and `boost_reason` against `feedback.preferred` and `feedback.resolved` to measure boost win-rate; use `response_fingerprint` to detect repeated weak answers. Assistant `payload_json.boost_provider` distinguishes Gemini vs OpenAI fallback.
- **LLM observability:** Persist `token_usage_json` and model/provider names on `response_variants` (`model_name` / `provider_name`) for primary vs boost — wiring is partial; see [`generation/llm.py`](backend/app/services/generation/llm.py) and [`generation/gemini_boost.py`](backend/app/services/generation/gemini_boost.py).
- **Embedding retrieval:** Add `backend="embedding"` to [`retrieve_chunks`](backend/app/services/retrieval.py); schema supports `RetrievalChunkHit` scoring data for hybrid ranking.
- **Dataset quality:** Query `retrieval_chunk_hits` joined to `feedback`/`message_outcomes` to find chunks that correlate with bad outcomes (low helpfulness, rephrases, confusion flags); use results to split, rewrite, or expand lecture content.
- **Production email:** Resend; remove reliance on **`dev_reset_token`** outside debug.
