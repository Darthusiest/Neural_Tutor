# LING 487 Tutor

Full-stack app for a course-specific tutor: **React + Vite** frontend, **Flask + SQLite** backend, **session cookie** auth (with **CSRF**, **rate limits**, and strict **password policy**), and **retrieval-first** answers from lecture JSON in **`lecture_chunks`** (v1 keyword scoring). Optional **Boosted Explanation** is intended to use **OpenAI** on the server only (`OPENAI_API_KEY`); [`backend/app/services/llm.py`](backend/app/services/llm.py) is still a stub until wired.

## Repository layout

| Path | Purpose |
|------|--------|
| [`backend/`](backend/) | Flask app: [`app/`](backend/app/), [`wsgi.py`](backend/wsgi.py), [`requirements.txt`](backend/requirements.txt) |
| [`backend/data/LING487_SUPER_TUTOR.json`](backend/data/LING487_SUPER_TUTOR.json) | Lecture corpus (`lectures[]` → `sections` with `heading` + `content`); load into SQLite via **`flask import-lectures`** |
| [`frontend/`](frontend/) | Vite + React: sidebar (sessions), chat panel, header (auth / admin) |
| [`progress/`](progress/) | Project log: how to use it in [`progress/README.md`](progress/README.md), dated notes in [`progress/entries/`](progress/entries/) |

### Backend code map (quick)

| Area | Location |
|------|----------|
| App factory, CORS, CLI | [`backend/app/__init__.py`](backend/app/__init__.py) |
| Config | [`backend/app/config.py`](backend/app/config.py) |
| `db`, `login_manager`, `csrf`, `limiter` | [`backend/app/extensions.py`](backend/app/extensions.py) |
| Auth + CSRF token route | [`backend/app/routes/auth.py`](backend/app/routes/auth.py) |
| Chat, sessions, feedback | [`backend/app/routes/chat.py`](backend/app/routes/chat.py) |
| JSON input + password/email checks + timing helpers | [`backend/app/utils/security.py`](backend/app/utils/security.py) |
| Lecture import + lexical retrieval + cache | [`lecture_loader.py`](backend/app/services/lecture_loader.py), [`retrieval.py`](backend/app/services/retrieval.py), [`lecture_data.py`](backend/app/services/lecture_data.py) |
| Chat turn orchestration + analytics persistence | [`chat_orchestrator.py`](backend/app/services/chat_orchestrator.py) |
| Lectures API (topics, summary, retrieve) | [`routes/lectures.py`](backend/app/routes/lectures.py) |
| Analytics / feedback / outcomes models | [`models/analytics.py`](backend/app/models/analytics.py) |
| Password reset email (Resend) | [`backend/app/services/reset_email.py`](backend/app/services/reset_email.py) |
| SPA `fetch` + CSRF | [`frontend/src/api/client.js`](frontend/src/api/client.js) |
| Auth + DB testing / schema notes | [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md), [`backend/docs/schema.md`](backend/docs/schema.md) |

## Current status

- **Auth:** Register / login / logout / `GET /api/auth/me`. **CSRF:** mutating requests need **`Content-Type: application/json`** and **`X-CSRFToken`** (see [`client.js`](frontend/src/api/client.js)). **Passwords:** 8+ chars with upper, lower, digit, and special (register + reset). **Password reset:** **`POST /api/auth/forgot-password`** persists a time-limited token and sends mail via **Resend** when **`RESEND_API_KEY`** and **`RESEND_FROM_EMAIL`** are set; link base is **`PASSWORD_RESET_BASE_URL`**. Without Resend, **`FLASK_DEBUG=1`** may include **`dev_reset_token`** in JSON (or set **`DEV_RETURN_RESET_TOKEN=1`** with Resend for manual QA). Details: [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md).
- **Course data:** `lecture_chunks` stores `source_excerpt`, `clean_explanation`, keywords, and optional sample Q/A ([`content.py`](backend/app/models/content.py)); **`import-lectures`** fills them from [`data/LING487_SUPER_TUTOR.json`](backend/data/LING487_SUPER_TUTOR.json) (optional per-section `clean_explanation` / `sample_question(s)` / `sample_answer` in JSON). Use **`--upsert`** to merge by `(lecture_number, topic)` without wiping the table. **`GET /api/lectures/topics`**, **`GET /api/lectures/<n>/summary`**, and **`POST /api/lectures/retrieve`** use [`lecture_data.py`](backend/app/services/lecture_data.py) ([`lectures.py`](backend/app/routes/lectures.py)); `backend=embedding` returns **501** until implemented.
- **Retrieval (v1):** Lexical, **token-aligned** scoring (no substring false positives like `cat` in `category`): field weights (`topic` > `keywords` > …), stopword-stripped queries, phrase bonuses, frequency and length normalization, lecture-number boost, and blended **confidence**. Module-level **`load_lecture_cache()`** keeps dict snapshots + per-chunk indices; **`invalidate_lecture_cache()`** on import. Tunables live at the top of [`retrieval.py`](backend/app/services/retrieval.py). **`RetrievalResult`** can carry **`diagnostics`** for logging.
- **Chat:** Sessions and messages persist. [`handle_chat_turn`](backend/app/services/chat_orchestrator.py) runs retrieval, builds **Course Answer** via [`format_course_answer`](backend/app/services/retrieval.py), persists **`retrieval_logs`** with per-chunk **`retrieval_chunk_hits`** (scores and field breakdown), **`response_variants`** (boost metadata + fingerprints), and on follow-up turns fills **`message_outcomes`** heuristics. **Boosted Explanation** may run when boost is on, confidence is below **`CONFIDENCE_THRESHOLD`**, or `mode` is `compare` / `summary` (LLM still stubbed in [`llm.py`](backend/app/services/llm.py) until wired).
- **UI:** ChatGPT-style layout; modes `chat` / `quiz` / `compare` / `summary`; boost checkbox; **`ProtectedRoute`** + **`ErrorBoundary`** (see [`App.jsx`](frontend/src/App.jsx), [`Layout.jsx`](frontend/src/components/Layout.jsx)); auth and reset flows; admin page hits insights stub.
- **Feedback:** `POST /api/feedback` accepts thumbs / preference plus optional enriched fields: `helpfulness_rating` (1-5), `resolved`, `follow_up_required`, `follow_up_type`, `explicit_confusion_flag`, `feedback_note`, `preference_strength`. All enriched fields are nullable; the endpoint is backward-compatible with the original thumb-only payload.

**Not done yet:** rich compare/summary/quiz-specific answer copy (beyond retrieval + boost triggers), OpenAI boost implementation, admin **insights** aggregates over stored analytics, optional Render manifests, email verification / account lockout / formal audit pipeline.

## Local setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Set FLASK_SECRET_KEY (required for sessions and CSRF).
```

If `DATABASE_URL` is unset or blank, the default is **`backend/ling487.db`** (absolute path; see [`config.py`](backend/app/config.py)).

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

**Database changes:** Flask-Migrate is enabled. For a **new** database, **`init-db`** creates current tables. For an **existing** `ling487.db` from an older revision, run **`flask --app wsgi db upgrade`** so analytics tables/columns match [`models/`](backend/app/models/). If the file is badly out of sync, back it up, remove it, then **`init-db`**, **`db upgrade`** (if using migration history), **`import-lectures`**, and re-register users as needed. See [`backend/docs/schema.md`](backend/docs/schema.md) and [`progress/scaffold-review-fixes.md`](progress/scaffold-review-fixes.md) (appendix on **NOT NULL** column additions).

**Auth, CSRF, curl, Resend:** see [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md). **Table reference:** [`backend/docs/schema.md`](backend/docs/schema.md).

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
| POST | `/api/chat` | `session_id`, `message`, `boost_toggle`, `mode` |
| POST | `/api/feedback` | `message_id`, optional thumb / `preferred` / enriched fields |
| GET | `/api/lectures/topics` | Authenticated; list lectures + chunk counts + section topics |
| GET | `/api/lectures/<n>/summary` | Authenticated; sections for lecture `n` |
| POST | `/api/lectures/retrieve` | `query`, optional `top_k`, optional `backend` (`keyword` / `embedding`; embedding returns 501 until implemented) |
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

Use **`RATELIMIT_STORAGE_URI`** (e.g. **Redis**) when running multiple Gunicorn workers so limits are shared (see [`.env.example`](backend/.env.example)).

## Answer format (product rule)

- **Course Answer** — Always returned; **only** from retrieved lecture sections when there are hits; otherwise a short no-match / off-scope message.
- **Boosted Explanation** — Separate field only when the backend generates it (never merged into Course Answer in the JSON response).

## Security (details)

- **Secrets** live in backend **`.env`** only (`os.getenv` in [`config.py`](backend/app/config.py)); never put API keys in the frontend ([`frontend/.env.example`](frontend/.env.example)).
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
- **Boost evaluation:** Compare `response_variants.boost_used` against `feedback.preferred` and `feedback.resolved` to measure boost win-rate; use `response_fingerprint` to detect repeated weak answers.
- **LLM integration:** Implement [`llm.py`](backend/app/services/llm.py) (boost + optional modes); populate `model_name`, `provider_name`, `token_usage_json`, and prompt version columns on `response_variants`.
- **Embedding retrieval:** Add `backend="embedding"` to [`retrieve_chunks`](backend/app/services/retrieval.py); schema supports `RetrievalChunkHit` scoring data for hybrid ranking.
- **Dataset quality:** Query `retrieval_chunk_hits` joined to `feedback`/`message_outcomes` to find chunks that correlate with bad outcomes (low helpfulness, rephrases, confusion flags); use results to split, rewrite, or expand lecture content.
- **Production email:** Resend; remove reliance on **`dev_reset_token`** outside debug.
