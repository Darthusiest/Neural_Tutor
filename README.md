# LING 487 Tutor

Full-stack app for a course-specific tutor: **React + Vite** frontend, **Flask + SQLite** backend, **session cookie** auth, and **retrieval-first** answers from lecture JSON loaded into **`lecture_chunks`** (v1 keyword scoring). Optional **Boosted Explanation** uses **OpenAI** on the server only (`OPENAI_API_KEY`); the integration in [`backend/app/services/llm.py`](backend/app/services/llm.py) is still a stub until wired.

## Repository layout

| Path | Purpose |
|------|--------|
| [`backend/`](backend/) | Flask app: [`app/`](backend/app/), [`wsgi.py`](backend/wsgi.py), dependencies in [`requirements.txt`](backend/requirements.txt) |
| [`backend/data/LING487_SUPER_TUTOR.json`](backend/data/LING487_SUPER_TUTOR.json) | Lecture/slide corpus (`lectures[]` → sections with `heading` + `content` bullets); import into SQLite via `flask import-lectures` |
| [`frontend/`](frontend/) | Vite + React UI: sidebar (sessions), main chat, header (auth / admin link) |
| [`progress/`](progress/) | Optional project log: dated notes under [`progress/entries/`](progress/entries/) |

## Current status (what works today)

- **Auth:** Register, login, logout, `GET /api/auth/me`. Password reset stores a token and completes reset via `POST`; **Resend** is not wired yet. In **`FLASK_DEBUG=1`**, forgot-password responses may include `dev_reset_token` for local testing only.
- **Chat:** Sessions and messages persist. `POST /api/chat` runs **keyword retrieval** over `lecture_chunks` ([`backend/app/services/retrieval.py`](backend/app/services/retrieval.py)), builds a **Course Answer** only from retrieved bullets ([`format_course_answer`](backend/app/services/retrieval.py)), logs hits in `retrieval_logs`, and may request a **Boosted Explanation** when the user toggles boost, confidence is low (below the ~0.35 threshold in code), or `mode` is `compare` / `summary` (LLM output is still empty until OpenAI is implemented).
- **UI:** ChatGPT-style shell; mode selector (`chat` / `quiz` / `compare` / `summary`); “Boosted explanation” checkbox; login, register, forgot/reset password routes; admin page calls insights API (stub JSON unless you implement aggregates).
- **Feedback:** `POST /api/feedback` accepts thumbs / preference for a message (UI hooks can be added later).

Not done yet: richer retrieval (compare/summary/quiz-specific behavior), OpenAI + Resend production flows, admin analytics aggregates, Render deploy configs in-repo.

## Local setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set FLASK_SECRET_KEY at minimum.
```

If you omit `DATABASE_URL`, the app uses **`backend/ling487.db`** (absolute path derived from the backend package). Override `DATABASE_URL` only if you want a different file or engine.

```bash
flask --app wsgi init-db
flask --app wsgi import-lectures   # loads data/LING487_SUPER_TUTOR.json → lecture_chunks
flask --app wsgi run --debug
```

Use `flask --app wsgi import-lectures /path/to/other.json` to override the file. Set **`LECTURE_JSON_PATH`** in `.env` to change the default path.

API defaults to `http://127.0.0.1:5000`. Health: `GET /api/health`.

**Note:** If `lecture_chunks` is empty, every question falls through to the off-topic / no-match Course Answer until you run `import-lectures`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# Production builds: set VITE_API_BASE_URL to the public API origin (no trailing slash).

npm run dev
```

Dev: `http://127.0.0.1:5173`. [`vite.config.js`](frontend/vite.config.js) proxies **`/api`** to Flask so the browser can use same-origin `fetch` and session cookies.

### Admin insights

`GET /api/admin/insights` returns **403** unless `users.is_admin` is true for the logged-in user. After `init-db`, promote one account (example):

```bash
sqlite3 backend/ling487.db "UPDATE users SET is_admin = 1 WHERE email = 'you@example.com';"
```

## HTTP API (v1 scaffold)

| Method | Path | Notes |
|--------|------|--------|
| GET | `/api/health` | Liveness |
| GET | `/api/auth/csrf` | Returns `csrf_token` (call before first mutating request; cookie session) |
| POST | `/api/auth/register` | Body: `email`, `password`; sets session; **requires** `X-CSRFToken` + `Content-Type: application/json` |
| POST | `/api/auth/login` | Same |
| POST | `/api/auth/logout` | Requires login |
| GET | `/api/auth/me` | `{ user: null \| { id, email, is_admin } }` |
| POST | `/api/auth/forgot-password` | Body: `email` |
| POST | `/api/auth/reset-password` | Body: `token`, `password` |
| GET | `/api/sessions` | List chats for current user |
| POST | `/api/sessions` | Body: optional `title`, `mode` |
| GET | `/api/sessions/<id>` | Single session |
| GET | `/api/sessions/<id>/messages` | Ordered messages + assistant variants |
| POST | `/api/chat` | Body: `session_id`, `message`, `boost_toggle`, `mode` |
| POST | `/api/feedback` | Body: `message_id`, optional `course_thumb`, `boost_thumb`, `preferred` |
| GET | `/api/admin/insights` | Admin-only stub |

## Answer format (product rule)

- **Course Answer** — Always returned; **grounded only in retrieved lecture sections** when keyword retrieval finds matches. If nothing matches, the user sees a short off-topic / no-match message.
- **Boosted Explanation** — Separate block only when the backend produces it (user toggle, low confidence, or compare/summary mode **once LLM is wired**). Never merged into the Course Answer in the API payload.

## Security

- Secrets only in backend **`.env`**, read with `os.getenv` (see [`backend/app/config.py`](backend/app/config.py)). Do not put API keys in the frontend.
- **CSRF:** [Flask-WTF](https://flask-wtf.palletsprojects.com/) protects `POST`/`PUT`/`PATCH`/`DELETE`. The SPA fetches `GET /api/auth/csrf` and sends **`X-CSRFToken`** on writes (see [`frontend/src/api/client.js`](frontend/src/api/client.js)). CORS allows that header for [`FRONTEND_ORIGIN`](backend/app/config.py).
- **Rate limits** ([Flask-Limiter](https://flask-limiter.readthedocs.io/)): tighter on auth (`register`, `login`, `forgot-password`, etc.), moderate on chat. Default storage is in-memory (`RATELIMIT_STORAGE_URI`, see [`.env.example`](backend/.env.example)); use **Redis** (e.g. `redis://localhost:6379`) behind multiple workers.
- **Passwords:** minimum **8** characters, **upper**, **lower**, **digit**, and **special** character (enforced on register and reset).
- **Auth hardening:** `IntegrityError` on concurrent duplicate signups; JSON/`Content-Type` validation; `SQLAlchemyError` rollback on auth/chat writes; password-reset path uses **`hmac.compare_digest`**, timing padding, and **`reject_login_password_check`** when the user is missing; security events logged under the **`auth.security`** logger.
- [`.gitignore`](.gitignore) covers `.env`, `.env.*`, `*.db`, `node_modules/`, `dist/`, `build/`, `instance/`, etc.

Not yet implemented (recommended for stricter production posture): **email verification**, **account lockout** after repeated failed logins, full **audit** pipeline, exponential backoff. Add **tests** under `backend/tests/` as you harden further.

## Deployment (Render)

- Run the Flask app from [`backend/`](backend/) (e.g. Gunicorn `wsgi:app`). Set `FRONTEND_ORIGIN`, `SESSION_COOKIE_SECURE=1`, `SECRET_KEY` / `FLASK_SECRET_KEY`, and production `DATABASE_URL` if you move off SQLite.
- Ship the frontend build with **`VITE_API_BASE_URL`** pointing at the API; ensure CORS allows that origin with **credentials** so session cookies work.

## Next steps

- Tune retrieval (stopwords, weighting, `top_k`) and add compare/summary/quiz-specific assembly.
- Implement [`backend/app/services/llm.py`](backend/app/services/llm.py) and Resend in forgot-password; remove reliance on `dev_reset_token` outside debug.
