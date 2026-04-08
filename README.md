# LING 487 Tutor

Full-stack scaffold for a course-specific tutor chatbot: **React + Vite** frontend, **Flask + SQLite** backend, session cookie auth, and placeholders for retrieval, OpenAI-boosted explanations (server-side only), and Resend password reset.

## Repository layout

- [`backend/`](backend/) — Flask app (`app/`), lecture data placeholder ([`backend/data/lecture_chunks.json`](backend/data/lecture_chunks.json)), [`backend/wsgi.py`](backend/wsgi.py)
- [`frontend/`](frontend/) — Vite + React UI with ChatGPT-like layout (sidebar, chat, header)

## Local setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — set FLASK_SECRET_KEY at minimum

flask --app wsgi init-db
flask --app wsgi run --debug
```

API defaults to `http://127.0.0.1:5000`. Health check: `GET /api/health`.

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# optional: VITE_API_BASE_URL for production builds

npm run dev
```

Dev server defaults to `http://127.0.0.1:5173`. The Vite config proxies `/api` to the Flask server so cookies and same-origin `fetch` work without CORS friction.

### Admin insights

`GET /api/admin/insights` requires `users.is_admin = 1`. After `init-db`, promote a user in SQLite (example):

```bash
sqlite3 backend/ling487.db "UPDATE users SET is_admin = 1 WHERE email = 'you@example.com';"
```

## Answer format (product rule)

- **Course Answer** — always returned; must be grounded in lecture chunks (retrieval is stubbed until wired).
- **Boosted Explanation** — optional; separate block when the user enables boost or retrieval confidence is low (OpenAI integration stub in [`backend/app/services/llm.py`](backend/app/services/llm.py)).

## Security

- API keys only in backend `.env` (`OPENAI_API_KEY`, `RESEND_API_KEY`, etc.), loaded via `os.getenv`.
- Root [`.gitignore`](.gitignore) excludes `.env`, `*.db`, `node_modules/`, build outputs.

## Deployment (Render)

- Deploy backend as a Python web service (start command e.g. `gunicorn wsgi:app`, install from `backend/`).
- Deploy frontend as static site or Node static server; set `VITE_API_BASE_URL` to the backend URL and ensure CORS + `credentials` match your cookie settings (`FRONTEND_ORIGIN`, `SESSION_COOKIE_SECURE=1` in production).

## Next steps (not in this scaffold)

- Load real lecture chunks into `lecture_chunks` (or keep JSON + importer).
- Implement keyword retrieval and deterministic Course Answer assembly.
- Wire OpenAI and Resend in the service modules and remove dev-only reset token leakage.
