# Scaffold review — 14 fixes implementation plan

> **Context**: This is a LING 487 Tutor chatbot (React + Vite frontend, Python + Flask backend, SQLite).
> A full scaffold review identified 14 issues across 3 severity tiers.
> Implement every fix below **in the order listed**. Mark each to-do as you go.
> Do NOT edit this plan file itself.

---

## Ground rules

- Do not change the database schema (no new columns or tables).
- Do not rename existing API routes or change JSON response shapes.
- Preserve all existing functionality — these are refactors, not rewrites.
- Add `pytest>=8.0,<9` to `backend/requirements.txt` when creating tests.
- After creating / moving code, verify imports compile: `cd backend && python -c "from app import create_app; create_app()"`.
- After frontend changes, verify: `cd frontend && npx vite build`.

---

## To-do list (14 items)

### HIGH — fix before production

#### 1. Eliminate `resend.api_key` global mutation

**File**: `backend/app/services/reset_email.py`

**Problem**: Line 64 (`resend.api_key = api_key`) writes to a module-level global on the `resend` package every request. Under threaded Gunicorn this is a race condition.

**Fix**: Use `resend.Resend(api_key=...)` client instance instead, scoped to the function call. The Resend SDK (`resend>=2.0`) supports this pattern. Replace the global assignment + `resend.Emails.send(...)` with a per-call client:

```python
client = resend.Resend(api_key=api_key)
client.emails.send({...})
```

Remove the line `resend.api_key = api_key` entirely. Do not store the client on a module global. The function already reads `api_key` from `current_app.config` each call — just pass it to the constructor.

Current code to replace (lines 64-74):

```python
    resend.api_key = api_key
    try:
        resend.Emails.send(
            {
                "from": from_email,
                "to": [to_email],
                "subject": subject,
                "html": html,
                "text": text,
            }
        )
        return ResetEmailResult.SENT
```

---

#### 2. Cache lecture chunks in memory for retrieval

**File**: `backend/app/services/retrieval.py`

**Problem**: Querying all `LectureChunk` rows on every chat request does a full table scan. With 500+ chunks this becomes a bottleneck.

**Implemented approach (current code)**: Load once per process inside an application context, cache **plain dict snapshots** of each row plus precomputed lexical indices (`_row_cache`, `_chunk_indices`). `retrieve()` / `retrieve_chunks()` read from that cache only. `invalidate_lecture_cache()` clears it; `load_lecture_cache()` rebuilds it (no arguments — require Flask app context, e.g. from `create_app()` or tests).

**Why session detachment is not required today**: ORM instances are **not** stored on module globals. The loader copies scalar columns into `dict`s in the same loop as the query; nothing cached is bound to `db.session`, so there is no lazy-load risk when the cache is used later outside a request.

**If you ever cache `LectureChunk` instances instead** (e.g. `_chunk_cache: list[LectureChunk]`): detach them after load so attribute access does not depend on an open session or trigger implicit loads:

```python
from app.extensions import db

rows = LectureChunk.query.order_by(LectureChunk.id).all()
for row in rows:
    db.session.expunge(row)
# or, if nothing else in this session needs the rows: db.session.expunge_all()
```

If the model gains **lazy-loaded relationships**, touch/materialize those attributes **before** `expunge`, or keep preferring **dict snapshots** for long-lived caches (simplest and session-free).

**Integration** (unchanged intent): call `load_lecture_cache()` after DB init in `create_app()`; after `import_lecture_json`, call `invalidate_lecture_cache()` then `load_lecture_cache()`.

---

#### 3. Add test infrastructure + initial tests

**New files**:
- `backend/tests/__init__.py` (empty)
- `backend/tests/conftest.py`
- `backend/tests/test_auth.py`
- `backend/tests/test_retrieval.py`

**Also edit**: `backend/requirements.txt` — append `pytest>=8.0,<9`.

**`conftest.py`** must:
- Create a test Flask app with `TESTING=True`, `WTF_CSRF_ENABLED=False`, in-memory SQLite (`sqlite://`), `FLASK_SECRET_KEY="test"`.
- Provide `@pytest.fixture` for `app`, `client` (Flask test client), and `db_session` that creates tables, yields, then drops.
- Provide a `register_user(client, email, password)` helper.

**`test_auth.py`** — at minimum:
- `test_register_and_login` — register, verify 201, login with same creds, verify 200.
- `test_register_duplicate` — register same email twice, verify 409.
- `test_login_wrong_password` — verify 401.
- `test_logout_unauthenticated` — verify 401.
- `test_forgot_password_returns_200_regardless` — valid and invalid emails both return 200 with uniform message.

**`test_retrieval.py`** — at minimum:
- `test_retrieve_empty_db` — returns confidence 0, empty chunks.
- `test_retrieve_keyword_match` — insert a `LectureChunk` row, query with a keyword from it, verify it appears in results.
- `test_confidence_scoring` — insert several chunks, query should rank the most relevant first.
- `test_format_course_answer` — pass a known chunk dict list, verify output string starts with "Course Answer:".

---

### MEDIUM — address during feature buildout

#### 4. Split analytics models out of `models/chat.py`

**Current file**: `backend/app/models/chat.py` (96 lines, 5 models)

**Fix**: Create `backend/app/models/analytics.py`. Move `RetrievalLog`, `ResponseVariant`, and `Feedback` into it. Keep `ChatSession` and `Message` in `models/chat.py`.

Update imports:
- `backend/app/models/__init__.py` — change imports to pull `RetrievalLog`, `ResponseVariant`, `Feedback` from `app.models.analytics` instead of `app.models.chat`.
- No other file needs changes because everything imports from `app.models` (the `__init__` barrel).

After the move, `models/chat.py` should only contain `ChatSession` and `Message`. `models/analytics.py` should contain `RetrievalLog`, `ResponseVariant`, and `Feedback`, all importing `db` from `app.extensions`.

---

#### 5. Extract chat orchestration into a service

**Current file**: `backend/app/routes/chat.py` lines 110-217 (the `chat()` function)

**Fix**: Create `backend/app/services/chat_orchestrator.py` with a function:

```python
def handle_chat_turn(
    session: ChatSession,
    text: str,
    boost_toggle: bool,
    mode: str,
) -> dict:
    """
    Run retrieval, build course answer, optionally boost, persist everything.
    Returns the JSON-serializable response dict.
    """
```

Move the core logic from lines 134-217 of `chat()` into this function. The route handler `chat()` in `routes/chat.py` should only:
1. Parse/validate the request.
2. Look up the session.
3. Call `handle_chat_turn(...)`.
4. Return `jsonify(result)`.

This sets up future mode dispatch: `handle_chat_turn` can internally branch on `mode` to call quiz/compare/summary handlers.

---

#### 6. Move `CONFIDENCE_THRESHOLD` to config

**Current location**: `backend/app/routes/chat.py` line 16: `CONFIDENCE_THRESHOLD = 0.35`

**Fix**:
- Add to `backend/app/config.py` in `class Config`: `CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))`.
- Add to `backend/.env.example`: `# CONFIDENCE_THRESHOLD=0.35`.
- In `chat_orchestrator.py` (or `routes/chat.py` if #5 hasn't been done yet), read it as `current_app.config["CONFIDENCE_THRESHOLD"]` instead of the module constant.
- Remove the `CONFIDENCE_THRESHOLD = 0.35` line from `routes/chat.py`.

---

#### 7. Move `_hash_token` to `utils/security.py`

**Current location**: `backend/app/routes/auth.py` lines 245-246:

```python
def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
```

**Fix**:
- Move this function to `backend/app/utils/security.py` as `hash_token` (drop the underscore — it's now a public utility). Add the `hashlib` import there if not already present.
- In `backend/app/routes/auth.py`, replace `from app.utils.security import (...)` to include `hash_token`, remove the local `_hash_token` definition, and replace all calls from `_hash_token(...)` to `hash_token(...)`.

---

#### 8. Merge duplicate timing functions

**Current file**: `backend/app/utils/security.py` lines 94-101:

```python
def burn_auth_timing_budget() -> None:
    """Fixed-cost work to narrow timing differences between auth code paths."""
    check_password_hash(_TIMING_HASH, "invalid")


def reject_login_password_check() -> None:
    """Approximate cost of a failed password check when the user does not exist."""
    check_password_hash(_TIMING_HASH, "invalid")
```

**Fix**: Remove `reject_login_password_check`. Keep `burn_auth_timing_budget` with a combined docstring:

```python
def burn_auth_timing_budget() -> None:
    """
    Fixed-cost hash check to equalize timing across auth code paths.
    Call when a real password check is skipped (user not found, forgot-password, etc.).
    """
    check_password_hash(_TIMING_HASH, "invalid")
```

In `backend/app/routes/auth.py`:
- Remove `reject_login_password_check` from the import.
- On line 94 (the `else` branch of login), change `reject_login_password_check()` to `burn_auth_timing_budget()`.

---

#### 9. Add pagination to sessions and messages

**File**: `backend/app/routes/chat.py`

**Fix for `list_sessions`** (line 23-41): Accept `?limit=` and `?offset=` query params. Default limit=50, max limit=200.

```python
@bp.route("/sessions", methods=["GET"])
@login_required
def list_sessions():
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))
    query = (
        ChatSession.query.filter_by(user_id=current_user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    total = query.count()
    rows = query.offset(offset).limit(limit).all()
    ...
    return jsonify({"sessions": out, "total": total, "limit": limit, "offset": offset})
```

**Fix for `list_messages`** (line 83-107): Same pattern. Default limit=100, max limit=500.

```python
    limit = min(int(request.args.get("limit", 100)), 500)
    offset = int(request.args.get("offset", 0))
    ...
    total = query.count()
    msgs = query.offset(offset).limit(limit).all()
    ...
    return jsonify({"messages": out, "total": total, "limit": limit, "offset": offset})
```

The frontend (`ChatPage.jsx`) currently calls these without params and will continue to work unchanged (uses defaults). No frontend changes required.

---

### LOW — cosmetic / hygiene

#### 10. Simplify admin auth check

**File**: `backend/app/routes/admin.py` line 13

**Current**:
```python
    if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
```

**Fix** — `@login_required` already guarantees `is_authenticated`. Simplify to:
```python
    if not current_user.is_admin:
```

---

#### 11. Add React error boundary

**New file**: `frontend/src/components/ErrorBoundary.jsx`

Create a class component that catches render errors and shows a fallback UI:

```jsx
import { Component } from 'react'

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="center-page">
          <h1>Something went wrong</h1>
          <p className="muted">Try refreshing the page.</p>
          <button
            type="button"
            className="link-btn"
            onClick={() => this.setState({ hasError: false })}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
```

**Edit `frontend/src/components/Layout.jsx`**: Wrap `<Outlet />` with `<ErrorBoundary>`:

```jsx
import { Outlet } from 'react-router-dom'
import { ErrorBoundary } from './ErrorBoundary'
import { Header } from './Header'

export function Layout({ user, onLogout }) {
  return (
    <div className="app-shell">
      <Header user={user} onLogout={onLogout} />
      <div className="app-body">
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </div>
    </div>
  )
}
```

---

#### 12. Create `<ProtectedRoute>` wrapper

**New file**: `frontend/src/components/ProtectedRoute.jsx`

```jsx
import { Navigate } from 'react-router-dom'

export function ProtectedRoute({ user, children }) {
  if (!user) return <Navigate to="/login" replace />
  return children
}
```

**Auth loading vs redirect**: In the shipped app, `App.jsx` keeps `user` as `undefined` until `/api/auth/me` finishes, and shows a full-page loading state **before** rendering `<Routes>`. Only then is `user` either an object or `null`, so `ProtectedRoute` never sees `undefined` and there is no “redirect flash” during initial auth. Optional hardening: add a `loading` prop to `ProtectedRoute` only if routes are ever rendered before auth resolves; otherwise the global gate in `App.jsx` is enough.

**Edit `frontend/src/App.jsx`**: Wrap authenticated routes with `<ProtectedRoute>`.

Current (lines 44-52):
```jsx
    <Routes>
      <Route
        element={<Layout user={user} onLogout={handleLogout} />}
      >
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ChatPage user={user} />} />
        <Route path="/chat/:sessionId" element={<ChatPage user={user} />} />
        <Route path="/admin" element={<AdminPage />} />
      </Route>
```

Change to:
```jsx
    <Routes>
      <Route
        element={<Layout user={user} onLogout={handleLogout} />}
      >
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<ProtectedRoute user={user}><ChatPage user={user} /></ProtectedRoute>} />
        <Route path="/chat/:sessionId" element={<ProtectedRoute user={user}><ChatPage user={user} /></ProtectedRoute>} />
        <Route path="/admin" element={<ProtectedRoute user={user}><AdminPage /></ProtectedRoute>} />
      </Route>
```

**Edit `frontend/src/pages/ChatPage.jsx`**: Remove the ad-hoc auth redirect `useEffect` (lines 50-54) and the `if (!user) return null` guard (lines 109-111). `ProtectedRoute` now handles this.

---

#### 13. Clean up `services/__init__.py` re-exports

**File**: `backend/app/services/__init__.py`

**Current**:
```python
from app.services.llm import generate_boosted_explanation
from app.services.retrieval import RetrievalResult, format_course_answer, retrieve

__all__ = [
    "retrieve",
    "RetrievalResult",
    "format_course_answer",
    "generate_boosted_explanation",
]
```

**Fix**: Replace with an empty docstring. No route file uses `from app.services import ...` — they all import from the specific submodule directly (`from app.services.retrieval import ...`, `from app.services.llm import ...`).

```python
"""Chat, retrieval, LLM, and email services."""
```

---

#### 14. Add upsert mode to lecture loader

**File**: `backend/app/services/lecture_loader.py`

**Current**: `import_lecture_json` (line 39) always does `LectureChunk.query.delete()` then inserts.

**Fix**: Add an `upsert: bool = False` parameter.

- When `upsert=False` (default): current behavior — delete all, then insert. No change.
- When `upsert=True`: skip the delete. For each section, look up an existing row by `(lecture_number, topic)`. If found, update `keywords`, `explanation`, `example_qa`. If not found, insert a new row.

Also update the `import-lectures` CLI command in `backend/app/__init__.py` to accept a `--upsert` flag:

```python
@app.cli.command("import-lectures")
@click.argument("json_path", type=click.Path(path_type=Path, exists=True), required=False)
@click.option("--upsert", is_flag=True, help="Merge into existing data instead of replacing.")
def import_lectures(json_path, upsert):
```

---

## Appendix: Alembic and NOT NULL columns on `lecture_chunks`

When adding **required** columns to [`backend/app/models/content.py`](backend/app/models/content.py) (e.g. `source_excerpt` as `nullable=False`) **after** the table already has rows, a single “add NOT NULL with no default” migration will fail.

Use one of these patterns:

1. **Multi-step (recommended)**  
   - Add the column `nullable=True`.  
   - Backfill existing rows (`UPDATE lecture_chunks SET source_excerpt = '' WHERE source_excerpt IS NULL`, or a real data migration).  
   - Alter the column to `nullable=False` (drop interim `server_default` if you used one only for migration).

2. **Single revision with server default**  
   - Add the column with `server_default=''` (or a sentinel); deploy; backfill real content; follow-up migration to remove `server_default` if inserts should not inherit it.

**Avoid**: `nullable=False` on a new column with no default while legacy rows exist — the database (and Alembic) will reject it until every row has a value.

This repo may not yet have an `alembic/versions/` tree; apply the above when migrations are introduced.

---

## Verification checklist

After all 14 items are done:

1. `cd backend && python -c "from app import create_app; create_app()"` — no import errors.
2. `cd backend && python -m pytest tests/ -v` — all tests pass.
3. `cd frontend && npx vite build` — builds without errors.
4. Manually verify: start backend (`flask --app wsgi run`), start frontend (`npm run dev`), register, login, send a chat message, check response renders.
