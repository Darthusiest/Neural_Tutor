# Local testing: auth and password reset

## 1. Environment

From `backend/`:

```bash
cp .env.example .env
```

Set at minimum:

| Variable | Purpose |
|----------|---------|
| `FLASK_SECRET_KEY` | Session + Flask-WTF CSRF signing (use a long random string) |
| `FRONTEND_ORIGIN` | Must match the SPA origin (default `http://127.0.0.1:5173`) |
| `PASSWORD_RESET_BASE_URL` | Page that reads `?token=` (default `http://127.0.0.1:5173/reset-password`) |

For real reset emails:

| Variable | Purpose |
|----------|---------|
| `RESEND_API_KEY` | From [resend.com](https://resend.com) API keys |
| `RESEND_FROM_EMAIL` | Verified sender, e.g. `onboarding@resend.dev` for quick tests |

Optional:

| Variable | Purpose |
|----------|---------|
| `DEV_RETURN_RESET_TOKEN=1` | In **debug**, JSON from forgot-password may include `dev_reset_token` even when Resend is configured |

Without `RESEND_API_KEY` / `RESEND_FROM_EMAIL`, forgot-password still returns **200** with a generic message; in **`FLASK_DEBUG=1`**, the JSON includes **`dev_reset_token`** so you can test `reset-password` without email.

## 2. Run the API

```bash
cd backend
source .venv/bin/activate   # after pip install -r requirements.txt
flask --app wsgi init-db
flask --app wsgi run --debug
```

## 3. CSRF + curl

Mutating routes require **`Content-Type: application/json`** and **`X-CSRFToken`**. Obtain a token and session cookie:

```bash
CSRF=$(curl -s -c cookies.txt http://127.0.0.1:5000/api/auth/csrf | jq -r .csrf_token)
```

Register:

```bash
curl -s -b cookies.txt -c cookies.txt -X POST http://127.0.0.1:5000/api/auth/register \
  -H "Content-Type: application/json" -H "X-CSRFToken: $CSRF" \
  -d '{"email":"you@example.com","password":"Abcd!2345"}'
```

Login (fetch a fresh CSRF after register if needed):

```bash
CSRF=$(curl -s -b cookies.txt -c cookies.txt http://127.0.0.1:5000/api/auth/csrf | jq -r .csrf_token)
curl -s -b cookies.txt -c cookies.txt -X POST http://127.0.0.1:5000/api/auth/login \
  -H "Content-Type: application/json" -H "X-CSRFToken: $CSRF" \
  -d '{"email":"you@example.com","password":"Abcd!2345"}'
```

Current user:

```bash
curl -s -b cookies.txt http://127.0.0.1:5000/api/auth/me
```

Logout:

```bash
CSRF=$(curl -s -b cookies.txt -c cookies.txt http://127.0.0.1:5000/api/auth/csrf | jq -r .csrf_token)
curl -s -b cookies.txt -X POST http://127.0.0.1:5000/api/auth/logout \
  -H "Content-Type: application/json" -H "X-CSRFToken: $CSRF"
```

## 4. Password reset flow

### Forgot (always 200 with same message shape for valid email format)

```bash
CSRF=$(curl -s -c cookies.txt http://127.0.0.1:5000/api/auth/csrf | jq -r .csrf_token)
curl -s -b cookies.txt -c cookies.txt -X POST http://127.0.0.1:5000/api/auth/forgot-password \
  -H "Content-Type: application/json" -H "X-CSRFToken: $CSRF" \
  -d '{"email":"you@example.com"}'
```

- With Resend configured: check the inbox (or Resend dashboard). The link is  
  `{PASSWORD_RESET_BASE_URL}?token=<token>`.
- Without Resend, with **`FLASK_DEBUG=1`**: parse **`dev_reset_token`** from JSON for the same URL pattern.

### Confirm reset

Password must meet policy (length, upper, lower, digit, special):

```bash
CSRF=$(curl -s -c cookies.txt http://127.0.0.1:5000/api/auth/csrf | jq -r .csrf_token)
curl -s -b cookies.txt -X POST http://127.0.0.1:5000/api/auth/reset-password \
  -H "Content-Type: application/json" -H "X-CSRFToken: $CSRF" \
  -d '{"token":"PASTE_TOKEN_HERE","password":"Xyz!9876"}'
```

## 5. Resend checklist

1. Create an API key and set `RESEND_API_KEY`.
2. Use a verified domain or Resend’s test sender **`onboarding@resend.dev`** and send **to your own** verified recipient per Resend onboarding rules.
3. Set `PASSWORD_RESET_BASE_URL` to the exact frontend route (see `frontend` reset page + query param `token`).
4. If emails fail, check server logs; the HTTP response for forgot-password stays generic.

## 6. SPA

The React app uses [`frontend/src/api/client.js`](../../frontend/src/api/client.js): it loads CSRF before POST/PUT/PATCH/DELETE automatically when `VITE_API_BASE_URL` is empty (Vite proxy) or set to the API.
