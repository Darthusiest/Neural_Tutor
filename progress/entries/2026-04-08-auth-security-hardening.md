# 2026-04-08 — Auth security hardening

## Summary

- **CSRF:** Flask-WTF `CSRFProtect`; `GET /api/auth/csrf`; SPA sends `X-CSRFToken` on mutating requests (see `frontend/src/api/client.js`); CORS allows that header.
- **Rate limiting:** Flask-Limiter on auth, chat create/chat/feedback, admin insights; `RATELIMIT_STORAGE_URI` config (default in-memory).
- **Registration:** `IntegrityError` / `SQLAlchemyError` handling with rollback; password policy (8+, upper, lower, digit, special); email validation via `email-validator`.
- **Login:** generic 401; dummy password hash check when user missing; failed attempts logged under `auth.security`.
- **Forgot password:** uniform 200 body; timing pad + `burn_auth_timing_budget`; invalid email still returns same message (no enumeration).
- **Reset:** token format bounds; `hmac.compare_digest` on hashes; timing pad; password policy on new password; DB rollback on errors.
- **JSON / Content-Type:** `parse_request_json` in `app/utils/security.py` for strict JSON on affected routes.

## Follow-ups

- Redis-backed limiter and unit tests; account lockout; email verification; exponential backoff (per README “not yet implemented”).
