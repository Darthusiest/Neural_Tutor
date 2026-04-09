# 2026-04-09 — Resend password reset + docs

## Summary

- Config: **`PASSWORD_RESET_BASE_URL`**, **`DEV_RETURN_RESET_TOKEN`** ([`backend/app/config.py`](../../backend/app/config.py), [`.env.example`](../../backend/.env.example)).
- New [`backend/app/services/reset_email.py`](../../backend/app/services/reset_email.py): Resend **`Emails.send`**, `ResetEmailResult`, **`resend_reset_is_configured()`**.
- [`auth.py`](../../backend/app/routes/auth.py): invalidate unused reset tokens per user before insert; after commit, send email; **`dev_reset_token`** only in debug when Resend not configured or **`DEV_RETURN_RESET_TOKEN=1`**.
- Docs: [`backend/docs/schema.md`](../../backend/docs/schema.md), [`backend/docs/AUTH_LOCAL.md`](../../backend/docs/AUTH_LOCAL.md); README links updated.
