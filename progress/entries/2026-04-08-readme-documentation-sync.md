# 2026-04-08 — README and env documentation sync

## Summary

- Refreshed root [`README.md`](../../README.md): backend **code map**, merged **current status**, explicit **CSRF + JSON** rules for all mutating routes, **per-route rate limit** table aligned with code, clearer **deployment** (Redis limiter, CORS headers), fixed **`requirements.txt`** link typo.
- [`backend/.env.example`](../../backend/.env.example): note that **`FLASK_SECRET_KEY`** backs sessions and CSRF.
- [`frontend/.env.example`](../../frontend/.env.example): short **CSRF / client.js** pointer.
