# 2026-04-08 — Documentation synced with codebase

## Summary

- Expanded root `README.md`: repository table, **current status** (implemented vs TODO), API table, boost/mode behavior, DB default path, security/deploy notes.
- Aligned `backend/.env.example`: `DATABASE_URL` documented as optional; removed misleading “relative to cwd” comment.
- `backend/app/config.py`: treat empty `DATABASE_URL` as unset so `cp .env.example .env` stays valid.
- Linked root README from `progress/README.md`.
