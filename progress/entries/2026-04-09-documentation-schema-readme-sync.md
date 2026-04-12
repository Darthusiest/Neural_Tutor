# Documentation sync (schema + README)

## Summary

Aligned root [`README.md`](../../README.md) and [`backend/docs/schema.md`](../../backend/docs/schema.md) with the current codebase: analytics layer (`retrieval_logs` enrichment, `retrieval_chunk_hits`, `response_variants` / `feedback` columns, `message_outcomes`), lexical retrieval + cache behavior, [`chat_orchestrator.py`](../../backend/app/services/chat_orchestrator.py), lectures API + [`lecture_data.py`](../../backend/app/services/lecture_data.py), UI guards (`ProtectedRoute`, `ErrorBoundary`), Flask-Migrate / `db upgrade`, and lecture route rate limits.

## Follow-ups

- Regenerate or extend [`backend/docs/schema.md`](../../backend/docs/schema.md) whenever models change; keep migration revisions in [`backend/migrations/versions/`](../../backend/migrations/versions/) in sync.
