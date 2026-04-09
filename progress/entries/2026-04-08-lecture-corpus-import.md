# 2026-04-08 — LING487_SUPER_TUTOR corpus in repo + retrieval

## Summary

- Added [`backend/data/LING487_SUPER_TUTOR.json`](../../backend/data/LING487_SUPER_TUTOR.json) (from `LING487_FINAL_SUPER_TUTOR` source: lectures with section headings and bullet `content`).
- New [`backend/app/services/lecture_loader.py`](../../backend/app/services/lecture_loader.py): `import_lecture_json` wipes `lecture_chunks` and inserts one row per section (topic = `{title} — {heading}`, keywords derived from text).
- [`backend/app/services/retrieval.py`](../../backend/app/services/retrieval.py): token overlap scoring, optional **lecture-number** boost from phrases like `lecture 8`, `top_k` hits; `format_course_answer` builds the Course Answer block from bullets only.
- CLI: `flask --app wsgi import-lectures [path]`; config default [`LECTURE_JSON_PATH`](../../backend/app/config.py) (env override).
- [`chat.py`](../../backend/app/routes/chat.py) uses real retrieval output for Course Answer when chunks exist.

## Follow-ups

- Re-run `import-lectures` after editing the JSON; optional migration to append-only imports instead of full replace.
