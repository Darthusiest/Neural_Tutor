#!/usr/bin/env python3
"""
Import structured lecture JSON into SQLite ``lecture_chunks`` and refresh the
lexical retrieval cache.

Usage (from ``backend/``)::

    python scripts/seed_lectures.py
    python scripts/seed_lectures.py data/LING487_SUPER_TUTOR.json
    python scripts/seed_lectures.py path/to/corpus.json --upsert

Equivalent Flask CLI::

    flask --app wsgi import-lectures [path] [--upsert]

Environment: ``DATABASE_URL`` or default ``ling487.db``; ``LECTURE_JSON_PATH``
for the default file when no path is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app import create_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed lecture_chunks from JSON.")
    parser.add_argument(
        "json_path",
        nargs="?",
        type=Path,
        help="Lecture corpus JSON (default: LECTURE_JSON_PATH from config)",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Merge by (lecture_number, topic) instead of replacing all rows.",
    )
    args = parser.parse_args()

    app = create_app()
    from app.services.lecture_loader import import_lecture_json
    from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

    path = args.json_path or app.config["LECTURE_JSON_PATH"]
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    with app.app_context():
        n = import_lecture_json(path, upsert=args.upsert)
        invalidate_lecture_cache()
        load_lecture_cache()
    print(f"Imported {n} lecture section(s) into lecture_chunks from {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
