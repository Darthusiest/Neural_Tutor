#!/usr/bin/env python3
"""Rough indexed chunk counts per KB concept (alias overlap in LectureChunk rows).

Uses the Flask app and database (same as the live tutor). For each concept in
``LING487_STRUCTURED_PIPELINE_KB.json``, counts chunks whose topic, keywords,
clean_explanation, or source_excerpt contains the concept id, canonical name,
or any alias (length ≥ 2).

Usage (from ``backend/``)::

    PYTHONPATH=. .venv/bin/python scripts/audit_kb_chunk_coverage.py

Exit code 1 if any concept has fewer than ``--min-chunks`` matches (default 2).
Use default app :class:`app.config.Config` (points DB at ``ling487.db`` unless
overridden).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app import create_app  # noqa: E402
from app.models import LectureChunk  # noqa: E402
from app.services.knowledge.concept_kb import get_kb  # noqa: E402
from app.services.knowledge.kb_chunk_audit import audit_kb_chunk_coverage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("Usage")[0].strip())
    parser.add_argument(
        "--min-chunks",
        type=int,
        default=2,
        help="Exit 1 if any concept has fewer than this many matching chunks.",
    )
    args = parser.parse_args()

    app = create_app()
    kb = get_kb()

    with app.app_context():
        rows = LectureChunk.query.all()
        result = audit_kb_chunk_coverage(kb, rows, min_chunks=args.min_chunks)

    for cid, n in sorted(result.counts.items()):
        print(f"{cid}\t{n}")

    if result.below_threshold:
        print("\nBelow threshold:", file=sys.stderr)
        for cid, n in result.below_threshold:
            print(f"  {cid}: {n}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
