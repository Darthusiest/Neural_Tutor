#!/usr/bin/env python3
"""
Offline boost evaluation: correlate boost_used with feedback / outcomes.

Usage (from repo root):
  cd backend && python scripts/boost_eval.py --days 30

Methodology (explicit):
- Joins ``response_variants`` to ``feedback`` on ``message_id`` in the UTC window.
- Reports counts of boost vs no-boost split by ``course_thumb`` when present.
- Token totals from ``token_usage_json`` are summed for rough cost context (same rules as admin).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow `python scripts/boost_eval.py` from backend/
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from datetime import datetime, timedelta, timezone  # noqa: E402

from sqlalchemy import text  # noqa: E402

from app import create_app  # noqa: E402
from app.config import Config  # noqa: E402
from app.extensions import db  # noqa: E402


def _sum_tokens(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    total = 0
    for key in ("primary", "boost"):
        block = d.get(key)
        if isinstance(block, dict):
            u = block.get("usage")
            if isinstance(u, dict) and isinstance(u.get("total_tokens"), int):
                total += u["total_tokens"]
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args()
    days = max(1, min(args.days, 365))
    app = create_app(Config)
    with app.app_context():
        until = datetime.now(timezone.utc).replace(tzinfo=None)
        since = until - timedelta(days=days)
        rows = db.session.execute(
            text(
                """
                SELECT rv.boost_used, f.course_thumb, COUNT(*) AS n,
                       SUM(CASE WHEN rv.token_usage_json IS NOT NULL THEN 1 ELSE 0 END) AS with_json
                FROM response_variants rv
                LEFT JOIN feedback f ON f.message_id = rv.message_id
                WHERE rv.created_at >= :since AND rv.created_at <= :until
                GROUP BY rv.boost_used, f.course_thumb
                """
            ),
            {"since": since, "until": until},
        ).fetchall()
        rvs = db.session.execute(
            text(
                """
                SELECT token_usage_json FROM response_variants
                WHERE created_at >= :since AND created_at <= :until
                """
            ),
            {"since": since, "until": until},
        ).fetchall()
        tok_sum = sum(_sum_tokens(r[0]) for r in rvs)

        print(f"Window UTC: {since.isoformat()}Z → {until.isoformat()}Z ({days} days)")
        print(f"Total estimated tokens (primary+boost usage in JSON): {tok_sum}")
        print("boost_used | course_thumb | count")
        for boost_used, thumb, n, _wj in rows:
            print(f"  {boost_used} | {thumb!r} | {n}")


if __name__ == "__main__":
    main()
