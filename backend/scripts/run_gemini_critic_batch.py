#!/usr/bin/env python3
"""Run Gemini critic for a persisted eval run in the **foreground** (no Flask background thread).

Admin ``POST /api/admin/eval/runs/<id>/critic`` spawns a daemon thread; **debug reloaders**
and process restarts kill that thread mid-batch, which looks like “stuck at 12 / 90”. Use this
script for reliable full-suite critic passes (use **``.venv/bin/python``** so ``dotenv`` /
``sqlalchemy`` imports resolve)::

    cd backend
    PYTHONPATH=. .venv/bin/python scripts/run_gemini_critic_batch.py --run-id 4 --force

Do **not** paste placeholder ``--run-id <id>`` into the shell: ``<`` is input redirection in
zsh/bash; use a numeric id (e.g. ``--run-id 11``).

Requires ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` and the same DB as the API (``DATABASE_URL`` / default SQLite).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app import create_app  # noqa: E402
from app.services.eval_critic import run_critic_for_eval_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Foreground Gemini critic batch for one evaluation run.")
    parser.add_argument("--run-id", type=int, required=True, help="evaluation_runs.id")
    parser.add_argument("--force", action="store_true", help="Bypass cached complete batch")
    parser.add_argument(
        "--modes",
        default="",
        help='Optional comma-separated effective modes (default: env CRITIC_CASE_MODES), e.g. "chat,compare"',
    )
    args = parser.parse_args(argv)

    modes_list = [x.strip().lower() for x in args.modes.split(",") if x.strip()]
    modes_arg = modes_list if modes_list else None

    app = create_app()
    with app.app_context():
        out = run_critic_for_eval_run(args.run_id, force=args.force, modes=modes_arg)

    print(json.dumps(out, indent=2))
    if out.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
