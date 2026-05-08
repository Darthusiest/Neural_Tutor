#!/usr/bin/env python3
"""Regenerate ``evaluation_outputs/`` artifacts from a persisted eval run.

This avoids rerunning the full chat pipeline when only the *visual* style of
the figures has changed. By default the most recent run in the database is
used; pass ``--run-id`` to pin a specific historical run.

Usage (from ``backend/``)::

    python scripts/regenerate_evaluation_outputs.py
    python scripts/regenerate_evaluation_outputs.py --run-id 5
    python scripts/regenerate_evaluation_outputs.py --out-dir ../paper/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app import create_app  # noqa: E402
from app.eval.evaluation_outputs import generate_evaluation_outputs  # noqa: E402
from app.models.evaluation import EvaluationCaseResult, EvaluationRun  # noqa: E402


def _repo_root() -> Path:
    return _BACKEND_ROOT.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-emit evaluation_outputs/ figures from a persisted eval run."
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=None,
        help="EvaluationRun.id to render (default: most recent persisted run).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory for the artifacts "
            "(default: <repo_root>/evaluation_outputs)."
        ),
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or (_repo_root() / "evaluation_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    app = create_app()
    with app.app_context():
        if args.run_id is None:
            run = (
                EvaluationRun.query.order_by(EvaluationRun.id.desc()).first()
            )
        else:
            run = EvaluationRun.query.get(args.run_id)
        if run is None:
            print("No persisted EvaluationRun found.", file=sys.stderr)
            return 1

        cases = (
            EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id)
            .order_by(EvaluationCaseResult.test_id)
            .all()
        )
        if not cases:
            print(
                f"EvaluationRun id={run.id} has no persisted cases.", file=sys.stderr
            )
            return 1

        generate_evaluation_outputs(cases, out_dir, current_run=run)
        print(
            f"Regenerated evaluation outputs from run_id={run.id} "
            f"(\"{run.run_name}\", dataset={run.dataset_name}, "
            f"{len(cases)} cases) -> {out_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
