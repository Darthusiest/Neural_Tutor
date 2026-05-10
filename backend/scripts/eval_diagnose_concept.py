#!/usr/bin/env python3
"""Summarize failing eval cases for one ``_case_concept`` bucket.

Uses the same bucket key as ``coverage_by_concept.png`` / ``coverage_phase_plan.csv``:
optional ``coverage_concept`` in suite JSON, else first ``must_include``, else category.

Usage (from ``backend/``)::

    PYTHONPATH=. python scripts/eval_diagnose_concept.py --concept cnn
    PYTHONPATH=. python scripts/eval_diagnose_concept.py --concept cnn --run-id 9 --write /tmp/cnn.md

"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app import create_app  # noqa: E402
from app.eval.analytics_common import parse_json_list  # noqa: E402
from app.eval.capability_analytics import _case_concept, primary_error_type_for_row  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.evaluation import EvaluationCaseResult, EvaluationRun  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diagnose eval failures for one coverage concept.")
    p.add_argument("--concept", required=True, help="Coverage bucket label (lowercase), e.g. cnn")
    p.add_argument("--run-id", type=int, default=None, help="EvaluationRun.id (default: latest run)")
    p.add_argument(
        "--write",
        type=Path,
        default=None,
        help="Optional path to write a markdown report",
    )
    args = p.parse_args(argv)
    needle = args.concept.strip().lower()
    if not needle:
        print("--concept must be non-empty", file=sys.stderr)
        return 1

    app = create_app()
    with app.app_context():
        if args.run_id is None:
            run = EvaluationRun.query.order_by(EvaluationRun.id.desc()).first()
        else:
            run = db.session.get(EvaluationRun, args.run_id)
        if run is None:
            print("No matching EvaluationRun.", file=sys.stderr)
            return 1

        cases = (
            EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id)
            .order_by(EvaluationCaseResult.test_id)
            .all()
        )
        if not cases:
            print(f"Run id={run.id} has no case rows.", file=sys.stderr)
            return 1

        matched = [c for c in cases if _case_concept(c) == needle]
        if not matched:
            print(
                f"No cases with coverage concept {needle!r} on run id={run.id} "
                f"({run.dataset_name!r}).",
                file=sys.stderr,
            )
            return 1

        fails = [c for c in matched if not c.pass_bool]
        err_hist: Counter[str] = Counter()
        for c in fails:
            pet = primary_error_type_for_row(c) or ""
            if pet:
                err_hist[pet] += 1
            for tag in parse_json_list(c.error_categories_json):
                if tag:
                    err_hist[f"tag:{tag}"] += 1

        lines = [
            f"# Concept diagnosis: `{needle}`",
            "",
            f"- Run id: {run.id}",
            f"- Dataset: {run.dataset_name}",
            f"- Cases in bucket: {len(matched)} (failed {len(fails)}, passed {len(matched) - len(fails)})",
            "",
            "## Primary error histogram",
            "",
        ]
        for k, v in err_hist.most_common(40):
            lines.append(f"- `{k}`: {v}")
        lines.extend(["", "## Failing test ids", ""])
        for c in fails:
            pet = primary_error_type_for_row(c) or ""
            q = (c.query_text or "").replace("\n", " ").strip()
            if len(q) > 160:
                q = q[:157] + "..."
            lines.append(f"- `{c.test_id}` — {pet} — {q}")

        text = "\n".join(lines).rstrip() + "\n"
        print(text)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(text, encoding="utf-8")
            print(f"\nWrote {args.write}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
