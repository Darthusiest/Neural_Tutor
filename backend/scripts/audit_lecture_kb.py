#!/usr/bin/env python3
"""Audit transcript-style lecture concept pack drafts.

Usage (from ``backend/``)::

    python scripts/audit_lecture_kb.py path/to/lecture_pack.json
    python scripts/audit_lecture_kb.py path/to/lecture_pack.json --write-clean
    python scripts/audit_lecture_kb.py path/to/lecture_pack.json --report reports/lecture_kb_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.knowledge.lecture_kb_audit import audit_lecture_kb_payload  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit transcript-style lecture concept pack drafts."
    )
    parser.add_argument("json_path", type=Path, help="Path to lecture concept pack JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write cleaned JSON (default: overwrite input when --write-clean is set).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write structured issue report JSON to this path.",
    )
    parser.add_argument(
        "--write-clean",
        action="store_true",
        help="Write sanitized payload (with export placeholders stripped).",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero when warnings are present.",
    )
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"Error: file not found: {args.json_path}", file=sys.stderr)
        return 1

    try:
        payload = json.loads(args.json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {args.json_path}: {exc}", file=sys.stderr)
        return 1

    result = audit_lecture_kb_payload(payload, strip_export_stubs=True)

    if args.write_clean:
        out_path = args.output or args.json_path
        _write_json(out_path, result.cleaned_payload)
        print(f"Wrote cleaned lecture concept pack JSON to {out_path}")

    if args.report:
        _write_json(args.report, result.summary_dict())
        print(f"Wrote audit report to {args.report}")

    print(
        f"Audit complete: {len(result.issues)} issue(s) "
        f"({result.warning_count} warning(s), {result.error_count} error(s))."
    )
    for issue in result.issues[:20]:
        print(
            f"- [{issue.severity}] {issue.code} "
            f"(lecture={issue.lecture_id}, concept={issue.concept_label}, field={issue.field})"
        )
    if len(result.issues) > 20:
        print(f"... plus {len(result.issues) - 20} more issue(s)")

    if result.error_count > 0:
        return 1
    if args.fail_on_warn and result.warning_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
