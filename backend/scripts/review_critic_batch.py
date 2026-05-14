#!/usr/bin/env python3
"""Triage a Gemini critic batch: pipeline/core failures vs harsh judge / adversarial rows.

Reads ``evaluation_outputs/critic/<batch_id>/manifest.json`` (+ optional ``critic_metrics.json``),
loads joined rows from the DB, and when ``critic_pass_rate`` is below ``--pass-threshold``
(default 0.68; match ``CRITIC_PASS_THRESHOLD``) writes:

- ``critic_failure_review.csv`` — failed cases with dimensions, rationale excerpt, triage bucket.
- ``REVIEW_REPORT.md`` — bucket stats, rubric-vs-pipeline cue, top examples per bucket.

**Triage buckets (heuristic)**

- ``adversarial_noise``: suite ``category == adversarial`` or ``error_tags`` mentions
  nonsense / off_topic / adversarial.
- ``clarification_edge``: suite ``category == clarification`` or clarification-related tags.
- ``core_course_query``: critic failed and (chatbot passed = disagreement, or primary error is
  hallucination / retrieval_miss / template_misuse).
- ``other``: remaining failures.

If ≥ ``--rubric-risk-ratio`` (default 0.45) of failures are ``adversarial_noise``, the report
flags likely **rubric calibration** rather than compare/renderer refactors.

Usage (from ``backend/``; use the project venv — bare ``python`` often lacks SQLAlchemy)::

    cd backend && PYTHONPATH=. .venv/bin/python scripts/review_critic_batch.py --latest
    PYTHONPATH=. .venv/bin/python scripts/review_critic_batch.py --batch-id 20260513T230601Z_4_27188b70

    # or: source .venv/bin/activate && PYTHONPATH=. python scripts/review_critic_batch.py --latest
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from sqlalchemy import asc  # noqa: E402

from app import create_app  # noqa: E402
from app.eval.analytics_common import parse_expected_behavior  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.evaluation import EvaluationCaseResult, EvaluationCriticResult  # noqa: E402
from app.services.eval_critic import _map_critic_primary  # noqa: E402

_ADV_TAGS = frozenset({"nonsense", "off_topic", "adversarial"})
_CORE_PRIMARIES = frozenset({"hallucination", "retrieval_miss", "template_misuse"})


def _resolve_batch_dir(batch_id: str | None, latest: bool) -> Path:
    root = _REPO_ROOT / "evaluation_outputs" / "critic"
    if not root.is_dir():
        raise SystemExit(f"No critic outputs directory: {root}")
    if batch_id:
        d = root / batch_id.strip()
        if not d.is_dir() or not (d / "manifest.json").is_file():
            raise SystemExit(f"Missing batch dir or manifest.json: {d}")
        return d
    if latest:
        candidates = sorted(
            [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(f"No critic batches under {root}")
        return candidates[0]
    raise SystemExit("Provide --batch-id or --latest")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dims(cr: EvaluationCriticResult) -> dict[str, str]:
    out = {
        "dim_grounded": "",
        "dim_accurate": "",
        "dim_complete": "",
        "dim_mode_compliant": "",
        "dim_no_hallucination": "",
    }
    try:
        raw = json.loads(cr.dimension_scores_json or "{}")
    except json.JSONDecodeError:
        return out
    if not isinstance(raw, dict):
        return out
    for k in ("grounded", "accurate", "complete", "mode_compliant", "no_hallucination"):
        v = raw.get(k)
        out[f"dim_{k}"] = "" if v is None else str(v)
    return out


def _error_cat_str(cr: EvaluationCriticResult) -> str:
    try:
        data = json.loads(cr.error_categories_json or "[]")
    except json.JSONDecodeError:
        return ""
    if isinstance(data, list):
        return ";".join(str(x) for x in data)
    return ""


def triage_bucket(case: EvaluationCaseResult, cr: EvaluationCriticResult) -> str:
    """Assign failure bucket; only meaningful when ``cr.critic_pass`` is False."""
    beh = parse_expected_behavior(case.expected_behavior_json)
    cat = (beh.get("category") or "").strip().lower()
    tags = {str(t).strip().lower() for t in (beh.get("error_tags") or [])}
    if cat == "adversarial" or (tags & _ADV_TAGS):
        return "adversarial_noise"
    if cat == "clarification" or "clarification" in tags or "underspecified" in tags:
        return "clarification_edge"
    primary = _map_critic_primary(cr)
    if case.pass_bool or (primary and primary in _CORE_PRIMARIES):
        return "core_course_query"
    return "other"


def _fmt_query(q: str, max_len: int = 160) -> str:
    s = (q or "").replace("\n", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review Gemini critic batch failures vs pipeline.")
    parser.add_argument("--batch-id", default="", help="Critic batch folder name under evaluation_outputs/critic/")
    parser.add_argument("--latest", action="store_true", help="Use most recently modified batch with manifest.json")
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.68,
        help="Below this triggers full report (default 0.68; match CRITIC_PASS_THRESHOLD)",
    )
    parser.add_argument(
        "--rubric-risk-ratio",
        type=float,
        default=0.45,
        help="If adversarial_noise failures / total failures ≥ this, flag rubric calibration (default 0.45)",
    )
    parser.add_argument(
        "--fail-under",
        action="store_true",
        help="Exit with code 1 when critic_pass_rate < pass-threshold",
    )
    args = parser.parse_args(argv)

    batch_dir = _resolve_batch_dir(args.batch_id.strip() or None, args.latest)
    manifest = _load_json(batch_dir / "manifest.json")
    run_id = int(manifest["evaluation_run_id"])
    batch_key = batch_dir.name

    metrics_path = batch_dir / "critic_metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        metrics = _load_json(metrics_path)

    app = create_app()
    with app.app_context():
        rows = (
            db.session.query(EvaluationCaseResult, EvaluationCriticResult)
            .join(
                EvaluationCriticResult,
                EvaluationCriticResult.case_result_id == EvaluationCaseResult.id,
            )
            .filter(
                EvaluationCaseResult.evaluation_run_id == run_id,
                EvaluationCriticResult.critic_batch_id == batch_key,
            )
            .order_by(asc(EvaluationCaseResult.test_id))
            .all()
        )

    if not rows:
        print(f"No DB rows for run_id={run_id} batch={batch_key}", file=sys.stderr)
        return 1

    n = len(rows)
    passed_n = sum(1 for _, cr in rows if cr.critic_pass)
    pass_rate = metrics.get("critic_pass_rate")
    if pass_rate is None:
        pass_rate = round(passed_n / max(1, n), 4)

    mean_score = metrics.get("critic_mean_score")
    if mean_score is None:
        scores = [float(cr.critic_score or 0.0) for _, cr in rows]
        mean_score = round(sum(scores) / max(1, len(scores)), 4)

    print(
        json.dumps(
            {
                "batch_dir": str(batch_dir.relative_to(_REPO_ROOT)),
                "evaluation_run_id": run_id,
                "critic_batch_id": batch_key,
                "cases_critiqued": n,
                "critic_pass_rate": pass_rate,
                "critic_mean_score": mean_score,
                "pass_threshold": args.pass_threshold,
            },
            indent=2,
        )
    )

    if float(pass_rate) >= args.pass_threshold:
        print("Pass rate meets threshold; no failure drill-down emitted.")
        return 1 if args.fail_under else 0

    failures = [(case, cr) for case, cr in rows if not cr.critic_pass]
    if not failures:
        print("Pass rate below threshold but no failing rows (unexpected); exiting.")
        return 1

    bucket_counts: Counter[str] = Counter()
    bucket_rows: dict[str, list[tuple[EvaluationCaseResult, EvaluationCriticResult]]] = defaultdict(list)
    for case, cr in failures:
        b = triage_bucket(case, cr)
        bucket_counts[b] += 1
        bucket_rows[b].append((case, cr))

    adv_fail = bucket_counts.get("adversarial_noise", 0)
    adv_ratio = adv_fail / max(1, len(failures))

    csv_path = batch_dir / "critic_failure_review.csv"
    fieldnames = [
        "test_id",
        "suite_category",
        "error_tags",
        "effective_mode",
        "chatbot_pass",
        "chatbot_score",
        "critic_pass",
        "critic_score",
        "dim_grounded",
        "dim_accurate",
        "dim_complete",
        "dim_mode_compliant",
        "dim_no_hallucination",
        "primary_error_type",
        "critic_error_categories",
        "rationale_excerpt",
        "user_query_one_line",
        "triage_bucket",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for case, cr in sorted(
            failures,
            key=lambda ic: float(ic[1].critic_score if ic[1].critic_score is not None else 999),
        ):
            beh = parse_expected_behavior(case.expected_behavior_json)
            tags = beh.get("error_tags") or []
            tag_s = ";".join(str(t) for t in tags) if isinstance(tags, list) else ""
            dims = _dims(cr)
            rationale = (cr.rationale_text or "").replace("\n", " ").strip()
            if len(rationale) > 800:
                rationale = rationale[:799] + "…"
            primary = _map_critic_primary(cr) or ""
            row_out = {
                "test_id": case.test_id,
                "suite_category": (beh.get("category") or "").strip(),
                "error_tags": tag_s,
                "effective_mode": (case.effective_mode or "").strip(),
                "chatbot_pass": "yes" if case.pass_bool else "no",
                "chatbot_score": "" if case.score is None else str(case.score),
                "critic_pass": "no",
                "critic_score": "" if cr.critic_score is None else str(cr.critic_score),
                "primary_error_type": primary,
                "critic_error_categories": _error_cat_str(cr),
                "rationale_excerpt": rationale,
                "user_query_one_line": _fmt_query(case.query_text or ""),
                "triage_bucket": triage_bucket(case, cr),
                **dims,
            }
            writer.writerow(row_out)

    report_path = batch_dir / "REVIEW_REPORT.md"
    lines: list[str] = [
        "# Critic batch review",
        "",
        f"- **Batch:** `{batch_key}`",
        f"- **evaluation_run_id:** {run_id}",
        f"- **Cases critiqued:** {n}",
        f"- **Critic pass rate:** {float(pass_rate) * 100:.1f}% (threshold {args.pass_threshold * 100:.0f}%)",
        f"- **Mean critic score:** {mean_score}",
        f"- **Failures analyzed:** {len(failures)}",
        "",
        "## Failure counts by triage bucket",
        "",
    ]
    for bucket in sorted(bucket_counts.keys()):
        c = bucket_counts[bucket]
        pct = 100.0 * c / len(failures)
        lines.append(f"- **{bucket}:** {c} ({pct:.1f}% of failures)")
    lines.append("")
    lines.append("## Rubric vs pipeline signal")
    lines.append("")
    if adv_ratio >= args.rubric_risk_ratio:
        lines.append(
            f"- **Flag:** About **{adv_ratio * 100:.0f}%** of failures are **`adversarial_noise`** "
            f"(≥ {args.rubric_risk_ratio * 100:.0f}% heuristic). "
            "Failures cluster on suite-tagged gibberish/off-topic/adversarial rows — **prioritize Gemini rubric "
            "calibration** ([`gemini_critic.py`](../app/services/critic/gemini_critic.py)) "
            "before large compare/retrieval refactors."
        )
    else:
        lines.append(
            "- **Signal:** Adversarial-tagged failures are not dominant; **spot-check "
            "`core_course_query`** rows in the CSV against lecture chunks — likely **pipeline** "
            "(retrieval, compare entities, composer)."
        )
    lines.extend(["", "## Primary error types (failed rows only)", ""])
    fp = metrics.get("failure_primary_counts") if isinstance(metrics.get("failure_primary_counts"), dict) else None
    if fp:
        for k, v in sorted(fp.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"- `{k}`: {v}")
    else:
        pc: Counter[str] = Counter()
        for _, cr in failures:
            pk = _map_critic_primary(cr) or "unknown"
            pc[pk] += 1
        for k, v in pc.most_common():
            lines.append(f"- `{k}`: {v}")

    def _score_key(item: tuple[EvaluationCaseResult, EvaluationCriticResult]) -> float:
        s = item[1].critic_score
        return float(s) if s is not None else 999.0

    lines.extend(["", "## Sample failures (worst critic score first, up to 10 per bucket)", ""])
    for bucket in ("core_course_query", "adversarial_noise", "clarification_edge", "other"):
        items = sorted(bucket_rows.get(bucket, []), key=_score_key)[:10]
        if not items:
            continue
        lines.append(f"### {bucket}")
        lines.append("")
        for case, cr in items:
            sc = cr.critic_score
            sc_s = f"{sc:.3f}" if sc is not None else "—"
            rat = _fmt_query(cr.rationale_text or "", 220)
            lines.append(f"- **`{case.test_id}`** (critic {sc_s}): {_fmt_query(case.query_text or '')}")
            lines.append(f"  - Rationale: {rat}")
        lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Drill-down CSV: [`critic_failure_review.csv`](./critic_failure_review.csv)",
            f"- Charts: `evaluation_summary.png`, `failure_modes.png`, …",
            "",
            "See **[critic_batch_review.md](../../../backend/docs/critic_batch_review.md)** for the human verification loop.",
            "",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path.relative_to(_REPO_ROOT)}")
    print(f"Wrote {report_path.relative_to(_REPO_ROOT)}")
    return 1 if args.fail_under else 0


if __name__ == "__main__":
    raise SystemExit(main())
