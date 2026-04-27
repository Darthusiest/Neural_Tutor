"""CLI: run static eval suite through production ``handle_chat_turn``."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import create_app
from app.eval.dataset import case_expected_behavior_dict, load_eval_dataset
from app.eval.report_markdown import (
    canonical_tags_and_retrieval_blob,
    fetch_previous_run_map,
    write_error_analysis_md,
    write_examples_md,
)
from app.eval.scoring import ScoringResult, score_eval_case
from app.extensions import db
from app.models import ChatSession, Message, RetrievalChunkHit, RetrievalLog, User
from app.models.evaluation import EvaluationCaseResult, EvaluationRun
from app.services.chat_orchestrator import handle_chat_turn

_VALID_API_MODES = frozenset({"auto", "chat", "quiz", "compare", "summary"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _reports_dir(ts: str) -> Path:
    d = _repo_root() / "reports" / "eval_runs" / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git_head() -> tuple[str | None, str | None]:
    try:
        root = str(_repo_root())
        c = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=root,
                check=False,
            ).stdout
            or ""
        ).strip()
        b = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=root,
                check=False,
            ).stdout
            or ""
        ).strip()
        return (c or None, b or None)
    except OSError:
        return None, None


def _resolve_user_api_mode(case) -> str:
    o = (case.mode_override or "").strip().lower()
    if o in _VALID_API_MODES:
        return o
    legacy = (case.mode or "").strip().lower()
    if legacy in _VALID_API_MODES:
        return legacy
    return "auto"


def _mode_request_source(case) -> str:
    if (case.mode_override or "").strip():
        return "override"
    if (case.mode or "").strip():
        return "legacy"
    return "implicit"


def _get_or_create_eval_user(_app, email: str) -> User:
    u = User.query.filter_by(email=email).first()
    if u:
        return u
    u = User(email=email, email_verified_at=datetime.now(timezone.utc).replace(tzinfo=None))
    pw = os.environ.get("EVAL_RUNNER_USER_PASSWORD", "eval-runner-not-for-login")
    u.set_password(pw)
    db.session.add(u)
    db.session.commit()
    return u


def _load_payload(assistant_id: int) -> dict[str, Any]:
    m = db.session.get(Message, assistant_id)
    if not m or not m.payload_json:
        return {}
    return json.loads(m.payload_json)


def _chunk_ids_for_log(message_id: int) -> list[int]:
    rlog = RetrievalLog.query.filter_by(message_id=message_id).first()
    if not rlog:
        return []
    hits = (
        RetrievalChunkHit.query.filter_by(retrieval_log_id=rlog.id)
        .order_by(RetrievalChunkHit.rank)
        .all()
    )
    return [h.lecture_chunk_id for h in hits if h.lecture_chunk_id is not None]


def _run_one_case(case, user: User, boost_toggle: bool) -> dict[str, Any]:
    s = ChatSession(user_id=user.id, title=f"eval:{case.id}", mode="auto")
    db.session.add(s)
    db.session.commit()
    user_mode = _resolve_user_api_mode(case)
    mrs = _mode_request_source(case)
    out = handle_chat_turn(
        s,
        case.query,
        boost_toggle,
        user_mode,
        mode_request_source=mrs,
    )
    msg_id = out["assistant_message_id"]
    payload = _load_payload(msg_id)
    rlog = RetrievalLog.query.filter_by(message_id=msg_id).first()
    pl_diag = payload.get("pipeline_diagnostics")
    if isinstance(pl_diag, str):
        try:
            pl_diag = json.loads(pl_diag)
        except json.JSONDecodeError:
            pl_diag = None
    mode_meta = payload.get("mode") or {}
    sc = score_eval_case(case, out.get("course_answer") or "", mode_meta, pl_diag)
    val_fail = None
    if pl_diag and isinstance(pl_diag.get("validation"), dict):
        val_fail = pl_diag.get("validation")
    elif rlog and rlog.validation_checks_json:
        try:
            val_fail = json.loads(rlog.validation_checks_json)
        except json.JSONDecodeError:
            val_fail = None
    cids = _chunk_ids_for_log(msg_id)
    return {
        "out": out,
        "payload": payload,
        "pipeline_diagnostics": pl_diag,
        "retrieval_log": rlog,
        "score": sc,
        "latency_ms": rlog.latency_ms if rlog else None,
        "validation": val_fail,
        "chunk_ids": cids,
    }


def _write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run LING 487 static eval through handle_chat_turn."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/l487_eval_suite.json",
        help="Path to eval JSON (relative to cwd, usually backend/)",
    )
    parser.add_argument("--run-name", default="default", help="Label stored on evaluation_runs")
    parser.add_argument(
        "--with-boost",
        action="store_true",
        help="Pass boost_toggle=True (default is False for stable scoring)",
    )
    args = parser.parse_args(argv)
    boost_toggle = bool(args.with_boost)

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = Path.cwd() / ds_path
    meta, cases = load_eval_dataset(ds_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _reports_dir(ts)

    app = create_app()
    with app.app_context():
        email = os.environ.get("EVAL_RUNNER_USER_EMAIL", "eval_runner@neural-tutor.local")
        user = _get_or_create_eval_user(app, email)
        gcommit, gbranch = _git_head()
        dname = f"{meta.get('name', 'eval')}"
        if meta.get("version"):
            dname = f"{dname}@{meta.get('version')}"
        struct_on = bool(app.config.get("STRUCTURED_PIPELINE_ENABLED"))
        run = EvaluationRun(
            run_name=args.run_name,
            dataset_name=dname,
            total_cases=len(cases),
            passed_cases=0,
            failed_cases=0,
            git_commit=gcommit,
            branch_name=gbranch,
            overall_score=None,
            notes_json=json.dumps(
                {
                    "dataset_path": str(ds_path),
                    "reports_dir": str(out_dir),
                    "structured_pipeline_enabled": struct_on,
                    "boost_toggle": boost_toggle,
                    "eval_user_email": email,
                }
            ),
        )
        db.session.add(run)
        db.session.commit()
        run_id = run.id

        row_results: list[dict[str, Any]] = []
        case_details: dict[str, dict[str, Any]] = {}
        passed_n = 0
        failed_n = 0

        for case in cases:
            try:
                one = _run_one_case(case, user, boost_toggle=boost_toggle)
            except Exception as e:  # noqa: BLE001
                failed_n += 1
                emsg = f"runner_exception:{e!s}"
                row_results.append(
                    {
                        "id": case.id,
                        "query": case.query,
                        "expected_mode": case.expected_mode,
                        "detected": "",
                        "effective": "",
                        "score": 0.0,
                        "pass": "false",
                        "q_mode": 0.0,
                        "q_content": 0.0,
                        "q_forbidden": 0.0,
                        "q_struct": 0.0,
                        "latency_ms": "",
                        "errors": emsg,
                    }
                )
                ecr = EvaluationCaseResult(
                    evaluation_run_id=run_id,
                    test_id=case.id,
                    query_text=case.query,
                    expected_mode=case.expected_mode,
                    detected_mode=None,
                    effective_mode=None,
                    expected_behavior_json=json.dumps(case_expected_behavior_dict(case)),
                    actual_response=None,
                    pass_bool=False,
                    score=0.0,
                    error_categories_json=json.dumps([emsg]),
                    validation_failures_json=None,
                    retrieval_chunk_ids_json=None,
                    latency_ms=None,
                )
                db.session.add(ecr)
                db.session.commit()
                case_details[case.id] = {
                    "query": case.query,
                    "actual_response": None,
                    "expected_behavior": case_expected_behavior_dict(case),
                    "detected": "",
                    "effective": "",
                    "canonical_tags": [],
                    "retrieval_blob_lower": "",
                    "scoring_errors": [emsg],
                    "pass": False,
                    "score": 0.0,
                }
                continue

            out = one["out"]
            sc: ScoringResult = one["score"]
            payload = one["payload"]
            mm = payload.get("mode") or {}
            if sc.pass_ok:
                passed_n += 1
            else:
                failed_n += 1
            pl_diag = one.get("pipeline_diagnostics")
            val_json = None
            if isinstance(pl_diag, dict):
                val_json = pl_diag.get("validation")
            if val_json is None and one.get("validation") is not None:
                val_json = one.get("validation")

            ecr = EvaluationCaseResult(
                evaluation_run_id=run_id,
                test_id=case.id,
                query_text=case.query,
                expected_mode=case.expected_mode,
                detected_mode=mm.get("detected"),
                effective_mode=mm.get("effective"),
                expected_behavior_json=json.dumps(case_expected_behavior_dict(case)),
                actual_response=out.get("course_answer"),
                pass_bool=sc.pass_ok,
                score=sc.score,
                error_categories_json=json.dumps(sc.error_categories),
                validation_failures_json=json.dumps(val_json) if val_json is not None else None,
                retrieval_chunk_ids_json=json.dumps(one.get("chunk_ids") or []),
                latency_ms=one.get("latency_ms"),
            )
            db.session.add(ecr)
            db.session.commit()

            tags, ret_blob = canonical_tags_and_retrieval_blob(
                case.raw,
                cli_pass=sc.pass_ok,
                course_answer=out.get("course_answer") or "",
                payload=payload,
                pl_diag=pl_diag if isinstance(pl_diag, dict) else None,
                chunk_ids=one.get("chunk_ids") or [],
            )
            case_details[case.id] = {
                "query": case.query,
                "actual_response": out.get("course_answer"),
                "expected_behavior": case_expected_behavior_dict(case),
                "detected": mm.get("detected"),
                "effective": mm.get("effective"),
                "canonical_tags": tags,
                "retrieval_blob_lower": ret_blob,
                "scoring_errors": list(sc.error_categories) if not sc.pass_ok else [],
                "pass": sc.pass_ok,
                "score": sc.score,
            }
            row_results.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "expected_mode": case.expected_mode,
                    "detected": mm.get("detected", ""),
                    "effective": mm.get("effective", ""),
                    "score": sc.score,
                    "pass": "true" if sc.pass_ok else "false",
                    "q_mode": sc.breakdown.get("mode", 0),
                    "q_content": sc.breakdown.get("required_content", 0),
                    "q_forbidden": sc.breakdown.get("forbidden", 0),
                    "q_struct": sc.breakdown.get("structure", 0),
                    "latency_ms": one.get("latency_ms") or "",
                    "errors": ";".join(sc.error_categories) if sc.error_categories else "",
                }
            )

        scores = [float(r["score"]) for r in row_results if "score" in r]
        if scores:
            run.overall_score = sum(scores) / len(scores)
        run.passed_cases = passed_n
        run.failed_cases = failed_n
        db.session.commit()

        summary = {
            "evaluation_run_id": run_id,
            "run_name": args.run_name,
            "dataset_name": dname,
            "dataset_path": str(ds_path),
            "created_at_utc": ts,
            "total_cases": len(cases),
            "passed": passed_n,
            "failed": failed_n,
            "mean_score": run.overall_score,
            "git_commit": gcommit,
            "git_branch": gbranch,
            "structured_pipeline_enabled": struct_on,
            "boost_toggle": boost_toggle,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        fields = [
            "id",
            "query",
            "expected_mode",
            "detected",
            "effective",
            "score",
            "pass",
            "q_mode",
            "q_content",
            "q_forbidden",
            "q_struct",
            "latency_ms",
            "errors",
        ]
        _write_csv(out_dir / "results.csv", row_results, fields)
        fail_rows = [r for r in row_results if r.get("pass") == "false"]
        _write_csv(out_dir / "failures.csv", fail_rows, fields)

        prev_run, prev_map = fetch_previous_run_map(run_id, dname)
        write_examples_md(
            out_dir / "examples.md",
            row_results=row_results,
            case_details=case_details,
            prev_run=prev_run,
            prev_by_id=prev_map,
            current_run_id=run_id,
        )
        write_error_analysis_md(
            out_dir / "error_analysis.md",
            row_results=row_results,
            case_details=case_details,
            dataset_name=dname,
        )

    print(
        f"Eval run {run_id} done: {passed_n}/{len(cases)} passed, mean score {run.overall_score}. "
        f"Reports: {out_dir}"
    )
    return 0 if failed_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
