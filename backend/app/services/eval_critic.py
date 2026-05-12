"""Orchestrate Gemini critic batches over stored :class:`EvaluationRun` rows."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from sqlalchemy import asc, desc
from sqlalchemy.exc import SQLAlchemyError

from app.eval.analytics_common import parse_expected_behavior, suite_category
from app.eval.evaluation_outputs import generate_evaluation_outputs
from app.extensions import db
from app.models import LectureChunk, Message, RetrievalChunkHit, RetrievalLog
from app.models.evaluation import EvaluationCaseResult, EvaluationCriticResult, EvaluationRun
from app.services.critic.gemini_critic import run_gemini_critic


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _parse_modes_csv(raw: str | None) -> frozenset[str]:
    """Comma-separated mode tokens from config."""
    s = (raw or "").strip()
    if not s:
        return frozenset({"chat", "compare", "summary"})
    out = {x.strip().lower() for x in s.split(",") if x.strip()}
    return frozenset(out) if out else frozenset({"chat", "compare", "summary"})


def resolved_critic_mode_allowlist(app: Any, modes: Sequence[str] | None) -> frozenset[str]:
    """POST ``modes`` list wins; empty or omitted uses ``CRITIC_CASE_MODES`` from config."""
    if modes is not None:
        allow = {str(m).strip().lower() for m in modes if str(m).strip()}
        if allow:
            return frozenset(allow)
    return _parse_modes_csv(getattr(app.config, "CRITIC_CASE_MODES", None))


def _critic_artifacts_dir(batch_id: str) -> Path:
    return _repo_root() / "evaluation_outputs" / "critic" / batch_id


def _load_critic_manifest(batch_id: str) -> dict[str, Any] | None:
    path = _critic_artifacts_dir(batch_id) / "manifest.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _effective_mode_for_critic(case: EvaluationCaseResult, payload: dict[str, Any]) -> str:
    """Match grading resolution: payload mode, case row, then expected JSON; normalize to lowercase."""
    mode_meta = payload.get("mode")
    mode_meta = mode_meta if isinstance(mode_meta, dict) else {}
    raw = str(mode_meta.get("effective") or case.effective_mode or "").strip().lower()
    if raw and raw != "auto":
        return raw
    expected_behavior = parse_expected_behavior(case.expected_behavior_json)
    fb = str(case.expected_mode or expected_behavior.get("expected_mode") or "").strip().lower()
    return fb or "auto"


def _manifest_modes_sort_key(manifest: dict[str, Any]) -> list[str]:
    raw = manifest.get("modes_filter")
    if not isinstance(raw, list):
        return []
    return sorted(str(x).strip().lower() for x in raw if str(x).strip())


def _load_payload(assistant_id: int | None) -> dict[str, Any]:
    if not assistant_id:
        return {}
    m = db.session.get(Message, assistant_id)
    if not m or not m.payload_json:
        return {}
    try:
        data = json.loads(m.payload_json)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _chunks_for_message(message_id: int | None, chunk_ids_fallback: list[int]) -> list[dict[str, Any]]:
    """Rebuild chunk dicts for the critic: prefer retrieval log order, else DB by ids."""
    ids_ordered: list[int] = []
    if message_id:
        rlog = RetrievalLog.query.filter_by(message_id=message_id).first()
        if rlog:
            hits = (
                RetrievalChunkHit.query.filter_by(retrieval_log_id=rlog.id)
                .order_by(RetrievalChunkHit.rank)
                .all()
            )
            ids_ordered = [h.lecture_chunk_id for h in hits if h.lecture_chunk_id is not None]
    if not ids_ordered and chunk_ids_fallback:
        ids_ordered = list(chunk_ids_fallback)

    if not ids_ordered:
        return []

    chunks = LectureChunk.query.filter(LectureChunk.id.in_(ids_ordered)).all()
    by_id: dict[int, LectureChunk] = {c.id: c for c in chunks}
    out: list[dict[str, Any]] = []
    for hid in ids_ordered:
        c = by_id.get(int(hid))
        if not c:
            continue
        text = (c.source_excerpt or c.clean_explanation or "").strip()
        out.append(
            {
                "id": c.id,
                "lecture_number": c.lecture_number,
                "topic": c.topic,
                "text": text[:8000],
            }
        )
    return out


def _parse_chunk_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[int] = []
    for x in data:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _usage_total_tokens(meta: Mapping[str, Any] | None) -> int:
    if not meta:
        return 0
    u = meta.get("usage")
    if not isinstance(u, dict):
        return 0
    for key in ("totalTokenCount", "total_token_count"):
        v = u.get(key)
        if isinstance(v, int) and v >= 0:
            return v
    return 0


def _error_categories_for_no_verdict(meta: Mapping[str, Any] | None) -> list[str]:
    """Tag rows where Gemini did not return a parseable verdict (not a teaching-quality failure)."""
    if not meta:
        return ["critic_no_response"]
    err = meta.get("error")
    if isinstance(err, str) and err.strip():
        return [err.strip()]
    if meta.get("block_reason"):
        return [f"prompt_blocked:{meta.get('block_reason')}"]
    return ["critic_no_response"]


def _map_critic_primary(cr: EvaluationCriticResult) -> str | None:
    if cr.critic_pass:
        return None
    # Distinguish "judge could not run / return JSON" from rubric failures the model assigned.
    if cr.critic_score is None:
        return "critic_pipeline_error"
    try:
        cats = json.loads(cr.error_categories_json or "[]")
    except json.JSONDecodeError:
        cats = []
    if not isinstance(cats, list) or not cats:
        return "shallow_explanation"
    first = str(cats[0]).strip().lower()
    if "halluc" in first or "unground" in first:
        return "hallucination"
    if "retriev" in first:
        return "retrieval_miss"
    if "mode" in first or "shape" in first:
        return "template_misuse"
    if "incomplete" in first or "vague" in first or "generic" in first:
        return "shallow_explanation"
    return "shallow_explanation"


class _CriticEvalRowProxy:
    """Presents critic scoring to chart code via EvaluationCaseResult-like attributes."""

    #: When True, analytics (e.g. failure_modes chart) use critic tags only, not chatbot postmortems.
    is_critic_eval_overlay = True

    __slots__ = ("_case", "_critic")

    def __init__(self, case: EvaluationCaseResult, critic: EvaluationCriticResult) -> None:
        self._case = case
        self._critic = critic

    def __getattr__(self, name: str) -> Any:
        if name == "pass_bool":
            return self._critic.critic_pass
        if name == "score":
            return self._critic.critic_score
        if name == "error_categories_json":
            return self._critic.error_categories_json
        if name == "primary_error_type":
            return _map_critic_primary(self._critic)
        return getattr(self._case, name)


def _latest_batch_id(run_id: int) -> str | None:
    row = (
        db.session.query(EvaluationCriticResult.critic_batch_id)
        .filter_by(evaluation_run_id=run_id)
        .order_by(desc(EvaluationCriticResult.created_at))
        .first()
    )
    return str(row[0]) if row and row[0] else None


def critic_batch_complete(run_id: int, batch_id: str) -> bool:
    """True when critic rows for this batch cover the intended scope (manifest or full run)."""
    run = db.session.get(EvaluationRun, run_id)
    if not run:
        return False
    n_crit = EvaluationCriticResult.query.filter_by(
        evaluation_run_id=run_id,
        critic_batch_id=batch_id,
    ).count()
    if n_crit <= 0:
        return False
    manifest = _load_critic_manifest(batch_id)
    ids = manifest.get("case_result_ids") if manifest else None
    if manifest is not None and isinstance(ids, list):
        target = len(ids)
        return target > 0 and n_crit == target
    n_cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run_id).count()
    return n_cases > 0 and n_cases == n_crit


def critic_cached_payload(
    run_id: int, modes: Sequence[str] | None = None
) -> dict[str, Any] | None:
    """If the latest critic batch is complete for the requested mode scope, return cached POST body."""
    from flask import current_app

    run = db.session.get(EvaluationRun, run_id)
    if run is None:
        return None
    allow = resolved_critic_mode_allowlist(current_app, modes)
    allow_key = sorted(allow)
    bid = _latest_batch_id(run_id)
    if not bid or not critic_batch_complete(run_id, bid):
        return None
    manifest = _load_critic_manifest(bid)
    if manifest is not None:
        if _manifest_modes_sort_key(manifest) != allow_key:
            return None
    started = datetime.now(timezone.utc).isoformat()
    summary = critic_summary_for_run(run_id) or {}
    return {
        "status": "cached",
        "evaluation_run_id": run_id,
        "critic_batch_id": bid,
        "started_at": started,
        "finished_at": started,
        "cases_total": int(run.total_cases or 0),
        "modes_filter": allow_key,
        **summary,
    }


def count_critic_target_cases(run_id: int, modes: Sequence[str] | None = None) -> int:
    """How many case rows would be critiqued for this run under ``modes`` (no Gemini calls)."""
    from flask import current_app

    allow = resolved_critic_mode_allowlist(current_app, modes)
    cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run_id).all()
    n = 0
    for case in cases:
        payload = _load_payload(case.assistant_message_id)
        if _effective_mode_for_critic(case, payload) in allow:
            n += 1
    return n


def run_critic_for_eval_run(
    run_id: int, *, force: bool = False, modes: Sequence[str] | None = None
) -> dict[str, Any]:
    """Run Gemini critic for cases whose effective mode is in ``modes``; write charts under critic/<batch>/."""
    run = db.session.get(EvaluationRun, run_id)
    if run is None:
        return {"error": "not_found", "evaluation_run_id": run_id}

    from flask import current_app

    allow = resolved_critic_mode_allowlist(current_app, modes)
    allow_list = sorted(allow)
    cap = int(current_app.config.get("CRITIC_MAX_TOKENS_PER_BATCH") or 0)
    started = datetime.now(timezone.utc).isoformat()

    if not force:
        hit = critic_cached_payload(run_id, modes=modes)
        if hit:
            return hit

    batch_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{run_id}_{uuid.uuid4().hex[:8]}"
    cases = (
        EvaluationCaseResult.query.filter_by(evaluation_run_id=run_id)
        .order_by(asc(EvaluationCaseResult.test_id))
        .all()
    )

    target_cases: list[EvaluationCaseResult] = []
    for case in cases:
        payload = _load_payload(case.assistant_message_id)
        eff = _effective_mode_for_critic(case, payload)
        if eff in allow:
            target_cases.append(case)

    if not target_cases:
        return {
            "error": "no_cases_in_scope",
            "evaluation_run_id": run_id,
            "modes_filter": allow_list,
            "cases_total": len(cases),
            "critic_target_cases": 0,
        }

    out_rel = Path("evaluation_outputs") / "critic" / batch_id
    out_dir = _repo_root() / out_rel
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "evaluation_run_id": run_id,
        "modes_filter": allow_list,
        "case_result_ids": [c.id for c in target_cases],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    token_accum = 0
    partial = False
    model_names: list[str] = []

    prompt_ver = str(current_app.config.get("CRITIC_PROMPT_VERSION", "v1"))

    for idx, case in enumerate(target_cases):
        if cap > 0 and token_accum >= cap:
            partial = True
            break

        payload = _load_payload(case.assistant_message_id)
        pl = payload.get("pipeline_diagnostics")
        if isinstance(pl, str):
            try:
                pl = json.loads(pl)
            except json.JSONDecodeError:
                pl = None
        if not isinstance(pl, dict):
            pl = {}
        rlog = None
        if case.assistant_message_id:
            rlog = RetrievalLog.query.filter_by(message_id=case.assistant_message_id).first()

        retrieved = _chunks_for_message(
            case.assistant_message_id,
            _parse_chunk_ids(case.retrieval_chunk_ids_json),
        )
        plan = pl.get("answer_plan") if isinstance(pl.get("answer_plan"), dict) else None
        effective = _effective_mode_for_critic(case, payload)

        expected_behavior = parse_expected_behavior(case.expected_behavior_json)
        t0 = time.perf_counter()
        verdict, meta = run_gemini_critic(
            user_question=case.query_text or "",
            course_answer=payload.get("course_answer") or case.actual_response or "",
            boosted_explanation=payload.get("boosted_explanation"),
            retrieved_chunks=retrieved,
            structured_plan=plan,
            expected_behavior=expected_behavior,
            mode=effective,
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if meta.get("model"):
            model_names.append(str(meta["model"]))
        token_accum += _usage_total_tokens(meta)

        err_cats = verdict.error_categories if verdict else _error_categories_for_no_verdict(meta)
        cr_row = EvaluationCriticResult(
            evaluation_run_id=run_id,
            case_result_id=case.id,
            critic_batch_id=batch_id,
            critic_score=verdict.score if verdict else None,
            critic_pass=bool(verdict.passed) if verdict else False,
            dimension_scores_json=json.dumps(verdict.dimensions) if verdict else None,
            error_categories_json=json.dumps(err_cats),
            rationale_text=(
                verdict.rationale
                if verdict
                else str(meta.get("error") or meta.get("raw_preview") or meta.get("body_preview") or "no_verdict")
            )[:16_000],
            category=suite_category(case),
            query_type_v2=rlog.query_type_v2 if rlog else None,
            answer_mode=rlog.answer_mode if rlog else None,
            model_name=str(meta.get("model")) if meta.get("model") else None,
            latency_ms=elapsed_ms,
            tokens_estimated=_usage_total_tokens(meta) or None,
            critic_prompt_version=prompt_ver,
        )
        db.session.add(cr_row)
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            raise

        delay_cases = float(current_app.config.get("CRITIC_INTER_CASE_DELAY_SEC", 0) or 0)
        if delay_cases > 0 and idx < len(target_cases) - 1:
            time.sleep(delay_cases)

    crit_rows = (
        EvaluationCriticResult.query.filter_by(
            evaluation_run_id=run_id,
            critic_batch_id=batch_id,
        )
        .order_by(asc(EvaluationCriticResult.case_result_id))
        .all()
    )
    by_case = {r.case_result_id: r for r in crit_rows}
    proxies: list[_CriticEvalRowProxy] = []
    for c in target_cases:
        cr = by_case.get(c.id)
        if cr:
            proxies.append(_CriticEvalRowProxy(c, cr))

    scores = [float(r._critic.critic_score or 0.0) for r in proxies]  # noqa: SLF001
    mean_score = round(sum(scores) / max(1, len(scores)), 4) if scores else 0.0
    passed_n = sum(1 for r in proxies if r.pass_bool)

    suffix = ""
    if len(target_cases) < len(cases):
        suffix = f" — critic: {'+'.join(allow_list)}"
    summary_ns = SimpleNamespace(
        dataset_name=f"{run.dataset_name} (Gemini critic){suffix}",
        overall_score=mean_score,
    )
    generate_evaluation_outputs(
        proxies,
        out_dir,
        current_run=None,
        include_regression=False,
        summary_run=summary_ns,
    )

    run_total = int(run.total_cases or len(cases))
    return {
        "status": "partial" if partial else "ok",
        "evaluation_run_id": run_id,
        "critic_batch_id": batch_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "cases_total": run_total,
        "critic_target_cases": len(target_cases),
        "modes_filter": allow_list,
        "cases_critiqued": len(proxies),
        "critic_passed_cases": passed_n,
        "critic_mean_score": mean_score,
        "critic_outputs_dir": str(out_rel),
        "token_accumulated_est": token_accum,
        "partial_due_to_token_cap": partial,
        "model_name": model_names[0] if model_names else None,
    }


def critic_summary_for_run(run_id: int) -> dict[str, Any] | None:
    run = db.session.get(EvaluationRun, run_id)
    if run is None:
        return None
    total = int(run.total_cases or 0)
    batch_id = _latest_batch_id(run_id)
    manifest = _load_critic_manifest(batch_id) if batch_id else None
    target_from_manifest: int | None = None
    modes_filter: list[str] | None = None
    if manifest and isinstance(manifest.get("case_result_ids"), list):
        target_from_manifest = len(manifest["case_result_ids"])
        modes_filter = _manifest_modes_sort_key(manifest) or None
    if not batch_id:
        return {
            "evaluation_run_id": run_id,
            "run_total_cases": total,
            "critic_target_cases": None,
            "modes_filter": modes_filter,
            "critic_batch_id": None,
            "critic_pass_rate": None,
            "critic_mean_score": None,
            "critic_mean_score_parsed_only": None,
            "critic_verdicts_parsed": 0,
            "critic_verdict_parse_rate": None,
            "cases_critiqued": 0,
            "critic_outputs_dir": None,
        }
    rows = EvaluationCriticResult.query.filter_by(
        evaluation_run_id=run_id,
        critic_batch_id=batch_id,
    ).all()
    critic_target = target_from_manifest if target_from_manifest is not None else total
    if not rows:
        return {
            "evaluation_run_id": run_id,
            "run_total_cases": total,
            "critic_target_cases": critic_target,
            "modes_filter": modes_filter,
            "critic_batch_id": batch_id,
            "critic_pass_rate": None,
            "critic_mean_score": None,
            "critic_mean_score_parsed_only": None,
            "critic_verdicts_parsed": 0,
            "critic_verdict_parse_rate": None,
            "cases_critiqued": 0,
            "critic_outputs_dir": str(Path("evaluation_outputs") / "critic" / batch_id),
        }
    n = len(rows)
    passed = sum(1 for r in rows if r.critic_pass)
    parsed_rows = [r for r in rows if r.critic_score is not None]
    verdicts_ok = len(parsed_rows)
    scores_all = [float(r.critic_score or 0.0) for r in rows]
    mean_all = round(sum(scores_all) / max(1, n), 4)
    scores_parsed = [float(r.critic_score) for r in parsed_rows]
    mean_parsed = (
        round(sum(scores_parsed) / max(1, len(scores_parsed)), 4) if scores_parsed else None
    )
    parse_rate = round(verdicts_ok / max(1, n), 4)
    return {
        "evaluation_run_id": run_id,
        "run_total_cases": total,
        "critic_target_cases": critic_target,
        "modes_filter": modes_filter,
        "critic_batch_id": batch_id,
        "critic_pass_rate": round(passed / max(1, n), 4),
        "critic_mean_score": mean_all,
        "critic_mean_score_parsed_only": mean_parsed,
        "critic_verdicts_parsed": verdicts_ok,
        "critic_verdict_parse_rate": parse_rate,
        "cases_critiqued": n,
        "critic_outputs_dir": str(Path("evaluation_outputs") / "critic" / batch_id),
    }


def _safe_error_categories(raw: str | None) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if x is not None]


def critic_cases_for_run(
    run_id: int,
    *,
    category: str | None = None,
    group_by: str = "query_type_v2",
) -> list[dict[str, Any]] | None:
    """Flat list of cases with critic + chatbot fields; optional filter by suite category."""
    run = db.session.get(EvaluationRun, run_id)
    if run is None:
        return None
    batch_id = _latest_batch_id(run_id)
    if not batch_id:
        return []
    q = (
        db.session.query(EvaluationCaseResult, EvaluationCriticResult)
        .join(
            EvaluationCriticResult,
            EvaluationCriticResult.case_result_id == EvaluationCaseResult.id,
        )
        .filter(
            EvaluationCaseResult.evaluation_run_id == run_id,
            EvaluationCriticResult.critic_batch_id == batch_id,
        )
        .order_by(asc(EvaluationCaseResult.test_id))
    )
    out: list[dict[str, Any]] = []
    for case, crit in q.all():
        if category and suite_category(case) != category:
            continue
        disagree = bool(case.pass_bool != crit.critic_pass)
        out.append(
            {
                "test_id": case.test_id,
                "query_text": case.query_text,
                "suite_category": suite_category(case),
                "query_type_v2": crit.query_type_v2 or "",
                "answer_mode": crit.answer_mode or "",
                "effective_mode": case.effective_mode,
                "chatbot_pass": case.pass_bool,
                "chatbot_score": case.score,
                "course_answer": case.actual_response,
                "critic_pass": crit.critic_pass,
                "critic_score": crit.critic_score,
                "critic_error_categories": _safe_error_categories(crit.error_categories_json),
                "critic_rationale": crit.rationale_text,
                "disagreement": disagree,
                "group_key": (
                    crit.query_type_v2 or "unknown"
                    if group_by == "query_type_v2"
                    else suite_category(case)
                    if group_by == "category"
                    else (crit.answer_mode or "unknown")
                ),
            }
        )
    return out
