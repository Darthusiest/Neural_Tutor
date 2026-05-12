from __future__ import annotations

import threading
from os.path import basename
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, send_file
from flask_login import current_user, login_required
from sqlalchemy.exc import OperationalError

from app.extensions import db, limiter
from app.models import EvaluationRun
from app.services.security_logging import log_security_event
from app.services.admin_insights import (
    chunk_analytics,
    compute_content_quality,
    compute_cost_summary,
    compute_insights_summary,
    compute_tokens_by_day,
    low_confidence_drill_down,
    render_low_confidence_csv,
    _parse_limit_offset,
)
from app.services.eval_admin import (
    evaluation_run_detail,
    evaluation_run_failures,
    list_evaluation_runs,
)

from app.services.eval_critic import (
    count_critic_target_cases,
    critic_cached_payload,
    critic_cases_for_run,
    critic_summary_for_run,
    run_critic_for_eval_run,
)

bp = Blueprint("admin", __name__)

_CRITIC_BG_LOCK = threading.Lock()
_CRITIC_BG_RUN_IDS: set[int] = set()


def critic_job_in_progress(run_id: int) -> bool:
    with _CRITIC_BG_LOCK:
        return run_id in _CRITIC_BG_RUN_IDS

_CRITIC_ARTIFACT_NAMES = frozenset(
    {
        "pipeline_diagram.png",
        "question_type_breakdown.png",
        "retrieval_accuracy.png",
        "evaluation_summary.png",
        "regression_comparison.png",
        "report_dashboard.png",
        "coverage_by_concept.png",
        "failure_modes.png",
        "example_answers.csv",
        "error_analysis.csv",
    }
)


def _admin_forbidden_response():
    if not current_user.is_admin:
        return jsonify({"error": "forbidden"}), 403
    if current_app.config.get("EMAIL_VERIFICATION_REQUIRED") and not getattr(
        current_user, "email_verified_at", None
    ):
        return jsonify({"error": "forbidden", "detail": "email_verification_required"}), 403
    log_security_event(
        "admin_api_access",
        user_id=current_user.id,
        user_email=current_user.email,
        metadata={"path": request.path},
    )
    return None


def _parse_days() -> int:
    raw = request.args.get("days", default="7")
    try:
        days = int(raw)
    except (TypeError, ValueError):
        days = 7
    return max(1, min(days, 365))


@bp.route("/insights", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def insights():
    if err := _admin_forbidden_response():
        return err
    return jsonify(compute_insights_summary(_parse_days()))


@bp.route("/insights/low-confidence", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def insights_low_confidence():
    if err := _admin_forbidden_response():
        return err
    limit, offset = _parse_limit_offset(
        request.args.get("limit"),
        request.args.get("offset"),
        max_limit=200,
    )
    return jsonify(low_confidence_drill_down(_parse_days(), limit, offset))


@bp.route("/insights/low-confidence.csv", methods=["GET"])
@login_required
@limiter.limit("30 per minute")
def insights_low_confidence_csv():
    if err := _admin_forbidden_response():
        return err
    body = render_low_confidence_csv(_parse_days())
    return Response(
        body,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="low-confidence-retrievals.csv"',
        },
    )


@bp.route("/insights/chunks", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def insights_chunks():
    if err := _admin_forbidden_response():
        return err
    try:
        lim = int(request.args.get("limit", "30"))
    except (TypeError, ValueError):
        lim = 30
    return jsonify(chunk_analytics(_parse_days(), lim))


@bp.route("/insights/tokens-by-day", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def insights_tokens_by_day():
    if err := _admin_forbidden_response():
        return err
    return jsonify(compute_tokens_by_day(_parse_days()))


@bp.route("/insights/cost-summary", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def insights_cost_summary():
    if err := _admin_forbidden_response():
        return err
    return jsonify(compute_cost_summary(_parse_days()))


@bp.route("/insights/content-quality", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def insights_content_quality():
    if err := _admin_forbidden_response():
        return err
    return jsonify(compute_content_quality(_parse_days()))


@bp.route("/eval/runs", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def eval_runs_list():
    """List persisted batch eval runs (``evaluation_runs``). Paths: ``/api/admin/eval/runs``."""
    if err := _admin_forbidden_response():
        return err
    try:
        lim = int(request.args.get("limit", "100"))
    except (TypeError, ValueError):
        lim = 100
    ds = (request.args.get("dataset") or "").strip() or None
    return jsonify(list_evaluation_runs(limit=lim, dataset_substring=ds))


@bp.route("/eval/runs/<int:run_id>", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def eval_run_get(run_id: int):
    if err := _admin_forbidden_response():
        return err
    body = evaluation_run_detail(run_id)
    if body is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(body)


@bp.route("/eval/runs/<int:run_id>/failures", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def eval_run_failures_get(run_id: int):
    if err := _admin_forbidden_response():
        return err
    body = evaluation_run_failures(run_id)
    if body is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(body)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _schema_outdated_critic_response(exc: OperationalError):
    """SQLite / Postgres missing-table after pull without ``db upgrade``."""
    raw = str(getattr(exc, "orig", exc) or exc).lower()
    if "evaluation_critic_results" in raw or "no such table" in raw:
        return (
            jsonify(
                {
                    "error": "schema_outdated",
                    "message": "Database is missing Gemini critic tables. Run: flask --app wsgi db upgrade",
                }
            ),
            503,
        )
    return None


@bp.route("/eval/runs/<int:run_id>/critic", methods=["POST"])
@login_required
@limiter.limit("6 per minute")
def eval_run_critic_post(run_id: int):
    """Run Gemini critic on all cases for a stored eval run (may take minutes)."""
    if err := _admin_forbidden_response():
        return err
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force"))
    modes = payload.get("modes")
    if modes is not None and not isinstance(modes, list):
        return (
            jsonify(
                {
                    "error": "invalid_modes",
                    "message": "modes must be a JSON array of strings (e.g. [\"chat\", \"compare\"])",
                }
            ),
            400,
        )
    modes_arg = modes if isinstance(modes, list) else None
    try:
        run = db.session.get(EvaluationRun, run_id)
        if run is None:
            return jsonify({"error": "not_found", "evaluation_run_id": run_id}), 404

        if not force:
            hit = critic_cached_payload(run_id, modes=modes_arg)
            if hit:
                return jsonify(hit), 200

        app = current_app._get_current_object()
        if count_critic_target_cases(run_id, modes=modes_arg) == 0:
            return (
                jsonify(
                    {
                        "error": "no_cases_in_scope",
                        "evaluation_run_id": run_id,
                        "cases_total": run.total_cases,
                        "critic_target_cases": 0,
                    }
                ),
                422,
            )

        if app.testing:
            out = run_critic_for_eval_run(run_id, force=force, modes=modes_arg)
            if out.get("error") == "not_found":
                return jsonify(out), 404
            if out.get("error") == "no_cases_in_scope":
                return jsonify(out), 422
            return jsonify(out), 200

        with _CRITIC_BG_LOCK:
            if run_id in _CRITIC_BG_RUN_IDS:
                return (
                    jsonify(
                        {
                            "status": "already_running",
                            "evaluation_run_id": run_id,
                            "run_total_cases": run.total_cases,
                            "message": "A critic job is already in progress for this run.",
                        }
                    ),
                    409,
                )
            _CRITIC_BG_RUN_IDS.add(run_id)

        def _job():
            try:
                with app.app_context():
                    run_critic_for_eval_run(run_id, force=force, modes=modes_arg)
            except Exception:
                app.logger.exception("Background critic failed run_id=%s", run_id)
            finally:
                with _CRITIC_BG_LOCK:
                    _CRITIC_BG_RUN_IDS.discard(run_id)

        threading.Thread(target=_job, daemon=True).start()
        return (
            jsonify(
                {
                    "status": "started",
                    "evaluation_run_id": run_id,
                    "run_total_cases": run.total_cases,
                    "message": (
                        "Critic is running on the server. Large suites can take many minutes; "
                        "this page refreshes stats automatically."
                    ),
                }
            ),
            202,
        )
    except OperationalError as e:
        if resp := _schema_outdated_critic_response(e):
            return resp
        raise


@bp.route("/eval/runs/<int:run_id>/critic", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def eval_run_critic_get(run_id: int):
    if err := _admin_forbidden_response():
        return err
    try:
        body = critic_summary_for_run(run_id)
    except OperationalError as e:
        if resp := _schema_outdated_critic_response(e):
            return resp
        raise
    if body is None:
        return jsonify({"error": "not_found"}), 404
    root = _repo_root()
    batch = body.get("critic_batch_id")
    rel = body.get("critic_outputs_dir")
    artifact_urls: dict[str, str] = {}
    if batch and rel:
        base_dir = root / rel
        if base_dir.is_dir():
            for name in sorted(_CRITIC_ARTIFACT_NAMES):
                if (base_dir / name).is_file():
                    artifact_urls[name] = f"/api/admin/eval/critic-image/{run_id}/{name}"
    body = {
        **body,
        "critic_job_in_progress": critic_job_in_progress(run_id),
        "artifact_urls": artifact_urls,
        "repo_relative_dir": rel,
        "absolute_dir": str(root / rel) if rel else None,
    }
    return jsonify(body)


@bp.route("/eval/runs/<int:run_id>/critic/cases", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def eval_run_critic_cases(run_id: int):
    if err := _admin_forbidden_response():
        return err
    category = (request.args.get("category") or "").strip() or None
    group_by = (request.args.get("group_by") or "query_type_v2").strip()
    if group_by not in ("query_type_v2", "category", "answer_mode"):
        group_by = "query_type_v2"
    try:
        rows = critic_cases_for_run(run_id, category=category, group_by=group_by)
    except OperationalError as e:
        if resp := _schema_outdated_critic_response(e):
            return resp
        raise
    if rows is None:
        return jsonify({"error": "not_found"}), 404
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        gk = row.get("group_key") or "unknown"
        by_group.setdefault(str(gk), []).append(row)
    return jsonify({"evaluation_run_id": run_id, "group_by": group_by, "groups": by_group, "flat": rows})


@bp.route("/eval/critic-image/<int:run_id>/<path:filename>", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def eval_critic_image(run_id: int, filename: str):
    if err := _admin_forbidden_response():
        return err
    safe = basename(filename)
    if safe != filename or safe not in _CRITIC_ARTIFACT_NAMES:
        return jsonify({"error": "forbidden"}), 404
    try:
        body = critic_summary_for_run(run_id)
    except OperationalError as e:
        if resp := _schema_outdated_critic_response(e):
            return resp
        raise
    if body is None or not body.get("critic_batch_id"):
        return jsonify({"error": "not_found"}), 404
    batch = request.args.get("batch") or body["critic_batch_id"]
    path = _repo_root() / "evaluation_outputs" / "critic" / batch / safe
    if not path.is_file():
        return jsonify({"error": "not_found", "path": str(path)}), 404
    mt = "image/png" if safe.endswith(".png") else "text/csv; charset=utf-8"
    return send_file(path, mimetype=mt)
