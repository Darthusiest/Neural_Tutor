from flask import Blueprint, Response, current_app, jsonify, request
from flask_login import current_user, login_required

from app.extensions import limiter
from app.services.audit import log_audit_event
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

bp = Blueprint("admin", __name__)


def _admin_forbidden_response():
    if not current_user.is_admin:
        return jsonify({"error": "forbidden"}), 403
    if current_app.config.get("EMAIL_VERIFICATION_REQUIRED") and not getattr(
        current_user, "email_verified_at", None
    ):
        return jsonify({"error": "forbidden", "detail": "email_verification_required"}), 403
    log_audit_event(
        "admin_api_access",
        actor_user_id=current_user.id,
        actor_email=current_user.email,
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
