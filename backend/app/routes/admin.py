from flask import Blueprint, Response, jsonify, request
from flask_login import current_user, login_required

from app.extensions import limiter
from app.services.admin_insights import (
    chunk_analytics,
    compute_insights_summary,
    low_confidence_drill_down,
    render_low_confidence_csv,
    _parse_limit_offset,
)

bp = Blueprint("admin", __name__)


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
    if not current_user.is_admin:
        return jsonify({"error": "forbidden"}), 403
    return jsonify(compute_insights_summary(_parse_days()))


@bp.route("/insights/low-confidence", methods=["GET"])
@login_required
@limiter.limit("60 per minute")
def insights_low_confidence():
    if not current_user.is_admin:
        return jsonify({"error": "forbidden"}), 403
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
    if not current_user.is_admin:
        return jsonify({"error": "forbidden"}), 403
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
    if not current_user.is_admin:
        return jsonify({"error": "forbidden"}), 403
    try:
        lim = int(request.args.get("limit", "30"))
    except (TypeError, ValueError):
        lim = 30
    return jsonify(chunk_analytics(_parse_days(), lim))
