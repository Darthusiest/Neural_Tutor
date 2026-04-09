from flask import Blueprint, jsonify
from flask_login import current_user, login_required

from app.extensions import limiter

bp = Blueprint("admin", __name__)


@bp.route("/insights", methods=["GET"])
@login_required
@limiter.limit("120 per minute")
def insights():
    if not current_user.is_admin:
        return jsonify({"error": "forbidden"}), 403
    return jsonify(
        {
            "stub": True,
            "message": "Aggregate confusing topics, latency, token usage, etc. (next step).",
        }
    )
