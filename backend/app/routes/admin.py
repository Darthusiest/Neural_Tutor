from flask import Blueprint, jsonify
from flask_login import current_user, login_required

bp = Blueprint("admin", __name__)


@bp.route("/insights", methods=["GET"])
@login_required
def insights():
    if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
        return jsonify({"error": "forbidden"}), 403
    return jsonify(
        {
            "stub": True,
            "message": "Aggregate confusing topics, latency, token usage, etc. (next step).",
        }
    )
