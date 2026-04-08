import secrets
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import db
from app.models import PasswordResetToken, User

bp = Blueprint("auth", __name__)


@bp.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"error": "email and password required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "email already registered"}), 409
    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({"user": {"id": user.id, "email": user.email}}), 201


@bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "invalid credentials"}), 401
    login_user(user)
    return jsonify({"user": {"id": user.id, "email": user.email}})


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({"ok": True})


@bp.route("/me", methods=["GET"])
def me():
    if not current_user.is_authenticated:
        return jsonify({"user": None})
    return jsonify(
        {
            "user": {
                "id": current_user.id,
                "email": current_user.email,
                "is_admin": current_user.is_admin,
            }
        }
    )


@bp.route("/forgot-password", methods=["POST"])
def forgot_password():
    """Request reset email via Resend (wire up in a later step)."""
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email required"}), 400
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"message": "if that account exists, email was sent"}), 200
    raw = secrets.token_urlsafe(32)
    pr = PasswordResetToken(
        user_id=user.id,
        token_hash=_hash_token(raw),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    db.session.add(pr)
    db.session.commit()
    from flask import current_app

    body = {
        "message": "If an account exists for this email, a reset link has been sent.",
    }
    if current_app.debug:
        body["dev_reset_token"] = raw
    return jsonify(body), 200


@bp.route("/reset-password", methods=["POST"])
def reset_password():
    data = request.get_json(silent=True) or {}
    token = data.get("token") or ""
    new_password = data.get("password") or ""
    if not token or not new_password:
        return jsonify({"error": "token and password required"}), 400
    th = _hash_token(token)
    pr = PasswordResetToken.query.filter_by(token_hash=th).first()
    if not pr or pr.used_at is not None or pr.expires_at < datetime.now(timezone.utc):
        return jsonify({"error": "invalid or expired token"}), 400
    user = db.session.get(User, pr.user_id)
    if not user:
        return jsonify({"error": "user not found"}), 400
    user.set_password(new_password)
    pr.used_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({"ok": True})


def _hash_token(raw: str) -> str:
    import hashlib

    return hashlib.sha256(raw.encode()).hexdigest()
