import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required, login_user, logout_user
from flask_wtf.csrf import generate_csrf
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.extensions import db, limiter
from app.models import PasswordResetToken, User
from app.services.reset_email import (
    ResetEmailResult,
    resend_reset_is_configured,
    send_password_reset_email,
)
from app.utils.security import (
    burn_auth_timing_budget,
    parse_request_json,
    reject_login_password_check,
    security_log,
    timing_pad,
    validate_email_format,
    validate_password_strength,
    validate_reset_token_format,
)

bp = Blueprint("auth", __name__)

DUMMY_TOKEN_HASH = "0" * 64
_FORGET_MIN_SECONDS = 0.085
_RESET_MIN_SECONDS = 0.085


@bp.route("/csrf", methods=["GET"])
@limiter.limit("60 per minute")
def get_csrf_token():
    return jsonify({"csrf_token": generate_csrf()})


@bp.route("/register", methods=["POST"])
@limiter.limit("5 per minute")
def register():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if msg := validate_email_format(email):
        return jsonify({"error": msg}), 400
    if msg := validate_password_strength(password):
        return jsonify({"error": msg}), 400

    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        security_log("register_conflict", email=email)
        return jsonify({"error": "email already registered"}), 409
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.exception("register failed")
        security_log("register_db_error", email=email)
        return jsonify({"error": "registration failed"}), 500

    login_user(user)
    return jsonify({"user": {"id": user.id, "email": user.email, "is_admin": user.is_admin}}), 201


@bp.route("/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    data, err = parse_request_json(request)
    if err:
        return err
    assert data is not None
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if validate_email_format(email):
        security_log("login_invalid_email_shape")
        return jsonify({"error": "invalid credentials"}), 401

    user = User.query.filter_by(email=email).first()
    if user:
        ok = user.check_password(password)
    else:
        reject_login_password_check()
        ok = False
    if not ok:
        security_log("login_failed", email=email)
        return jsonify({"error": "invalid credentials"}), 401

    login_user(user)
    return jsonify({"user": {"id": user.id, "email": user.email, "is_admin": user.is_admin}})


@bp.route("/logout", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
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
@limiter.limit("5 per minute")
def forgot_password():
    """Request password reset. Sends email via Resend when configured; uniform JSON body."""
    import time as _time

    t0 = _time.monotonic()
    burn_auth_timing_budget()
    try:
        data, err = parse_request_json(request)
        if err:
            return err
        assert data is not None
        email = (data.get("email") or "").strip().lower()

        if msg := validate_email_format(email):
            security_log("forgot_invalid_email")
            body = {
                "message": "If an account exists for this email, a reset link has been sent."
            }
            return jsonify(body), 200

        user = User.query.filter_by(email=email).first()
        body = {
            "message": "If an account exists for this email, a reset link has been sent.",
        }
        if user:
            raw = secrets.token_urlsafe(32)
            PasswordResetToken.query.filter(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.used_at.is_(None),
            ).delete(synchronize_session=False)
            pr = PasswordResetToken(
                user_id=user.id,
                token_hash=_hash_token(raw),
                expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            db.session.add(pr)
            try:
                db.session.commit()
            except SQLAlchemyError:
                db.session.rollback()
                current_app.logger.exception("forgot_password_commit_failed")
                security_log("forgot_db_error", email=email)
            else:
                email_result = send_password_reset_email(user.email, raw)
                if email_result == ResetEmailResult.FAILED:
                    security_log("forgot_email_failed", email=email)
                resend_ok = resend_reset_is_configured()
                force_dev = current_app.config.get("DEV_RETURN_RESET_TOKEN", False)
                if current_app.debug and (not resend_ok or force_dev):
                    body = dict(body)
                    body["dev_reset_token"] = raw
        return jsonify(body), 200
    finally:
        timing_pad(t0, _FORGET_MIN_SECONDS)


@bp.route("/reset-password", methods=["POST"])
@limiter.limit("10 per minute")
def reset_password():
    import time as _time

    t0 = _time.monotonic()
    burn_auth_timing_budget()
    try:
        data, err = parse_request_json(request)
        if err:
            return err
        assert data is not None
        token = (data.get("token") or "").strip()
        new_password = data.get("password") or ""

        if msg := validate_reset_token_format(token):
            security_log("reset_invalid_token_shape")
            return jsonify({"error": "invalid or expired token"}), 400

        if msg := validate_password_strength(new_password):
            return jsonify({"error": msg}), 400

        th = _hash_token(token)
        stored_hex = DUMMY_TOKEN_HASH
        pr = PasswordResetToken.query.filter_by(token_hash=th).first()
        if pr is not None:
            stored_hex = pr.token_hash

        if not hmac.compare_digest(stored_hex.encode("utf-8"), th.encode("utf-8")):
            security_log("reset_token_mismatch")
            return jsonify({"error": "invalid or expired token"}), 400

        assert pr is not None

        now = datetime.now(timezone.utc)
        if pr.used_at is not None or pr.expires_at < now:
            security_log("reset_token_stale", token_id=pr.id)
            return jsonify({"error": "invalid or expired token"}), 400

        user = db.session.get(User, pr.user_id)
        if not user:
            db.session.rollback()
            security_log("reset_user_missing", token_id=pr.id)
            return jsonify({"error": "invalid or expired token"}), 400

        user.set_password(new_password)
        pr.used_at = now
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            current_app.logger.exception("reset_password_commit_failed")
            security_log("reset_db_error")
            return jsonify({"error": "password reset failed"}), 500

        return jsonify({"ok": True}), 200
    finally:
        timing_pad(t0, _RESET_MIN_SECONDS)


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
