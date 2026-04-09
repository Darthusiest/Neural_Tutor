from __future__ import annotations

import logging
import re
import time
from typing import Any

from email_validator import EmailNotValidError, validate_email
from flask import Request, request
from werkzeug.exceptions import BadRequest
from werkzeug.security import check_password_hash, generate_password_hash

log = logging.getLogger("auth.security")

_TIMING_HASH = generate_password_hash(
    "__auth_timing__", method="pbkdf2:sha256", salt_length=16
)

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{20,256}$")


def parse_request_json(req: Request) -> tuple[dict[str, Any] | None, tuple | None]:
    """
    Parse JSON body. Returns (data, None) or (None, (jsonify_response, status_code)).
    """
    from flask import jsonify

    ct = (req.content_type or "").split(";")[0].strip().lower()
    if ct != "application/json":
        return None, (jsonify({"error": "Content-Type must be application/json"}), 415)
    try:
        data = req.get_json(force=False, silent=False)
    except BadRequest:
        return None, (jsonify({"error": "invalid JSON"}), 400)
    if data is not None and not isinstance(data, dict):
        return None, (jsonify({"error": "JSON body must be an object"}), 400)
    return data if data is not None else {}, None


def validate_email_format(email: str) -> str | None:
    """Return error message or None if valid."""
    if not email or not isinstance(email, str):
        return "valid email required"
    email = email.strip().lower()
    if len(email) > 254:
        return "email is too long"
    try:
        validate_email(email, check_deliverability=False)
    except EmailNotValidError:
        return "invalid email format"
    return None


def validate_password_strength(password: str) -> str | None:
    """Return error message or None if password meets policy."""
    if not password or not isinstance(password, str):
        return "password required"
    if len(password) < 8:
        return "password must be at least 8 characters"
    if len(password) > 256:
        return "password is too long"
    if not re.search(r"[A-Z]", password):
        return "password must contain an uppercase letter"
    if not re.search(r"[a-z]", password):
        return "password must contain a lowercase letter"
    if not re.search(r"\d", password):
        return "password must contain a digit"
    if not re.search(r'[!@#$%^&*(),.?":{}\[\]|_\-\\/<>`~+=\';]', password):
        return "password must contain a special character"
    return None


def validate_reset_token_format(token: str) -> str | None:
    if not token or not isinstance(token, str):
        return "token required"
    if not _TOKEN_RE.match(token):
        return "invalid token format"
    return None


def timing_pad(start: float, minimum_seconds: float) -> None:
    """Reduce timing side-channels by padding fast code paths."""
    elapsed = time.monotonic() - start
    if elapsed < minimum_seconds:
        time.sleep(minimum_seconds - elapsed)


def security_log(event: str, **fields: Any) -> None:
    """Structured security / audit logging."""
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
    log.warning("%s %s ip=%s", event, parts, request.remote_addr if request else "-")


def burn_auth_timing_budget() -> None:
    """Fixed-cost work to narrow timing differences between auth code paths."""
    check_password_hash(_TIMING_HASH, "invalid")


def reject_login_password_check() -> None:
    """Approximate cost of a failed password check when the user does not exist."""
    check_password_hash(_TIMING_HASH, "invalid")
