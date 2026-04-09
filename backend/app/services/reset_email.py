"""
Send password-reset messages via Resend.

All configuration comes from Flask `current_app.config` (env-backed). Never pass
API keys as function arguments from callers.
"""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlencode

import resend
from flask import current_app


class ResetEmailResult(Enum):
    """Outcome of attempting to send a reset email."""

    SENT = "sent"
    SKIPPED_NO_CONFIG = "skipped"
    FAILED = "failed"


def send_password_reset_email(to_email: str, plaintext_token: str) -> ResetEmailResult:
    """
    Queue a single transactional email with a link to the frontend reset page.

    Returns:
        SENT if Resend accepted the message.
        SKIPPED_NO_CONFIG if API key or from-address is missing.
        FAILED if Resend raised or the request failed (caller should log; HTTP layer
        should still use a generic forgot-password response).
    """
    api_key = (current_app.config.get("RESEND_API_KEY") or "").strip()
    from_email = (current_app.config.get("RESEND_FROM_EMAIL") or "").strip()
    base_url = (current_app.config.get("PASSWORD_RESET_BASE_URL") or "").strip().rstrip("/")

    if not api_key or not from_email:
        current_app.logger.warning(
            "Password reset email skipped: set RESEND_API_KEY and RESEND_FROM_EMAIL"
        )
        return ResetEmailResult.SKIPPED_NO_CONFIG

    if not base_url:
        current_app.logger.error("PASSWORD_RESET_BASE_URL is empty; cannot build reset link")
        return ResetEmailResult.FAILED

    query = urlencode({"token": plaintext_token})
    link = f"{base_url}?{query}"

    subject = "Reset your LING 487 Tutor password"
    html = (
        "<p>You requested a password reset for your LING 487 Tutor account.</p>"
        f'<p><a href="{link}">Set a new password</a></p>'
        "<p>If you did not request this, you can ignore this email.</p>"
    )
    text = (
        "You requested a password reset for your LING 487 Tutor account.\n\n"
        f"Open this link to set a new password:\n{link}\n\n"
        "If you did not request this, ignore this email.\n"
    )

    resend.api_key = api_key
    try:
        resend.Emails.send(
            {
                "from": from_email,
                "to": [to_email],
                "subject": subject,
                "html": html,
                "text": text,
            }
        )
        return ResetEmailResult.SENT
    except Exception:
        current_app.logger.exception(
            "Resend failed while sending password reset to %s", to_email
        )
        return ResetEmailResult.FAILED


def resend_reset_is_configured() -> bool:
    """True when both Resend env vars are non-empty (used for dev token policy)."""
    api_key = (current_app.config.get("RESEND_API_KEY") or "").strip()
    from_email = (current_app.config.get("RESEND_FROM_EMAIL") or "").strip()
    return bool(api_key and from_email)
