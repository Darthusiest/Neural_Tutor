"""Persist security audit events (server-side only)."""

from __future__ import annotations

import json
from typing import Any

from flask import Request, has_request_context, request

from app.extensions import db
from app.models import AuditEvent


def log_audit_event(
    event_type: str,
    *,
    actor_user_id: int | None = None,
    actor_email: str | None = None,
    severity: str = "info",
    metadata: dict[str, Any] | None = None,
    req: Request | None = None,
) -> None:
    """Best-effort insert; never raises to callers (logs on failure)."""
    try:
        meta_json: str | None = None
        if metadata:
            meta_json = json.dumps(metadata, default=str)[:8000]
        r = req or (request if has_request_context() else None)
        ip = (r.remote_addr if r else None) or None
        ua = (r.headers.get("User-Agent", "")[:500] if r else None) or None
        row = AuditEvent(
            actor_user_id=actor_user_id,
            actor_email=(actor_email[:255] if actor_email else None),
            event_type=event_type[:64],
            severity=severity[:16] if severity else None,
            ip=ip[:64] if ip else None,
            user_agent=ua,
            metadata_json=meta_json,
        )
        db.session.add(row)
        db.session.commit()
    except Exception:
        db.session.rollback()
        try:
            from flask import current_app

            current_app.logger.exception("audit_log_failed event_type=%s", event_type)
        except Exception:
            pass
