"""ORM model for the ``audit_events`` table — append-only **security event** rows (no secrets).

The physical table name stays ``audit_events`` (migration **005**). Columns ``actor_user_id`` /
``actor_email`` mean **the user who triggered the event** (email is denormalized so we can log
events before a user row is resolved, e.g. failed login).

For **inserting** rows, use :func:`app.services.security_logging.log_security_event`.
"""

from __future__ import annotations

from app.extensions import db


class SecurityLogEntry(db.Model):
    __tablename__ = "audit_events"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now(), index=True)
    actor_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    actor_email = db.Column(db.String(255), nullable=True, index=True)
    event_type = db.Column(db.String(64), nullable=False, index=True)
    severity = db.Column(db.String(16), nullable=True)
    ip = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)
    metadata_json = db.Column(db.Text, nullable=True)

    user = db.relationship("User", backref="security_log_entries")
