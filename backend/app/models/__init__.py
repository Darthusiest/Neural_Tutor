"""SQLAlchemy ORM models — public re-exports for ``from app.models import …``.

Layout by domain:

- **Auth / account:** :class:`User`, :class:`EmailVerificationToken`, :class:`PasswordResetToken`
- **Chat:** :class:`ChatSession`, :class:`Message`
- **Course content:** :class:`LectureChunk`
- **Analytics / pipeline (per assistant turn):** :class:`RetrievalLog`, :class:`RetrievalChunkHit`,
  :class:`ResponseVariant`, :class:`Feedback`, :class:`MessageOutcome`
- **Security event log (DB table ``audit_events``):** :class:`SecurityLogEntry` in :mod:`app.models.security_log`

``__all__`` lists symbols intended for external imports; keep it in sync when adding models.
"""

from app.models.analytics import (
    Feedback,
    MessageOutcome,
    ResponseVariant,
    RetrievalChunkHit,
    RetrievalLog,
)
from app.models.security_log import SecurityLogEntry
from app.models.chat import ChatSession, Message
from app.models.content import LectureChunk
from app.models.email_verification import EmailVerificationToken
from app.models.password_reset import PasswordResetToken
from app.models.user import User

__all__ = [
    # auth
    "User",
    "EmailVerificationToken",
    "PasswordResetToken",
    # chat
    "ChatSession",
    "Message",
    # content
    "LectureChunk",
    # analytics / retrieval
    "RetrievalLog",
    "RetrievalChunkHit",
    "ResponseVariant",
    "Feedback",
    "MessageOutcome",
    # security event log (audit_events table)
    "SecurityLogEntry",
]
