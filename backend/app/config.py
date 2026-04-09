import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SQLITE = f"sqlite:///{_BACKEND_ROOT / 'ling487.db'}"
_DEFAULT_LECTURE_JSON = _BACKEND_ROOT / "data" / "LING487_SUPER_TUTOR.json"


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or _DEFAULT_SQLITE
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"

    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
    RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "")

    # Frontend URL for reset links in emails (no trailing ?; token appended as ?token=...)
    PASSWORD_RESET_BASE_URL = os.getenv(
        "PASSWORD_RESET_BASE_URL",
        "http://127.0.0.1:5173/reset-password",
    ).rstrip("/")

    # Debug-only: include dev_reset_token in JSON when Resend is configured (see AUTH_LOCAL.md).
    DEV_RETURN_RESET_TOKEN = os.getenv("DEV_RETURN_RESET_TOKEN", "0") == "1"

    LECTURE_JSON_PATH = Path(
        os.getenv("LECTURE_JSON_PATH", str(_DEFAULT_LECTURE_JSON))
    )

    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = None
    WTF_CSRF_SSL_STRICT = False

    RATELIMIT_STORAGE_URI = os.getenv("RATELIMIT_STORAGE_URI", "memory://")
