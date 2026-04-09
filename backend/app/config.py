import json
import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SQLITE = f"sqlite:///{_BACKEND_ROOT / 'ling487.db'}"
_DEFAULT_LECTURE_JSON = _BACKEND_ROOT / "data" / "LING487_SUPER_TUTOR.json"

_DEFAULT_RETRIEVAL_FIELD_WEIGHTS: dict[str, float] = {
    "topic": 3.0,
    "keywords": 2.5,
    "sample_questions": 2.0,
    "clean_explanation": 1.2,
    "source_excerpt": 1.0,
    "sample_answer": 0.7,
}
_DEFAULT_RETRIEVAL_PHRASE_FIELD_WEIGHT: dict[str, float] = {
    "topic": 1.0,
    "keywords": 0.95,
    "sample_questions": 0.85,
    "clean_explanation": 0.55,
    "source_excerpt": 0.5,
    "sample_answer": 0.35,
}


def _merge_weight_dict(default: dict[str, float], env_json: str | None) -> dict[str, float]:
    out = dict(default)
    raw = (env_json or "").strip()
    if not raw:
        return out
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                out[str(k)] = float(v)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return out


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL") or _DEFAULT_SQLITE
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"

    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5173")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "60"))
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

    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

    # Auto keyword list length cap (section keywords = curated + derived, capped here).
    LECTURE_KEYWORD_CAP = int(os.getenv("LECTURE_KEYWORD_CAP", "48"))

    # Lexical retrieval field weights (JSON object, partial overrides merge into defaults).
    RETRIEVAL_FIELD_WEIGHTS: dict[str, float] = _merge_weight_dict(
        _DEFAULT_RETRIEVAL_FIELD_WEIGHTS,
        os.getenv("RETRIEVAL_FIELD_WEIGHTS_JSON"),
    )
    RETRIEVAL_PHRASE_FIELD_WEIGHT: dict[str, float] = _merge_weight_dict(
        _DEFAULT_RETRIEVAL_PHRASE_FIELD_WEIGHT,
        os.getenv("RETRIEVAL_PHRASE_FIELD_WEIGHT_JSON"),
    )

    # Future: dense / hybrid retrieval (not implemented in v1 keyword path).
    RETRIEVAL_HYBRID_ENABLED = os.getenv("RETRIEVAL_HYBRID_ENABLED", "0") == "1"
    EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "")

    # v2 summary mode: max chunks returned for a single-lecture summary (ranked lexically).
    SUMMARY_MAX_CHUNKS = int(os.getenv("SUMMARY_MAX_CHUNKS", "48"))


class TestConfig(Config):
    """In-memory SQLite + no CSRF for pytest."""

    TESTING = True
    SECRET_KEY = "test-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
