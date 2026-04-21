import json
import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SQLITE = f"sqlite:///{_BACKEND_ROOT / 'ling487.db'}"
_DEFAULT_LECTURE_JSON = _BACKEND_ROOT / "data" / "LING487_SUPER_TUTOR.json"
_DEFAULT_PIPELINE_KB_JSON = _BACKEND_ROOT / "data" / "LING487_STRUCTURED_PIPELINE_KB.json"

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

    EMAIL_VERIFICATION_BASE_URL = os.getenv(
        "EMAIL_VERIFICATION_BASE_URL",
        "http://127.0.0.1:5173/verify-email",
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

    # Dense / hybrid retrieval (embedding path requires backfilled vectors; see embed-chunks CLI).
    EMBEDDING_RETRIEVAL_ENABLED = os.getenv("EMBEDDING_RETRIEVAL_ENABLED", "0") == "1"
    RETRIEVAL_HYBRID_ENABLED = os.getenv("RETRIEVAL_HYBRID_ENABLED", "0") == "1"
    EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-3-small")
    HYBRID_LEXICAL_WEIGHT = float(os.getenv("HYBRID_LEXICAL_WEIGHT", "0.45"))
    HYBRID_EMBEDDING_WEIGHT = float(os.getenv("HYBRID_EMBEDDING_WEIGHT", "0.55"))

    # Study: optional structured pipeline + optional LLM polish for study copy (compare/summary/quiz).
    STRUCTURED_STUDY_PIPELINE_ENABLED = os.getenv("STRUCTURED_STUDY_PIPELINE_ENABLED", "0") == "1"
    STUDY_MODE_LLM_POLISH = os.getenv("STUDY_MODE_LLM_POLISH", "0") == "1"

    # Auth: email verification, lockout, audit (see migrations).
    EMAIL_VERIFICATION_REQUIRED = os.getenv("EMAIL_VERIFICATION_REQUIRED", "0") == "1"
    LOGIN_MAX_ATTEMPTS = int(os.getenv("LOGIN_MAX_ATTEMPTS", "8"))
    LOGIN_LOCKOUT_MINUTES = int(os.getenv("LOGIN_LOCKOUT_MINUTES", "15"))

    # Production safety: never expose dev_reset_token in JSON unless explicitly allowed (dev/QA only).
    ALLOW_DEV_RESET_TOKEN_IN_JSON = os.getenv("ALLOW_DEV_RESET_TOKEN_IN_JSON", "0") == "1"
    # True when not in debug and dev token override is off (used with ALLOW_DEV_RESET_TOKEN_IN_JSON).
    PRODUCTION_LIKE = os.getenv("FLASK_ENV", "").lower() == "production" or os.getenv(
        "PRODUCTION_MODE", "0"
    ) == "1"

    # LLM cost analytics (optional caps; tokens from response_variants.token_usage_json).
    LLM_MONTHLY_TOKEN_CAP = int(os.getenv("LLM_MONTHLY_TOKEN_CAP", "0")) or None  # 0 = unset
    LLM_MONTHLY_TOKEN_WARN_FRACTION = float(os.getenv("LLM_MONTHLY_TOKEN_WARN_FRACTION", "0.8"))
    LLM_COST_USD_PER_MTOKENS = float(os.getenv("LLM_COST_USD_PER_MTOKENS", "0") or 0) or None
    LLM_SPIKE_DAY_OVER_DAY_RATIO = float(os.getenv("LLM_SPIKE_DAY_OVER_DAY_RATIO", "2.5"))

    # v2 summary mode: max chunks returned for a single-lecture summary (ranked lexically).
    SUMMARY_MAX_CHUNKS = int(os.getenv("SUMMARY_MAX_CHUNKS", "48"))

    # Chat / pipeline retrieval: chunks requested for grounding (first pass).
    CHAT_RETRIEVAL_TOP_K = max(1, min(int(os.getenv("CHAT_RETRIEVAL_TOP_K", "5")), 100))
    # When validation hard-fails, retry retrieval with top_k + this extra budget.
    PIPELINE_RETRY_TOP_K_EXTRA = max(0, min(int(os.getenv("PIPELINE_RETRY_TOP_K_EXTRA", "6")), 50))

    # OpenAI sampling (primary Course Answer + optional OpenAI boost paths).
    OPENAI_TEMPERATURE_COURSE_ANSWER = float(os.getenv("OPENAI_TEMPERATURE_COURSE_ANSWER", "0.4"))
    OPENAI_TEMPERATURE_BOOST = float(os.getenv("OPENAI_TEMPERATURE_BOOST", "0.45"))
    OPENAI_TEMPERATURE_DEFAULT = float(os.getenv("OPENAI_TEMPERATURE_DEFAULT", "0.5"))

    # Structured reasoning pipeline (concept KB + answer plan + validation).
    KB_JSON_PATH = Path(os.getenv("KB_JSON_PATH", str(_DEFAULT_PIPELINE_KB_JSON)))
    STRUCTURED_PIPELINE_ENABLED = os.getenv("STRUCTURED_PIPELINE_ENABLED", "1") == "1"
    # Entity-scored chunk ranking + per-compare evidence bundles (recommended on).
    ENTITY_EVIDENCE_SCORING_ENABLED = os.getenv("ENTITY_EVIDENCE_SCORING_ENABLED", "1") == "1"
    # Extra retrieval pass when validation hard-fails (wider top_k).
    PIPELINE_RETRIEVAL_RETRY_ENABLED = os.getenv("PIPELINE_RETRIEVAL_RETRY_ENABLED", "1") == "1"
    # Pass section specs + constraints into primary LLM user prompt (when LLM path is used).
    SECTION_CONTRACTS_ENABLED = os.getenv("SECTION_CONTRACTS_ENABLED", "1") == "1"
    # Primary Course Answer: OpenAI when key present. PRIMARY_COURSE_ANSWER_OPENAI wins; else LLM_ANSWER_GENERATION.
    _primary_llm = os.getenv("PRIMARY_COURSE_ANSWER_OPENAI")
    if _primary_llm is None:
        _primary_llm = os.getenv("LLM_ANSWER_GENERATION", "1")
    PRIMARY_COURSE_ANSWER_OPENAI = _primary_llm == "1"
    LLM_ANSWER_GENERATION = PRIMARY_COURSE_ANSWER_OPENAI  # backward-compatible alias

    # Secondary boost: Gemini only by default (see generate_boosted_explanation in gemini_boost).
    # Set OPENAI_BOOST_FALLBACK=1 to call OpenAI for Boosted Explanation when Gemini is missing or fails.
    OPENAI_BOOST_FALLBACK = os.getenv("OPENAI_BOOST_FALLBACK", "0") == "1"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT_SEC", "60"))
    GEMINI_TEMPERATURE_BOOST = float(os.getenv("GEMINI_TEMPERATURE_BOOST", "0.4"))
    GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))


class TestConfig(Config):
    """In-memory SQLite + no CSRF for pytest."""

    TESTING = True
    SECRET_KEY = "test-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    OPENAI_API_KEY = ""
    PRIMARY_COURSE_ANSWER_OPENAI = False
    LLM_ANSWER_GENERATION = False
    GEMINI_API_KEY = ""
    GOOGLE_API_KEY = ""
    OPENAI_BOOST_FALLBACK = False
    EMBEDDING_RETRIEVAL_ENABLED = False
    RETRIEVAL_HYBRID_ENABLED = False
    STRUCTURED_STUDY_PIPELINE_ENABLED = False
    STUDY_MODE_LLM_POLISH = False
    ENTITY_EVIDENCE_SCORING_ENABLED = True
    PIPELINE_RETRIEVAL_RETRY_ENABLED = False
    SECTION_CONTRACTS_ENABLED = True
    EMAIL_VERIFICATION_REQUIRED = False
    ALLOW_DEV_RESET_TOKEN_IN_JSON = False
    PRODUCTION_LIKE = False
