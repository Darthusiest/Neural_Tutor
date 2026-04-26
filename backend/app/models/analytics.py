"""
Optimization-oriented analytics models for the Neural Tutor chatbot.

Five tables capture the full decision chain from query → retrieval → response → feedback → outcome,
structured so downstream analysis can improve retrieval ranking, boost triggering, prompt quality,
and dataset coverage without additional schema work.

Table names are ``__tablename__`` on each class (e.g. ``retrieval_logs``, ``response_variants``).
Python class names are for code only; renaming them would require a DB migration and import updates.
"""

from __future__ import annotations

from app.extensions import db


class RetrievalLog(db.Model):
    """One retrieval event per assistant message.

    Stores query-level features and aggregate scoring signals needed to
    calibrate CONFIDENCE_THRESHOLD, detect weak-coverage topics, and compare
    retrieval backends.

    ``message_id`` is unique: at most one log row per assistant reply that ran retrieval.
    """

    __tablename__ = "retrieval_logs"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("chat_sessions.id"), nullable=True, index=True
    )
    message_id = db.Column(
        db.Integer, db.ForeignKey("messages.id"), nullable=True, unique=True, index=True
    )

    # Query features (from the user turn that led to this assistant reply)
    user_question = db.Column(db.Text, nullable=False)
    # Space-joined lexical tokens from retrieval diagnostics (for analytics, not the raw string).
    normalized_query = db.Column(db.Text, nullable=True)
    query_tokens_json = db.Column(db.Text, nullable=True)
    # Short label from top chunk topic — heuristic, not a separate classifier.
    detected_topic = db.Column(db.String(512), nullable=True)
    lecture_numbers_detected_json = db.Column(db.Text, nullable=True)

    # Retrieval config snapshot
    retrieval_backend = db.Column(db.String(32), nullable=True)
    top_k_requested = db.Column(db.SmallInteger, nullable=True)

    # Aggregate scoring signals
    num_chunks_scored = db.Column(db.Integer, nullable=True)
    num_chunks_hit = db.Column(db.Integer, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    top_score = db.Column(db.Float, nullable=True)
    second_score = db.Column(db.Float, nullable=True)
    score_margin = db.Column(db.Float, nullable=True)
    query_coverage = db.Column(db.Float, nullable=True)

    # Flags for fast filtering in analytics queries
    is_low_confidence = db.Column(db.Boolean, nullable=True)
    is_off_topic = db.Column(db.Boolean, nullable=True)

    # Performance
    latency_ms = db.Column(db.Integer, nullable=True)
    token_usage_json = db.Column(db.Text, nullable=True)

    # Deprecated: chunk ids are now in RetrievalChunkHit.
    # Kept nullable for existing rows; not written by new code.
    retrieved_chunk_ids = db.Column(db.Text, nullable=True)

    # Structured reasoning pipeline (optional; chat when STRUCTURED_PIPELINE_ENABLED).
    # "v2" distinguishes pipeline answer_intent / plan from legacy query_type strings elsewhere.
    query_type_v2 = db.Column(db.String(64), nullable=True)
    sub_questions_json = db.Column(db.Text, nullable=True)
    answer_mode = db.Column(db.String(64), nullable=True)
    validation_passed = db.Column(db.Boolean, nullable=True)
    validation_checks_json = db.Column(db.Text, nullable=True)
    generic_answer_flag = db.Column(db.Boolean, nullable=True)
    missing_comparison_side_flag = db.Column(db.Boolean, nullable=True)
    answer_plan_json = db.Column(db.Text, nullable=True)

    # Mode routing (chat / quiz / compare / summary) — mirrors the `mode` block on assistant
    # payload_json so wrong-routing can be debugged with plain SQL without JSON parsing.
    mode_detected = db.Column(db.String(16), nullable=True)
    mode_effective = db.Column(db.String(16), nullable=True)
    mode_overridden = db.Column(db.Boolean, nullable=True)
    mode_confidence = db.Column(db.Float, nullable=True)
    mode_ambiguous = db.Column(db.Boolean, nullable=True)
    mode_signals_json = db.Column(db.Text, nullable=True)
    mode_request_source = db.Column(db.String(16), nullable=True)

    created_at = db.Column(db.DateTime, server_default=db.func.now())

    session = db.relationship("ChatSession")
    message = db.relationship("Message", back_populates="retrieval_log")
    chunk_hits = db.relationship(
        "RetrievalChunkHit",
        back_populates="retrieval_log",
        cascade="all, delete-orphan",
        order_by="RetrievalChunkHit.rank",
    )
    # One optional ResponseVariant (generated text) linked to this retrieval.
    response_variant = db.relationship(
        "ResponseVariant", back_populates="retrieval_log", uselist=False
    )


class RetrievalChunkHit(db.Model):
    """One ranked lecture chunk in the retrieval result set for a :class:`RetrievalLog`.

    "Hit" means this chunk was scored and listed (rank order), not necessarily a click or web hit.
    Use for chunk-level analytics: weak outcomes, field weights, content quality.
    """

    __tablename__ = "retrieval_chunk_hits"
    __table_args__ = (
        db.Index("ix_chunk_hit_log_rank", "retrieval_log_id", "rank"),
    )

    id = db.Column(db.Integer, primary_key=True)
    retrieval_log_id = db.Column(
        db.Integer,
        db.ForeignKey("retrieval_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lecture_chunk_id = db.Column(
        db.Integer,
        db.ForeignKey("lecture_chunks.id"),
        nullable=False,
        index=True,
    )

    rank = db.Column(db.SmallInteger, nullable=False)
    score = db.Column(db.Float, nullable=False)
    # Whether this chunk was treated as input to the final answer path (vs diagnostic-only listing).
    selected_for_answer = db.Column(db.Boolean, nullable=False, default=True)

    # Score decomposition from _ScoreParts
    token_score = db.Column(db.Float, nullable=True)
    phrase_score = db.Column(db.Float, nullable=True)
    lecture_bonus = db.Column(db.Float, nullable=True)
    strong_field_token_score = db.Column(db.Float, nullable=True)
    matched_query_terms = db.Column(db.SmallInteger, nullable=True)
    phrase_events = db.Column(db.SmallInteger, nullable=True)

    # Per-field token score breakdown (JSON dict: field_name -> float)
    field_scores_json = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, server_default=db.func.now())

    retrieval_log = db.relationship("RetrievalLog", back_populates="chunk_hits")
    lecture_chunk = db.relationship("LectureChunk")


class ResponseVariant(db.Model):
    """Stored primary (and optional boost) text for one assistant message + generation metadata.

    Name is historical ("variant"); in practice there is one row per assistant message
    (``message_id`` unique). Used for analytics: boost reasons, model/provider, tokens, fingerprint.
    """

    __tablename__ = "response_variants"

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(
        db.Integer, db.ForeignKey("messages.id"), unique=True, nullable=False
    )
    retrieval_log_id = db.Column(
        db.Integer, db.ForeignKey("retrieval_logs.id"), nullable=True, index=True
    )

    # Generated content
    course_answer = db.Column(db.Text, nullable=False)
    boosted_explanation = db.Column(db.Text, nullable=True)

    # Boost decomposition
    boost_used = db.Column(db.Boolean, nullable=True)
    boost_reason = db.Column(db.String(64), nullable=True)
    boost_auto_triggered = db.Column(db.Boolean, nullable=True)
    boost_toggle_user_selected = db.Column(db.Boolean, nullable=True)

    # Generation metadata
    model_name = db.Column(db.String(128), nullable=True)
    provider_name = db.Column(db.String(64), nullable=True)
    course_answer_prompt_version = db.Column(db.String(32), nullable=True)
    boost_prompt_version = db.Column(db.String(32), nullable=True)
    token_usage_json = db.Column(db.Text, nullable=True)

    # Length / fingerprint for cost and duplicate analysis
    course_answer_length = db.Column(db.Integer, nullable=True)
    boosted_answer_length = db.Column(db.Integer, nullable=True)
    response_fingerprint = db.Column(db.String(40), nullable=True, index=True)

    created_at = db.Column(db.DateTime, server_default=db.func.now())

    message = db.relationship("Message", back_populates="response_variant")
    retrieval_log = db.relationship("RetrievalLog", back_populates="response_variant")


class Feedback(db.Model):
    """Explicit user feedback on an assistant message (API / UI), not inferred behavior.

    One row per assistant message at most (``message_id`` unique).
    """

    __tablename__ = "feedback"

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(
        db.Integer, db.ForeignKey("messages.id"), unique=True, nullable=False
    )

    # Thumbs / preference: small string codes from the client (e.g. up/down), not booleans.
    course_thumb = db.Column(db.String(8), nullable=True)
    boost_thumb = db.Column(db.String(8), nullable=True)
    preferred = db.Column(db.String(16), nullable=True)

    # Enriched signals
    helpfulness_rating = db.Column(db.SmallInteger, nullable=True)
    resolved = db.Column(db.Boolean, nullable=True)
    follow_up_required = db.Column(db.Boolean, nullable=True)
    follow_up_type = db.Column(db.String(32), nullable=True)
    explicit_confusion_flag = db.Column(db.Boolean, nullable=True)
    feedback_note = db.Column(db.Text, nullable=True)
    preference_strength = db.Column(db.String(16), nullable=True)

    created_at = db.Column(db.DateTime, server_default=db.func.now())

    message = db.relationship("Message", back_populates="feedback")


class MessageOutcome(db.Model):
    """Inferred follow-up behavior *after* an assistant message (filled on the next user turn).

    ``message_id`` refers to the **assistant** message being judged, not the follow-up user message.
    Heuristic only (token overlap, keywords); not a classifier model.
    """

    __tablename__ = "message_outcomes"

    id = db.Column(db.Integer, primary_key=True)
    message_id = db.Column(
        db.Integer, db.ForeignKey("messages.id"), unique=True, nullable=False, index=True
    )

    had_follow_up = db.Column(db.Boolean, nullable=True)
    follow_up_count = db.Column(db.SmallInteger, nullable=True)
    follow_up_type = db.Column(db.String(32), nullable=True)
    was_rephrased = db.Column(db.Boolean, nullable=True)
    user_changed_topic_after = db.Column(db.Boolean, nullable=True)
    answer_resolved = db.Column(db.Boolean, nullable=True)

    created_at = db.Column(db.DateTime, server_default=db.func.now())

    message = db.relationship("Message", back_populates="message_outcome")
