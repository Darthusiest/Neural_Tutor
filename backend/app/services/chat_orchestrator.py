"""Compose retrieval, course answer, optional boost, and persistence for one chat turn."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from flask import current_app

from app.extensions import db
from app.models import (
    ChatSession,
    Message,
    MessageOutcome,
    ResponseVariant,
    RetrievalChunkHit,
    RetrievalLog,
)
from app.services.answers.answer_planning import build_answer_plan
from app.services.generation.boost_triggers import should_use_gemini_boost
from app.services.generation.gemini_boost import generate_gemini_boosted_explanation
from app.services.generation.llm import generate_boosted_explanation as generate_openai_boost_fallback
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import build_structured_query
from app.services.query_understanding import analyze_query
from app.services.conversational_responses import classify_no_match_query, varied_no_chunk_course_answer
from app.services.reasoning_pipeline import PipelineResult, pipeline_diagnostics_dict, run_reasoning_pipeline
from app.services.retrieval_v2 import EnhancedRetrievalResult
from app.services.retrieval import (
    format_course_answer,
    tokenize_query_terms,
)
from app.services.retrieval_v2 import retrieve_enhanced

logger = logging.getLogger(__name__)

# Keywords that suggest the user wants a simpler/different explanation
_CLARIFY_KEYWORDS = frozenset(
    "clarify explain again confused unclear rephrase repeat elaborate".split()
)
_SIMPLER_KEYWORDS = frozenset("simpler simple easier basic dumb layman".split())
_DEEPER_KEYWORDS = frozenset("deeper detail more expand further depth".split())
_EXAMPLE_KEYWORDS = frozenset("example instance sample show demonstrate".split())


def _classify_follow_up(text: str) -> str | None:
    """Heuristic follow-up type from user message text."""
    tokens = set(tokenize_query_terms(text))
    if tokens & _SIMPLER_KEYWORDS:
        return "simpler"
    if tokens & _CLARIFY_KEYWORDS:
        return "clarify"
    if tokens & _DEEPER_KEYWORDS:
        return "deeper"
    if tokens & _EXAMPLE_KEYWORDS:
        return "example"
    return None


def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard-style overlap of query tokens between two strings."""
    ta = set(tokenize_query_terms(a))
    tb = set(tokenize_query_terms(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _populate_previous_outcome(session: ChatSession, current_user_text: str) -> None:
    """Fill in MessageOutcome for the most recent assistant message (if missing)."""
    prev_messages = (
        Message.query.filter_by(session_id=session.id)
        .order_by(Message.created_at.desc())
        .limit(3)
        .all()
    )
    if len(prev_messages) < 2:
        return

    prev_assistant = None
    prev_user_text = None
    for m in prev_messages:
        if prev_assistant is None and m.role == "assistant":
            prev_assistant = m
        elif prev_assistant is not None and m.role == "user":
            prev_user_text = m.content_text
            break

    if prev_assistant is None or prev_assistant.message_outcome is not None:
        return

    follow_up_type = _classify_follow_up(current_user_text)
    was_rephrased = False
    if prev_user_text:
        was_rephrased = _token_overlap_ratio(prev_user_text, current_user_text) > 0.6

    topic_changed = follow_up_type is None and not was_rephrased

    outcome = MessageOutcome(
        message_id=prev_assistant.id,
        had_follow_up=True,
        follow_up_count=1,
        follow_up_type=follow_up_type or ("rephrase" if was_rephrased else "new_topic"),
        was_rephrased=was_rephrased,
        user_changed_topic_after=topic_changed,
        answer_resolved=None,
    )
    db.session.add(outcome)


def _response_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _plan_and_sq_for_gemini_boost(text: str, r: EnhancedRetrievalResult):
    """Build :class:`AnswerPlan` + :class:`StructuredQuery` for legacy (non-structured) chat path."""
    kb = get_kb()
    intent = r.query_intent
    if intent is None:
        intent = analyze_query(text)
    sq = build_structured_query(intent, kb=kb)
    plan = build_answer_plan(sq, r.chunks, r.supporting_chunks, kb=kb)
    return plan, sq


def handle_chat_turn(
    session: ChatSession,
    text: str,
    boost_toggle: bool,
    mode: str,
) -> dict[str, Any]:
    """
    Run retrieval, build course / boosted answers, persist messages and logs.
    Caller must not commit before this; this function commits on success.
    """
    threshold = float(current_app.config.get("CONFIDENCE_THRESHOLD", 0.35))

    session.mode = mode

    # Retroactively populate outcome for the previous assistant message
    _populate_previous_outcome(session, text)

    db.session.add(
        Message(session_id=session.id, role="user", content_text=text, payload_json=None)
    )
    db.session.flush()

    t0 = time.perf_counter()
    structured_on = bool(current_app.config.get("STRUCTURED_PIPELINE_ENABLED"))
    pipeline_extra: dict[str, Any] | None = None
    pr: PipelineResult | None = None
    no_match_kind: str | None = None
    if structured_on:
        pr = run_reasoning_pipeline(text, top_k=5)
        r = pr.enhanced_result
        pipeline_extra = pipeline_diagnostics_dict(pr)
        if not r.chunks:
            no_match_kind = classify_no_match_query(text)
            course_answer = varied_no_chunk_course_answer(no_match_kind)
        else:
            course_answer = pr.course_answer
    else:
        r = retrieve_enhanced(text, top_k=5)
        if not r.chunks:
            no_match_kind = classify_no_match_query(text)
            course_answer = varied_no_chunk_course_answer(no_match_kind)
        else:
            course_answer = format_course_answer(r.chunks)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    diag = r.diagnostics

    low_confidence = r.confidence < threshold
    qt = r.query_intent.query_type if getattr(r, "query_intent", None) else None
    answer_intent = pr.structured_query.answer_intent if pr else None
    subq_n = len(pr.structured_query.sub_questions) if pr else 0
    validation_for_boost = pr.validation if pr else None

    if not r.chunks:
        need_boost, boost_reason = False, None
    else:
        need_boost, boost_reason = should_use_gemini_boost(
            user_query=text,
            confidence=r.confidence,
            validation=validation_for_boost,
            confidence_threshold=threshold,
            boost_toggle=boost_toggle,
            mode=mode,
            query_type=qt,
            answer_intent=answer_intent,
            subquestion_count=subq_n,
        )

    boosted = None
    boost_provider: str | None = None
    boost_usage_meta: dict[str, Any] | None = None
    primary_llm_usage: dict[str, Any] = pr.primary_llm_usage if pr else {}
    primary_for_log = pipeline_extra.get("primary_model") if pipeline_extra else None
    val_sev = (
        pipeline_extra.get("validation", {}).get("severity") if pipeline_extra else None
    )

    if need_boost:
        gemini_key = current_app.config.get("GEMINI_API_KEY") or current_app.config.get(
            "GOOGLE_API_KEY"
        )
        if gemini_key and r.chunks:
            if pr is not None:
                boosted, gmeta = generate_gemini_boosted_explanation(
                    text,
                    course_answer,
                    pr.answer_plan,
                    r.chunks or [],
                    pr.structured_query,
                )
            else:
                plan_l, sq_l = _plan_and_sq_for_gemini_boost(text, r)
                boosted, gmeta = generate_gemini_boosted_explanation(
                    text,
                    course_answer,
                    plan_l,
                    r.chunks or [],
                    sq_l,
                )
            if boosted:
                boost_provider = gmeta.get("provider", "gemini")
                boost_usage_meta = gmeta

        use_openai_fallback = bool(current_app.config.get("OPENAI_BOOST_FALLBACK")) and bool(
            current_app.config.get("OPENAI_API_KEY")
        )
        if not boosted and use_openai_fallback:
            ctx = json.dumps(r.chunks) if r.chunks else "[]"
            boosted, ometa = generate_openai_boost_fallback(text, ctx)
            if boosted:
                boost_provider = "openai"
                boost_usage_meta = ometa

    logger.info(
        "chat_turn structured=%s confidence=%.3f primary_model=%s validation_severity=%s "
        "need_boost=%s boost_reason=%s boost_provider=%s",
        structured_on,
        r.confidence,
        primary_for_log,
        val_sev,
        need_boost,
        boost_reason,
        boost_provider,
    )

    assistant = Message(
        session_id=session.id,
        role="assistant",
        content_text=None,
        payload_json=json.dumps(
            {
                "course_answer": course_answer,
                "boosted_explanation": boosted,
                "confidence": r.confidence,
                "query_type": (
                    r.query_intent.query_type.value
                    if getattr(r, "query_intent", None) and r.query_intent.query_type
                    else None
                ),
                "structured_pipeline": structured_on,
                "pipeline_diagnostics": pipeline_extra,
                "primary_model": pipeline_extra.get("primary_model") if pipeline_extra else None,
                "validation_severity": (
                    pipeline_extra.get("validation", {}).get("severity") if pipeline_extra else None
                ),
                "boost_provider": boost_provider,
                "boost_reason": boost_reason,
                "query_complexity": pipeline_extra.get("query_complexity") if pipeline_extra else None,
                "no_match_kind": no_match_kind,
            }
        ),
    )
    db.session.add(assistant)
    db.session.flush()

    def _token_usage_blob() -> str | None:
        parts: dict[str, Any] = {}
        if primary_llm_usage:
            parts["primary"] = primary_llm_usage
        if boost_usage_meta:
            parts["boost"] = boost_usage_meta
        if not parts:
            return None
        return json.dumps(parts)

    def _primary_log_token_json() -> str | None:
        if not primary_llm_usage:
            return None
        return json.dumps(primary_llm_usage)

    rv_model = None
    rv_provider = None
    if primary_for_log == "openai" and primary_llm_usage:
        rv_model = primary_llm_usage.get("model") or current_app.config.get(
            "OPENAI_CHAT_MODEL", "gpt-4o-mini"
        )
        rv_provider = "openai"
    elif primary_for_log == "rule_based":
        rv_provider = "rule_based"

    # --- RetrievalLog (enriched) ---
    log = RetrievalLog(
        session_id=session.id,
        message_id=assistant.id,
        user_question=text,
        normalized_query=" ".join(diag.query_tokens) if diag else None,
        query_tokens_json=json.dumps(diag.query_tokens) if diag else None,
        detected_topic=r.detected_topic,
        lecture_numbers_detected_json=(
            json.dumps(diag.lecture_numbers_detected) if diag else None
        ),
        retrieval_backend=diag.retrieval_backend if diag else "keyword",
        top_k_requested=diag.top_k_requested if diag else None,
        num_chunks_scored=diag.num_chunks_scored if diag else None,
        num_chunks_hit=diag.num_chunks_hit if diag else None,
        confidence=r.confidence,
        top_score=diag.top_score if diag else None,
        second_score=diag.second_score if diag else None,
        score_margin=diag.score_margin if diag else None,
        query_coverage=diag.query_coverage if diag else None,
        is_low_confidence=low_confidence,
        is_off_topic=len(r.chunks) == 0,
        latency_ms=latency_ms,
        token_usage_json=_primary_log_token_json(),
        query_type_v2=(pipeline_extra.get("answer_intent") if pipeline_extra else None),
        sub_questions_json=json.dumps(pipeline_extra.get("sub_questions", [])) if pipeline_extra else None,
        answer_mode=pipeline_extra.get("answer_mode") if pipeline_extra else None,
        validation_passed=pipeline_extra.get("validation", {}).get("passed") if pipeline_extra else None,
        validation_checks_json=json.dumps(pipeline_extra.get("validation", {})) if pipeline_extra else None,
        generic_answer_flag=(
            bool(pipeline_extra.get("validation", {}).get("flags", {}).get("generic_answer"))
            if pipeline_extra
            else None
        ),
        missing_comparison_side_flag=(
            bool(pipeline_extra.get("validation", {}).get("flags", {}).get("missing_comparison_side"))
            if pipeline_extra
            else None
        ),
        answer_plan_json=json.dumps(pipeline_extra.get("answer_plan", {})) if pipeline_extra else None,
    )
    db.session.add(log)
    db.session.flush()

    # --- RetrievalChunkHit (one per selected chunk) ---
    if diag:
        for hit in diag.chunk_hits:
            db.session.add(
                RetrievalChunkHit(
                    retrieval_log_id=log.id,
                    lecture_chunk_id=hit.chunk_id,
                    rank=hit.rank,
                    score=hit.score,
                    selected_for_answer=True,
                    token_score=hit.token_score,
                    phrase_score=hit.phrase_score,
                    lecture_bonus=hit.lecture_bonus,
                    strong_field_token_score=hit.strong_field_token_score,
                    matched_query_terms=hit.matched_query_terms,
                    phrase_events=hit.phrase_events,
                    field_scores_json=json.dumps(hit.field_scores) if hit.field_scores else None,
                )
            )

    # --- ResponseVariant (enriched) ---
    boost_used = boosted is not None
    rv = ResponseVariant(
        message_id=assistant.id,
        retrieval_log_id=log.id,
        course_answer=course_answer,
        boosted_explanation=boosted,
        boost_used=boost_used,
        boost_reason=boost_reason,
        boost_auto_triggered=bool(need_boost and not boost_toggle and boost_reason),
        boost_toggle_user_selected=boost_toggle,
        model_name=rv_model,
        provider_name=rv_provider,
        course_answer_prompt_version=None,
        boost_prompt_version=None,
        token_usage_json=_token_usage_blob(),
        course_answer_length=len(course_answer),
        boosted_answer_length=len(boosted) if boosted else None,
        response_fingerprint=_response_fingerprint(course_answer),
    )
    db.session.add(rv)
    db.session.commit()

    return {
        "assistant_message_id": assistant.id,
        "course_answer": course_answer,
        "boosted_explanation": boosted,
        "retrieval_confidence": r.confidence,
        "boost_applied": boost_used,
    }
