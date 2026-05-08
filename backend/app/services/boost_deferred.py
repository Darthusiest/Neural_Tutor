"""Lazy-compute constrained Boosted Explanation (POST /api/chat/boost/<id>)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from flask import current_app

from app.extensions import db
from app.models import LectureChunk, Message, ResponseVariant, RetrievalLog
from app.services.answers.concept_constraints import (
    ConceptConstraints,
    build_concept_constraints,
    collect_allowed_evidence_lines,
    line_has_forbidden,
)
from app.services.generation.boost_provider import boost_provider_chain
from app.services.generation.gemini_boost import generate_gemini_constrained_boost
from app.services.generation.llm import generate_openai_constrained_boost
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import build_structured_query
from app.services.query_understanding import QueryType, analyze_query

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_SHALLOW_DRAFT_CHARS = 600
_SHALLOW_EVIDENCE_LINES = 2

_MARKER_PHRASES = (
    "a useful clarification is",
    "in standard speech processing terms",
    "in standard machine learning terms",
    "more generally",
)


def course_answer_is_shallow(course_answer: str, allowed_evidence: list[str]) -> bool:
    """True when deferred boost may consider adding marker-phrased external clarification."""
    body = (course_answer or "").strip()
    if len(body) < _SHALLOW_DRAFT_CHARS:
        return True
    if len(allowed_evidence) <= _SHALLOW_EVIDENCE_LINES:
        return True
    return False


def _evidence_fallback_from_course_answer(course_answer: str) -> list[str]:
    """
    When retrieval yields no chunk lines pure enough for ``collect_allowed_evidence_lines``,
    ground prompt + validation on sentences from the rendered Course Answer block.
    """
    body = (course_answer or "").strip()
    low = body.lower()
    if low.startswith("course answer:"):
        body = body.split(":", 1)[1].strip()
    if not body:
        return []
    out: list[str] = []
    for raw in body.replace("\r\n", "\n").split("\n"):
        ln = raw.strip().lstrip("-*•").strip()
        if len(ln) >= 12:
            out.append(ln[:400])
        if len(out) >= 5:
            break
    if out:
        return out
    return [body[:800]] if body else []


def _sentence_evidence_overlap_ok(sentence: str, allowed_blob: str) -> bool:
    """Cheap overlap gate for non-marker sentences."""
    sn = sentence.strip().lower()
    if len(sn) < 12:
        return True
    words = [w for w in re.findall(r"[a-z]{3,}", sn) if len(w) >= 3]
    if not words:
        return True
    hits = sum(1 for w in words if w in allowed_blob)
    return hits >= max(1, len(words) // 4)


def _boost_body_after_prefix(text: str) -> str:
    t = (text or "").strip()
    low = t.lower()
    marker = "boosted explanation:"
    idx = low.find(marker)
    if idx < 0:
        return t
    return t[idx + len(marker) :].lstrip()


def _has_unmarked_external_addition(stripped: str, allowed_blob: str) -> bool:
    body = _boost_body_after_prefix(stripped)
    for raw in _SENTENCE_SPLIT.split(body):
        piece = raw.strip()
        if not piece:
            continue
        sl = piece.lower()
        if any(sl.startswith(m) for m in _MARKER_PHRASES):
            continue
        if not _sentence_evidence_overlap_ok(piece, allowed_blob):
            return True
    return False


def _cap_marker_prefixed_sentences(text: str, max_markers: int = 2) -> str:
    """Keep only the first ``max_markers`` sentences that begin with a framing phrase."""
    body = (text or "").strip()
    if not body.lower().startswith("boosted explanation"):
        body = "Boosted Explanation:\n\n" + body
    prefix = "Boosted Explanation:"
    rest = body[len(prefix) :].lstrip()
    sentences = _SENTENCE_SPLIT.split(rest)
    kept: list[str] = []
    marker_count = 0
    for s in sentences:
        piece = s.strip()
        if not piece:
            continue
        sl = piece.lower()
        is_marker = any(sl.startswith(m) for m in _MARKER_PHRASES)
        if is_marker:
            if marker_count >= max_markers:
                continue
            marker_count += 1
        kept.append(piece)
    return prefix + "\n\n" + " ".join(kept)


def _marker_segments_joined(body: str) -> str:
    parts: list[str] = []
    for raw in _SENTENCE_SPLIT.split(body):
        piece = raw.strip()
        if not piece:
            continue
        sl = piece.lower()
        if any(sl.startswith(m) for m in _MARKER_PHRASES):
            parts.append(piece)
    return " ".join(parts)


def _too_many_novel_terms_in_marker_segments(stripped: str, allowed_blob: str) -> bool:
    body = _boost_body_after_prefix(stripped)
    marker_blob = _marker_segments_joined(body)
    if not marker_blob.strip():
        return False
    ab = allowed_blob.lower()
    tokens: set[str] = set()
    for m in re.finditer(r"\b[A-Z][a-z]{3,}\b|\b[A-Z]{2,}\b", marker_blob):
        tok = m.group(0)
        tl = tok.lower()
        if tl in ab:
            continue
        tokens.add(tl)
    return len(tokens) > 1


def _lecture_chunk_to_dict(row: LectureChunk) -> dict[str, Any]:
    src = row.source_excerpt or ""
    return {
        "id": row.id,
        "chunk_key": row.chunk_key,
        "lecture_number": row.lecture_number,
        "topic": row.topic,
        "keywords": row.keywords,
        "source_excerpt": src,
        "source_text": src,
        "clean_explanation": row.clean_explanation or "",
        "sample_questions": row.sample_questions,
        "sample_answer": row.sample_answer,
        "chunk_type": row.chunk_type,
        "concept_family": row.concept_family,
    }


def chunks_from_retrieval_log(log: RetrievalLog) -> list[dict[str, Any]]:
    """Rebuild retrieval chunk dicts from persisted hits (rank order)."""
    out: list[dict[str, Any]] = []
    for hit in log.chunk_hits:
        if not hit.selected_for_answer:
            continue
        lc = hit.lecture_chunk
        if lc is None:
            lc = db.session.get(LectureChunk, hit.lecture_chunk_id)
        if lc is None:
            continue
        out.append(_lecture_chunk_to_dict(lc))
    return out


def _target_concept_label(constraints: ConceptConstraints, sq_dict: dict[str, Any] | None) -> str:
    if constraints.target_concepts:
        cid = constraints.target_concepts[0]
        kb = get_kb()
        meta = kb.get_concept_by_id(cid)
        if meta:
            return meta.name or cid
        return cid
    cids = (sq_dict or {}).get("concept_ids") or []
    if cids:
        return str(cids[0])
    return (sq_dict or {}).get("raw_query") or "topic"


def _strip_forbidden_sentences(text: str, constraints: ConceptConstraints) -> str:
    body = (text or "").strip()
    if not body.lower().startswith("boosted explanation"):
        body = "Boosted Explanation:\n\n" + body
    prefix = "Boosted Explanation:"
    rest = body[len(prefix) :].lstrip()
    sentences = _SENTENCE_SPLIT.split(rest)
    kept: list[str] = []
    for s in sentences:
        piece = s.strip()
        if not piece:
            continue
        if line_has_forbidden(piece, constraints):
            continue
        kept.append(piece)
    if not kept:
        return ""
    return prefix + "\n\n" + " ".join(kept)


def _validate_boost_output(
    text: str | None,
    constraints: ConceptConstraints,
    *,
    allowed_evidence_lines: list[str] | None = None,
    allow_external_clarification: bool = False,
) -> tuple[str | None, str | None]:
    """Return (clean_text, None) or (None, reason)."""
    if not text or not str(text).strip():
        return None, "empty_boost"
    stripped = _strip_forbidden_sentences(text, constraints)
    if not stripped.strip():
        return None, "all_sentences_forbidden"
    rest = stripped
    if line_has_forbidden(rest, constraints):
        return None, "forbidden_after_strip"
    if not constraints.is_relational and constraints.forbidden_terms:
        al = rest.lower()
        for t in constraints.forbidden_terms:
            if len(t) > 2 and t in al:
                return None, f"forbidden_term:{t[:20]}"

    allowed = list(allowed_evidence_lines or [])
    allowed_blob = " ".join(allowed).lower()

    if allow_external_clarification:
        if _has_unmarked_external_addition(stripped, allowed_blob):
            return None, "unmarked_external"
        stripped = _cap_marker_prefixed_sentences(stripped)
        if _too_many_novel_terms_in_marker_segments(stripped, allowed_blob):
            return None, "too_many_new_terms"

    return stripped, None


def run_constrained_boost_for_message(
    message: Message,
    *,
    user_id: int,
) -> dict[str, Any]:
    """
    Idempotent: if boost_status is not pending, returns current payload fields.
    Persists boosted_explanation + boost_status on success; skipped on validation failure.
    """
    session = message.session
    if session is None or session.user_id != user_id:
        return {"error": "forbidden", "code": 403}

    if message.role != "assistant":
        return {"error": "not_assistant_message", "code": 400}

    payload = json.loads(message.payload_json or "{}")
    if "boost_status" not in payload:
        be = payload.get("boosted_explanation")
        return {
            "boost_status": "ready" if be else "skipped",
            "boosted_explanation": be,
            "assistant_message_id": message.id,
        }

    status = payload.get("boost_status") or "skipped"
    if status in ("ready", "skipped", "failed"):
        return {
            "boost_status": status,
            "boosted_explanation": payload.get("boosted_explanation"),
            "boost_skip_reason": payload.get("boost_skip_reason"),
            "assistant_message_id": message.id,
        }

    log = message.retrieval_log
    if not log or not log.chunk_hits:
        payload["boost_status"] = "skipped"
        payload["boost_skip_reason"] = "no_retrieval_log"
        message.payload_json = json.dumps(payload)
        db.session.commit()
        return {
            "boost_status": "skipped",
            "boosted_explanation": None,
            "boost_skip_reason": "no_retrieval_log",
            "assistant_message_id": message.id,
        }

    course_answer = payload.get("course_answer") or ""
    user_q = log.user_question or ""
    mode_eff = (payload.get("mode") or {}).get("effective") or "chat"

    sq_dict = None
    pd = payload.get("pipeline_diagnostics")
    if isinstance(pd, dict):
        sq_dict = pd.get("structured_query")

    chunks = chunks_from_retrieval_log(log)
    kb = get_kb()

    snap = payload.get("boost_constraints")
    if isinstance(snap, dict) and snap.get("target_concepts"):
        constraints = ConceptConstraints.from_dict(snap)
    elif sq_dict and sq_dict.get("concept_ids") is not None:
        raw_q = sq_dict.get("raw_query") or user_q
        intent = analyze_query(str(raw_q))
        try:
            qt_val = sq_dict.get("query_type")
            if qt_val:
                intent.query_type = QueryType(str(qt_val))
        except Exception:
            pass
        mode_routing = payload.get("mode_routing") or {}
        sq = build_structured_query(intent, kb=kb, mode_routing=mode_routing)
        constraints = build_concept_constraints(sq, kb)
    else:
        intent = analyze_query(user_q)
        sq = build_structured_query(
            intent, kb=kb, mode_routing=payload.get("mode_routing") or {}
        )
        constraints = build_concept_constraints(sq, kb)

    allowed = collect_allowed_evidence_lines(chunks, constraints, max_lines=5)
    kb_meta = (
        kb.get_concept_by_id(constraints.target_concepts[0])
        if constraints.target_concepts
        else None
    )
    allow_external_clarification = bool(
        kb_meta
        and kb_meta.allow_external_clarification
        and course_answer_is_shallow(course_answer, allowed)
    )

    if not allowed and not allow_external_clarification:
        payload["boost_status"] = "skipped"
        payload["boost_skip_reason"] = "no_evidence_lines"
        message.payload_json = json.dumps(payload)
        _sync_response_variant(message, payload, None, False)
        db.session.commit()
        return {
            "boost_status": "skipped",
            "boosted_explanation": None,
            "boost_skip_reason": "no_evidence_lines",
            "assistant_message_id": message.id,
        }

    evidence_lines = list(allowed)
    if not evidence_lines and allow_external_clarification:
        evidence_lines = _evidence_fallback_from_course_answer(course_answer)
        if not evidence_lines:
            payload["boost_status"] = "skipped"
            payload["boost_skip_reason"] = "no_evidence_lines"
            message.payload_json = json.dumps(payload)
            _sync_response_variant(message, payload, None, False)
            db.session.commit()
            return {
                "boost_status": "skipped",
                "boosted_explanation": None,
                "boost_skip_reason": "no_evidence_lines",
                "assistant_message_id": message.id,
            }

    target_label = _target_concept_label(constraints, sq_dict)
    forbidden_list = sorted(constraints.forbidden_terms)

    chain = boost_provider_chain()
    if not any(a.has_key for a in chain):
        payload["boost_status"] = "skipped"
        payload["boost_skip_reason"] = "no_boost_api_key"
        payload["boosted_explanation"] = None
        message.payload_json = json.dumps(payload)
        _sync_response_variant(message, payload, None, False)
        db.session.commit()
        return {
            "boost_status": "skipped",
            "boosted_explanation": None,
            "boost_skip_reason": "no_boost_api_key",
            "assistant_message_id": message.id,
        }

    boosted = None
    meta: dict[str, Any] = {}
    for attempt in chain:
        if not attempt.has_key:
            continue
        if attempt.provider == "openai":
            boosted, meta = generate_openai_constrained_boost(
                user_question=user_q,
                target_concept=target_label,
                allowed_evidence_lines=evidence_lines,
                forbidden_terms=forbidden_list,
                draft_answer=course_answer,
                mode=mode_eff,
                allow_external_clarification=allow_external_clarification,
            )
            if boosted:
                meta = {**meta, "provider": "openai"}
                break
        elif attempt.provider == "gemini":
            boosted, meta = generate_gemini_constrained_boost(
                user_question=user_q,
                target_concept=target_label,
                allowed_evidence_lines=evidence_lines,
                forbidden_terms=forbidden_list,
                draft_answer=course_answer,
                mode=mode_eff,
                allow_external_clarification=allow_external_clarification,
            )
            if boosted:
                break

    if boosted is None and (meta or {}).get("error"):
        err = meta.get("error")
        logger.info("deferred_boost_no_output message_id=%s provider_error=%s", message.id, err)
        payload["boost_status"] = "skipped"
        payload["boost_skip_reason"] = "boost_provider_error"
        payload["boosted_explanation"] = None
        message.payload_json = json.dumps(payload)
        _sync_response_variant(message, payload, None, False)
        db.session.commit()
        return {
            "boost_status": "skipped",
            "boosted_explanation": None,
            "boost_skip_reason": "boost_provider_error",
            "assistant_message_id": message.id,
        }

    clean, reason = _validate_boost_output(
        boosted,
        constraints,
        allowed_evidence_lines=evidence_lines,
        allow_external_clarification=allow_external_clarification,
    )
    if clean is None:
        logger.info("deferred_boost_discarded message_id=%s reason=%s", message.id, reason)
        payload["boost_status"] = "skipped"
        payload["boost_skip_reason"] = reason or "validation_failed"
        payload["boosted_explanation"] = None
        message.payload_json = json.dumps(payload)
        _sync_response_variant(message, payload, None, False)
        db.session.commit()
        return {
            "boost_status": "skipped",
            "boosted_explanation": None,
            "boost_skip_reason": payload["boost_skip_reason"],
            "assistant_message_id": message.id,
        }

    payload["boosted_explanation"] = clean
    payload["boost_status"] = "ready"
    payload["boost_provider"] = meta.get("provider")
    payload.pop("boost_skip_reason", None)
    message.payload_json = json.dumps(payload)
    _sync_response_variant(message, payload, clean, True)
    db.session.commit()
    return {
        "boost_status": "ready",
        "boosted_explanation": clean,
        "assistant_message_id": message.id,
        "boost_provider": meta.get("provider"),
    }


def _sync_response_variant(
    message: Message,
    payload: dict[str, Any],
    boosted: str | None,
    boost_used: bool,
) -> None:
    rv = message.response_variant
    if rv is None:
        return
    rv.boosted_explanation = boosted
    rv.boost_used = boost_used
    if boosted:
        rv.boosted_answer_length = len(boosted)
    else:
        rv.boosted_answer_length = None
