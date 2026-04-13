"""Aggregate analytics for admin insights (read-only SQL over analytics tables)."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, text

from app.extensions import db
from app.models.analytics import Feedback, MessageOutcome, ResponseVariant, RetrievalLog
from app.models.content import LectureChunk


def _utc_window(days: int) -> tuple[datetime, datetime]:
    """Inclusive since, inclusive until in naive UTC (matches typical SQLite storage)."""
    until = datetime.now(timezone.utc).replace(tzinfo=None)
    since = until - timedelta(days=max(1, min(int(days), 365)))
    return since, until


def _pct(part: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(100.0 * part / total, 2)


def _group_count_rows(
    model: type,
    column: Any,
    since: datetime,
    until: datetime,
) -> dict[str, int]:
    rows = (
        db.session.query(column, func.count())
        .filter(model.created_at >= since, model.created_at <= until)
        .group_by(column)
        .all()
    )
    out: dict[str, int] = {}
    for key, n in rows:
        k = "null" if key is None else str(key)
        out[k] = int(n)
    return out


def compute_insights_summary(days: int = 7) -> dict[str, Any]:
    """
    Return dashboard-friendly aggregates for the last ``days`` (clamped 1–365), UTC window.

    Uses existing tables only; no new migrations.
    """
    since, until = _utc_window(days)

    log_q = db.session.query(RetrievalLog).filter(
        RetrievalLog.created_at >= since,
        RetrievalLog.created_at <= until,
    )
    total_logs = log_q.count()

    insufficient = total_logs < 1

    # Volume
    distinct_sessions = (
        db.session.query(RetrievalLog.session_id)
        .filter(
            RetrievalLog.created_at >= since,
            RetrievalLog.created_at <= until,
            RetrievalLog.session_id.isnot(None),
        )
        .distinct()
        .count()
    )

    # Retrieval KPIs
    avg_conf = (
        db.session.query(func.avg(RetrievalLog.confidence))
        .filter(
            RetrievalLog.created_at >= since,
            RetrievalLog.created_at <= until,
            RetrievalLog.confidence.isnot(None),
        )
        .scalar()
    )
    avg_lat = (
        db.session.query(func.avg(RetrievalLog.latency_ms))
        .filter(
            RetrievalLog.created_at >= since,
            RetrievalLog.created_at <= until,
            RetrievalLog.latency_ms.isnot(None),
        )
        .scalar()
    )

    no_chunk = log_q.filter(
        (RetrievalLog.num_chunks_hit == 0)
        | (RetrievalLog.is_off_topic == True)  # noqa: E712
    ).count()

    low_conf = log_q.filter(RetrievalLog.is_low_confidence == True).count()  # noqa: E712

    # Pipeline breakdowns
    by_qtype = _group_count_rows(RetrievalLog, RetrievalLog.query_type_v2, since, until)
    by_mode = _group_count_rows(RetrievalLog, RetrievalLog.answer_mode, since, until)

    passed_rows = (
        db.session.query(RetrievalLog.validation_passed, func.count())
        .filter(
            RetrievalLog.created_at >= since,
            RetrievalLog.created_at <= until,
        )
        .group_by(RetrievalLog.validation_passed)
        .all()
    )
    validation_passed: dict[str, int] = {}
    for key, n in passed_rows:
        label = "null" if key is None else ("true" if key else "false")
        validation_passed[label] = int(n)

    # Severity from JSON (SQLite json_extract)
    severity_counts: dict[str, int] = {}
    try:
        raw_sev = db.session.execute(
            text(
                """
                SELECT json_extract(validation_checks_json, '$.severity') AS sev, COUNT(*)
                FROM retrieval_logs
                WHERE created_at >= :since AND created_at <= :until
                  AND validation_checks_json IS NOT NULL
                  AND TRIM(validation_checks_json) != ''
                GROUP BY sev
                """
            ),
            {"since": since, "until": until},
        ).fetchall()
        for sev, cnt in raw_sev:
            label = sev if sev else "null"
            severity_counts[str(label)] = int(cnt)
    except Exception:
        severity_counts = {}

    # Boost (per response variant in window)
    rv_total = (
        db.session.query(func.count(ResponseVariant.id))
        .filter(
            ResponseVariant.created_at >= since,
            ResponseVariant.created_at <= until,
        )
        .scalar()
        or 0
    )
    rv_boost = (
        db.session.query(func.count(ResponseVariant.id))
        .filter(
            ResponseVariant.created_at >= since,
            ResponseVariant.created_at <= until,
            ResponseVariant.boost_used == True,  # noqa: E712
        )
        .scalar()
        or 0
    )
    boost_reason_rows = (
        db.session.query(ResponseVariant.boost_reason, func.count())
        .filter(
            ResponseVariant.created_at >= since,
            ResponseVariant.created_at <= until,
        )
        .group_by(ResponseVariant.boost_reason)
        .all()
    )
    by_boost_reason: dict[str, int] = {}
    for key, n in boost_reason_rows:
        k = "null" if key is None else str(key)
        by_boost_reason[k] = int(n)

    # Feedback
    fb_total = (
        db.session.query(func.count(Feedback.id))
        .filter(
            Feedback.created_at >= since,
            Feedback.created_at <= until,
        )
        .scalar()
        or 0
    )
    thumb_rows = (
        db.session.query(Feedback.course_thumb, func.count())
        .filter(
            Feedback.created_at >= since,
            Feedback.created_at <= until,
        )
        .group_by(Feedback.course_thumb)
        .all()
    )
    course_thumb: dict[str, int] = {}
    for key, n in thumb_rows:
        k = "null" if key is None else str(key)
        course_thumb[k] = int(n)

    helpful_avg = (
        db.session.query(func.avg(Feedback.helpfulness_rating))
        .filter(
            Feedback.created_at >= since,
            Feedback.created_at <= until,
            Feedback.helpfulness_rating.isnot(None),
        )
        .scalar()
    )

    # Message outcomes
    mo_total = (
        db.session.query(func.count(MessageOutcome.id))
        .filter(
            MessageOutcome.created_at >= since,
            MessageOutcome.created_at <= until,
        )
        .scalar()
        or 0
    )
    mo_resolved_known = (
        db.session.query(func.count(MessageOutcome.id))
        .filter(
            MessageOutcome.created_at >= since,
            MessageOutcome.created_at <= until,
            MessageOutcome.answer_resolved.isnot(None),
        )
        .scalar()
        or 0
    )
    mo_resolved_yes = (
        db.session.query(func.count(MessageOutcome.id))
        .filter(
            MessageOutcome.created_at >= since,
            MessageOutcome.created_at <= until,
            MessageOutcome.answer_resolved == True,  # noqa: E712
        )
        .scalar()
        or 0
    )

    return {
        "window": {
            "days": max(1, min(int(days), 365)),
            "since": since.isoformat() + "Z",
            "until": until.isoformat() + "Z",
            "timezone_note": "Timestamps are UTC; filters use created_at on each table.",
        },
        "volume": {
            "retrieval_events": total_logs,
            "distinct_sessions": distinct_sessions,
        },
        "retrieval": {
            "avg_confidence": float(avg_conf) if avg_conf is not None else None,
            "avg_latency_ms": float(avg_lat) if avg_lat is not None else None,
            "pct_no_chunks_or_off_topic": _pct(no_chunk, total_logs),
            "pct_low_confidence_flag": _pct(low_conf, total_logs),
        },
        "pipeline": {
            "by_query_type_v2": dict(by_qtype),
            "by_answer_mode": dict(by_mode),
            "validation_passed": validation_passed,
            "validation_severity": severity_counts,
        },
        "boost": {
            "response_variants_in_window": int(rv_total),
            "pct_boost_used": _pct(int(rv_boost), int(rv_total)) if rv_total else None,
            "by_boost_reason": by_boost_reason,
        },
        "feedback": {
            "rows": int(fb_total),
            "course_thumb": course_thumb,
            "avg_helpfulness_rating": round(float(helpful_avg), 2) if helpful_avg is not None else None,
        },
        "outcomes": {
            "rows": int(mo_total),
            "pct_answer_resolved_true": _pct(int(mo_resolved_yes), int(mo_resolved_known))
            if mo_resolved_known
            else None,
        },
        "insufficient_data": insufficient,
        "models_and_tokens": _rollup_models_tokens(since, until),
    }


def _parse_limit_offset(limit_raw: str | None, offset_raw: str | None, *, max_limit: int = 200) -> tuple[int, int]:
    try:
        limit = int(limit_raw) if limit_raw is not None else 50
    except (TypeError, ValueError):
        limit = 50
    try:
        offset = int(offset_raw) if offset_raw is not None else 0
    except (TypeError, ValueError):
        offset = 0
    limit = max(1, min(limit, max_limit))
    offset = max(0, offset)
    return limit, offset


def low_confidence_drill_down(days: int, limit: int, offset: int) -> dict[str, Any]:
    """Paged retrieval logs flagged low-confidence (no user PII)."""
    since, until = _utc_window(days)
    q = (
        db.session.query(RetrievalLog)
        .filter(
            RetrievalLog.created_at >= since,
            RetrievalLog.created_at <= until,
            RetrievalLog.is_low_confidence == True,  # noqa: E712
        )
        .order_by(RetrievalLog.created_at.desc())
    )
    total = q.count()
    rows = q.offset(offset).limit(limit).all()
    items = []
    for log in rows:
        items.append(
            {
                "retrieval_log_id": log.id,
                "message_id": log.message_id,
                "session_id": log.session_id,
                "created_at": (log.created_at.isoformat() + "Z") if log.created_at else None,
                "user_question": (log.user_question or "")[:800],
                "confidence": log.confidence,
                "query_type_v2": log.query_type_v2,
                "answer_mode": log.answer_mode,
            }
        )
    return {
        "window": {
            "days": max(1, min(int(days), 365)),
            "since": since.isoformat() + "Z",
            "until": until.isoformat() + "Z",
        },
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


def low_confidence_csv_rows(days: int, max_rows: int = 5000) -> list[list[Any]]:
    """Flat rows for CSV export (capped)."""
    since, until = _utc_window(days)
    rows = (
        db.session.query(RetrievalLog)
        .filter(
            RetrievalLog.created_at >= since,
            RetrievalLog.created_at <= until,
            RetrievalLog.is_low_confidence == True,  # noqa: E712
        )
        .order_by(RetrievalLog.created_at.desc())
        .limit(max_rows)
        .all()
    )
    out: list[list[Any]] = []
    for log in rows:
        out.append(
            [
                log.id,
                log.message_id,
                log.session_id,
                (log.created_at.isoformat() + "Z") if log.created_at else "",
                (log.user_question or "").replace("\n", " ")[:2000],
                log.confidence,
                log.query_type_v2 or "",
                log.answer_mode or "",
            ]
        )
    return out


def chunk_analytics(days: int, limit: int) -> dict[str, Any]:
    """Top lecture chunks in low-confidence retrievals vs overall hit frequency."""
    since, until = _utc_window(days)
    lim = max(1, min(int(limit), 100))

    weak_rows = db.session.execute(
        text(
            """
            SELECT h.lecture_chunk_id, COUNT(*) AS hit_count
            FROM retrieval_chunk_hits h
            JOIN retrieval_logs r ON r.id = h.retrieval_log_id
            WHERE r.created_at >= :since AND r.created_at <= :until
              AND r.is_low_confidence = 1
            GROUP BY h.lecture_chunk_id
            ORDER BY hit_count DESC
            LIMIT :lim
            """
        ),
        {"since": since, "until": until, "lim": lim},
    ).fetchall()

    all_rows = db.session.execute(
        text(
            """
            SELECT h.lecture_chunk_id, COUNT(*) AS hit_count
            FROM retrieval_chunk_hits h
            JOIN retrieval_logs r ON r.id = h.retrieval_log_id
            WHERE r.created_at >= :since AND r.created_at <= :until
            GROUP BY h.lecture_chunk_id
            ORDER BY hit_count DESC
            LIMIT :lim
            """
        ),
        {"since": since, "until": until, "lim": lim},
    ).fetchall()

    def _enrich(rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for chunk_id, cnt in rows:
            chunk = db.session.get(LectureChunk, int(chunk_id))
            out.append(
                {
                    "lecture_chunk_id": int(chunk_id),
                    "hit_count": int(cnt),
                    "lecture_number": chunk.lecture_number if chunk else None,
                    "topic": (chunk.topic[:200] if chunk and chunk.topic else None),
                    "chunk_key": chunk.chunk_key if chunk else None,
                }
            )
        return out

    return {
        "window": {
            "days": max(1, min(int(days), 365)),
            "since": since.isoformat() + "Z",
            "until": until.isoformat() + "Z",
        },
        "top_chunks_in_low_confidence_retrievals": _enrich(list(weak_rows)),
        "top_chunks_overall": _enrich(list(all_rows)),
    }


def _tokens_from_openai_usage(usage: dict[str, Any]) -> int | None:
    if not isinstance(usage, dict):
        return None
    t = usage.get("total_tokens")
    return int(t) if isinstance(t, int) else None


def _tokens_from_gemini_usage(usage: dict[str, Any]) -> int | None:
    if not isinstance(usage, dict):
        return None
    for k in ("totalTokenCount", "total_tokens"):
        v = usage.get(k)
        if isinstance(v, int):
            return v
    return None


def _tokens_from_block(block: dict[str, Any]) -> int:
    """Best-effort total tokens from primary/boost JSON blob."""
    if not isinstance(block, dict):
        return 0
    u = block.get("usage")
    if isinstance(u, dict):
        t = _tokens_from_openai_usage(u)
        if t is not None:
            return t
        t2 = _tokens_from_gemini_usage(u)
        if t2 is not None:
            return t2
    return 0


def _rollup_models_tokens(since: datetime, until: datetime) -> dict[str, Any]:
    rvs = (
        db.session.query(ResponseVariant)
        .filter(
            ResponseVariant.created_at >= since,
            ResponseVariant.created_at <= until,
        )
        .all()
    )
    by_provider: dict[str, int] = {}
    by_model: dict[str, int] = {}
    total_tokens = 0
    with_counts = 0
    for rv in rvs:
        if rv.provider_name:
            by_provider[rv.provider_name] = by_provider.get(rv.provider_name, 0) + 1
        if rv.model_name:
            by_model[rv.model_name] = by_model.get(rv.model_name, 0) + 1
        if not rv.token_usage_json:
            continue
        try:
            d = json.loads(rv.token_usage_json)
        except json.JSONDecodeError:
            continue
        row_total = 0
        for key in ("primary", "boost"):
            block = d.get(key)
            if isinstance(block, dict):
                row_total += _tokens_from_block(block)
        if row_total > 0:
            total_tokens += row_total
            with_counts += 1

    return {
        "response_variants_in_window": len(rvs),
        "sum_total_tokens_estimated": total_tokens if total_tokens else None,
        "response_variants_with_token_totals": with_counts,
        "by_provider": by_provider,
        "by_primary_model_name": by_model,
    }


def render_low_confidence_csv(days: int) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "retrieval_log_id",
            "message_id",
            "session_id",
            "created_at_utc",
            "user_question",
            "confidence",
            "query_type_v2",
            "answer_mode",
        ]
    )
    for row in low_confidence_csv_rows(days):
        w.writerow(row)
    return buf.getvalue()
