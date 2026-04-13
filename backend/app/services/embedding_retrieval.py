"""Dense retrieval over ``lecture_chunks.embedding_blob`` + hybrid fusion with lexical scores."""

from __future__ import annotations

from typing import Any

import numpy as np
from flask import current_app

from app.models import LectureChunk
from app.services.embedding_api import openai_embed_one
from app.services.retrieval import (
    ChunkHitDiag,
    RetrievalDiagnostics,
    RetrievalResult,
    _row_to_public_dict,
    score_chunks_keyword,
)


def _orm_to_row_dict(r: LectureChunk) -> dict[str, Any]:
    return {
        "id": r.id,
        "chunk_key": r.chunk_key,
        "lecture_number": r.lecture_number,
        "topic": r.topic,
        "keywords": r.keywords,
        "source_excerpt": r.source_excerpt,
        "clean_explanation": r.clean_explanation,
        "sample_questions": r.sample_questions,
        "sample_answer": r.sample_answer,
        "chunk_type": getattr(r, "chunk_type", None),
        "concept_family": getattr(r, "concept_family", None),
    }


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embedding_config() -> tuple[str, str]:
    key = (current_app.config.get("OPENAI_API_KEY") or "").strip()
    model = (current_app.config.get("EMBEDDING_MODEL_ID") or "text-embedding-3-small").strip()
    return key, model


def retrieve_embedding_only(
    query: str,
    *,
    top_k: int,
    lecture_filter: int | None,
    summary_rank: bool,
) -> RetrievalResult:
    """
    Cosine similarity of query embedding against stored chunk vectors.
    """
    api_key, model = _embedding_config()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for embedding retrieval")

    q_emb = np.array(openai_embed_one(query, api_key=api_key, model=model), dtype=np.float32)
    qn = float(np.linalg.norm(q_emb))
    if qn < 1e-12:
        raise ValueError("query embedding is degenerate")

    q = LectureChunk.query
    if lecture_filter is not None:
        q = q.filter_by(lecture_number=lecture_filter)
    rows = [r for r in q.all() if r.embedding_blob and r.embedding_dim]
    if not rows:
        raise ValueError(
            "No chunk embeddings in database. Run: flask --app wsgi embed-chunks"
        )

    scored: list[tuple[LectureChunk, float]] = []
    for r in rows:
        v = _blob_to_vec(r.embedding_blob)
        if r.embedding_dim and len(v) != int(r.embedding_dim):
            continue
        if len(v) != len(q_emb):
            continue
        scored.append((r, _cosine(q_emb, v)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_rows = scored[:top_k]
    if not top_rows:
        return RetrievalResult(chunks=[], confidence=0.0, detected_topic=None, diagnostics=None)

    best = top_rows[0][1]
    second = top_rows[1][1] if len(top_rows) > 1 else 0.0
    eps = 1e-9
    conf = min(0.99, max(0.05, best))
    chunk_diags: list[ChunkHitDiag] = []
    public_chunks: list[dict[str, Any]] = []
    for rank, (row, sc) in enumerate(top_rows, start=1):
        public_chunks.append(_row_to_public_dict(_orm_to_row_dict(row)))
        chunk_diags.append(
            ChunkHitDiag(
                chunk_id=row.id,
                rank=rank,
                score=sc,
                token_score=0.0,
                phrase_score=0.0,
                lecture_bonus=0.0,
                strong_field_token_score=0.0,
                matched_query_terms=0,
                phrase_events=0,
                field_scores={"dense": sc},
            )
        )

    diag = RetrievalDiagnostics(
        query_tokens=query.split(),
        lecture_numbers_detected=[],
        retrieval_backend="embedding",
        top_k_requested=top_k,
        num_chunks_scored=len(rows),
        num_chunks_hit=len(top_rows),
        top_score=best,
        second_score=second,
        score_margin=(best - second) / (best + eps) if best > 0 else 0.0,
        query_coverage=0.0,
        chunk_hits=chunk_diags,
    )
    return RetrievalResult(
        chunks=public_chunks,
        confidence=conf,
        detected_topic=(top_rows[0][0].topic or None) if top_rows else None,
        diagnostics=diag,
    )


def retrieve_hybrid(
    query: str,
    *,
    top_k: int,
    lecture_filter: int | None,
    summary_rank: bool,
) -> RetrievalResult:
    """
    Weighted fusion of normalized lexical score and dense cosine (same chunk ids).
    """
    w_lex = float(current_app.config.get("HYBRID_LEXICAL_WEIGHT", 0.45))
    w_emb = float(current_app.config.get("HYBRID_EMBEDDING_WEIGHT", 0.55))
    s = w_lex + w_emb
    if s > 0:
        w_lex, w_emb = w_lex / s, w_emb / s

    from app.services import retrieval as R

    lex = score_chunks_keyword(
        query,
        R._row_cache,
        top_k=max(top_k * 4, 24),
        lecture_filter=lecture_filter,
        summary_rank=summary_rank,
    )
    if not lex.chunks:
        return lex

    dense = retrieve_embedding_only(
        query, top_k=max(top_k * 4, 32), lecture_filter=lecture_filter, summary_rank=False
    )
    dense_by_id: dict[int, float] = {}
    if dense.diagnostics:
        for ch in dense.diagnostics.chunk_hits:
            dense_by_id[ch.chunk_id] = ch.score

    lex_by_id: dict[int, float] = {}
    if lex.diagnostics:
        for ch in lex.diagnostics.chunk_hits:
            lex_by_id[ch.chunk_id] = ch.score
    n_lex = max(lex_by_id.values()) if lex_by_id else 1.0
    if n_lex <= 0:
        n_lex = 1.0
    lex_norm = {cid: float(v) / n_lex for cid, v in lex_by_id.items()}

    combined: dict[int, float] = {}
    all_ids = set(lex_norm.keys()) | set(dense_by_id.keys())
    for cid in all_ids:
        ln = lex_norm.get(cid, 0.0)
        dn = dense_by_id.get(cid, 0.0)
        combined[cid] = w_lex * ln + w_emb * dn

    # Map id -> chunk dict from lex first, then dense
    by_id: dict[int, dict[str, Any]] = {}
    for c in lex.chunks:
        by_id[c["id"]] = c
    for c in dense.chunks:
        by_id.setdefault(c["id"], c)

    ranked = sorted(combined.keys(), key=lambda i: combined[i], reverse=True)[:top_k]
    out_chunks = [by_id[i] for i in ranked if i in by_id]

    best_c = combined[ranked[0]] if ranked else 0.0
    second_c = combined[ranked[1]] if len(ranked) > 1 else 0.0
    eps = 1e-9
    diag = RetrievalDiagnostics(
        query_tokens=query.split(),
        lecture_numbers_detected=[],
        retrieval_backend="hybrid",
        top_k_requested=top_k,
        num_chunks_scored=len(all_ids),
        num_chunks_hit=len(out_chunks),
        top_score=best_c,
        second_score=second_c,
        score_margin=(best_c - second_c) / (best_c + eps) if best_c > 0 else 0.0,
        query_coverage=0.0,
        chunk_hits=[],
    )
    return RetrievalResult(
        chunks=out_chunks,
        confidence=min(0.99, max(0.05, best_c)),
        detected_topic=out_chunks[0].get("topic") if out_chunks else None,
        diagnostics=diag,
    )
