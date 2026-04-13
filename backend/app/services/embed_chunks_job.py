"""Batch-compute OpenAI embeddings for all ``lecture_chunks`` rows."""

from __future__ import annotations

import numpy as np
from flask import current_app

from app.extensions import db
from app.models import LectureChunk
from app.services.embedding_api import openai_embed_batch


def run_embed_chunks(*, batch_size: int = 32) -> tuple[int, str]:
    """
    Fill ``embedding_blob`` / ``embedding_dim`` / ``embedding_model`` for every chunk.

    Returns (number of rows updated, model id).
    """
    api_key = (current_app.config.get("OPENAI_API_KEY") or "").strip()
    model = (current_app.config.get("EMBEDDING_MODEL_ID") or "text-embedding-3-small").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embed-chunks")

    rows = LectureChunk.query.order_by(LectureChunk.id).all()
    updated = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        texts: list[str] = []
        for r in batch:
            ce = (r.clean_explanation or "")[:4000]
            se = (r.source_excerpt or "")[:1500]
            texts.append(f"{ce}\n\n---\n{se}")
        vecs = openai_embed_batch(texts, api_key=api_key, model=model)
        if len(vecs) != len(batch):
            raise RuntimeError("embedding batch size mismatch from OpenAI")
        for r, v in zip(batch, vecs):
            arr = np.array(v, dtype=np.float32)
            r.embedding_blob = arr.tobytes()
            r.embedding_dim = len(v)
            r.embedding_model = model
            updated += 1
        db.session.commit()
        current_app.logger.info("embed-chunks: committed %s rows (total %s)", len(batch), updated)

    return updated, model
