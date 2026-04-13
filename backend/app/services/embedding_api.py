"""OpenAI embeddings HTTP client (urllib; no extra deps)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def openai_embed_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str,
    timeout_sec: int = 120,
) -> list[list[float]]:
    """
    Call ``POST /v1/embeddings``. Returns one vector per input string (same order).
    """
    if not api_key or not texts:
        return []
    body = json.dumps({"input": texts, "model": model}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode() if e.fp else ""
        raise RuntimeError(f"OpenAI embeddings HTTP {e.code}: {err_body[:500]}") from e
    data = payload.get("data") or []
    ordered = sorted(data, key=lambda x: int(x.get("index", 0)))
    return [list(map(float, item["embedding"])) for item in ordered]


def openai_embed_one(text: str, *, api_key: str, model: str, timeout_sec: int = 60) -> list[float]:
    vecs = openai_embed_batch([text], api_key=api_key, model=model, timeout_sec=timeout_sec)
    return vecs[0] if vecs else []
