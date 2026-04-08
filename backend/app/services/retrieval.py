"""Keyword retrieval over lecture chunks (v1). Not implemented in scaffold."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    confidence: float
    detected_topic: str | None


def retrieve(query: str, top_k: int = 5) -> RetrievalResult:
    """Placeholder: returns empty hits and zero confidence."""
    return RetrievalResult(chunks=[], confidence=0.0, detected_topic=None)
