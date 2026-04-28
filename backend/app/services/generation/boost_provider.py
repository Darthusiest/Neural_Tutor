"""Resolve the Boosted Explanation provider chain (primary → fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from flask import current_app


_VALID = ("openai", "gemini", "none")


@dataclass(frozen=True)
class BoostAttempt:
    provider: str
    has_key: bool


def _norm(p: str | None) -> str:
    p = (p or "").strip().lower()
    return p if p in _VALID else "none"


def _has_key_for(provider: str) -> bool:
    cfg = current_app.config
    if provider == "openai":
        return bool(cfg.get("OPENAI_API_KEY"))
    if provider == "gemini":
        return bool(cfg.get("GEMINI_API_KEY") or cfg.get("GOOGLE_API_KEY"))
    return False


def boost_provider_chain() -> list[BoostAttempt]:
    """
    Ordered providers to try: primary first, then fallback (deduped, ``none`` skipped).

    Each entry has ``has_key`` so callers can avoid making a doomed network request.
    """
    cfg = current_app.config
    primary = _norm(cfg.get("BOOST_PRIMARY_PROVIDER", "openai"))
    fallback = _norm(cfg.get("BOOST_FALLBACK_PROVIDER", "gemini"))
    chain: list[BoostAttempt] = []
    seen: set[str] = set()
    for p in (primary, fallback):
        if p == "none" or p in seen:
            continue
        seen.add(p)
        chain.append(BoostAttempt(provider=p, has_key=_has_key_for(p)))
    return chain


def first_runnable_provider(chain: Iterable[BoostAttempt]) -> BoostAttempt | None:
    for a in chain:
        if a.has_key:
            return a
    return None
