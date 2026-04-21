"""Deterministic API mode detection (chat / quiz / compare / summary) for natural-language routing.

No LLM: regex, phrase lists, and priority tie-breaking (quiz > compare > summary > chat).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from app.services.query_understanding import (
    QueryIntent,
    QueryType,
    extract_compare_entities,
)

logger = logging.getLogger(__name__)

# Priority when multiple families score high (first wins).
_MODE_PRIORITY: tuple[ApiMode, ...] = ("quiz", "compare", "summary", "chat")

_MIN_SCORE = 0.32
_AMBIGUITY_GAP = 0.12


def _normalize(raw: str) -> str:
    s = raw.strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    return s


# --- Scoring helpers (return score in [0,1], list of signal tags) ---


def _score_quiz(q: str) -> tuple[float, list[str]]:
    signals: list[str] = []
    score = 0.0
    patterns: list[tuple[str, str, float]] = [
        (r"\bquiz\s+me\b", "phrase:quiz_me", 0.95),
        (r"\btest\s+me\b", "phrase:test_me", 0.92),
        (r"\bgive\s+me\s+a\s+quiz\b", "phrase:give_quiz", 0.95),
        (r"\bpractice\s+quiz\b", "phrase:practice_quiz", 0.9),
        (r"\bask\s+me\s+questions\b", "phrase:ask_me_questions", 0.88),
        (r"\bask\s+questions\s+about\b", "phrase:ask_questions_about", 0.85),
        (r"\bcheck\s+my\s+understanding\b", "phrase:check_understanding", 0.88),
        (r"\bmultiple\s+choice\s+me\s+on\b", "phrase:mc_me_on", 0.9),
        (r"\bdrill\s+me\b", "phrase:drill_me", 0.75),
    ]
    for pat, tag, wt in patterns:
        if re.search(pat, q):
            signals.append(tag)
            score = max(score, wt)
    if re.search(r"\bpop\s*quiz\b", q):
        signals.append("keyword:pop_quiz")
        score = max(score, 0.8)
    return score, signals


def _score_compare(q: str) -> tuple[float, list[str]]:
    signals: list[str] = []
    score = 0.0
    strong: list[tuple[str, str, float]] = [
        (r"\bcompare\b", "keyword:compare", 0.72),
        (r"\bcontrast\b", "keyword:contrast", 0.7),
        (r"\bdifferences?\s+between\b", "phrase:differences_between", 0.88),
        (r"\bdifference\s+between\b", "phrase:difference_between", 0.88),
        (r"\bhow\s+is\s+.+\s+different\s+from\b", "pattern:how_different_from", 0.9),
        (r"\bdistinguish\s+.+\s+from\b", "pattern:distinguish_from", 0.85),
        (r"\bversus\b", "keyword:versus", 0.65),
        (r"\bvs\.?\b", "keyword:vs", 0.62),
    ]
    for pat, tag, wt in strong:
        if re.search(pat, q, re.IGNORECASE):
            signals.append(tag)
            score = max(score, wt)
    # Two-sided "X and Y" after compare / difference phrasing
    if re.search(
        r"(?:^|\b)(?:compare|contrast|difference(?:s)?\s+between)\s+.+?\b(?:and|vs\.?|versus)\b",
        q,
        re.IGNORECASE,
    ):
        signals.append("structure:compare_binary")
        score = max(score, 0.78)
    ce = extract_compare_entities(q)
    if ce and len(ce) >= 2:
        signals.append("entity:multi_compare")
        score = max(score, 0.75)
    return score, signals


def _score_summary(q: str) -> tuple[float, list[str]]:
    signals: list[str] = []
    score = 0.0
    patterns: list[tuple[str, str, float]] = [
        (r"\bsummarize\b", "keyword:summarize", 0.82),
        (r"\bsummarise\b", "keyword:summarise", 0.82),
        (r"\bsummary\s+of\b", "phrase:summary_of", 0.85),
        (r"\bgive\s+me\s+a\s+recap\b", "phrase:give_recap", 0.88),
        (r"\brecap\b", "keyword:recap", 0.72),
        (r"\boverview\s+of\b", "phrase:overview_of", 0.8),
        (r"\bbrief\s+overview\b", "phrase:brief_overview", 0.78),
        (r"\bsummarize\s+lecture\b", "phrase:summarize_lecture", 0.9),
        (r"\bsummarize\s+topic\b", "phrase:summarize_topic", 0.85),
        (r"\bmain\s+ideas\s+(?:of|in|from)\b", "phrase:main_ideas", 0.8),
        (r"\btl;?dr\b", "keyword:tldr", 0.7),
    ]
    for pat, tag, wt in patterns:
        if re.search(pat, q, re.IGNORECASE):
            signals.append(tag)
            score = max(score, wt)
    if re.search(r"\bhigh-?level\s+(?:summary|overview)\b", q, re.IGNORECASE):
        signals.append("phrase:high_level")
        score = max(score, 0.75)
    return score, signals


@dataclass
class ModeDetectionResult:
    mode: str  # chat | quiz | compare | summary
    confidence: float
    signals: list[str]
    ambiguous: bool = False
    candidate_modes: list[str] | None = None
    override_allowed: bool = True


def detect_query_mode(raw_query: str) -> ModeDetectionResult:
    """
    Rule-based mode detection. Fallback is **chat** (general Q&A / definitions / explanations).
    """
    q = _normalize(raw_query)
    if not q:
        return ModeDetectionResult(
            mode="chat",
            confidence=0.2,
            signals=["empty:query"],
        )

    qz, sig_qz = _score_quiz(q)
    cp, sig_cp = _score_compare(q)
    sm, sig_sm = _score_summary(q)

    scores: dict[str, float] = {
        "quiz": qz,
        "compare": cp,
        "summary": sm,
        "chat": 0.45,
    }

    # Mixed-intent: strong signals in multiple families
    strong = [(m, scores[m]) for m in ("quiz", "compare", "summary") if scores[m] >= _MIN_SCORE]
    strong.sort(key=lambda x: -x[1])
    ambiguous = False
    candidates: list[str] | None = None
    if len(strong) >= 2 and (strong[0][1] - strong[1][1]) < _AMBIGUITY_GAP:
        ambiguous = True
        candidates = [strong[0][0], strong[1][0]]
        logger.info(
            "query_mode: ambiguous query=%r top=%s second=%s",
            q[:120],
            strong[0],
            strong[1],
        )

    # Priority pass: first mode in quiz > compare > summary meeting threshold
    chosen: str | None = None
    chosen_score = 0.0
    chosen_signals: list[str] = []
    for m in ("quiz", "compare", "summary"):
        s = scores[m]
        if s >= _MIN_SCORE:
            chosen = m
            chosen_score = s
            if m == "quiz":
                chosen_signals = sig_qz
            elif m == "compare":
                chosen_signals = sig_cp
            else:
                chosen_signals = sig_sm
            break

    if chosen is None:
        # Soft fallback: best-scoring specialized mode above a low floor (priority on ties)
        ranked = sorted(
            ((scores[m], m) for m in ("quiz", "compare", "summary")),
            key=lambda x: (-x[0], _MODE_PRIORITY.index(x[1])),
        )
        best_alt_s, best_alt = ranked[0]
        if best_alt_s >= 0.22:
            chosen = best_alt
            chosen_score = best_alt_s
            chosen_signals = {"quiz": sig_qz, "compare": sig_cp, "summary": sig_sm}[best_alt]
        else:
            return ModeDetectionResult(
                mode="chat",
                confidence=min(0.85, 0.5 + 0.1 * len(q.split())),
                signals=["fallback:chat"],
                ambiguous=ambiguous,
                candidate_modes=candidates,
            )

    conf = min(0.99, max(0.35, chosen_score))
    if ambiguous:
        conf *= 0.85

    return ModeDetectionResult(
        mode=chosen,
        confidence=conf,
        signals=chosen_signals or [f"mode:{chosen}"],
        ambiguous=ambiguous,
        candidate_modes=candidates,
    )


def resolve_effective_mode(user_mode: str, detection: ModeDetectionResult) -> tuple[str, bool]:
    """
    ``user_mode``: ``auto`` (or empty) uses detection; otherwise explicit ``chat`` / ``quiz`` /
    ``compare`` / ``summary`` overrides.

    Returns ``(effective_mode, mode_was_overridden)``.
    """
    u = (user_mode or "auto").strip().lower()
    if u in ("", "auto"):
        return detection.mode, False
    if u in ("chat", "quiz", "compare", "summary"):
        return u, True
    return detection.mode, False


def apply_effective_api_mode(
    intent: QueryIntent,
    original_query: str,
    effective_api_mode: str,
) -> QueryIntent:
    """
    Map high-level API mode onto :class:`QueryIntent`.

    **chat**: keep full :func:`analyze_query` classification (definitions, synthesis, etc.).
    **quiz / compare / summary**: coerce ``query_type`` so retrieval strategies match.
    """
    if effective_api_mode == "chat":
        return intent

    if effective_api_mode == "quiz":
        return replace(intent, query_type=QueryType.QUIZ)

    if effective_api_mode == "summary":
        return replace(intent, query_type=QueryType.SUMMARY)

    # compare
    ce = extract_compare_entities(original_query)
    if ce and len(ce) >= 2:
        return replace(
            intent,
            query_type=QueryType.COMPARE,
            compare_entities=ce,
            compare_concepts=(ce[0].strip(), ce[1].strip()),
        )
    return replace(intent, query_type=QueryType.COMPARE)
