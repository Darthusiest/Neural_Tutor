"""
Lexical (keyword) retrieval over ``lecture_chunks`` with field-weighted scoring.

This module is intentionally structured so a future **embedding** or **hybrid** backend
can implement the same outward behavior (``retrieve_chunks``, ``RetrievalResult``) by:
  - adding ``backend="embedding"`` handling that returns dense similarity scores, or
  - fusing ``lexical_score`` with ``dense_score`` in a small orchestrator without
    changing ``format_course_answer`` or chunk row shapes.

Design: **token-aligned** matching (whole tokens only) eliminates substring false positives
like ``cat`` → ``category``. Per-chunk caches hold **ordered token sequences** and
**per-field Counters** built at load time.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from app.models import LectureChunk

# ---------------------------------------------------------------------------
# Public API types (stable for callers + future hybrid retrieval)
# ---------------------------------------------------------------------------


@dataclass
class ChunkHitDiag:
    """Per-chunk scoring diagnostics for analytics persistence."""

    chunk_id: int
    rank: int
    score: float
    token_score: float
    phrase_score: float
    lecture_bonus: float
    strong_field_token_score: float
    matched_query_terms: int
    phrase_events: int
    field_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalDiagnostics:
    """Aggregate retrieval-event diagnostics attached to RetrievalResult."""

    query_tokens: list[str]
    lecture_numbers_detected: list[int]
    retrieval_backend: str
    top_k_requested: int
    num_chunks_scored: int
    num_chunks_hit: int
    top_score: float
    second_score: float
    score_margin: float
    query_coverage: float
    chunk_hits: list[ChunkHitDiag] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    confidence: float
    detected_topic: str | None
    diagnostics: RetrievalDiagnostics | None = None


# ---------------------------------------------------------------------------
# Tunable lexical scoring config (single place to adjust behavior)
# ---------------------------------------------------------------------------

# Relative importance per field — only used for *token* hits and phrase bonuses.
# Tune here without touching scoring math.
FIELD_WEIGHTS: dict[str, float] = {
    "topic": 3.0,
    "keywords": 2.5,
    "sample_questions": 2.0,
    "clean_explanation": 1.2,
    "source_excerpt": 1.0,
    "sample_answer": 0.7,
}

# Extra weight for phrase matches in “strong” fields (topic / keywords / sample_questions).
PHRASE_FIELD_WEIGHT: dict[str, float] = {
    "topic": 1.0,
    "keywords": 0.95,
    "sample_questions": 0.85,
    "clean_explanation": 0.55,
    "source_excerpt": 0.5,
    "sample_answer": 0.35,
}

# Each repeated token hit beyond the first adds ``FREQ_GAMMA`` up to ``FREQ_EXTRA_CAP`` extras.
_FREQ_GAMMA = 0.32
_FREQ_EXTRA_CAP = 2

# Soft length normalization: longer chunks don’t dominate purely from token mass.
_LENGTH_NORM_LAMBDA = 0.24

# Explicit lecture / week / class references in the query.
_LECTURE_NUMBER_BONUS = 2.85

# Phrase bonuses (bigram / trigram), scaled by PHRASE_FIELD_WEIGHT; cap per chunk.
_BIGRAM_BONUS_BASE = 1.05
_TRIGRAM_BONUS_BASE = 1.45
_MAX_PHRASE_EVENTS = 5

# Confidence blending (interpretable weights, sum ≈ 1).
_CONF_W_NORM = 0.20
_CONF_W_MARGIN = 0.22
_CONF_W_COVERAGE = 0.28
_CONF_W_PHRASE = 0.14
_CONF_W_STRONG = 0.11
_CONF_W_LEC = 0.05

# Saturating scale for raw score → [0,1) term inside confidence.
_CONF_SCORE_SATURATION = 14.0


# ---------------------------------------------------------------------------
# Stopwords & short-token policy
# ---------------------------------------------------------------------------

_QUERY_STOPWORDS: frozenset[str] = frozenset(
    """
    what how why when where who whom which whose
    explain tells tell describe summary summarize overview
    about between difference different compare comparison versus vs
    the a an is are was were be been being
    do does did done got get
    me my we us our you your they them their
    it its this that these those
    on of at in to for from by as if or not no
    can could should would will may might must shall
    need just like lot really very much some any all each every
    into out up down over
    quiz give show teach help learn
    """.split()
)

# Two-letter tokens are usually noise; allow if not in this junk set *after* stopword pass.
_TWO_LETTER_DROP: frozenset[str] = frozenset(
    "an as at be by do go he if in is it me my no of on or so to up us we am id ok ox".split()
)


# ---------------------------------------------------------------------------
# In-memory cache: raw rows + structured lexical index per chunk
# ---------------------------------------------------------------------------

_row_cache: list[dict[str, Any]] = []
_chunk_indices: dict[int, "ChunkLexicalIndex"] = {}


@dataclass
class ChunkLexicalIndex:
    """Precomputed per-chunk structures for fast lexical scoring."""

    chunk_id: int
    lecture_number: int
    row: dict[str, Any]
    # Ordered tokens per field (for consecutive phrase detection).
    field_tokens: dict[str, list[str]] = field(default_factory=dict)
    # Token counts per field (frequency / strength).
    field_counts: dict[str, Counter[str]] = field(default_factory=dict)
    total_tokens: int = 0


def invalidate_lecture_cache() -> None:
    global _row_cache, _chunk_indices
    _row_cache = []
    _chunk_indices = {}


def load_lecture_cache() -> None:
    """Load DB rows and build lexical indices. Requires Flask app context."""
    global _row_cache, _chunk_indices
    rows = LectureChunk.query.order_by(LectureChunk.id).all()
    _row_cache = []
    _chunk_indices = {}
    for r in rows:
        d = {
            "id": r.id,
            "lecture_number": r.lecture_number,
            "topic": r.topic,
            "keywords": r.keywords,
            "source_excerpt": r.source_excerpt,
            "clean_explanation": r.clean_explanation,
            "sample_questions": r.sample_questions,
            "sample_answer": r.sample_answer,
        }
        _row_cache.append(d)
        idx = _build_chunk_lexical_index(d)
        _chunk_indices[r.id] = idx


# ---------------------------------------------------------------------------
# Optional stemming hook (stdlib-only default: light suffix rules).
# Replace with ``stemmer.lemmatize`` later without changing scoring structure.
# ---------------------------------------------------------------------------

def _default_light_stem(tok: str) -> str:
    """Conservative English trimming; safe default without NLTK."""
    if len(tok) <= 2:
        return tok
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 4 and tok.endswith("es") and not tok.endswith("ses"):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    # Only strip -ing on longer words to avoid ``running`` → ``runn``.
    if len(tok) > 7 and tok.endswith("ing"):
        root = tok[:-3]
        if len(root) >= 4:
            return root
    if len(tok) > 4 and tok.endswith("ed"):
        root = tok[:-2]
        return root if len(root) > 2 else tok
    return tok


# Optional: set to ``your_stemmer.stem`` from NLTK/snowball; kept None for stdlib-only installs.
EXTERNAL_STEMMER: Callable[[str], str] | None = None


def _term_families(token: str) -> frozenset[str]:
    """Query/chunk token equivalence set (original + stem + light variants)."""
    base = {token}
    stem = _default_light_stem(token)
    base.add(stem)
    if EXTERNAL_STEMMER is not None:
        try:
            base.add(EXTERNAL_STEMMER(token))
        except Exception:
            pass
    return frozenset(t for t in base if t and len(t) >= 1)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize_text_to_list(text: str) -> list[str]:
    """Split text into lowercase tokens (chunk fields + query raw pass)."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def tokenize_query_terms(q: str) -> list[str]:
    """
    Tokens used for *matching*: lowercase, drop stopwords and noisy short forms.

    Intended for **queries** only. Chunk fields keep full tokens except we never
    rely on substring containment across word boundaries.
    """
    raw = tokenize_text_to_list(q)
    out: list[str] = []
    for t in raw:
        if t in _QUERY_STOPWORDS:
            continue
        if len(t) < 2:
            continue
        if len(t) == 2 and t in _TWO_LETTER_DROP:
            continue
        out.append(t)
    return out


# Backward-compatible name: older code called ``tokenize_query``.
def tokenize_query(q: str) -> list[str]:
    return tokenize_query_terms(q)


def lecture_numbers_mentioned(q: str) -> set[int]:
    """Lecture / week / class numbers explicitly referenced in the query."""
    ql = q.lower()
    nums: set[int] = set()
    for m in re.finditer(r"(?:lecture|lec\.?|week|class)\s*#?\s*(\d+)", ql):
        nums.add(int(m.group(1)))
    for m in re.finditer(r"\blec\s*(\d+)\b", ql):
        nums.add(int(m.group(1)))
    return nums


# ---------------------------------------------------------------------------
# Field text extraction (single place if JSON shape changes)
# ---------------------------------------------------------------------------

def _keywords_as_text(chunk: dict[str, Any]) -> str:
    try:
        kws = json.loads(chunk.get("keywords") or "[]")
    except json.JSONDecodeError:
        kws = []
    if isinstance(kws, list):
        return " ".join(str(x) for x in kws)
    return str(kws)


def _sample_questions_as_text(chunk: dict[str, Any]) -> str:
    raw = chunk.get("sample_questions")
    if not raw:
        return ""
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return " ".join(str(x) for x in arr)
    except json.JSONDecodeError:
        pass
    return str(raw)


def _field_text_map(chunk: dict[str, Any]) -> dict[str, str]:
    return {
        "topic": chunk.get("topic") or "",
        "keywords": _keywords_as_text(chunk),
        "clean_explanation": chunk.get("clean_explanation") or "",
        "source_excerpt": chunk.get("source_excerpt") or "",
        "sample_questions": _sample_questions_as_text(chunk),
        "sample_answer": chunk.get("sample_answer") or "",
    }


def _build_chunk_lexical_index(row: dict[str, Any]) -> ChunkLexicalIndex:
    texts = _field_text_map(row)
    field_tokens: dict[str, list[str]] = {}
    field_counts: dict[str, Counter[str]] = {}
    total = 0
    for fname in FIELD_WEIGHTS:
        toks = tokenize_text_to_list(texts.get(fname, ""))
        field_tokens[fname] = toks
        field_counts[fname] = Counter(toks)
        total += len(toks)
    return ChunkLexicalIndex(
        chunk_id=int(row["id"]),
        lecture_number=int(row["lecture_number"]),
        row=row,
        field_tokens=field_tokens,
        field_counts=field_counts,
        total_tokens=total,
    )


# ---------------------------------------------------------------------------
# Phrase extraction (on normalized query tokens)
# ---------------------------------------------------------------------------

def _query_bigrams_trigrams(tokens: list[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    bigs: list[tuple[str, str]] = []
    tris: list[tuple[str, str, str]] = []
    for i in range(len(tokens) - 1):
        bigs.append((tokens[i], tokens[i + 1]))
    for i in range(len(tokens) - 2):
        tris.append((tokens[i], tokens[i + 1], tokens[i + 2]))
    return bigs, tris


def _consecutive_match(
    seq: list[str],
    a_fam: frozenset[str],
    b_fam: frozenset[str],
) -> bool:
    for i in range(len(seq) - 1):
        if seq[i] in a_fam and seq[i + 1] in b_fam:
            return True
    return False


def _consecutive_trigram(
    seq: list[str],
    fa: frozenset[str],
    fb: frozenset[str],
    fc: frozenset[str],
) -> bool:
    for i in range(len(seq) - 2):
        if seq[i] in fa and seq[i + 1] in fb and seq[i + 2] in fc:
            return True
    return False


# ---------------------------------------------------------------------------
# Scoring one chunk
# ---------------------------------------------------------------------------

@dataclass
class _ScoreParts:
    """Diagnostics for confidence; kept internal."""

    token_score: float = 0.0
    phrase_score: float = 0.0
    lecture_bonus: float = 0.0
    strong_field_token_score: float = 0.0  # topic + keywords token contributions only
    matched_query_terms: int = 0
    phrase_events: int = 0
    field_scores: dict[str, float] = field(default_factory=dict)


def _score_chunk_lexical(
    idx: ChunkLexicalIndex,
    query_tokens: list[str],
    lecture_hit: bool,
) -> tuple[float, _ScoreParts]:
    parts = _ScoreParts()
    if not query_tokens and not lecture_hit:
        return 0.0, parts

    families = [_term_families(t) for t in query_tokens]
    matched_mask = [False] * len(query_tokens)

    # --- token overlap + frequency (per field, additive across fields) ---
    for fname, w in FIELD_WEIGHTS.items():
        ctr = idx.field_counts.get(fname, Counter())
        field_contrib = 0.0
        for i, fam in enumerate(families):
            raw_c = max(ctr.get(v, 0) for v in fam)
            if raw_c <= 0:
                continue
            matched_mask[i] = True
            extra = min(raw_c - 1, _FREQ_EXTRA_CAP)
            contrib = w * (1.0 + _FREQ_GAMMA * extra)
            parts.token_score += contrib
            field_contrib += contrib
            if fname in ("topic", "keywords"):
                parts.strong_field_token_score += contrib
        if field_contrib > 0:
            parts.field_scores[fname] = field_contrib

    # --- phrase bonuses (consecutive tokens in field token stream) ---
    bigs, tris = _query_bigrams_trigrams(query_tokens)
    phrase_events = 0

    def _phrase_loop() -> None:
        nonlocal phrase_events
        for fname in FIELD_WEIGHTS:
            if phrase_events >= _MAX_PHRASE_EVENTS:
                return
            seq = idx.field_tokens.get(fname, [])
            if len(seq) < 2:
                continue
            scale = PHRASE_FIELD_WEIGHT.get(fname, 0.4)

            for (ta, tb) in bigs:
                if phrase_events >= _MAX_PHRASE_EVENTS:
                    return
                if _consecutive_match(seq, _term_families(ta), _term_families(tb)):
                    parts.phrase_score += _BIGRAM_BONUS_BASE * scale
                    phrase_events += 1

            for (ta, tb, tc) in tris:
                if phrase_events >= _MAX_PHRASE_EVENTS:
                    return
                if _consecutive_trigram(
                    seq,
                    _term_families(ta),
                    _term_families(tb),
                    _term_families(tc),
                ):
                    parts.phrase_score += _TRIGRAM_BONUS_BASE * scale
                    phrase_events += 1

    _phrase_loop()
    parts.phrase_events = phrase_events

    if lecture_hit:
        parts.lecture_bonus = _LECTURE_NUMBER_BONUS

    raw = parts.token_score + parts.phrase_score + parts.lecture_bonus
    if idx.total_tokens > 0:
        raw /= 1.0 + _LENGTH_NORM_LAMBDA * math.log1p(idx.total_tokens)

    parts.matched_query_terms = sum(1 for m in matched_mask if m)
    return raw, parts


def _confidence_from_parts(
    best_score: float,
    second_score: float,
    parts: _ScoreParts,
    query_tokens: list[str],
    lecture_hit: bool,
) -> float:
    """Map diagnostic stats to [0, 1] for UI and logging."""
    eps = 1e-6
    norm_best = best_score / (best_score + _CONF_SCORE_SATURATION)
    margin = (best_score - second_score) / (best_score + eps)
    margin_c = min(1.0, margin * 1.15)

    n_q = max(len(query_tokens), 1)
    coverage = parts.matched_query_terms / n_q

    phrase_c = min(1.0, parts.phrase_events / 3.5)

    denom = parts.token_score + parts.phrase_score + eps
    strong_ratio = (
        (parts.strong_field_token_score + 0.35 * parts.phrase_score) / denom
        if denom > 0
        else 0.0
    )
    strong_c = min(1.0, strong_ratio)

    conf = (
        _CONF_W_NORM * norm_best
        + _CONF_W_MARGIN * margin_c
        + _CONF_W_COVERAGE * coverage
        + _CONF_W_PHRASE * phrase_c
        + _CONF_W_STRONG * strong_c
    )
    if lecture_hit and parts.lecture_bonus > 0:
        conf += _CONF_W_LEC
    return float(min(1.0, max(0.0, conf)))


def score_chunks_keyword(query: str, rows: list[dict[str, Any]], top_k: int) -> RetrievalResult:
    """
    Rank chunks using the lexical index. ``rows`` is ignored in favor of the global
    cache indices (callers still pass ``_row_cache`` for API stability).
    """
    del rows  # single source of truth: _chunk_indices aligned with _row_cache

    q_tokens = tokenize_query_terms(query)
    lec_nums = lecture_numbers_mentioned(query)

    def _empty_diag() -> RetrievalDiagnostics:
        return RetrievalDiagnostics(
            query_tokens=q_tokens,
            lecture_numbers_detected=sorted(lec_nums),
            retrieval_backend="keyword",
            top_k_requested=top_k,
            num_chunks_scored=len(_chunk_indices),
            num_chunks_hit=0,
            top_score=0.0,
            second_score=0.0,
            score_margin=0.0,
            query_coverage=0.0,
        )

    if not _chunk_indices:
        return RetrievalResult(
            chunks=[], confidence=0.0, detected_topic=None, diagnostics=_empty_diag()
        )

    scored: list[tuple[float, ChunkLexicalIndex, _ScoreParts]] = []
    for cid, idx in _chunk_indices.items():
        lec_hit = idx.lecture_number in lec_nums
        if not q_tokens and not lec_hit:
            scored.append((0.0, idx, _ScoreParts()))
            continue
        raw, parts = _score_chunk_lexical(idx, q_tokens, lec_hit)
        scored.append((raw, idx, parts))

    scored.sort(key=lambda x: x[0], reverse=True)
    best, second = scored[0][0], scored[1][0] if len(scored) > 1 else 0.0
    num_hit = sum(1 for s, _, _ in scored if s > 0)

    if best <= 0:
        return RetrievalResult(
            chunks=[], confidence=0.0, detected_topic=None, diagnostics=_empty_diag()
        )

    top_entries = [(s, idx, p) for s, idx, p in scored if s > 0][:top_k]
    if not top_entries:
        return RetrievalResult(
            chunks=[], confidence=0.0, detected_topic=None, diagnostics=_empty_diag()
        )

    _, best_idx, best_parts = top_entries[0]
    conf = _confidence_from_parts(
        best,
        second,
        best_parts,
        q_tokens,
        best_idx.lecture_number in lec_nums,
    )

    if not q_tokens and lec_nums:
        conf = max(conf, 0.42)

    top_rows = [idx.row for _, idx, _ in top_entries]
    detected = (top_rows[0].get("topic") or "").split("—")[0].strip() if top_rows else None

    # Build per-chunk diagnostics for the selected entries
    eps = 1e-6
    n_q = max(len(q_tokens), 1)
    chunk_diags: list[ChunkHitDiag] = []
    for rank_0, (score, cidx, parts) in enumerate(top_entries):
        chunk_diags.append(
            ChunkHitDiag(
                chunk_id=cidx.chunk_id,
                rank=rank_0 + 1,
                score=score,
                token_score=parts.token_score,
                phrase_score=parts.phrase_score,
                lecture_bonus=parts.lecture_bonus,
                strong_field_token_score=parts.strong_field_token_score,
                matched_query_terms=parts.matched_query_terms,
                phrase_events=parts.phrase_events,
                field_scores=dict(parts.field_scores),
            )
        )

    diag = RetrievalDiagnostics(
        query_tokens=q_tokens,
        lecture_numbers_detected=sorted(lec_nums),
        retrieval_backend="keyword",
        top_k_requested=top_k,
        num_chunks_scored=len(_chunk_indices),
        num_chunks_hit=num_hit,
        top_score=best,
        second_score=second,
        score_margin=(best - second) / (best + eps),
        query_coverage=best_parts.matched_query_terms / n_q,
        chunk_hits=chunk_diags,
    )

    return RetrievalResult(
        chunks=[_row_to_public_dict(r) for r in top_rows],
        confidence=conf,
        detected_topic=detected or None,
        diagnostics=diag,
    )


def retrieve_chunks(
    query: str,
    *,
    top_k: int = 5,
    backend: Literal["keyword", "embedding"] = "keyword",
) -> RetrievalResult:
    """
    Retrieve top matching chunks.

    ``backend="keyword"`` — lexical engine (this module).

    ``backend="embedding"`` — reserved: should eventually call a dense retriever and
    return the same ``RetrievalResult`` shape (optionally merging with lexical scores).
    """
    if backend == "embedding":
        raise NotImplementedError("embedding retrieval is not implemented yet")
    if backend != "keyword":
        raise ValueError(f"unknown retrieval backend: {backend!r}")
    return score_chunks_keyword(query, _row_cache, top_k)


def retrieve(query: str, top_k: int = 5) -> RetrievalResult:
    """Backward-compatible alias for keyword retrieval."""
    return retrieve_chunks(query, top_k=top_k, backend="keyword")


def _row_to_public_dict(row: dict[str, Any]) -> dict[str, Any]:
    src = row["source_excerpt"]
    return {
        "id": row["id"],
        "lecture_number": row["lecture_number"],
        "topic": row["topic"],
        "keywords": row["keywords"],
        "source_excerpt": src,
        "source_text": src,
        "clean_explanation": row["clean_explanation"],
        "sample_questions": row["sample_questions"],
        "sample_answer": row["sample_answer"],
    }


def format_course_answer(chunks: list[dict[str, Any]]) -> str:
    """Build the mandatory Course Answer block from retrieved chunks only."""
    lines: list[str] = ["Course Answer:", ""]
    for c in chunks:
        num = c.get("lecture_number")
        topic = c.get("topic", "")
        expl = (c.get("clean_explanation") or "").strip()
        if not expl:
            expl = (c.get("source_excerpt") or "").strip()
        lines.append(f"Lecture {num} — {topic}")
        for part in expl.split("\n"):
            part = part.strip()
            if part:
                lines.append(f"- {part}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Legacy helpers (optional diagnostics / admin). Safe to keep; not used in hot path.
# ---------------------------------------------------------------------------

def build_retrieval_blob(chunk: Mapping[str, Any]) -> str:
    """Single lowercased string of all fields — useful for grep/debug only."""
    texts = _field_text_map(dict(chunk))
    return " ".join(texts[k] for k in FIELD_WEIGHTS if k in texts).lower()


def build_topic_blob(chunk: Mapping[str, Any]) -> str:
    """Topic + keyword text for debugging."""
    d = dict(chunk)
    return ((d.get("topic") or "") + " " + _keywords_as_text(d)).lower()
