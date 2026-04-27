"""Deterministic direct-answer selection for chat / definition / compare modes.

The "direct answer" is the very first sentence the user reads. Before this
module existed it was always derived at render time from
``primary_chunks[0]``'s first bullet — which means *retrieval drift* showed up
twice (the first chunk wasn't always the most concept-grounded chunk in the
pool, and even when it was, its first bullet wasn't always a definition).

:func:`select_direct_answer` ports the spec's per-mode rules into one place:

- **Compare (two-entity)**: synthesizes a deterministic contrast string from
  the V2 evidence bundles. Always mentions both labels and uses the
  ``"…while … focuses on …"`` template so the validator's both-side check
  passes.
- **Compare (multi-entity)**: short list of entity labels plus the existing
  comparison axes.
- **Chat / direct definition / multi-step / scoped**: ranks candidate
  sentences from target-scoped chunks using definition cues + target-alias
  hits + (negative) forbidden-term hits, then returns the top sentence
  normalized for terminal punctuation.
- **Summary / quiz / synthesis / "may include incorrect statements"**:
  returns ``None`` — the renderers in those modes own their own opening and
  shouldn't be overridden.
"""

from __future__ import annotations

import re
from typing import Any

from app.services.answers.concept_constraints import (
    ConceptConstraints,
    has_definition_cue,
    is_line_concept_pure,
    line_has_forbidden,
    line_has_target,
)  # noqa: F401  (line_has_forbidden re-exported for tests / downstream callers)
from app.services.answers.entity_retrieval import (
    EvidenceBundleLike,
    _term_hits,
    score_chunk_for_entity,
)
from app.services.knowledge.concept_kb import ConceptKB
from app.services.knowledge.structured_query import StructuredQuery


# Modes that should *not* produce a direct answer. The renderer for each of
# these owns its own opening (lecture summary blurb, quiz prompt, multi-
# lecture synthesis lede), so overriding the opener with a single sentence
# would break the format.
_NO_DIRECT_ANSWER_MODES = frozenset(
    {
        "lecture_summary",
        "teaching_plus_check",
        "cross_lecture_synthesis",
        "compare_multi",
    }
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_PREFIX_RE = re.compile(
    r"^(?:#+\s+|course answer\s*:|direct answer\s*:|definition\s*:|answer\s*:)\s*",
    re.IGNORECASE,
)
_BULLET_PREFIX_RE = re.compile(r"^[-•*]\s*")
_MIN_LEN = 30
_MAX_LEN = 280

# Lead-noun-phrase extractor for the compare contrast: we want the head noun
# of the bundle's first core line so we can fill ``axisA`` / ``axisB``. The
# pattern grabs everything between the first occurrence of a target alias
# and the first sentence terminator (or up to ~80 chars). It's intentionally
# coarse — we'd rather pull "spatial features extracted via convolution" than
# nothing.
_LEAD_PHRASE_AFTER_CUE_RE = re.compile(
    r"\b(?:is|are|focuses on|extracts?|computes?|maps?|uses?|"
    r"models?|captures?|operates on)\s+(.{8,80}?)(?:[.;,!?]|\s+(?:and|while|whereas|but)\s|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_decorations(line: str) -> str:
    """Remove markdown / heading prefixes and bullets so the line is readable as prose."""
    text = (line or "").strip()
    if not text:
        return ""
    text = _BULLET_PREFIX_RE.sub("", text)
    text = _HEADING_PREFIX_RE.sub("", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Naive sentence split that keeps each piece's terminal punctuation."""
    body = (text or "").strip()
    if not body:
        return []
    raw_lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    sentences: list[str] = []
    for ln in raw_lines:
        clean = _strip_decorations(ln)
        if not clean:
            continue
        for piece in _SENTENCE_SPLIT_RE.split(clean):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def _normalize_terminal(text: str) -> str:
    body = text.strip()
    if not body:
        return body
    if not re.search(r"[.!?]$", body):
        body = body + "."
    return body


def _looks_like_skip(line: str) -> bool:
    """Throw out heading-like, table-like, or example-marker fragments."""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("###") or stripped.startswith("##"):
        return True
    if stripped.startswith("|") or stripped.endswith("|"):
        return True
    if stripped.lower().startswith(("for example", "e.g.", "example:", "the key idea:")):
        return True
    return False


def _topic_head(chunk: dict[str, Any]) -> str:
    """First clause of the chunk's ``topic`` — used for the compare-axis fallback."""
    raw = str(chunk.get("topic", "")).strip()
    if not raw:
        return ""
    head = re.split(r"\s*[—\-:|]\s*", raw, maxsplit=1)[0].strip()
    return head


# ---------------------------------------------------------------------------
# Chat / definition path
# ---------------------------------------------------------------------------


def _chunk_is_target_scoped(
    chunk: dict[str, Any], constraints: ConceptConstraints
) -> bool:
    """True when the chunk's ``topic`` or ``keywords`` mention a target alias.

    Used as the gating signal for non-relational chat queries: candidate
    sentences must come *either* from a chunk that's clearly about the
    target, *or* mention the target alias themselves. Without this gate,
    same-lecture neighbour topics (e.g. *"Forward pass: compute output …"*
    showing up next to *"CNNs and Residuals — CNN"*) leak into the opener.
    """
    if not constraints.target_aliases:
        return False
    topic = (
        str(chunk.get("topic", ""))
        + " "
        + str(chunk.get("keywords", ""))
    ).lower()
    if not topic.strip():
        return False
    for term in constraints.target_aliases:
        if term and _term_hits(topic, term) > 0:
            return True
    return False


def _candidate_sentences_for_chat(
    chunks: list[dict[str, Any]],
    constraints: ConceptConstraints | None,
    *,
    kb: ConceptKB | None,
    primary_concept_id: str | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Sentence pool for chat ranking — pairs each sentence with its source chunk.

    Pulls from ``clean_explanation`` first, then ``source_excerpt``, then
    ``sample_answer``. Pairing the sentence with its chunk lets the ranker
    use chunk-level signals (topic head, entity score) when scoring.

    For non-relational chat queries, a sentence must come from a
    target-scoped chunk *or* mention a target alias itself — otherwise
    same-lecture neighbour topics drift into the opener (e.g. *"Forward
    pass: compute output …"* leaking into a *What is CNN?* answer). The
    relational case keeps the looser gate so shared-vocabulary sentences
    survive.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_target_scoped = (
            constraints is not None and _chunk_is_target_scoped(chunk, constraints)
        )
        sources = [
            chunk.get("clean_explanation"),
            chunk.get("source_excerpt"),
            chunk.get("sample_answer"),
        ]
        for src in sources:
            if not src:
                continue
            for sentence in _split_sentences(str(src)):
                if _looks_like_skip(sentence):
                    continue
                length = len(sentence)
                if length < _MIN_LEN or length > _MAX_LEN:
                    continue
                key = sentence.strip().lower()[:240]
                if key in seen:
                    continue
                if constraints is not None and not is_line_concept_pure(sentence, constraints):
                    continue
                if (
                    constraints is not None
                    and constraints.target_aliases
                    and not constraints.is_relational
                    and not chunk_target_scoped
                    and not line_has_target(sentence, constraints)
                ):
                    continue
                seen.add(key)
                out.append((sentence, chunk))
    return out


def _rank_chat_sentence(
    sentence: str,
    chunk: dict[str, Any],
    constraints: ConceptConstraints | None,
) -> tuple[float, int]:
    """Score a candidate sentence; higher is better, length tiebreaker."""
    score = 0.0
    if has_definition_cue(sentence):
        score += 2.0
    line_lower = sentence.lower()
    if constraints and constraints.target_aliases:
        alias_hits = sum(
            1 for term in constraints.target_aliases if _term_hits(line_lower, term) > 0
        )
        score += float(alias_hits)
    topic_head = _topic_head(chunk).lower()
    if topic_head:
        if any(cue in topic_head for cue in ("definition", "core idea", "overview", "introduction")):
            score += 1.0
        if constraints:
            for term in constraints.target_aliases:
                if term and _term_hits(topic_head, term) > 0:
                    score += 1.0
                    break
    # Bonus for sentences from a chunk whose ``topic`` mentions the target —
    # this disambiguates two sibling chunks under the same lecture header
    # (e.g. *"CNNs and Residuals — CNN"* vs *"CNNs and Residuals —
    # Residuals"* for a *What is CNN?* query) where keywords overlap heavily
    # but the trailing topic suffix names exactly one of them.
    if constraints is not None and constraints.target_aliases:
        full_topic = str(chunk.get("topic", "")).lower()
        keywords_blob = str(chunk.get("keywords", "")).lower()
        topic_hit = any(
            term and _term_hits(full_topic, term) > 0
            for term in constraints.target_aliases
        )
        keyword_hit = any(
            term and _term_hits(keywords_blob, term) > 0
            for term in constraints.target_aliases
        )
        if topic_hit:
            score += 1.5
        elif keyword_hit:
            score += 0.5
    if constraints and constraints.forbidden_terms:
        forbidden_hits = sum(
            1 for term in constraints.forbidden_terms if _term_hits(line_lower, term) > 0
        )
        score -= 2.0 * forbidden_hits
    length = len(sentence)
    if 60 <= length <= 200:
        score += 0.4
    return score, -length


def _select_chat_direct_answer(
    chunks: list[dict[str, Any]],
    constraints: ConceptConstraints | None,
    kb: ConceptKB | None,
) -> str | None:
    """Pick the highest-ranking definition-style sentence from ``chunks``.

    Returns ``None`` when ``constraints`` is missing or has no target
    aliases — without a target alias set we have no signal to score "is this
    sentence about the right concept", so it's safer to defer to the legacy
    renderer than to risk shipping a worse opener. Live pipeline calls
    always thread constraints through; the ``None`` short-circuit keeps
    legacy unit-test call sites that bypass the pipeline (e.g.
    ``build_answer_plan(sq, chunks, supporting, kb=kb)``) on the original
    code path.
    """
    if constraints is None or not constraints.target_aliases:
        return None
    primary_id = (
        constraints.target_concepts[0] if constraints.target_concepts else None
    )
    candidates = _candidate_sentences_for_chat(
        chunks, constraints, kb=kb, primary_concept_id=primary_id
    )
    if not candidates:
        return None
    scored = [
        (_rank_chat_sentence(sent, chunk, constraints), sent) for sent, chunk in candidates
    ]
    scored.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
    best_score, best_sentence = scored[0]
    if best_score[0] < 0:
        return None
    return _normalize_terminal(best_sentence)


# ---------------------------------------------------------------------------
# Compare path (two-entity)
# ---------------------------------------------------------------------------


def _aliases_for_bundle(bundle: EvidenceBundleLike) -> list[str]:
    aliases = list(getattr(bundle, "aliases", []) or [])
    label = getattr(bundle, "label", "") or ""
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        if not raw:
            return
        term = str(raw).strip().lower()
        if not term or term in seen:
            return
        seen.add(term)
        out.append(term)

    _add(getattr(bundle, "concept_id", None))
    _add(label)
    for alias in aliases:
        _add(alias)
    return out


def _axis_phrase_for_bundle(bundle: EvidenceBundleLike) -> str:
    """Pull a short noun phrase that captures the bundle's mechanism / focus."""
    aliases = _aliases_for_bundle(bundle)
    for line in list(bundle.core_lines)[:6]:
        text = _strip_decorations(line)
        if not text:
            continue
        match = _LEAD_PHRASE_AFTER_CUE_RE.search(text)
        if match:
            phrase = match.group(1).strip().strip(",;:")
            if phrase and len(phrase) <= 80:
                return phrase
    # Fallback to the topic head of the first evidence chunk.
    for chunk in bundle.evidence_chunks[:3]:
        head = _topic_head(chunk)
        if head and head.lower() not in {a.lower() for a in aliases}:
            return head
    # Final fallback: first core line lowercased + truncated.
    for line in bundle.core_lines[:3]:
        text = _strip_decorations(line)
        if text:
            return text[:80].rstrip(" ,;:.") + ("…" if len(text) > 80 else "")
    return "its own mechanism"


def _select_compare_direct_answer(
    sq: StructuredQuery,
    bundles: list[EvidenceBundleLike] | None,
) -> str | None:
    if not bundles or len(bundles) < 2:
        # Try to fall back to comparison_axes via the structured query's
        # compare entities — but only if both sides are nameable.
        if sq.intent.compare_entities and len(sq.intent.compare_entities) >= 2:
            a, b = sq.intent.compare_entities[0], sq.intent.compare_entities[1]
            return (
                f"{a} and {b} are related, but they differ along role, "
                "computation, and typical use."
            )
        if sq.intent.compare_concepts:
            a, b = sq.intent.compare_concepts
            return (
                f"{a} and {b} are related, but they differ along role, "
                "computation, and typical use."
            )
        return None
    bundle_a = bundles[0]
    bundle_b = bundles[1]
    label_a = bundle_a.label or bundle_a.concept_id
    label_b = bundle_b.label or bundle_b.concept_id
    axis_a = _axis_phrase_for_bundle(bundle_a)
    axis_b = _axis_phrase_for_bundle(bundle_b)
    return (
        f"{label_a} and {label_b} are related, but {label_a} focuses on "
        f"{axis_a}, while {label_b} focuses on {axis_b}."
    )


def _select_compare_multi_direct_answer(
    sq: StructuredQuery,
    bundles: list[EvidenceBundleLike] | None,
) -> str | None:
    """Short list opener for 3+ entity compare. Returns ``None`` per spec.

    The compare_multi renderer already has its own opener (the architecture
    matrix table) and the spec calls out that compare_multi sits in the
    ``_NO_DIRECT_ANSWER_MODES`` set, so this exists only as a placeholder for
    parity with the per-mode dispatch table.
    """
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def select_direct_answer(
    sq: StructuredQuery,
    *,
    chunks: list[dict[str, Any]],
    bundles: list[EvidenceBundleLike] | None = None,
    constraints: ConceptConstraints | None = None,
    kb: ConceptKB | None = None,
) -> str | None:
    """Mode-aware direct answer or ``None`` when the renderer should keep its lede.

    ``bundles`` should be passed in compare order (``[bundle_a, bundle_b]``)
    when available — that's what the compare renderer expects. ``constraints``
    is required for chat / definition ranking; without it the chat path
    returns ``None`` so callers fall back to the legacy direct-answer
    derivation in :func:`answer_generation._direct_answer_and_skip`.
    """
    mode = sq.answer_intent
    if mode in _NO_DIRECT_ANSWER_MODES:
        return None
    if sq.response_constraints.allow_incorrect_statements:
        return None

    if mode == "compare":
        return _select_compare_direct_answer(sq, bundles)

    return _select_chat_direct_answer(chunks, constraints, kb)
