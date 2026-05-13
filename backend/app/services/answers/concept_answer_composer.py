"""Rule-based Concept Answer Composer: classify retrieved lines by semantic role,
dedupe/purity-filter, and assemble the tutor-style narrative (Course Answer → paragraphs).

Uses generic lexical cues only—no topic-specific hardcoding.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.answers.concept_constraints import (
    ConceptConstraints,
    has_definition_cue,
    is_line_concept_pure,
    line_has_forbidden,
    line_has_target,
)
from app.services.answers.direct_answer import _chunk_is_target_scoped, _split_sentences
from app.services.answers.entity_retrieval import _term_hits
from app.services.knowledge.structured_query import StructuredQuery

# Semantic roles (string constants).
DEFINITION = "definition"
MECHANISM = "mechanism"
KEY_IDEA = "key_idea"
EXAMPLE = "example"
RELEVANCE = "relevance"

_MECHANISM_RE = re.compile(
    r"\b(?:uses|processes|extracts?|computes?|maps?|captures?|converts?|works?\s+by|operates|"
    r"applies|performs|takes|outputs|produces|transforms|combines|generates|builds|runs|implements)\b",
    re.IGNORECASE,
)

_KEY_IDEA_PHRASE_RE = re.compile(
    r"\b(?:key\s+idea|main\s+idea|core\s+idea|in\s+short|in\s+essence|"
    r"essentially|the\s+goal\s+of)\b",
    re.IGNORECASE,
)

_EXAMPLE_RE = re.compile(
    r"\b(?:for\s+example|e\.g\.|imagine|think\s+of|like\s+a|analogous\s+to|"
    r"such\s+as|consider|suppose|pretend)\b",
    re.IGNORECASE,
)

_RELEVANCE_RE = re.compile(
    r"\b(?:matters|important|used\s+for|used\s+to|helps?\s+(?:with|to|us)|"
    r"enables?|allows?|why\s+(?:we|this)|in\s+practice|critical\s+for|essential\s+for|useful\s+for)\b",
    re.IGNORECASE,
)

# Imported lazily from answer_generation in helpers that need them — avoids init-order surprises.
_AG_IMPORTS_DONE = False
_SENTENCE_SPLIT_PATTERN = None
_CONTRAST_CUE_PATTERN = None
_NUMERIC_EXAMPLE_PATTERN = None
_GENERIC_FILLER_PATTERNS = None


def _ag():
    global _AG_IMPORTS_DONE, _SENTENCE_SPLIT_PATTERN, _CONTRAST_CUE_PATTERN
    global _NUMERIC_EXAMPLE_PATTERN, _GENERIC_FILLER_PATTERNS
    if not _AG_IMPORTS_DONE:
        from app.services.answers import answer_generation as ag

        _SENTENCE_SPLIT_PATTERN = ag._SENTENCE_SPLIT_PATTERN
        _CONTRAST_CUE_PATTERN = ag._CONTRAST_CUE_PATTERN
        _NUMERIC_EXAMPLE_PATTERN = ag._NUMERIC_EXAMPLE_PATTERN
        _GENERIC_FILLER_PATTERNS = ag._GENERIC_FILLER_PATTERNS
        _AG_IMPORTS_DONE = True
    return (
        _SENTENCE_SPLIT_PATTERN,
        _CONTRAST_CUE_PATTERN,
        _NUMERIC_EXAMPLE_PATTERN,
        _GENERIC_FILLER_PATTERNS,
    )


def _is_generic_filler(line: str) -> bool:
    _, _, _, patterns = _ag()
    return any(p.search(line) for p in (patterns or ()))


def classify_line(line: str) -> str | None:
    """Return one semantic role for ``line``, or ``None`` if no heuristic matches."""
    text = (line or "").strip()
    if len(text) < 12:
        return None

    ssp, contrast_pat, num_pat, _ = _ag()
    num_pat = num_pat or re.compile("$")

    # Priority: explicit definitional cues first (often overlaps “for example, X is a…”).
    if has_definition_cue(text):
        return DEFINITION
    if _EXAMPLE_RE.search(text) or num_pat.search(text):
        return EXAMPLE
    if _RELEVANCE_RE.search(text):
        return RELEVANCE
    if _KEY_IDEA_PHRASE_RE.search(text):
        return KEY_IDEA
    # Prefer mechanism verbs before the short-sentence KEY_IDEA heuristic — otherwise
    # crisp mechanism lines like “It computes …” classify as KEY_IDEA via length.
    if _MECHANISM_RE.search(text):
        return MECHANISM
    # Short summary-like sentence (no contrast scaffold).
    if (
        len(text) <= 110
        and text.endswith(".")
        and (contrast_pat is None or not contrast_pat.search(text))
    ):
        if ssp:
            parts = [s.strip() for s in ssp.split(text) if s.strip()]
            if len(parts) <= 1:
                return KEY_IDEA

    return None


def _composer_skip_line(line: str) -> bool:
    """Drop markdown/table debris only — do not strip ``For example`` / ``key idea`` lines."""
    stripped = (line or "").strip()
    if not stripped:
        return True
    if stripped.startswith("###") or stripped.startswith("##"):
        return True
    if stripped.startswith("|") or stripped.endswith("|"):
        return True
    return False


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(".!?:—-")


def _score_sentence_for_bucket(
    line: str,
    role: str,
    constraints: ConceptConstraints | None,
) -> tuple[float, int]:
    """Higher tuple compares better."""
    score = 0.0
    if role == DEFINITION and has_definition_cue(line):
        score += 2.0
    ll = line.lower()
    if constraints and constraints.target_aliases:
        alias_hits = sum(
            1 for term in constraints.target_aliases if term and _term_hits(ll, term) > 0
        )
        score += float(alias_hits)
    leng = len(line)
    if 60 <= leng <= 200:
        score += 0.4
    forbidden_penalty = 0
    if constraints and constraints.forbidden_terms:
        forbidden_penalty = sum(
            1 for term in constraints.forbidden_terms if term and _term_hits(ll, term) > 0
        )
    score -= 2.0 * forbidden_penalty
    return score, -leng


def _maybe_take_sentence_for_constraints(
    sentence: str,
    chunk: dict[str, Any],
    constraints: ConceptConstraints | None,
) -> bool:
    if constraints is None:
        return True
    if not is_line_concept_pure(sentence, constraints):
        return False
    if constraints.is_relational:
        return True
    if not constraints.target_aliases:
        return True
    if constraints.forbidden_terms and line_has_forbidden(sentence, constraints):
        # Mixed sentences may survive purity; drop forbidden-only lines
        if not line_has_target(sentence, constraints):
            return False
    if _chunk_is_target_scoped(chunk, constraints):
        return True
    return line_has_target(sentence, constraints)


def collect_role_buckets(
    plan: AnswerPlan,
    chunks: list[dict[str, Any]],
    *,
    constraints: ConceptConstraints | None,
    max_per_role: int = 2,
) -> dict[str, Any]:
    """Gather purity-passing sentences into role buckets (≤ ``max_per_role`` each)."""
    from app.services.answers import answer_generation as ag

    primary = ag._primary_chunks_ordered(plan, chunks)

    supporting_ids = list(plan.supporting_chunk_ids or [])[:2]
    supporting: list[dict[str, Any]] = chunks_by_ids(chunks, supporting_ids)

    role_candidates: dict[str, list[tuple[tuple[float, int], str]]] = {
        DEFINITION: [],
        MECHANISM: [],
        KEY_IDEA: [],
        EXAMPLE: [],
        RELEVANCE: [],
    }
    seen_norms: set[str] = set()
    ordered_sentences: list[str] = []

    def consume_chunk(chunk: dict[str, Any]) -> None:
        nonlocal ordered_sentences
        sources = [
            chunk.get("clean_explanation"),
            chunk.get("source_excerpt"),
            chunk.get("sample_answer"),
        ]
        for src in sources:
            if not src:
                continue
            for sentence in _split_sentences(str(src)):
                if len(sentence) < 15:
                    continue
                if _composer_skip_line(sentence):
                    continue
                if _is_generic_filler(sentence):
                    continue
                if not _maybe_take_sentence_for_constraints(sentence, chunk, constraints):
                    continue
                norm = _normalize_for_dedupe(sentence)
                if not norm or norm in seen_norms:
                    continue

                role = classify_line(sentence)
                if not role:
                    continue

                seen_norms.add(norm)
                ordered_sentences.append(sentence)
                sc = _score_sentence_for_bucket(sentence, role, constraints)
                role_candidates[role].append((sc, sentence))

    for c in primary:
        consume_chunk(c)
    for c in supporting:
        consume_chunk(c)

    buckets: dict[str, list[str]] = {}
    for role, cand_list in role_candidates.items():
        cand_list.sort(key=lambda x: x[0], reverse=True)
        picked: list[str] = []
        seen_role: set[str] = set()
        for _sc, sent in cand_list:
            key = _normalize_for_dedupe(sent)
            if key in seen_role:
                continue
            seen_role.add(key)
            picked.append(sent)
            if len(picked) >= max_per_role:
                break
        buckets[role] = picked

    return {"buckets": buckets, "ordered_sentences": ordered_sentences}


def _lead_mechanism_phrase(line: str) -> str:
    """Short noun/verb phrase from a mechanism-like sentence for fallback openers."""
    m = re.search(
        r"\b(?:to|that)\s+([^.!?]{12,120}?)(?:[.;,!]|$)",
        line,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()
    return "process the inputs introduced in this lecture"


_HEADING_LABEL_STRIP = re.compile(
    r"^(?:the\s+)?(?:key|main|core)\s+idea\s*:\s*",
    re.IGNORECASE,
)


def _strip_key_section_labels(text: str) -> str:
    return _HEADING_LABEL_STRIP.sub("", (text or "").strip()).strip()


def _ensure_terminal_period(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if t[-1] not in ".!?":
        t += "."
    return t


# Heads kept capitalized after transitions like ``This means that softmax …``.
_PROPER_HEAD_TERMS = frozenset(
    {"softmax", "mfcc", "mfccs", "cnn", "fft", "dct", "mel", "relu", "gpu", "nlp"}
)

_PRONOUN_SUBJECT_PREFIX_RE = re.compile(
    r"^(?:It|They|This|That|These|Those)\s+",
    re.IGNORECASE,
)


def _lower_first_safe(text: str) -> str:
    """Lowercase first character unless it begins a likely proper noun."""
    t = (text or "").strip()
    if not t:
        return ""
    first_word = t.split()[0]
    lw = first_word.lower().rstrip(".!?,;:—")
    if lw in _PROPER_HEAD_TERMS:
        return t
    if t[0].isupper():
        return t[0].lower() + t[1:]
    return t


def _strip_leading_pronoun_subject(text: str) -> str:
    return _PRONOUN_SUBJECT_PREFIX_RE.sub("", (text or "").strip()).strip()


def _is_complete_sentence(text: str) -> bool:
    """Cheap clause completeness check for stitched explanations."""
    t = (text or "").strip()
    if len(t) < 12:
        return False
    tl = t.lower()
    if tl.startswith(
        (
            "it ",
            "they ",
            "there ",
            "this ",
            "that ",
            "these ",
            "those ",
            "the ",
            "a ",
            "an ",
        )
    ):
        return True
    if _MECHANISM_RE.search(t):
        return True
    if has_definition_cue(t):
        return True
    # Subject–verb opener like ``Softmax converts …``.
    return bool(
        re.match(
            r"^[A-Za-z][\w\-]*\s+(?:uses|computes|maps|transforms|converts|captures|extracts|"
            r"applies|combines|generates|outputs|normalizes|processes)\b",
            t,
            re.IGNORECASE,
        )
    )


def _looks_like_initial_verb_fragment(line: str) -> bool:
    """Fragment led by imperative-like verb (``Captures spatial …``) needing a subject."""
    s = (line or "").strip()
    if not s:
        return False
    first = s.split(maxsplit=1)[0]
    fl = first.lower().rstrip(".!?")
    if fl in {"it", "they", "the", "this", "that", "these", "those", "there", "a", "an"}:
        return False
    # Starts with capitalized verb-ish token + remainder (not ``Softmax converts`` — two-token pattern above).
    return bool(
        re.match(
            r"^[A-Z][a-z]{2,}(?:es|s|ed)?\s+\S",
            s,
        )
        and not re.match(
            r"^[A-Za-z][\w\-]*\s+(?:uses|computes|maps|transforms|converts|captures|extracts)\b",
            s,
            re.IGNORECASE,
        )
    )


def _reads_like_standalone_process_sentence(line: str) -> bool:
    """True when the line already opens like a mechanism sentence (no ``It works by`` needed)."""
    s = (line or "").strip()
    if not s:
        return False
    return bool(
        re.match(
            r"^(It |They |There |This\b|The\b|[\w\-]+\s+(uses|computes|maps|transforms|converts|captures|"
            r"works\s+by|applies|runs|builds|outputs|normalizes|extracts)\b)",
            s,
            re.IGNORECASE,
        )
    )


def _format_mechanism_sentence(line: str, concept_label: str) -> str:
    """Turn a mechanism line into a standalone grammatical sentence."""
    raw = _strip_key_section_labels(line).strip()
    if not raw:
        return ""
    if _reads_like_standalone_process_sentence(raw):
        return _ensure_terminal_period(raw)
    lab = (concept_label or "").strip()
    core = raw.rstrip(".!?").strip()
    if _looks_like_initial_verb_fragment(core):
        frag_rest = core[0].lower() + core[1:] if core else ""
        if lab:
            return _ensure_terminal_period(f"{lab} {frag_rest}")
        return ""
    inner = core
    if inner and inner[0].isupper() and not raw.startswith(
        ("It ", "The ", "This ", "They ", "There ")
    ):
        inner = inner[0].lower() + inner[1:]
    inner = inner.strip()
    return _ensure_terminal_period(f"It works by {inner}") if inner else ""


def _split_interpretation_and_result(key_line: str, ag: Any) -> tuple[str, str]:
    """Split one key-idea line into bridge vs result clauses when possible."""
    cleaned = _strip_key_section_labels(key_line)
    cleaned = cleaned.strip()
    if not cleaned:
        return "", ""
    parts = [p.strip() for p in ag._SENTENCE_SPLIT_PATTERN.split(cleaned) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    if ";" in cleaned:
        a, b = cleaned.split(";", 1)
        return a.strip(), b.strip()
    if " — " in cleaned:
        a, b = cleaned.split(" — ", 1)
        return a.strip(), b.strip()
    return cleaned, ""


def _norm_overlap(a: str, b: str) -> bool:
    """True when ``b`` largely repeats ``a`` (avoid stitched duplicates)."""
    if not a or not b:
        return False
    na, nb = _normalize_for_dedupe(a), _normalize_for_dedupe(b)
    if na == nb:
        return True
    if len(nb) > 20 and nb in na:
        return True
    if len(na) > 20 and na in nb:
        return True
    return False


def _strip_numeric_illustration_sentences(paragraph: str, numeric_pat: Any) -> str:
    """Drop sentences containing bracket/tuple numeric lifts so the example block can own them."""
    if not paragraph or not numeric_pat or not numeric_pat.search(paragraph):
        return paragraph
    ssp, _, _, _ = _ag()
    parts = [s.strip() for s in ssp.split(paragraph) if s.strip()]
    kept = [s for s in parts if not numeric_pat.search(s)]
    return " ".join(kept).strip()


def _sentences_expanded_for_dedupe(para: str, ag: Any) -> list[str]:
    """Split prose including em-dash clauses so sentence dedupe catches ``A — B`` pairs."""
    raw_parts: list[str] = []
    for segment in re.split(r"\s+[—\-]\s+", para):
        for sent in ag._SENTENCE_SPLIT_PATTERN.split(segment):
            st = sent.strip()
            if st:
                raw_parts.append(st)
    return raw_parts if raw_parts else ([para.strip()] if para.strip() else [])


def _strip_paragraphs_repeating_prior_sentences(paragraphs: list[str], ag: Any) -> list[str]:
    """Remove sentences from later paragraphs that repeat a sentence seen in an earlier paragraph."""
    seen: set[str] = set()
    out: list[str] = []
    for para in paragraphs:
        kept_sents: list[str] = []
        for sent in _sentences_expanded_for_dedupe(para, ag):
            sn = _normalize_for_dedupe(sent)
            if sn and sn in seen:
                continue
            if sn:
                seen.add(sn)
            kept_sents.append(sent.rstrip())
        block = " ".join(kept_sents).strip()
        if block:
            out.append(block)
    return out


def _drop_duplicate_example_sentences(example_text: str, opening_para: str, ag: Any) -> str:
    """Remove sentences from example body that repeat the opening paragraph verbatim."""
    if not example_text or not opening_para:
        return example_text
    open_norms = {
        _normalize_for_dedupe(s)
        for s in ag._SENTENCE_SPLIT_PATTERN.split(opening_para)
        if s.strip()
    }
    kept: list[str] = []
    for sent in ag._SENTENCE_SPLIT_PATTERN.split(example_text):
        sn = _normalize_for_dedupe(sent)
        if sn and sn in open_norms:
            continue
        kept.append(sent.strip())
    return " ".join(kept).strip()


def _finalize_transition_clause(
    raw_clause: str,
    *,
    transition_prefix: str,
    concept_label: str,
    mech_line: str,
) -> str | None:
    """Normalize clause after ``This means that`` / ``As a result,``; drop bad fragments."""
    inner = _strip_key_section_labels(raw_clause).strip()
    inner = _strip_leading_pronoun_subject(inner).strip().rstrip(".!?")
    if not inner:
        return None
    inner_adj = _lower_first_safe(inner)
    probe = inner_adj[0].upper() + inner_adj[1:] if inner_adj else ""
    candidate_body = probe
    lab = (concept_label or "").strip()
    if not _is_complete_sentence(probe):
        if lab:
            merged = f"{lab} {inner_adj}".strip()
            if _is_complete_sentence(merged):
                candidate_body = merged
            else:
                return None
        else:
            return None
    if _norm_overlap(mech_line, candidate_body):
        return None
    tail = candidate_body
    fw = tail.split()[0].lower().rstrip(".!,?:;—") if tail.split() else ""
    if fw not in _PROPER_HEAD_TERMS and tail:
        tail_out = tail[0].lower() + tail[1:]
    else:
        tail_out = tail
    return _ensure_terminal_period(f"{transition_prefix}{tail_out}")


def _build_coherent_explanation_paragraph(
    buckets: dict[str, list[str]],
    ordered: list[str],
    contrast_pat: Any,
    opening_norm_keys: set[str],
    ag: Any,
    *,
    concept_label: str,
    prefers_contrast: bool,
) -> tuple[str, set[str]]:
    """Three-beat explanation: mechanism → interpretation → result (single flowing paragraph).

    Uses transitions "This means that …" and "As a result, …" instead of concatenating raw lines.
    Returns ``(paragraph_text, segment_norm_keys_used)`` for downstream dedupe with ``The key idea:``.
    """
    used_segment_norms: set[str] = set()

    mech_line = ""
    mech_how = ""  # "mechanism" | "contrast" | "definition" | ""
    label_for_mech = concept_label or ""

    mechs = buckets.get(MECHANISM) or []
    if prefers_contrast and contrast_pat:
        for cand in mechs:
            if contrast_pat.search(cand):
                nk = _normalize_for_dedupe(cand)
                if nk and nk not in opening_norm_keys:
                    mech_line = cand
                    mech_how = "mechanism"
                    used_segment_norms.add(nk)
                    break

    if not mech_line:
        for cand in mechs:
            nk = _normalize_for_dedupe(cand)
            if nk and nk not in opening_norm_keys:
                mech_line = cand
                mech_how = "mechanism"
                used_segment_norms.add(nk)
                break

    if not mech_line and contrast_pat:
        for sent in ordered:
            if contrast_pat.search(sent):
                nk = _normalize_for_dedupe(sent)
                if nk and nk not in opening_norm_keys:
                    mech_line = sent
                    mech_how = "contrast"
                    used_segment_norms.add(nk)
                    break

    keys = list(buckets.get(KEY_IDEA) or [])
    interp_line = ""
    result_line = ""

    if len(keys) >= 2:
        k0, k1 = keys[0], keys[1]
        n0, n1 = _normalize_for_dedupe(k0), _normalize_for_dedupe(k1)
        if n0 and n0 not in opening_norm_keys:
            interp_line = k0
            used_segment_norms.add(n0)
        if n1 and n1 not in opening_norm_keys and n1 != n0:
            result_line = k1
            used_segment_norms.add(n1)
    elif len(keys) == 1:
        a, b = _split_interpretation_and_result(keys[0], ag)
        nk = _normalize_for_dedupe(keys[0])
        if nk not in opening_norm_keys:
            interp_line = a
            used_segment_norms.add(nk)
            result_line = b

    if not mech_line:
        for cand in buckets.get(DEFINITION) or []:
            nk = _normalize_for_dedupe(cand)
            if nk and nk not in opening_norm_keys:
                mech_line = cand
                mech_how = "definition"
                used_segment_norms.add(nk)
                break

    sentences: list[str] = []

    if mech_line:
        if mech_how in ("mechanism", "contrast"):
            m_sent = _format_mechanism_sentence(mech_line, label_for_mech)
        else:
            m_sent = _ensure_terminal_period(_strip_key_section_labels(mech_line))
        if m_sent:
            sentences.append(m_sent)

    if interp_line:
        bridge = _finalize_transition_clause(
            interp_line,
            transition_prefix="This means that ",
            concept_label=label_for_mech,
            mech_line=mech_line,
        )
        if bridge:
            sentences.append(bridge)

    if result_line:
        tail_ref = interp_line or mech_line
        bridge_r = _finalize_transition_clause(
            result_line,
            transition_prefix="As a result, ",
            concept_label=label_for_mech,
            mech_line=tail_ref,
        )
        if bridge_r:
            sentences.append(bridge_r)

    if len(sentences) < 2 and mech_line:
        for cand in buckets.get(DEFINITION) or []:
            nk = _normalize_for_dedupe(cand)
            if nk in opening_norm_keys or nk in used_segment_norms:
                continue
            inner = _strip_key_section_labels(cand).strip().rstrip(".!?")
            inner = _strip_leading_pronoun_subject(inner).strip()
            inner = _lower_first_safe(inner)
            if inner and not _norm_overlap(mech_line, inner):
                probe = inner[0].upper() + inner[1:] if inner else ""
                if label_for_mech and not _is_complete_sentence(probe):
                    inner = _lower_first_safe(f"{label_for_mech.strip()} {inner}")
                    probe = inner[0].upper() + inner[1:] if inner else ""
                if _is_complete_sentence(probe):
                    sentences.append(_ensure_terminal_period(f"In practice, {inner}"))
                    used_segment_norms.add(nk)
                    break

    if not sentences:
        return "", used_segment_norms

    paragraph = " ".join(sentences)
    return paragraph, used_segment_norms


_PROCESS_STEP_VERB_RE = re.compile(
    r"\b(?:uses|computes|maps|extracts|applies|combines|takes|produces|transforms|generates|"
    r"converts|captures|normalizes|builds)\b",
    re.IGNORECASE,
)


_HARDMAX_PEER_RE = re.compile(r"\bhard-?max\b", re.IGNORECASE)


def _strip_hardmax_peer_sentences(text: str) -> str:
    """Remove sentences naming hardmax/hard-max from softmax-only tutor copy."""
    body = (text or "").strip()
    if not body or not _HARDMAX_PEER_RE.search(body):
        return body
    from app.services.answers import answer_generation as ag

    kept: list[str] = []
    for sent in ag._SENTENCE_SPLIT_PATTERN.split(body):
        piece = sent.strip()
        if not piece or _HARDMAX_PEER_RE.search(piece):
            continue
        kept.append(piece)
    return " ".join(kept).strip()


QueryTypeLabel = Literal[
    "definition",
    "mechanism",
    "step_by_step",
    "comparison",
    "why",
    "limitation",
]


_STEP_BY_STEP_QUERY_RE = re.compile(
    r"\bstep[- ]by[- ]step\b|"
    r"\bwalk me through\b|"
    r"\bprocess of\b|"
    r"\b(steps?|stages?|phases?)\b\s+(?:used|involved|to|of)\b|"
    r"\bcompute\b.*\bfrom\b|"
    r"\bhow .* (?:computed|derived|built|constructed)\b",
    re.IGNORECASE,
)

_COMPARISON_QUERY_RE = re.compile(
    r"\b(difference|differ|vs\.?|versus|contrast|compare|comparison)\b",
    re.IGNORECASE,
)

_LIMITATION_QUERY_RE = re.compile(
    r"\bwhy not\b|\blimitations?\b|\bcan'?t\b|\bdoes not\b|\bfails?\b",
    re.IGNORECASE,
)

_MECHANISM_QUERY_RE = re.compile(
    r"\bhow does\b|\bhow is\b|\bhow do\b|\brole of\b",
    re.IGNORECASE,
)


def classify_query_type(
    sq: StructuredQuery | None,
    *,
    answer_mode: str | None = None,
) -> QueryTypeLabel:
    """Strict query-type taxonomy used by the composer templates."""
    mode = (answer_mode or (sq.answer_intent if sq else "")).strip().lower()
    if mode in {"compare", "compare_multi"}:
        return "comparison"
    raw = (sq.intent.original_query if sq else "").strip().lower()
    if _COMPARISON_QUERY_RE.search(raw):
        return "comparison"
    if _STEP_BY_STEP_QUERY_RE.search(raw):
        return "step_by_step"
    if _LIMITATION_QUERY_RE.search(raw):
        return "limitation"
    if raw.startswith("why "):
        return "why"
    if _MECHANISM_QUERY_RE.search(raw):
        return "mechanism"
    if re.match(r"\s*(?:what\s+is|what\s+are|define|explain)\b", raw):
        return "definition"
    return "definition"


def _classify_query_signals(sq: StructuredQuery | None) -> dict[str, Any]:
    """Legacy query signals shim kept for older tests/callers."""
    qtype = classify_query_type(sq)
    raw = (sq.intent.original_query if sq else "").lower()
    wants_steps = qtype == "step_by_step"
    wants_example = bool(re.search(r"\bexample\b|\bshow me\b|\billustrate\b", raw))
    wants_why = qtype in {"why", "limitation"}
    wants_contrast = qtype == "comparison"
    # Prefer explicit definitional depth before ``step'' / ``computed'' process cues.
    if qtype == "definition":
        depth = "short"
    elif qtype == "step_by_step":
        depth = "process"
    elif qtype in {"why", "limitation"}:
        depth = "medium"
    elif re.search(r"\bexplain\b|\btell me about\b|\bin detail\b", raw):
        depth = "long"
    else:
        depth = "medium"
    return {
        "wants_steps": wants_steps,
        "wants_example": wants_example,
        "wants_why": wants_why,
        "wants_contrast": wants_contrast,
        "depth": depth,
    }


def _normalize_step_sentence(sentence: str, concept_label: str) -> str:
    raw = _strip_key_section_labels(sentence).strip()
    if not raw:
        return ""
    # Strip common leading subjects so steps read as imperative actions.
    if concept_label:
        raw = re.sub(
            rf"^{re.escape(concept_label)}\s+",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
    raw = _PRONOUN_SUBJECT_PREFIX_RE.sub("", raw).strip()
    raw = re.sub(r"^It works by\s+", "", raw, flags=re.IGNORECASE).strip()
    if not raw:
        return ""
    if raw[-1] in ".!?":
        raw = raw[:-1]
    if not raw:
        return ""
    return raw[0].upper() + raw[1:] + "."


def _compose_numbered_steps(
    buckets: dict[str, list[str]],
    ordered: list[str],
    *,
    opening_norm_keys: set[str],
    concept_label: str,
) -> list[str] | None:
    pool: list[str] = []
    pool.extend(buckets.get(MECHANISM) or [])
    for role in (KEY_IDEA, DEFINITION):
        for line in buckets.get(role) or []:
            if _PROCESS_STEP_VERB_RE.search(line):
                pool.append(line)
    for sent in ordered:
        if _PROCESS_STEP_VERB_RE.search(sent):
            pool.append(sent)

    seen: set[str] = set()
    steps: list[str] = []
    for sent in pool:
        nk = _normalize_for_dedupe(sent)
        if nk in seen or nk in opening_norm_keys:
            continue
        step_line = _normalize_step_sentence(sent, concept_label)
        if not step_line:
            continue
        probe = step_line[0].upper() + step_line[1:] if step_line else ""
        if not _is_complete_sentence(probe):
            continue
        steps.append(step_line)
        seen.add(nk)
        if len(steps) >= 4:
            break
    if len(steps) < 3:
        return None
    return steps[:5]


def _strong_why_it_matters(buckets: dict[str, list[str]], _concept_label: str) -> str | None:
    pool: list[str] = []
    pool.extend(buckets.get(RELEVANCE) or [])
    pool.extend(buckets.get(KEY_IDEA) or [])
    blob = " ".join(pool).strip()
    if not blob:
        return None
    cap = None
    task = None
    m = re.search(r"\bused\s+to\s+([^,.;:]{8,160})", blob, re.IGNORECASE)
    if m:
        cap = m.group(1).strip()
    if not cap:
        m = re.search(r"\ballows\s+(?:the\s+)?(?:system\s+)?to\s+([^,.;:]{8,160})", blob, re.IGNORECASE)
        if m:
            cap = m.group(1).strip()
    if not cap:
        m = re.search(r"\benables\s+([^,.;:]{8,160})", blob, re.IGNORECASE)
        if m:
            cap = m.group(1).strip()
    m = re.search(r"\b(?:for|during)\s+([^,.;:]{8,120})", blob, re.IGNORECASE)
    if m:
        task = m.group(1).strip()
    if not task:
        m = re.search(r"\bin\s+([^,.;:]{8,120})", blob, re.IGNORECASE)
        if m:
            task = m.group(1).strip()
    if cap and task:
        return (
            f"This matters because it allows the system to {cap.rstrip('.')}, "
            f"which is important for {task.rstrip('.')}."
        )
    if cap:
        return f"This matters because it allows the system to {cap.rstrip('.')}."
    if task:
        return f"This matters because it supports {task.rstrip('.')}."
    return None


def _looks_broken(text: str) -> bool:
    if re.search(
        r"\bIt works by [a-z]+\s+(?:uses|converts|extracts|computes|maps|applies|combines|"
        r"takes|produces|transforms)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    if re.search(
        r"\b(This means that|As a result,)\s+(?:They|This|That|These|Those|[A-Z][a-z]+s)\b",
        text,
    ):
        return True
    return False


def _failsafe_render(
    *,
    raw_direct_answer: str,
    buckets: dict[str, list[str]],
    opening_para: str,
    concept_label: str,
    key_idea: str,
    why_it_matters: str,
) -> str:
    opening = (raw_direct_answer or "").strip()
    lab = concept_label or ""
    if not opening:
        mechs = buckets.get(MECHANISM) or []
        if mechs:
            opening = _format_mechanism_sentence(mechs[0], lab)
        if not opening:
            opening = (opening_para or "").strip()
    lines = ["Course Answer:", "", opening, "", "The key idea:", key_idea, "", why_it_matters]
    return "\n".join(lines).rstrip()


def compose_concept_answer(
    plan: AnswerPlan,
    evidence: list[dict[str, Any]],
    structured_query: StructuredQuery | None,
    *,
    constraints: ConceptConstraints | None = None,
) -> str:
    """Compose strict query-aware tutor narrative."""
    from app.services.answers import answer_generation as ag

    primary = ag._primary_chunks_ordered(plan, evidence)
    if not primary:
        return (
            "Course Answer:\n\n"
            "I couldn't tie that question to specific notes yet. "
            "Try again with a class vocabulary term (e.g. softmax, attention, MFCC)—"
            "a sharper prompt usually surfaces a concrete example."
        )

    raw_direct_answer, _skip = ag._direct_answer_and_skip(plan, primary)
    concept_label = ag._primary_concept_label(plan, primary, structured_query)
    uf = set(ag._user_forbidden_set(structured_query))
    if constraints is not None and constraints.forbidden_terms:
        uf.update(
            t.strip().lower()
            for t in constraints.forbidden_terms
            if t and str(t).strip()
        )
    if constraints is not None and raw_direct_answer and not constraints.is_relational:
        if constraints.target_aliases and not line_has_target(raw_direct_answer, constraints):
            raw_direct_answer = ""
        if (
            raw_direct_answer
            and constraints.forbidden_terms
            and line_has_forbidden(raw_direct_answer, constraints)
            and not line_has_target(raw_direct_answer, constraints)
        ):
            raw_direct_answer = ""
    if uf and raw_direct_answer and ag._line_contains_user_forbidden(raw_direct_answer, uf):
        raw_direct_answer = ""

    legacy_example_prefetch = ag._example_intuition_block(primary)

    query_type = classify_query_type(structured_query, answer_mode=plan.answer_mode)

    collected = collect_role_buckets(plan, evidence, constraints=constraints)
    buckets: dict[str, list[str]] = collected["buckets"]
    ordered: list[str] = collected["ordered_sentences"]

    def _first_numeric_illustration() -> str:
        """Bracket / tuple numerics may classify as DEFINITION—still lift for the example block."""
        pat = ag._NUMERIC_EXAMPLE_PATTERN
        for sent in ordered:
            if pat.search(sent):
                return sent.strip()
        for role in (DEFINITION, MECHANISM, KEY_IDEA, EXAMPLE, RELEVANCE):
            for sent in buckets.get(role) or []:
                if pat.search(sent):
                    return sent.strip()
        return ""

    numeric_pick = _first_numeric_illustration()
    placeholder = ag._EXAMPLE_INTUITION_PLACEHOLDER.strip()
    legacy_is_placeholder = (legacy_example_prefetch or "").strip() == placeholder
    rc = structured_query.response_constraints if structured_query else None
    blocked = bool(rc and (rc.no_examples or rc.intuition_only))
    # Planner ``include_example`` is False when chunks lack quiz/sample hooks — still surface
    # bracket/tuple illustrations from ``clean_explanation`` when the legacy hook isn't the
    # explicit no-example placeholder (Task 6 stripped fixtures hit the placeholder path).
    want_example_block = not blocked and (
        plan.include_example or (not legacy_is_placeholder and bool(numeric_pick))
    )
    if query_type in {"definition", "step_by_step", "why", "limitation", "comparison"}:
        want_example_block = False
    if query_type == "definition" and not numeric_pick:
        want_example_block = False

    contrast_pat = ag._CONTRAST_CUE_PATTERN
    max_expl_sentences = 3
    np_pat = ag._NUMERIC_EXAMPLE_PATTERN

    # Opening paragraph (optionally strip numeric illustration sentences for example-block ownership)
    opening_src = raw_direct_answer
    if not opening_src.strip():
        defs = buckets.get(DEFINITION) or []
        if defs:
            opening_src = defs[0]
    if not opening_src.strip():
        mechs_o = buckets.get(MECHANISM) or []
        phrase = _lead_mechanism_phrase(mechs_o[0]) if mechs_o else "process the inputs introduced in this lecture"
        label_o = concept_label or "This topic"
        opening_src = f"{label_o} is a method described in the course materials, used to {phrase.strip()}."
    opening_src_for_para = opening_src
    if want_example_block and numeric_pick:
        opening_src_for_para = _strip_numeric_illustration_sentences(opening_src_for_para, np_pat)
    opening_para = ag._natural_opening_sentence(opening_src_for_para, concept_label)

    opening_norm_keys = set()
    if opening_src:
        for sentence in ag._SENTENCE_SPLIT_PATTERN.split(opening_src.strip()):
            k = _normalize_for_dedupe(sentence)
            if k:
                opening_norm_keys.add(k)

    lab = concept_label or ""

    explanation_para = ""
    numbered_steps: list[str] | None = None
    explanation_segment_norms: set[str] = set()
    if query_type in {"mechanism"}:
        explanation_para, explanation_segment_norms = _build_coherent_explanation_paragraph(
            buckets,
            ordered,
            contrast_pat,
            opening_norm_keys,
            ag,
            concept_label=lab,
            prefers_contrast=False,
        )
    elif query_type == "step_by_step":
        numbered_steps = _compose_numbered_steps(
            buckets,
            ordered,
            opening_norm_keys=opening_norm_keys,
            concept_label=lab,
        )
    elif query_type in {"why", "limitation"}:
        for cand in (buckets.get(RELEVANCE) or []) + (buckets.get(KEY_IDEA) or []):
            nk = _normalize_for_dedupe(cand)
            if nk and nk not in opening_norm_keys:
                explanation_para = _ensure_terminal_period(_strip_key_section_labels(cand))
                explanation_segment_norms.add(nk)
                break

    if explanation_para:
        explanation_para = ag._truncate_to_first_sentences(explanation_para, max_sentences=max_expl_sentences)
        if want_example_block and numeric_pick:
            explanation_para = _strip_numeric_illustration_sentences(explanation_para, np_pat)

    used_norms_for_key: set[str] = set(opening_norm_keys)
    used_norms_for_key.update(explanation_segment_norms)

    paragraphs: list[str] = [opening_para]
    if explanation_para:
        paragraphs.append(explanation_para)
    paragraphs = _strip_paragraphs_repeating_prior_sentences(paragraphs, ag)

    # Example block — mirror legacy `_example_intuition_block` priority (sample/question/excerpt)
    # so tests that strip ``sample_answer`` / ``source_excerpt`` still skip the block when the
    # materials only carried a numeric cue in ``clean_explanation``. Fall back to the EXAMPLE
    # role bucket when legacy yields no concrete example.
    example_block_lines: list[str] = []
    if want_example_block:
        legacy_example = legacy_example_prefetch
        bucket_lines = buckets.get(EXAMPLE) or []
        num_pat = ag._NUMERIC_EXAMPLE_PATTERN

        example_text = ""
        leg_stripped = (legacy_example or "").strip()
        if ag._has_concrete_example(legacy_example):
            # Prefer legacy text; if it lacks a numeric illustration but retrieved sentences
            # carry one (often ``clean_explanation`` lines classified as DEFINITION), lift it.
            if num_pat.search(leg_stripped):
                example_text = leg_stripped
            else:
                example_text = numeric_pick or leg_stripped
        elif leg_stripped == placeholder:
            # Task 6 — no sample/excerpt example available; do not substitute role-bucket lines.
            example_text = ""
        else:
            example_text = (bucket_lines[0] if bucket_lines else "").strip()
        if uf and example_text and ag._line_contains_user_forbidden(example_text, uf):
            example_text = ""
        if example_text:
            example_text = _drop_duplicate_example_sentences(example_text, opening_para, ag)
        if example_text and constraints is not None and not constraints.is_relational:
            if constraints.target_concepts == ["softmax"]:
                example_text = _strip_hardmax_peer_sentences(example_text)
        if example_text:
            example_block_lines = ag._format_example_block(example_text)

    paragraphs = ag._dedupe_paragraphs([p for p in paragraphs if p])
    if (
        constraints is not None
        and not constraints.is_relational
        and constraints.target_concepts == ["softmax"]
        and len(paragraphs) > 1
    ):
        paragraphs = [paragraphs[0]] + [
            _strip_hardmax_peer_sentences(p) for p in paragraphs[1:]
        ]
        paragraphs = [p for p in paragraphs if (p or "").strip()]
    if uf:
        paragraphs = [p for p in paragraphs if not ag._line_contains_user_forbidden(p, uf)]

    if rc and rc.one_sentence:
        body = (paragraphs[0] if paragraphs else opening_para or "").strip()
        single = ag._truncate_to_first_sentences(body, max_sentences=1)
        return f"Course Answer:\n\n{single}".rstrip()

    rendered_lines: list[str] = ["Course Answer:", ""]
    for paragraph in paragraphs:
        rendered_lines.append(paragraph)
        rendered_lines.append("")

    if numbered_steps:
        for idx, step in enumerate(numbered_steps, start=1):
            rendered_lines.append(f"{idx}. {step}")
        rendered_lines.append("")

    if example_block_lines:
        rendered_lines.extend(example_block_lines)
        rendered_lines.append("")

    # Key idea — prefer bucket line not overlapping explanation/opening
    key_candidates = buckets.get(KEY_IDEA) or []
    key_idea = ""
    for cand in key_candidates:
        nk = _normalize_for_dedupe(cand)
        if nk and nk not in used_norms_for_key:
            key_idea = ag._truncate_to_first_sentences(cand, max_sentences=1)
            break
    if not key_idea:
        pool_mechanism = buckets.get(MECHANISM) or []
        fallback_expl_lines = (
            [pool_mechanism[0]]
            if pool_mechanism
            else [
                ln
                for ln in ordered[:16]
                if ln and _normalize_for_dedupe(ln) not in used_norms_for_key
            ][:8]
        )
        key_idea = ag._truncate_to_first_sentences(
            ag._key_idea_sentence(opening_src, fallback_expl_lines, concept_label),
            max_sentences=1,
        )
    if _norm_overlap(opening_para, key_idea):
        replacement = ""
        for line in (buckets.get(KEY_IDEA) or []) + (buckets.get(DEFINITION) or []):
            if not _norm_overlap(opening_para, line):
                replacement = ag._truncate_to_first_sentences(line, max_sentences=1)
                break
        if replacement:
            key_idea = replacement
        elif concept_label:
            key_idea = f"{concept_label} is the anchor concept for this question."

    if uf and key_idea and ag._line_contains_user_forbidden(key_idea, uf):
        key_idea = (
            f"{concept_label} is the anchor idea here."
            if concept_label
            else "The anchor idea is the definition in your notes."
        )

    if (
        constraints is not None
        and not constraints.is_relational
        and constraints.target_concepts == ["softmax"]
    ):
        key_idea = _strip_hardmax_peer_sentences(key_idea) or key_idea

    rendered_lines.extend(["The key idea:", key_idea, ""])

    max_why = 2
    why_it_matters = _strong_why_it_matters(buckets, lab) or ""
    if not why_it_matters:
        why_it_matters = ag._truncate_to_first_sentences(
            ag._grounded_why_it_matters(plan, primary, concept_label, user_forbidden=uf),
            max_sentences=max_why,
        )
    else:
        why_it_matters = ag._truncate_to_first_sentences(why_it_matters, max_sentences=max_why)
    if why_it_matters.startswith("This matters because"):
        why_it_matters = "That matters because" + why_it_matters[len("This matters because") :]

    if (
        constraints is not None
        and not constraints.is_relational
        and constraints.target_concepts == ["softmax"]
    ):
        why_it_matters = _strip_hardmax_peer_sentences(why_it_matters)

    rendered_lines.append(why_it_matters)
    out = "\n".join(rendered_lines).rstrip()

    if uf and ag._line_contains_user_forbidden(out, uf):
        lines = out.split("\n")
        kept = [ln for ln in lines if not ag._line_contains_user_forbidden(ln, uf)]
        out = "\n".join(kept).strip()
    return out
