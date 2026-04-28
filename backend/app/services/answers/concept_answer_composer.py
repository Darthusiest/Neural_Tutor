"""Rule-based Concept Answer Composer: classify retrieved lines by semantic role,
dedupe/purity-filter, and assemble the tutor-style narrative (Course Answer → paragraphs).

Uses generic lexical cues only—no topic-specific hardcoding.
"""

from __future__ import annotations

import re
from typing import Any

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
    r"\b(?:uses|processes|extracts?|computes?|maps?|works?\s+by|operates|applies|performs|"
    r"takes|outputs|produces|transforms|combines|generates|builds|runs|implements)\b",
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


def _reads_like_standalone_process_sentence(line: str) -> bool:
    """True when the line already opens like a mechanism sentence (no ``It works by`` needed)."""
    s = (line or "").strip()
    if not s:
        return False
    return bool(
        re.match(
            r"^(It |They |There |This\b|The\b|[\w\-]+\s+(uses|computes|maps|transforms|"
            r"works\s+by|applies|runs|builds|outputs|normalizes|extracts)\b)",
            s,
            re.IGNORECASE,
        )
    )


def _format_it_works_by_sentence(line: str) -> str:
    """Turn a mechanism line into a first explanation sentence (what it does)."""
    raw = _strip_key_section_labels(line)
    raw = raw.strip()
    if not raw:
        return ""
    if _reads_like_standalone_process_sentence(raw):
        return _ensure_terminal_period(raw)
    core = raw.rstrip(".!?")
    if core and core[0].isupper() and not raw.startswith(
        ("It ", "The ", "This ", "They ", "There ")
    ):
        core = core[0].lower() + core[1:]
    inner = core.strip()
    return _ensure_terminal_period(f"It works by {inner}")


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


def _build_coherent_explanation_paragraph(
    buckets: dict[str, list[str]],
    ordered: list[str],
    contrast_pat: Any,
    opening_norm_keys: set[str],
    ag: Any,
) -> tuple[str, set[str]]:
    """Three-beat explanation: mechanism → interpretation → result (single flowing paragraph).

    Uses transitions "This means that …" and "As a result, …" instead of concatenating raw lines.
    Returns ``(paragraph_text, segment_norm_keys_used)`` for downstream dedupe with ``The key idea:``.
    """
    used_segment_norms: set[str] = set()

    mech_line = ""
    mech_how = ""  # "mechanism" | "contrast" | "definition" | ""
    for cand in buckets.get(MECHANISM) or []:
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
            m_sent = _format_it_works_by_sentence(mech_line)
        else:
            m_sent = _ensure_terminal_period(_strip_key_section_labels(mech_line))
        if m_sent:
            sentences.append(m_sent)

    if interp_line:
        inner_i = _strip_key_section_labels(interp_line).strip()
        inner_i = inner_i.rstrip(".!?")
        if inner_i and not _norm_overlap(mech_line, inner_i):
            bridge = _ensure_terminal_period(f"This means that {inner_i}")
            sentences.append(bridge)

    if result_line:
        inner_r = _strip_key_section_labels(result_line).strip()
        inner_r = inner_r.rstrip(".!?")
        tail_ref = interp_line or mech_line
        if inner_r and not _norm_overlap(tail_ref, inner_r):
            sentences.append(_ensure_terminal_period(f"As a result, {inner_r}"))
    if len(sentences) < 2 and mech_line:
        for cand in buckets.get(DEFINITION) or []:
            nk = _normalize_for_dedupe(cand)
            if nk in opening_norm_keys or nk in used_segment_norms:
                continue
            inner = _strip_key_section_labels(cand).strip().rstrip(".!?")
            if inner and not _norm_overlap(mech_line, inner):
                sentences.append(_ensure_terminal_period(f"In practice, {inner}"))
                used_segment_norms.add(nk)
                break

    if not sentences:
        return "", used_segment_norms

    paragraph = " ".join(sentences)
    return paragraph, used_segment_norms


def compose_concept_answer(
    plan: AnswerPlan,
    evidence: list[dict[str, Any]],
    structured_query: StructuredQuery | None,
    *,
    constraints: ConceptConstraints | None = None,
) -> str:
    """Compose tutor narrative matching :func:`render_tutor_style_answer` layout."""
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
    uf = ag._user_forbidden_set(structured_query)
    if uf and raw_direct_answer and ag._line_contains_user_forbidden(raw_direct_answer, uf):
        raw_direct_answer = ""

    legacy_example_prefetch = ag._example_intuition_block(primary)

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

    contrast_pat = ag._CONTRAST_CUE_PATTERN

    # Opening paragraph
    opening_src = raw_direct_answer
    if not opening_src.strip():
        defs = buckets.get(DEFINITION) or []
        if defs:
            opening_src = defs[0]
    if not opening_src.strip():
        mechs = buckets.get(MECHANISM) or []
        phrase = _lead_mechanism_phrase(mechs[0]) if mechs else "process the inputs introduced in this lecture"
        label = concept_label or "This topic"
        opening_src = f"{label} is a method described in the course materials, used to {phrase.strip()}."
    opening_para = ag._natural_opening_sentence(opening_src, concept_label)

    opening_norm_keys = set()
    if opening_src:
        for sentence in ag._SENTENCE_SPLIT_PATTERN.split(opening_src.strip()):
            k = _normalize_for_dedupe(sentence)
            if k:
                opening_norm_keys.add(k)

    explanation_para, explanation_segment_norms = _build_coherent_explanation_paragraph(
        buckets,
        ordered,
        contrast_pat,
        opening_norm_keys,
        ag,
    )
    used_norms_for_key: set[str] = set(opening_norm_keys)
    used_norms_for_key.update(explanation_segment_norms)

    paragraphs: list[str] = [opening_para]
    if explanation_para:
        paragraphs.append(
            ag._truncate_to_first_sentences(explanation_para, max_sentences=5)
        )

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
            example_block_lines = ag._format_example_block(example_text)

    paragraphs = ag._dedupe_paragraphs([p for p in paragraphs if p])
    if uf:
        paragraphs = [p for p in paragraphs if not ag._line_contains_user_forbidden(p, uf)]

    rendered_lines: list[str] = ["Course Answer:", ""]
    for paragraph in paragraphs:
        rendered_lines.append(paragraph)
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

    if uf and key_idea and ag._line_contains_user_forbidden(key_idea, uf):
        key_idea = (
            f"{concept_label} is the anchor idea here."
            if concept_label
            else "The anchor idea is the definition in your notes."
        )

    rendered_lines.extend(["The key idea:", key_idea, ""])

    # Closer — relevance bucket first
    rel_lines = buckets.get(RELEVANCE) or []
    why_it_matters = ""
    if rel_lines:
        rel0 = rel_lines[0].strip()
        rl = rel0.lower()
        if rl.startswith("that matters because"):
            why_it_matters = ag._truncate_to_first_sentences(rel0, max_sentences=2)
        elif rl.startswith(("this matters because", "it matters because")):
            why_it_matters = ag._truncate_to_first_sentences(rel0, max_sentences=2)
        else:
            # Strip redundant causal prefixes so we prepend canonical wording once.
            rest = rel0
            if rl.startswith("because "):
                rest = rest[8:].strip()
                rest = rest[0].upper() + rest[1:] if rest else rest
            why_it_matters = ag._truncate_to_first_sentences(
                f"That matters because {rest}",
                max_sentences=2,
            )
    if not why_it_matters:
        why_it_matters = ag._truncate_to_first_sentences(
            ag._grounded_why_it_matters(plan, primary, concept_label, user_forbidden=uf),
            max_sentences=2,
        )

    rendered_lines.append(why_it_matters)
    out = "\n".join(rendered_lines).rstrip()

    if uf and ag._line_contains_user_forbidden(out, uf):
        lines = out.split("\n")
        kept = [ln for ln in lines if not ag._line_contains_user_forbidden(ln, uf)]
        out = "\n".join(kept).strip()
    return out
