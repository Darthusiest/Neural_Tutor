"""Structured course answer text from an :class:`AnswerPlan` (rule-based by default).

For chat-shaped intents (``direct_definition``, ``multi_step_explanation``,
``scoped_explanation``, ``simplified_reteach``, ``cross_lecture_synthesis``, …)
the default output is a natural tutor-tone narrative composed by
:func:`concept_answer_composer.compose_concept_answer`: role-classified evidence
→ opening sentence → explanation paragraph → optional numeric/example lift →
``The key idea:`` → ``That matters because …``. Legacy helper :func:`render_tutor_style_answer`
remains for reference but is no longer the primary path.

Compare / compare_multi / lecture_summary / quiz paths and
exotic response constraints (``no_examples``, ``intuition_only``,
``exact_explanation_count``, ``repeat_explanation_times``,
``allow_incorrect_statements``) keep the legacy four-section markdown layout
(Direct Answer / Explanation / Example / Why it matters).

Student-facing text only—no lecture IDs, keyword dumps, or retrieval jargon.
"""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from typing import Any

from app.services.answers.concept_constraints import ConceptConstraints, build_concept_constraints
from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.answers.compare_render import (
    format_multi_entity_compare_markdown,
    format_two_entity_compare_markdown,
)
from app.services.knowledge.concept_kb import ConceptKB, get_kb
from app.services.knowledge.structured_query import StructuredQuery
from app.services.retrieval import _sample_questions_as_text


def _bullet_lines_from_chunk(lecture_chunk: dict[str, Any]) -> list[str]:
    expl = (lecture_chunk.get("clean_explanation") or "").strip()
    if not expl:
        expl = (lecture_chunk.get("source_excerpt") or "").strip()
    lines: list[str] = []
    for raw_line in expl.split("\n"):
        line = raw_line.strip()
        if line:
            lines.append(line)
    return lines[:16]


def _strip_bullet_prefix(line: str) -> str:
    return re.sub(r"^[-•*]\s*", "", line.strip()).strip()


def _compose_direct_answer_with_count(lines: list[str]) -> tuple[str, int]:
    """Direct answer text and how many leading lines of ``lines`` it consumed (for explanation dedup)."""
    if not lines:
        return "", 0
    first = _strip_bullet_prefix(lines[0])
    if not first:
        return "", 0
    has_terminal = bool(re.search(r"[.!?]\s*$", first))
    if len(first) >= 100 and has_terminal:
        return first[:420], 1
    if len(lines) >= 2:
        second = _strip_bullet_prefix(lines[1])
        if second and (len(first) < 100 or not has_terminal):
            merged = f"{first.rstrip('.')} — {second}"
            if len(merged) <= 450:
                return merged, 2
    return (_first_sentence_or_line(first) or first[:420]), 1


def _first_sentence_or_line(text: str, max_len: int = 420) -> str:
    """First sentence if clear; else first line; capped for a short 'Direct Answer'."""
    trimmed = text.strip()
    if not trimmed:
        return ""
    # Prefer sentence boundary in first segment
    sentence_match = re.match(r"([^.!?]+[.!?])(\s|$)", trimmed[:800])
    if sentence_match:
        return sentence_match.group(1).strip()
    first_line = trimmed.split("\n")[0].strip()
    return first_line[:max_len] + ("…" if len(first_line) > max_len else "")


def _dedupe_lines(lines: list[str], cap: int = 22) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for candidate in lines:
        key = candidate.strip().lower()[:240]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(candidate.strip())
        if len(out) >= cap:
            break
    return out


def _primary_chunks_ordered(plan: AnswerPlan, all_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_chunk_ids = plan.primary_chunk_ids or []
    ordered = chunks_by_ids(all_chunks, primary_chunk_ids)
    if ordered:
        return ordered
    return list(all_chunks)[:8]


# Sentinel for the legacy "Think of the explanation above as the core
# picture…" placeholder — used by the tutor renderer to detect the
# no-good-example case and skip the example block (Task 6 fallback rule).
_EXAMPLE_INTUITION_PLACEHOLDER = (
    "Think of the explanation above as the core picture—ask if you want a different angle "
    "or a walkthrough with numbers."
)


def _example_intuition_block(primary: list[dict[str, Any]]) -> str:
    for chunk in primary[:3]:
        sample_answer = (chunk.get("sample_answer") or "").strip()
        if sample_answer and sample_answer not in ("[]", "null"):
            return sample_answer[:600]
    for chunk in primary[:3]:
        paired_question = _sample_questions_as_text(chunk).strip()
        if paired_question and paired_question not in ("[]", "null", "None"):
            return (
                f"A question the materials pair with this topic: {paired_question[:500]}"
                if len(paired_question) < 400
                else paired_question[:600]
            )
    if primary:
        excerpt = (primary[0].get("source_excerpt") or "").strip()
        if len(excerpt) > 40:
            return _first_sentence_or_line(excerpt[:500]) or excerpt[:280]
    return _EXAMPLE_INTUITION_PLACEHOLDER


def _has_concrete_example(example_text: str) -> bool:
    """``True`` when ``example_text`` is real content (not the no-example placeholder)."""
    body = (example_text or "").strip()
    if not body:
        return False
    return body != _EXAMPLE_INTUITION_PLACEHOLDER.strip()


def _user_forbidden_set(sq: StructuredQuery | None) -> set[str]:
    if sq is None or not sq.response_constraints.forbidden_topics:
        return set()
    return {
        t.strip().lower()
        for t in sq.response_constraints.forbidden_topics
        if t and str(t).strip()
    }


def _line_contains_user_forbidden(line: str, forb: set[str]) -> bool:
    if not line or not forb:
        return False
    low = line.lower()
    for f in forb:
        if not f:
            continue
        if f in low:
            return True
        if len(f) > 3 and f.endswith("s") and f[:-1] in low:
            return True
        if len(f) > 3 and not f.endswith("s") and (f + "s") in low:
            return True
    return False


def _drop_lines_matching_forbidden(lines: list[str], forb: set[str]) -> list[str]:
    if not forb:
        return list(lines)
    return [ln for ln in lines if not _line_contains_user_forbidden(ln, forb)]


def _evidence_text_blob(primary: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for c in primary:
        for k in ("topic", "keywords", "clean_explanation", "source_excerpt"):
            parts.append(str(c.get(k) or ""))
    return " ".join(parts).lower()


def _related_concepts_in_evidence(
    plan: AnswerPlan,
    primary: list[dict[str, Any]],
    kb: ConceptKB,
    *,
    user_forbidden: set[str] | None = None,
) -> list[str]:
    """At most one KB ``related`` name that actually appears in primary chunk text."""
    blob = _evidence_text_blob(primary)
    uf = {x.strip().lower() for x in (user_forbidden or set()) if x and str(x).strip()}
    out: list[str] = []
    for name in plan.include_related_concepts or []:
        raw = (name or "").strip()
        if not raw:
            continue
        nl = raw.lower()
        if any(f and f in nl for f in uf):
            continue
        meta = kb.get_concept(raw)
        if meta:
            terms = [meta.name.lower()] + [a.lower() for a in meta.aliases]
            if any(t and t in blob for t in terms):
                out.append(raw)
        elif raw.lower() in blob:
            out.append(raw)
        if out:
            break
    return out[:1]


def _why_matters_block(plan: AnswerPlan, structured_query: StructuredQuery, primary: list[dict[str, Any]]) -> str:
    """Tutor-style closing—no lecture IDs, scope lists, or 'graph' jargon."""
    kb = get_kb()
    uf = {
        t.strip().lower()
        for t in (structured_query.response_constraints.forbidden_topics or [])
        if t and str(t).strip()
    }
    parts: list[str] = []
    related_ev = _related_concepts_in_evidence(plan, primary, kb, user_forbidden=uf)
    if related_ev:
        parts.append(
            f"You’ll see how this connects to {related_ev[0]} in the materials you already have for this topic."
        )
    if plan.comparison_axes and plan.answer_mode == "compare":
        parts.append(
            "Getting the contrast right matters when you interpret model behavior or compare architectures."
        )
    if not parts and structured_query.concept_ids:
        cid = structured_query.concept_ids[0]
        meta = kb.get_concept_by_id(cid)
        label = (meta.name if meta else cid) or "this idea"
        parts.append(
            f"Understanding {label} clearly helps you follow the rest of the topics in this course."
        )
    if not parts:
        parts.append(
            "Clear intuition for this topic makes the next ideas in the course easier to follow."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tutor-style narrative renderer (chat-mode only)
# ---------------------------------------------------------------------------

# Modes that should flow as a tutor narrative rather than the legacy
# four-section markdown layout (compare / quiz / lecture_summary short-circuit earlier).
_CHAT_NARRATIVE_MODES = frozenset(
    {
        "direct_definition",
        "multi_step_explanation",
        "scoped_explanation",
        "simplified_reteach",
        "teaching_plus_check",
        "cross_lecture_synthesis",
    }
)

_CONTRAST_CUE_PATTERN = re.compile(
    r"\b(vs\.?|versus|instead|whereas|while|unlike|rather than|"
    r"however|in contrast|differs|differ\sfrom|hardmax|hard-max)\b",
    re.IGNORECASE,
)

# Captures bracketed numeric arrays like "[2, 5]" or "[0.12, 0.88]" and
# inline numeric tuples like "0.12, 0.88" — used to lift numeric examples
# onto their own line for readability.
_NUMERIC_EXAMPLE_PATTERN = re.compile(
    r"\[[\s\-+0-9.,]+\]|"
    r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?){1,}",
)

# Generic filler phrases that the legacy `_why_matters_block` (and similarly
# bland prose) tends to emit. Lines containing any of these are dropped before
# being used as opening / contrast / key-idea source material.
_GENERIC_FILLER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"you[’']ll keep running into", re.IGNORECASE),
    re.compile(r"you will keep running into", re.IGNORECASE),
    re.compile(r"this topic connects to", re.IGNORECASE),
    re.compile(r"solid intuition here makes the next topics", re.IGNORECASE),
    re.compile(r"notation and vocabulary pay off later", re.IGNORECASE),
    re.compile(
        r"think of the explanation above as the core picture", re.IGNORECASE
    ),
    re.compile(r"see the explanation below for how the notes develop", re.IGNORECASE),
    re.compile(r"captures sound shape", re.IGNORECASE),
    re.compile(r"essence of sound", re.IGNORECASE),
    re.compile(r"compact representation", re.IGNORECASE),
    re.compile(r"fingerprint of sound", re.IGNORECASE),
    re.compile(r"keeps reappearing alongside", re.IGNORECASE),
    re.compile(r"clear intuition for this topic", re.IGNORECASE),
    re.compile(r"high-level picture", re.IGNORECASE),
    re.compile(r"\bbig picture\b", re.IGNORECASE),
    re.compile(r"\bstay(s)?\s+high.level\b", re.IGNORECASE),
    re.compile(r"in the materials you already have", re.IGNORECASE),
    re.compile(r"building blocks for later topics", re.IGNORECASE),
    re.compile(r"^concept a\s*:", re.IGNORECASE),
    re.compile(r"^concept b\s*:", re.IGNORECASE),
    re.compile(r"forward pass:\s*compute output", re.IGNORECASE),
    re.compile(r"\[1,\s*2\]\s*→", re.IGNORECASE),
)

_STRICT_BANNED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"captures sound shape", re.IGNORECASE),
    re.compile(r"essence of sound", re.IGNORECASE),
    re.compile(r"compact representation", re.IGNORECASE),
    re.compile(r"forward pass:\s*compute output", re.IGNORECASE),
    re.compile(r"concept a\s*:", re.IGNORECASE),
    re.compile(r"concept b\s*:", re.IGNORECASE),
)

# Splits a paragraph into sentences for sentence-level dedupe. Conservative
# regex (no lookbehind on abbreviations) — close enough for tutor copy.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(])")


def _is_generic_filler(line: str) -> bool:
    return any(pattern.search(line) for pattern in _GENERIC_FILLER_PATTERNS)


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower().rstrip(".!?:—-")


def strict_clarification_answer(
    query_type: str | None = None,
    *,
    reason: str | None = None,
) -> str:
    kind = (query_type or "").strip().lower()
    if kind == "step_by_step":
        body = (
            "I don't have enough lecture material to give you a clean step-by-step answer for that. "
            'Try asking with a specific course term (e.g. "How is the mel filterbank computed?").'
        )
    elif kind == "comparison" or (reason or "").startswith("compare_"):
        body = (
            "I don't have enough clean lecture evidence to compare those concepts without mixing them. "
            "Try naming two explicit concepts from class so I can contrast them directly."
        )
    else:
        body = (
            "I don't have enough lecture-grounded evidence to answer that cleanly yet. "
            "Try a narrower question with one explicit class concept."
        )
    return f"Course Answer:\n\n{body}"


def _query_type_for_audit(plan: AnswerPlan, sq: StructuredQuery) -> str:
    try:
        from app.services.answers.concept_answer_composer import classify_query_type

        return classify_query_type(sq, answer_mode=plan.answer_mode)
    except Exception:
        if plan.answer_mode in {"compare", "compare_multi"}:
            return "comparison"
        return "definition"


def _opening_line(answer: str) -> str:
    lines = [ln.strip() for ln in (answer or "").split("\n") if ln.strip()]
    for ln in lines:
        if ln in {"Course Answer:", "The key idea:"}:
            continue
        if re.match(r"^\d+\.\s+", ln):
            continue
        return ln
    return ""


def _key_idea_line(answer: str) -> str:
    lines = [ln.strip() for ln in (answer or "").split("\n")]
    for idx, ln in enumerate(lines):
        if ln.strip() == "The key idea:":
            for follow in lines[idx + 1 :]:
                if follow.strip():
                    return follow.strip()
    return ""


def _why_line(answer: str) -> str:
    lines = [ln.strip() for ln in (answer or "").split("\n") if ln.strip()]
    for ln in reversed(lines):
        if ln in {"Course Answer:", "The key idea:"}:
            continue
        if re.match(r"^\d+\.\s+", ln):
            continue
        return ln
    return ""


def _explanation_text(answer: str, opening: str, key_idea: str, why: str) -> str:
    lines = [ln.strip() for ln in (answer or "").split("\n") if ln.strip()]
    out: list[str] = []
    for ln in lines:
        if ln in {"Course Answer:", "The key idea:", opening, key_idea, why}:
            continue
        if ln.startswith("Think of it this way:"):
            continue
        if re.match(r"^\d+\.\s+", ln):
            continue
        out.append(ln)
    return " ".join(out).strip()


def _assert_section_disjoint(opening: str, explanation: str, key_idea: str, why: str) -> bool:
    parts = [opening, explanation, key_idea, why]
    norm = [_normalize_for_dedupe(p) for p in parts]
    for i in range(len(norm)):
        if not norm[i]:
            continue
        for j in range(i + 1, len(norm)):
            if not norm[j]:
                continue
            if norm[i] == norm[j]:
                return False
            if len(norm[i]) > 24 and norm[i] in norm[j]:
                # The key-idea line is often an intentional one-sentence recap of the mechanism
                # paragraph; treating that as overlap should not force a clarification-only reply.
                if i == 2 or j == 2:
                    continue
                return False
            if len(norm[j]) > 24 and norm[j] in norm[i]:
                if i == 2 or j == 2:
                    continue
                return False
    return True


def _mentions_target_in_opening(
    opening: str,
    sq: StructuredQuery,
    constraints: ConceptConstraints | None,
) -> bool:
    if sq.answer_intent in {"compare", "compare_multi", "cross_lecture_synthesis"}:
        return True
    low = (opening or "").lower()
    if not low:
        return False
    if constraints is not None and not constraints.is_relational and constraints.target_aliases:
        return any(term and term in low for term in constraints.target_aliases)
    if not sq.concept_ids:
        return True
    kb = get_kb()
    concept = kb.get_concept_by_id(sq.concept_ids[0])
    if concept is None:
        return True
    aliases = [concept.name.lower(), *[a.lower() for a in concept.aliases[:8]]]
    return any(alias and alias in low for alias in aliases)


def audit_rendered_answer(
    text: str,
    query_type: str,
    plan: AnswerPlan,
    sq: StructuredQuery,
    *,
    concept_constraints: ConceptConstraints | None = None,
) -> str | None:
    if plan.requires_clarification:
        return strict_clarification_answer(query_type, reason=plan.clarification_reason)
    body = (text or "").strip()
    if not body:
        return strict_clarification_answer(query_type)
    # Compare answers quote scoped lecture lines; those may contain phrasing we
    # ban in chat-mode composed output (e.g. "Forward pass: compute output").
    if query_type != "comparison" and any(
        p.search(body) for p in _STRICT_BANNED_PATTERNS
    ):
        return strict_clarification_answer(query_type)

    uf_global = _user_forbidden_set(sq)
    if uf_global and _line_contains_user_forbidden(body, uf_global):
        return strict_clarification_answer(query_type)

    if sq.response_constraints.one_sentence:
        opening = _opening_line(body)
        if not opening:
            return strict_clarification_answer(query_type)
        if not _mentions_target_in_opening(opening, sq, concept_constraints):
            return strict_clarification_answer(query_type)
        return body

    if query_type in {"definition", "mechanism", "step_by_step", "why", "limitation"}:
        if "The key idea:" not in body:
            return strict_clarification_answer(query_type)

    if query_type == "step_by_step":
        step_lines = [
            ln.strip()
            for ln in body.split("\n")
            if re.match(r"^\d+\.\s+\S", ln.strip())
        ]
        if len(step_lines) < 3:
            return strict_clarification_answer(query_type)

    if query_type == "comparison":
        from app.services.answers.answer_validation import _must_match_compare_contract

        if not _must_match_compare_contract(body, sq, get_kb()):
            return strict_clarification_answer(query_type)
        # Multi-entity compare uses a dedicated table layout (not tutor sections).
        if "### Compared architectures" in body:
            return body
        sentence_counts: Counter[str] = Counter()
        for raw in _SENTENCE_SPLIT_PATTERN.split(body):
            t = raw.strip().lower()
            if len(t) > 20:
                sentence_counts[t] += 1
        if any(c >= 3 for c in sentence_counts.values()):
            return strict_clarification_answer(query_type)

    opening = _opening_line(body)
    key_idea = _key_idea_line(body)
    why = _why_line(body)
    explanation = _explanation_text(body, opening, key_idea, why)
    if not _assert_section_disjoint(opening, explanation, key_idea, why):
        return strict_clarification_answer(query_type)
    if not _mentions_target_in_opening(opening, sq, concept_constraints):
        return strict_clarification_answer(query_type)
    return body


def _clean_explanation_lines(
    explanation_lines: list[str],
    *,
    direct_answer: str = "",
) -> list[str]:
    """Sentence-level cleanup applied before the renderer reads explanation lines.

    Drops generic filler, deduplicates lines that are the same sentence as
    the direct answer (after whitespace/punctuation normalization), and
    deduplicates lines that repeat each other.
    """
    seen: set[str] = set()
    direct_answer_keys: set[str] = set()
    if direct_answer:
        for sentence in _SENTENCE_SPLIT_PATTERN.split(direct_answer.strip()):
            normalized = _normalize_for_dedupe(sentence)
            if normalized:
                direct_answer_keys.add(normalized)
    cleaned: list[str] = []
    for raw_line in explanation_lines:
        line = _strip_bullet_prefix(raw_line)
        if not line or _is_generic_filler(line):
            continue
        normalized = _normalize_for_dedupe(line)
        if not normalized or normalized in seen or normalized in direct_answer_keys:
            continue
        seen.add(normalized)
        cleaned.append(line)
    return cleaned


def _dedupe_paragraphs(paragraphs: list[str]) -> list[str]:
    """Drop paragraphs that repeat earlier ones (same sentence after normalization)."""
    seen: set[str] = set()
    out: list[str] = []
    for paragraph in paragraphs:
        normalized = _normalize_for_dedupe(paragraph)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(paragraph)
    return out


def _truncate_to_first_sentences(text: str, max_sentences: int = 2) -> str:
    """Keep at most ``max_sentences`` sentences—prevents dense text blocks."""
    trimmed = text.strip()
    if not trimmed:
        return ""
    sentences = [s for s in _SENTENCE_SPLIT_PATTERN.split(trimmed) if s.strip()]
    if len(sentences) <= max_sentences:
        return trimmed
    return " ".join(sentences[:max_sentences]).strip()


def _primary_concept_label(
    plan: AnswerPlan,
    primary: list[dict[str, Any]],
    structured_query: StructuredQuery | None = None,
) -> str:
    """Best human-readable concept name for grounding the closing sentence.

    Topic strings in the corpus often carry a section suffix (e.g.
    ``Softmax — Core Idea`` or ``Lecture 4: Backpropagation``). For tutor
    closers we just want the bare concept name—strip lecture prefixes and any
    trailing dash-delimited section qualifier.
    """
    uf = set()
    if structured_query and structured_query.response_constraints.forbidden_topics:
        uf = {
            t.strip().lower()
            for t in structured_query.response_constraints.forbidden_topics
            if t and str(t).strip()
        }
    if structured_query and structured_query.concept_ids:
        kb = get_kb()
        meta = kb.get_concept_by_id(structured_query.concept_ids[0])
        if meta and (meta.name or "").strip():
            return (meta.name or "").strip()
    if primary:
        topic_value = (primary[0].get("topic") or "").strip()
        if topic_value:
            cleaned = re.sub(
                r"^lecture\s+\d+\s*[:\-—]\s*", "", topic_value, flags=re.IGNORECASE
            ).strip()
            cleaned = re.split(r"\s+[—\-–:]\s+", cleaned, maxsplit=1)[0].strip()
            if cleaned:
                cl = cleaned.lower()
                if not any(f and f in cl for f in uf):
                    return cleaned
    if plan.include_related_concepts:
        return plan.include_related_concepts[0]
    return ""


def _natural_opening_sentence(direct_answer: str, concept_label: str) -> str:
    """Tutor-tone opening paragraph (≤ 2 sentences) for the response.

    Strips legacy section labels that sometimes leak in (`Direct Answer:`,
    `Definition:`, `Answer:`) and trims long compositions back to the first two
    sentences so the response opens with a short, readable paragraph.
    """
    text = (direct_answer or "").strip()
    if not text:
        if concept_label:
            return f"Here is how the course frames {concept_label}."
        return "Here is what the notes say about this topic."
    text = re.sub(
        r"^(direct answer|definition|answer)\s*[:\-—]\s*", "", text, flags=re.IGNORECASE
    )
    return _truncate_to_first_sentences(text, max_sentences=2)


def _format_example_block(example_text: str) -> list[str]:
    """Lines for a 'Think of it this way' example block, with numeric arrays lifted out."""
    body = (example_text or "").strip()
    if not body:
        return []
    intro = "Think of it this way:"
    block: list[str] = [intro, ""]
    numeric_match = _NUMERIC_EXAMPLE_PATTERN.search(body)
    if numeric_match:
        before = body[: numeric_match.start()].strip(" .,—-:")
        match_text = numeric_match.group(0).strip()
        after = body[numeric_match.end():].strip(" .,—-:")
        if before:
            sentence = before
            if not sentence.endswith((".", ":", "?", "!")):
                sentence = sentence + ":"
            block.append(sentence)
            block.append("")
        block.append(match_text)
        if after:
            block.append("")
            tail = after
            if not tail.endswith((".", "?", "!")):
                tail = tail + "."
            block.append(tail)
    else:
        block.append(body)
    return block


def _contrast_paragraphs(
    explanation_lines: list[str], concept_label: str
) -> list[str]:
    """Up to two short contrast / clarification paragraphs (each ≤ 2 sentences)."""
    candidates: list[str] = []
    for line in explanation_lines:
        if not _CONTRAST_CUE_PATTERN.search(line):
            continue
        clean = _strip_bullet_prefix(line)
        if not clean:
            continue
        candidates.append(_truncate_to_first_sentences(clean, max_sentences=2))
    if not candidates:
        return []
    paragraphs: list[str] = [candidates[0]]
    for follow_up in candidates[1:]:
        if _normalize_for_dedupe(follow_up) == _normalize_for_dedupe(candidates[0]):
            continue
        if concept_label and concept_label.lower() in follow_up.lower():
            paragraphs.append(follow_up)
            break
    return paragraphs


def _key_idea_sentence(
    direct_answer: str, explanation_lines: list[str], concept_label: str
) -> str:
    """Single distilled sentence for the 'The key idea:' highlight."""
    short_candidates: list[str] = []
    for line in explanation_lines[:8]:
        cleaned = _strip_bullet_prefix(line)
        if cleaned and 12 <= len(cleaned) <= 160:
            short_candidates.append(cleaned)
    if concept_label:
        for cleaned in short_candidates:
            if concept_label.lower() in cleaned.lower():
                return _first_sentence_or_line(cleaned, max_len=160) or cleaned
    if short_candidates:
        return _first_sentence_or_line(short_candidates[0], max_len=160) or short_candidates[0]
    if direct_answer:
        return _first_sentence_or_line(direct_answer, max_len=160) or direct_answer[:160]
    if concept_label:
        return f"{concept_label} is the anchor concept here."
    return "Stay close to the course definition."


def _grounded_why_it_matters(
    plan: AnswerPlan,
    primary: list[dict[str, Any]],
    concept_label: str,
    *,
    user_forbidden: set[str] | None = None,
) -> str:
    """Short, concept-tied closer (1–2 sentences).

    Always begins with a causal cue (``"That matters because"``) so validators
    like ``must_answer_how_or_why`` keep passing for chat-style intents.
    Intentionally avoids the legacy generic phrasings (``"You'll keep running
    into ..."`` / ``"This topic connects to ..."``) — :func:`_is_generic_filler`
    would also strip those if they ever leaked in.
    """
    name = concept_label or "this idea"
    nl_name = name.lower()
    uf = {x.strip().lower() for x in (user_forbidden or set()) if x and str(x).strip()}
    if any(f and f in nl_name for f in uf):
        name = "this idea"
    kb = get_kb()
    related_ev = _related_concepts_in_evidence(
        plan, primary, kb, user_forbidden=user_forbidden
    )
    if related_ev:
        related_phrase = related_ev[0]
        out = (
            f"That matters because {name} keeps reappearing alongside {related_phrase}, "
            "so a clean grasp here makes the next idea easier to read."
        )
        if not uf or not _line_contains_user_forbidden(out, uf):
            return out
    if primary:
        topic_value = (primary[0].get("topic") or "").strip()
        if topic_value and topic_value.lower() != name.lower():
            head = re.split(r"\s+[—\-–:]\s+", topic_value, maxsplit=1)[0].strip()
            out = (
                f"That matters because understanding {name} clearly frames how you interpret "
                f"the ideas bundled under {head}."
            )
            if not uf or not _line_contains_user_forbidden(out, uf):
                return out
    return (
        f"That matters because clear intuition for {name} is what lets the next idea land "
        "without feeling arbitrary."
    )


def render_tutor_style_answer(
    plan: AnswerPlan,
    evidence: list[dict[str, Any]],
    structured_query: StructuredQuery | None = None,
) -> str:
    """Tutor-tone narrative answer for chat-mode replies.

    Replaces the legacy ``### Direct Answer / Explanation / Example / Why it matters``
    section markdown with a flowing, teaching-style response. The only explicit
    label kept in the output is ``"The key idea:"``.

    Layout (each section is its own short paragraph, blank line between):

    1. **Opening** — the ``direct_answer`` from the pipeline, redundancy-trimmed
       and capped at two sentences.
    2. **Contrast / clarification (optional)** — up to two short paragraphs
       lifted from explanation lines that contain contrast cues
       (``vs``, ``instead``, ``unlike``, ``hardmax``, …).
    3. **Concrete example (optional)** — the example block from
       ``example_lines``; bracketed numeric arrays are lifted onto their own
       line so the example reads visually.
    4. **The key idea:** — one short, concept-mentioning sentence pulled from
       the cleaned explanation lines.
    5. **Closer** — short ``That matters because …`` sentence grounded in the
       concept name and related-concept list (never the legacy generic copy).

    Cleanup applied before rendering:
    - generic filler phrases (``"You'll keep running into …"`` /
      ``"This topic connects to …"`` / leftover scaffold lines) are stripped;
    - sentences already used in the opening are removed from the explanation
      pool so they cannot reappear in contrast / key-idea;
    - the final answer is paragraph-deduped so no paragraph repeats earlier
      content verbatim.

    Intentionally scoped to chat-mode answer modes — compare, compare_multi,
    lecture_summary, and cross_lecture_synthesis remain on the legacy layout
    (they own deterministic per-mode renderers elsewhere).
    """
    primary = _primary_chunks_ordered(plan, evidence)
    if not primary:
        return (
            "Course Answer:\n\n"
            "I couldn't tie that question to specific notes yet. "
            "Try again with a class vocabulary term (e.g. softmax, attention, MFCC)—"
            "a sharper prompt usually surfaces a concrete example."
        )

    raw_direct_answer, lines_consumed_by_direct_answer = _direct_answer_and_skip(plan, primary)
    raw_explanation_lines = _build_explanation_bullets(
        plan, evidence, primary, skip_first_chunk_lines=lines_consumed_by_direct_answer
    )
    concept_label = _primary_concept_label(plan, primary, structured_query)
    uf = _user_forbidden_set(structured_query)
    if uf and raw_direct_answer and _line_contains_user_forbidden(raw_direct_answer, uf):
        raw_direct_answer = ""
    cleaned_explanation_lines = _clean_explanation_lines(
        raw_explanation_lines, direct_answer=raw_direct_answer
    )
    cleaned_explanation_lines = _drop_lines_matching_forbidden(cleaned_explanation_lines, uf)

    paragraphs: list[str] = []
    paragraphs.append(_natural_opening_sentence(raw_direct_answer, concept_label))

    contrast = _contrast_paragraphs(cleaned_explanation_lines, concept_label)
    paragraphs.extend(contrast)

    example_block_lines: list[str] = []
    if plan.include_example:
        example_text = _example_intuition_block(primary)
        if uf and _line_contains_user_forbidden(example_text, uf):
            example_text = ""
        if _has_concrete_example(example_text):
            example_block_lines = _format_example_block(example_text)

    paragraphs = _dedupe_paragraphs([p for p in paragraphs if p])
    if uf:
        paragraphs = [p for p in paragraphs if not _line_contains_user_forbidden(p, uf)]

    rendered_lines: list[str] = ["Course Answer:", ""]
    for paragraph in paragraphs:
        rendered_lines.append(paragraph)
        rendered_lines.append("")

    if example_block_lines:
        rendered_lines.extend(example_block_lines)
        rendered_lines.append("")

    key_idea = _truncate_to_first_sentences(
        _key_idea_sentence(raw_direct_answer, cleaned_explanation_lines, concept_label),
        max_sentences=1,
    )
    if uf and _line_contains_user_forbidden(key_idea, uf):
        key_idea = (
            f"{concept_label} is the anchor idea here."
            if concept_label
            else "The anchor idea is the definition in your notes."
        )
    rendered_lines.extend(["The key idea:", key_idea, ""])

    why_it_matters = _truncate_to_first_sentences(
        _grounded_why_it_matters(plan, primary, concept_label, user_forbidden=uf),
        max_sentences=2,
    )
    rendered_lines.append(why_it_matters)

    out = "\n".join(rendered_lines).rstrip()
    if uf and _line_contains_user_forbidden(out, uf):
        # Last pass: drop forbidden mentions anywhere (e.g. stray contrast lines).
        lines = out.split("\n")
        kept: list[str] = []
        for ln in lines:
            if _line_contains_user_forbidden(ln, uf):
                continue
            kept.append(ln)
        out = "\n".join(kept).strip()
    return out


def _build_explanation_bullets(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    skip_first_chunk_lines: int = 0,
) -> list[str]:
    """Bullets for ### Explanation (compare vs default). Skips lines already used in Direct Answer."""
    if plan.answer_mode == "compare":
        # One heading per plan section; at most a few lines under each (no per-line
        # "First idea / In one line" scaffolding—that produced hundreds of repeated labels).
        compare_bullets: list[str] = []
        for section in plan.sections:
            if not section.chunk_ids:
                continue
            heading = section.heading
            excerpt_lines: list[str] = []
            for lecture_chunk in chunks_by_ids(all_chunks, section.chunk_ids):
                for bullet_line in _bullet_lines_from_chunk(lecture_chunk)[:4]:
                    excerpt_lines.append(bullet_line.strip())
                    if len(excerpt_lines) >= 4:
                        break
                if len(excerpt_lines) >= 4:
                    break
            excerpt_lines = _dedupe_lines(excerpt_lines, cap=4)
            if not excerpt_lines:
                continue
            compare_bullets.append(f"**{heading}:** {excerpt_lines[0]}")
            for extra_line in excerpt_lines[1:]:
                compare_bullets.append(extra_line)
        if plan.comparison_axes:
            compare_bullets.append(
                "**Contrast to keep in mind:** " + "; ".join(plan.comparison_axes[:4])
            )
        return _dedupe_lines(compare_bullets, cap=22)

    explanation_bullets: list[str] = []
    if not primary:
        return ["Add more detail by asking a follow-up with a specific term from class."]
    first_lines = _bullet_lines_from_chunk(primary[0])
    if not first_lines:
        raw = (primary[0].get("clean_explanation") or primary[0].get("source_excerpt") or "").strip()
        first_lines = [raw_line.strip() for raw_line in raw.split("\n") if raw_line.strip()] if raw else []
    if first_lines:
        explanation_bullets.extend(first_lines[skip_first_chunk_lines:])
    for lecture_chunk in primary[1:]:
        explanation_bullets.extend(_bullet_lines_from_chunk(lecture_chunk))
    # Cap supporting material to reduce retrieval contamination (unrelated chunks).
    for supporting_id in plan.supporting_chunk_ids[:3]:
        supporting_chunk = next((x for x in all_chunks if x.get("id") == supporting_id), None)
        if supporting_chunk:
            explanation_bullets.extend(_bullet_lines_from_chunk(supporting_chunk)[:2])
    return _dedupe_lines(explanation_bullets, cap=16)


def _direct_answer_and_skip(
    plan: AnswerPlan, primary: list[dict[str, Any]]
) -> tuple[str, int]:
    """Direct answer text and number of first-chunk lines consumed (non-compare).

    When the planner produced a deterministic ``plan.direct_answer`` (via
    :func:`direct_answer.select_direct_answer`), prefer it. The skip count is
    derived by checking how many leading bullet lines of the first chunk
    appear as substrings of the direct answer (case-insensitive, normalized
    whitespace) so the explanation bullets don't repeat the opening sentence.
    """
    plan_direct_answer = (plan.direct_answer or "").strip()
    if plan_direct_answer:
        skip = 0
        if primary:
            first_lines = _bullet_lines_from_chunk(primary[0])
            answer_norm = re.sub(r"\s+", " ", plan_direct_answer.lower()).strip()
            for line in first_lines:
                line_norm = re.sub(
                    r"\s+", " ", _strip_bullet_prefix(line).lower()
                ).strip()
                if not line_norm:
                    continue
                if line_norm in answer_norm or (
                    len(line_norm) > 20 and line_norm[:80] in answer_norm
                ):
                    skip += 1
                else:
                    break
        return plan_direct_answer, skip

    if plan.answer_mode == "compare" and plan.comparison_axes:
        return "; ".join(plan.comparison_axes[:3]), 0
    if not primary:
        return "I don’t have a short direct line for that phrasing—see the bullets below.", 0
    first_lines = _bullet_lines_from_chunk(primary[0])
    if not first_lines:
        raw = (primary[0].get("clean_explanation") or primary[0].get("source_excerpt") or "").strip()
        if raw:
            paragraph_lines = [raw_line.strip() for raw_line in raw.split("\n") if raw_line.strip()]
            if paragraph_lines:
                return _compose_direct_answer_with_count(paragraph_lines)
            return raw[:400], 0
        return "See the explanation below for how the notes develop this idea.", 0
    return _compose_direct_answer_with_count(first_lines)


def generate_structured_answer(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
    *,
    concept_constraints: ConceptConstraints | None = None,
) -> str:
    """Build **Course Answer:** with the tutor four-section layout (aligned with OpenAI primary path)."""
    query_type = _query_type_for_audit(plan, structured_query)
    if plan.requires_clarification:
        return strict_clarification_answer(query_type, reason=plan.clarification_reason)

    rc = structured_query.response_constraints
    if rc.allow_incorrect_statements:
        refusal_message = (
            "Course Answer:\n\n"
            "### Direct Answer\n"
            "I can’t mix deliberately false statements with true ones in a tutor response.\n\n"
            "### Explanation\n"
            "- If you want practice, ask for a short quiz with separate options, or ask for common "
            "misconceptions explained *as* misconceptions.\n\n"
            "### Example / Intuition\n"
            "Try: “What is a typical mistake people make about softmax vs hardmax?”\n\n"
            "### Why it matters\n"
            "Clear, correct explanations are safer for learning than blended true/false prompts."
        )
        return refusal_message

    if plan.answer_mode == "lecture_summary":
        from app.services.answers.summary_render import format_summary_markdown

        return format_summary_markdown(plan, all_chunks, structured_query)

    if plan.answer_mode == "teaching_plus_check":
        from app.services.answers.quiz_render import format_quiz_markdown

        return format_quiz_markdown(plan, all_chunks, structured_query)

    if plan.answer_mode == "compare_multi" and plan.evidence_bundles:
        entity_bundles = list(plan.evidence_bundles.values())
        rendered_multi = format_multi_entity_compare_markdown(
            entity_bundles, all_chunks, structured_query, plan=plan
        )
        audited = audit_rendered_answer(
            rendered_multi,
            query_type,
            plan,
            structured_query,
            concept_constraints=concept_constraints,
        )
        return audited or strict_clarification_answer(query_type)

    if plan.answer_mode == "compare" and len(plan.evidence_bundles) >= 2:
        bundle_concept_ids = list(plan.evidence_bundles.keys())
        left_bundle = plan.evidence_bundles[bundle_concept_ids[0]]
        right_bundle = plan.evidence_bundles[bundle_concept_ids[1]]
        rendered_compare = format_two_entity_compare_markdown(
            plan, all_chunks, structured_query, left_bundle, right_bundle
        )
        audited = audit_rendered_answer(
            rendered_compare,
            query_type,
            plan,
            structured_query,
            concept_constraints=concept_constraints,
        )
        return audited or strict_clarification_answer(query_type)

    primary = _primary_chunks_ordered(plan, all_chunks)
    if not primary:
        return (
            "Course Answer:\n\n"
            "### Direct Answer\n"
            "I couldn’t tie that question to specific notes yet.\n\n"
            "### Explanation\n"
            "- Ask again using a vocabulary term from class (e.g. softmax, attention, MFCC).\n\n"
            "### Example / Intuition\n"
            "A sharper question usually unlocks a concrete example on the next try.\n\n"
            "### Why it matters\n"
            "Staying close to the course vocabulary keeps answers aligned with what you’re graded on."
        )

    # Chat-shaped intents flow as a tutor narrative via :func:`compose_concept_answer`.
    # The narrow exception is structured-explanation constraints
    # (``exact_explanation_count`` / ``repeat_explanation_times``), which
    # explicitly request the numbered-subsection / repeated-block layout —
    # those keep the legacy markdown so the dedicated copy still applies.
    keeps_legacy_structured_layout = (
        rc.exact_explanation_count is not None
        or rc.repeat_explanation_times is not None
    )
    if plan.answer_mode in _CHAT_NARRATIVE_MODES and not keeps_legacy_structured_layout:
        from app.services.answers.concept_answer_composer import compose_concept_answer

        # Honor ``no_examples`` / ``intuition_only`` by suppressing the example
        # block at the call site, without mutating the planner's plan instance.
        plan_for_render = plan
        if rc.no_examples or rc.intuition_only:
            plan_for_render = dataclasses.replace(plan, include_example=False)
        purity_constraints = concept_constraints
        if purity_constraints is None:
            purity_constraints = build_concept_constraints(structured_query, get_kb())
        composed = compose_concept_answer(
            plan_for_render,
            all_chunks,
            structured_query,
            constraints=purity_constraints,
        )
        audited = audit_rendered_answer(
            composed,
            query_type,
            plan,
            structured_query,
            concept_constraints=purity_constraints,
        )
        return audited or strict_clarification_answer(query_type)

    direct_answer_text, lines_consumed_by_direct_answer = _direct_answer_and_skip(plan, primary)
    explanation_bullets = _build_explanation_bullets(
        plan, all_chunks, primary, skip_first_chunk_lines=lines_consumed_by_direct_answer
    )
    example_intuition_text = _example_intuition_block(primary)
    why_it_matters_text = _why_matters_block(plan, structured_query, primary)

    requested_distinct_explanations = rc.exact_explanation_count
    wants_numbered_explanation_subsections = (
        requested_distinct_explanations is not None and requested_distinct_explanations >= 2
    )
    if wants_numbered_explanation_subsections:
        while len(explanation_bullets) < requested_distinct_explanations:
            explanation_bullets.append(
                "Another angle on the same idea from the notes (distinct wording): see the preceding bullets."
            )
        explanation_bullets = explanation_bullets[:requested_distinct_explanations]

    course_answer_lines: list[str] = [
        "Course Answer:",
        "",
        "### Direct Answer",
        "",
        direct_answer_text,
        "",
        "### Explanation",
        "",
    ]
    max_numbered_explanation_sections = 12
    if wants_numbered_explanation_subsections:
        for section_index in range(
            min(requested_distinct_explanations, max_numbered_explanation_sections)
        ):
            if section_index < len(explanation_bullets):
                subsection_body = explanation_bullets[section_index]
            elif explanation_bullets:
                subsection_body = explanation_bullets[-1]
            else:
                subsection_body = ""
            course_answer_lines.append(f"#### Explanation {section_index + 1}")
            course_answer_lines.append("")
            course_answer_lines.append(subsection_body or "(See course text.)")
            course_answer_lines.append("")
    else:
        for bullet_text in explanation_bullets:
            course_answer_lines.append(f"- {bullet_text}")
        if len(explanation_bullets) == 0:
            course_answer_lines.append(
                "- The notes may pack the idea into a short block—say if you want it slower or with a diagram."
            )

    repeat_explanation_count = rc.repeat_explanation_times
    if repeat_explanation_count is not None and repeat_explanation_count >= 2:
        repeated_explanation_markdown = (
            "\n".join(f"- {bullet_text}" for bullet_text in explanation_bullets)
            if explanation_bullets
            else direct_answer_text
        )
        course_answer_lines.extend(
            [
                "",
                "### Repeated explanation (as requested)",
                "",
                repeated_explanation_markdown,
            ]
        )

    if rc.intuition_only:
        course_answer_lines.extend(
            [
                "",
                "### Example / Intuition",
                "",
                "(Technical training details omitted for intuition-only request.)",
                "",
                "### Why it matters",
                "",
                why_it_matters_text,
            ]
        )
        rendered = "\n".join(course_answer_lines).rstrip()
        if keeps_legacy_structured_layout:
            return rendered
        audited = audit_rendered_answer(
            rendered,
            query_type,
            plan,
            structured_query,
            concept_constraints=concept_constraints,
        )
        return audited or strict_clarification_answer(query_type)

    if rc.no_examples:
        course_answer_lines.extend(
            [
                "",
                "### Why it matters",
                "",
                why_it_matters_text,
            ]
        )
        rendered = "\n".join(course_answer_lines).rstrip()
        if keeps_legacy_structured_layout:
            return rendered
        audited = audit_rendered_answer(
            rendered,
            query_type,
            plan,
            structured_query,
            concept_constraints=concept_constraints,
        )
        return audited or strict_clarification_answer(query_type)

    course_answer_lines.extend(
        [
            "",
            "### Example / Intuition",
            "",
            example_intuition_text,
            "",
            "### Why it matters",
            "",
            why_it_matters_text,
        ]
    )
    rendered = "\n".join(course_answer_lines).rstrip()
    if keeps_legacy_structured_layout:
        return rendered
    audited = audit_rendered_answer(
        rendered,
        query_type,
        plan,
        structured_query,
        concept_constraints=concept_constraints,
    )
    return audited or strict_clarification_answer(query_type)
