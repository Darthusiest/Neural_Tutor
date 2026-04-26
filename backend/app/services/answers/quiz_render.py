"""Deterministic Course Answer markdown for quiz mode.

Renders a static, retrieval-grounded quiz block of up to 3 questions plus an
inline ``Answer Key:`` section. The shape is intentionally distinct from the
standard four-block Course Answer (``### Direct Answer`` / ``### Explanation``
/ ``### Example / Intuition`` / ``### Why it matters``) so that quiz mode
cannot fall back into chat-style output.

Question type policy (deterministic; degrades gracefully when evidence is thin):

- Q1 — short answer (drawn from the strongest evidence chunk)
- Q2 — multiple choice (correct option is the chunk's topic head; distractors
  are other retrieved chunk topics, padded with fixed decoys only when fewer
  than two real distractors exist)
- Q3 — true/false (a true statement lifted from the chunk's clean explanation)

Random ordering of MC options is seeded from the original query so output is
stable across runs and easy to assert on in tests.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from app.services.answers.answer_planning import AnswerPlan, chunks_by_ids
from app.services.knowledge.structured_query import StructuredQuery


_TOPIC_DELIMITER_RE = re.compile(r"\s*[—\-:|]\s*")

_MC_LETTERS = ("A", "B", "C", "D")

_FALLBACK_DECOYS = (
    "Different section (review another lecture)",
    "Unrelated course topic",
    "Placeholder distractor",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _topic_head(topic: str | None) -> str:
    if not topic:
        return ""
    return _TOPIC_DELIMITER_RE.split(str(topic), maxsplit=1)[0].strip()


def _first_sentence(text: str, *, max_len: int = 320) -> str:
    body = (text or "").strip()
    if not body:
        return ""
    match = re.match(r"([^.!?\n]+[.!?])", body)
    if match:
        return match.group(1).strip()[:max_len]
    first_line = body.split("\n", 1)[0].strip()
    return first_line[:max_len]


def _sample_questions(chunk: dict[str, Any]) -> list[str]:
    raw = chunk.get("sample_questions") or "[]"
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if not isinstance(raw, str):
        return []
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(arr, list):
        return [str(x).strip() for x in arr if str(x).strip()]
    return []


def _select_evidence(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
    *,
    max_chunks: int = 3,
) -> list[dict[str, Any]]:
    """Up to ``max_chunks`` lecture-scoped or topic-scoped chunks, deduped by topic head."""
    lecture_numbers = list(structured_query.intent.lecture_numbers or [])
    primary_ids = list(plan.primary_chunk_ids or [])
    primary = chunks_by_ids(all_chunks, primary_ids) or list(all_chunks)

    if len(lecture_numbers) == 1:
        target = lecture_numbers[0]
        primary = [c for c in primary if c.get("lecture_number") == target] or [
            c for c in all_chunks if c.get("lecture_number") == target
        ]

    selected: list[dict[str, Any]] = []
    seen_heads: set[str] = set()
    for chunk in primary:
        head = _topic_head(chunk.get("topic")).lower()
        if not head or head in seen_heads:
            continue
        seen_heads.add(head)
        selected.append(chunk)
        if len(selected) >= max_chunks:
            break

    if len(selected) < max_chunks:
        for chunk in primary:
            if chunk in selected:
                continue
            selected.append(chunk)
            if len(selected) >= max_chunks:
                break
    return selected[:max_chunks]


def _quiz_header(structured_query: StructuredQuery, evidence: list[dict[str, Any]]) -> str:
    """`Quiz: Lecture N` for single-lecture queries, otherwise `Quiz: <topic>`."""
    lecture_numbers = list(structured_query.intent.lecture_numbers or [])
    if len(lecture_numbers) == 1:
        return f"Quiz: Lecture {lecture_numbers[0]}"
    intent = structured_query.intent
    if intent.detected_concepts:
        return f"Quiz: {str(intent.detected_concepts[0]).strip()}"
    if evidence:
        head = _topic_head(evidence[0].get("topic"))
        if head:
            return f"Quiz: {head}"
    raw = (intent.original_query or "").strip()
    return f"Quiz: {raw[:60] if raw else 'course topic'}"


def _topic_for_question(chunk: dict[str, Any], default: str) -> str:
    head = _topic_head(chunk.get("topic"))
    return head or default


def _short_answer_question(chunk: dict[str, Any], topic: str) -> tuple[str, str]:
    """`(stem, answer_key_line)` for the short-answer slot."""
    sample = _sample_questions(chunk)
    if sample:
        stem = sample[0]
    else:
        stem = f"In your own words, what is **{topic}** in this course?"
    answer = _first_sentence(
        chunk.get("clean_explanation") or chunk.get("source_excerpt") or "",
        max_len=320,
    )
    if not answer:
        answer = f"A short, course-grounded definition of {topic}."
    return stem, answer


def _mc_distractors(
    correct_topic: str,
    pool: list[dict[str, Any]],
    rng: random.Random,
) -> list[str]:
    """Exactly 3 distractor strings: real sibling-topic heads first, then fixed decoys."""
    distractors: list[str] = []
    seen: set[str] = {correct_topic.lower()}
    candidates = list(pool)
    rng.shuffle(candidates)
    for chunk in candidates:
        head = _topic_head(chunk.get("topic"))
        if not head or head.lower() in seen:
            continue
        seen.add(head.lower())
        distractors.append(head)
        if len(distractors) >= 3:
            break
    pad_index = 0
    while len(distractors) < 3 and pad_index < len(_FALLBACK_DECOYS) * 2:
        decoy = _FALLBACK_DECOYS[pad_index % len(_FALLBACK_DECOYS)]
        if decoy.lower() not in seen:
            distractors.append(decoy)
            seen.add(decoy.lower())
        pad_index += 1
    return distractors[:3]


def _multiple_choice_question(
    chunk: dict[str, Any],
    topic: str,
    pool: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[str, list[tuple[str, str]], str, str]:
    """Build MC stem + lettered options.

    Returns ``(stem, [(letter, text), ...], correct_letter, correct_text)``.
    """
    sample = _sample_questions(chunk)
    if sample:
        stem = sample[0]
    else:
        stem = f"Which of the following best describes **{topic}**?"
    distractors = _mc_distractors(topic, pool, rng)
    options_pool = [topic] + distractors
    rng.shuffle(options_pool)
    lettered: list[tuple[str, str]] = list(zip(_MC_LETTERS, options_pool))
    correct_letter = next(
        (letter for letter, text in lettered if text.lower() == topic.lower()),
        _MC_LETTERS[0],
    )
    return stem, lettered, correct_letter, topic


def _true_false_question(chunk: dict[str, Any], topic: str) -> tuple[str, str]:
    """`(stem, answer_key_line)` for the true/false slot. Always emits a *true* statement."""
    sentence = _first_sentence(
        chunk.get("clean_explanation") or chunk.get("source_excerpt") or "",
        max_len=320,
    )
    if not sentence:
        sentence = f"{topic} appears in the course notes for this section."
    stem = f"True or false: {sentence}"
    return stem, "True"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def format_quiz_markdown(
    plan: AnswerPlan,
    all_chunks: list[dict[str, Any]],
    structured_query: StructuredQuery,
) -> str:
    """Return quiz markdown: ``Quiz: ...`` header + numbered questions + ``Answer Key:``.

    Never emits Course Answer headings (``### Direct Answer``, ``### Explanation``,
    ``### Example / Intuition``, ``### Why it matters``). Falls back to a clarification
    message when no evidence is available — preserving the existing "no hallucination"
    behavior for empty / off-topic queries.
    """
    evidence = _select_evidence(plan, all_chunks, structured_query, max_chunks=3)
    header = _quiz_header(structured_query, evidence)

    if not evidence:
        return (
            f"{header}\n\n"
            "I couldn't pull enough course material to build a quiz on that.\n\n"
            "Try a more specific term from the syllabus (e.g. softmax, MFCC, attention) "
            "or a lecture number."
        )

    rng = random.Random(structured_query.intent.original_query or header)

    # Topic strings used for question stems (and as the MC correct option).
    fallback_topic = (
        structured_query.intent.detected_concepts[0]
        if structured_query.intent.detected_concepts
        else _topic_head(evidence[0].get("topic")) or "this topic"
    )
    topics = [_topic_for_question(chunk, fallback_topic) for chunk in evidence]

    question_blocks: list[str] = []
    answer_key_lines: list[str] = []

    short_stem, short_answer = _short_answer_question(evidence[0], topics[0])
    question_blocks.append(f"1. {short_stem}")
    answer_key_lines.append(f"1. {short_answer}")

    if len(evidence) >= 2:
        # Scope MC distractors to the same lecture as the question when the query is
        # lecture-bound; otherwise allow any retrieved chunk. This prevents a lecture-N
        # quiz from drawing distractors out of unrelated lectures.
        lecture_numbers = list(structured_query.intent.lecture_numbers or [])
        if len(lecture_numbers) == 1:
            target_lecture = lecture_numbers[0]
            mc_pool = [
                c for c in all_chunks
                if c is not evidence[1] and c.get("lecture_number") == target_lecture
            ]
        else:
            mc_pool = [c for c in all_chunks if c is not evidence[1]]
        mc_stem, mc_options, mc_letter, mc_correct = _multiple_choice_question(
            evidence[1], topics[1], mc_pool, rng
        )
        mc_lines = [f"2. {mc_stem}"]
        for letter, text in mc_options:
            mc_lines.append(f"   {letter}) {text}")
        question_blocks.append("\n".join(mc_lines))
        answer_key_lines.append(f"2. {mc_letter}) {mc_correct}")

    if len(evidence) >= 3:
        tf_stem, tf_answer = _true_false_question(evidence[2], topics[2])
        question_blocks.append(f"3. {tf_stem}")
        answer_key_lines.append(f"3. {tf_answer}")

    parts: list[str] = [header, ""]
    parts.extend(question_blocks[0:1])
    for block in question_blocks[1:]:
        parts.append("")
        parts.append(block)
    parts.extend(["", "Answer Key:", ""])
    parts.extend(answer_key_lines)
    return "\n".join(parts).rstrip()
