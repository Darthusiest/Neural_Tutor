"""OpenAI-backed explanations (server-side only; key never sent to frontend)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.answers.answer_planning import AnswerPlan
    from app.services.knowledge.structured_query import StructuredQuery

from flask import current_app

from app.services.generation.generation_input import (
    build_generation_input,
    format_generation_prompt_user_message,
)
from app.services.generation.output_cleanup import clean_output, enforce_structure

logger = logging.getLogger(__name__)

# Primary Course Answer when PRIMARY_COURSE_ANSWER_OPENAI is on: final student-facing tutor text only.
_COURSE_ANSWER_SYSTEM_PROMPT = """You are a LING 487 tutor. You produce the FINAL answer shown to a student.

The user message has: the question, concept names, a teaching-style hint, and course text under Primary Content and Supporting Content (prose only—no metadata).

Grounding: Use only ideas supported by that Primary and Supporting text. Do not invent course facts. At most a tiny clarification for readability.

Voice: Human tutor, not a retrieval system or database.

Paraphrase the course text in your own words—do not copy its wording.

Sections: Each ### block must add new information. Do not repeat the same idea or phrasing across blocks. Example / Intuition must be a fresh numeric example or analogy (not a restatement of Direct Answer or Explanation).

FORBIDDEN in your output: keyword lists, chunk or lecture IDs, lecture scope, "concept graph", debug strings, "retrieved", "indexed", "chunk", "materials show", or other internal phrasing.

OUTPUT FORMAT (STRICT). Start with the exact line "Course Answer:" then a blank line, then:

### Direct Answer
1–2 sentences in plain English.

### Explanation
How it works; details not already stated in Direct Answer. Short bullets only when helpful.

### Example / Intuition
One concrete numeric example or analogy.

### Why it matters
Why this matters in LING 487 / NLP—natural prose only, no lecture lists or IDs.

Compare questions: Direct Answer states the main distinction; Explanation covers both sides; Example shows the contrast; Why it matters ties to course goals—all without system jargon."""


def _openai_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.5,
) -> tuple[str | None, dict[str, Any]]:
    """
    Minimal chat-completions call (stdlib only).

    Returns (assistant_text, usage_meta).
    """
    api_key = current_app.config.get("OPENAI_API_KEY")
    if not api_key:
        return None, {}

    model = current_app.config.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    timeout = int(current_app.config.get("OPENAI_TIMEOUT_SEC", 60))
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.warning("OpenAI HTTP error: %s %s", e.code, e.read()[:500])
        return None, {"error": f"http_{e.code}"}
    except OSError as e:
        logger.warning("OpenAI request failed: %s", e)
        return None, {"error": str(e)}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, {"error": "invalid_json"}

    usage = data.get("usage") or {}
    text = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content")
    )
    if not text or not str(text).strip():
        return None, usage
    return str(text).strip(), usage


def generate_boosted_explanation(
    user_question: str,
    retrieved_context: str,
) -> tuple[str | None, dict]:
    """
    OpenAI-only **Boosted Explanation** (optional fallback).

    Primary product path uses Gemini: :func:`app.services.generation.gemini_boost.generate_boosted_explanation`.
    This function is used when ``OPENAI_BOOST_FALLBACK`` is enabled and Gemini is unavailable.

    Returns (text, usage_meta).
    """
    if not current_app.config.get("OPENAI_API_KEY"):
        return None, {}

    system = (
        "You are a teaching assistant for LING 487. Produce a **Boosted Explanation** that "
        "clarifies the student's question using ONLY ideas supported by the RETRIEVED_CONTEXT. "
        "If context is thin, say so and stay course-relevant—do not invent facts. "
        "Start the response with the exact line:\n\nBoosted Explanation:\n\n"
        "then continue. Do not restate the full Course Answer verbatim."
    )
    user = (
        f"STUDENT_QUESTION:\n{user_question}\n\n"
        f"RETRIEVED_CONTEXT (JSON lecture chunks):\n{retrieved_context}"
    )
    text, usage = _openai_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.45,
    )
    return text, usage


def generate_comparison_boost(
    concept_a: str,
    concept_b: str,
    course_answer_block: str,
) -> tuple[str | None, dict]:
    """
    Optional narrative comparing two concepts, grounded in the course answer text.
    """
    if not current_app.config.get("OPENAI_API_KEY"):
        return None, {}

    system = (
        "You help students compare two ideas from LING 487. Using ONLY the COURSE_ANSWER material below, "
        "add a short comparison that highlights similarities and differences. "
        "If the material does not support a claim, omit it. "
        "Start with the exact line:\n\nBoosted Explanation:\n\n"
    )
    user = (
        f"CONCEPT_A: {concept_a}\n"
        f"CONCEPT_B: {concept_b}\n\n"
        f"COURSE_ANSWER (ground truth):\n{course_answer_block}"
    )
    return _openai_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.45,
    )


def generate_plan_constrained_answer(
    plan: "AnswerPlan",
    chunks: list[dict[str, Any]],
    sq: "StructuredQuery",
) -> tuple[str | None, dict[str, Any]]:
    """
    Optional LLM path: clean generation input → tutor prompt → ``clean_output`` → ``enforce_structure``.

    Raw chunks are never sent as JSON with metadata; only teaching text from
    :func:`build_generation_input`.
    """
    if not current_app.config.get("OPENAI_API_KEY"):
        return None, {}

    clean_input = build_generation_input(sq, plan, chunks)
    user = format_generation_prompt_user_message(clean_input)
    raw, usage = _openai_chat(
        [{"role": "system", "content": _COURSE_ANSWER_SYSTEM_PROMPT}, {"role": "user", "content": user}],
        temperature=0.4,
    )
    if not raw:
        return None, usage

    filtered = clean_output(raw)
    final = enforce_structure(filtered)
    if not final.strip().lower().startswith("course answer"):
        final = "Course Answer:\n\n" + final
    return final.strip(), usage
