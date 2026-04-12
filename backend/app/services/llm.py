"""OpenAI-backed explanations (server-side only; key never sent to frontend)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.services.answer_planning import AnswerPlan
    from app.services.structured_query import StructuredQuery

from flask import current_app

logger = logging.getLogger(__name__)


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
    Boosted explanation grounded in retrieved JSON context.

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
    Optional LLM path: generate **Course Answer:** text following ``plan``, grounded only in ``chunks``.
    Falls back to caller rule-based generation when API is unavailable.
    """
    if not current_app.config.get("OPENAI_API_KEY"):
        return None, {}

    system = (
        "You write the **Course Answer** section for LING 487. "
        "Follow ANSWER_PLAN sections in order. Use ONLY RETRIEVED_CHUNKS for facts; "
        "do not invent citations or topics. Start with the exact line:\n\nCourse Answer:\n\n"
        "then use ### headings matching the plan when multiple sections exist."
    )
    payload = {
        "student_question": sq.intent.original_query,
        "answer_plan": plan.to_dict(),
        "retrieved_chunks": chunks[:20],
    }
    user = "STRUCTURED_INPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)[:120_000]
    text, usage = _openai_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.35,
    )
    if not text:
        return None, usage
    if not text.strip().lower().startswith("course answer"):
        text = "Course Answer:\n\n" + text
    return text, usage
