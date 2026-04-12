"""Gemini-only secondary boost (never the primary Course Answer)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from flask import current_app

from app.services.answer_planning import AnswerPlan
from app.services.structured_query import StructuredQuery

logger = logging.getLogger(__name__)


def generate_gemini_boosted_explanation(
    user_question: str,
    course_answer_block: str,
    plan: AnswerPlan,
    chunks: list[dict[str, Any]],
    sq: StructuredQuery,
) -> tuple[str | None, dict[str, Any]]:
    """
    Produce **Boosted Explanation** only: clarity, examples, simplification, grounded in course.

    Does not replace or rewrite the Course Answer. Returns ``(text, meta)``; ``text`` starts
    with ``Boosted Explanation:\\n\\n`` when successful.
    """
    api_key = current_app.config.get("GEMINI_API_KEY") or current_app.config.get("GOOGLE_API_KEY")
    if not api_key:
        return None, {}

    model = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")
    timeout = int(current_app.config.get("GEMINI_TIMEOUT_SEC", "60"))

    system_instruction = (
        "You are a teaching assistant for LING 487. Produce a **Boosted Explanation** that "
        "clarifies or expands the student's understanding using ONLY the COURSE_ANSWER and "
        "RETRIEVED_CHUNKS. Do not contradict the Course Answer. Do not introduce facts not "
        "supported by the materials. If material is thin, say so briefly. "
        "Start with the exact line:\n\nBoosted Explanation:\n\n"
    )
    payload = {
        "student_question": user_question,
        "answer_plan_summary": plan.to_dict(),
        "course_answer": course_answer_block[:80_000],
        "retrieved_chunks": chunks[:16],
    }
    user_text = (
        f"{system_instruction}\n\nSTUDENT_QUESTION:\n{user_question}\n\n"
        f"STRUCTURED_CONTEXT_JSON:\n{json.dumps(payload, ensure_ascii=False)[:100_000]}"
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = json.dumps(
        {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_text}],
                }
            ],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        logger.warning("Gemini HTTP error: %s %s", e.code, e.read()[:500])
        return None, {"error": f"http_{e.code}", "provider": "gemini"}
    except OSError as e:
        logger.warning("Gemini request failed: %s", e)
        return None, {"error": str(e), "provider": "gemini"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, {"error": "invalid_json", "provider": "gemini"}

    try:
        text = (
            data["candidates"][0]["content"]["parts"][0].get("text")
        )
    except (KeyError, IndexError, TypeError):
        return None, {"error": "no_text", "provider": "gemini"}

    if not text or not str(text).strip():
        return None, {"provider": "gemini"}

    out = str(text).strip()
    if not out.lower().startswith("boosted explanation"):
        out = "Boosted Explanation:\n\n" + out
    return out, {"provider": "gemini", "model": model}
