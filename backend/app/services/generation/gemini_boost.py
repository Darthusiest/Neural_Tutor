"""Gemini-only secondary boost (never the primary Course Answer)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from flask import current_app

from app.services.answers.answer_planning import AnswerPlan
from app.services.knowledge.structured_query import StructuredQuery

logger = logging.getLogger(__name__)

_GEMINI_BOOST_CONSTRAINED_PREAMBLE = (
    "You are improving clarity ONLY.\n\n"
    "Rules:\n"
    "- Do NOT introduce any new concepts.\n"
    "- ONLY use ideas present in allowed_evidence_lines.\n"
    "- If draft contains forbidden concepts, REMOVE them.\n"
    "- Do NOT mention forbidden_terms.\n"
    "- Do NOT expand beyond the topic.\n"
    "- Keep explanation concise and tutor-like.\n\n"
    "Goal: rewrite for clarity, not add content.\n"
    "Start with the exact line:\n\nBoosted Explanation:\n\n"
)

_GEMINI_BOOST_EXTENDED_PREAMBLE = (
    "You are improving clarity and may add ONE short standard-knowledge clarification.\n\n"
    "Rules:\n"
    "- Start with the rewritten clarity-only body grounded in allowed_evidence_lines.\n"
    "- You MAY add at most TWO sentences of widely-known clarification about target_concept ONLY.\n"
    "- Each added sentence MUST start with one of: 'A useful clarification is', "
    "'In standard speech processing terms', 'In standard machine learning terms', or 'More generally'.\n"
    "- Do NOT contradict the draft_answer.\n"
    "- Do NOT mention any forbidden_terms.\n"
    "- Do NOT introduce more than ONE new technical term beyond target_concept and its standard pipeline steps.\n"
    "- Do NOT expand into a separate topic or new full lecture.\n"
    "- Keep total length under 6 sentences.\n"
    "Start with the exact line:\n\nBoosted Explanation:\n\n"
)


def generate_gemini_constrained_boost(
    *,
    user_question: str,
    target_concept: str,
    allowed_evidence_lines: list[str],
    forbidden_terms: list[str],
    draft_answer: str,
    mode: str,
    allow_external_clarification: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    """Constrained boost for deferred endpoint (2s default timeout)."""
    api_key = current_app.config.get("GEMINI_API_KEY") or current_app.config.get("GOOGLE_API_KEY")
    if not api_key:
        return None, {}

    model = current_app.config.get("GEMINI_MODEL", "gemini-1.5-flash")
    timeout = int(current_app.config.get("BOOST_TIMEOUT_SEC", 2))
    g_temp = float(current_app.config.get("GEMINI_TEMPERATURE_BOOST", 0.4))
    g_max_tokens = int(current_app.config.get("GEMINI_MAX_OUTPUT_TOKENS", 2048))

    lines = [ln[:400] for ln in (allowed_evidence_lines or [])[:5]]
    preamble = (
        _GEMINI_BOOST_EXTENDED_PREAMBLE if allow_external_clarification else _GEMINI_BOOST_CONSTRAINED_PREAMBLE
    )
    payload = {
        "target_concept": target_concept,
        "allowed_evidence_lines": lines,
        "forbidden_terms": list(forbidden_terms or [])[:40],
        "draft_answer": (draft_answer or "")[:8000],
        "mode": mode or "chat",
        "allow_external_clarification": allow_external_clarification,
    }
    user_text = (
        f"{preamble}\n\nSTUDENT_QUESTION:\n{user_question[:4000]}\n\n"
        f"BOOST_INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)[:50_000]}"
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    body = json.dumps(
        {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": g_temp, "maxOutputTokens": g_max_tokens},
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
        text = data["candidates"][0]["content"]["parts"][0].get("text")
    except (KeyError, IndexError, TypeError):
        return None, {"error": "no_text", "provider": "gemini"}

    if not text or not str(text).strip():
        return None, {"provider": "gemini", "model": model}

    meta: dict[str, Any] = {"provider": "gemini", "model": model}
    um = data.get("usageMetadata")
    if isinstance(um, dict):
        meta["usage"] = um

    out = str(text).strip()
    if not out.lower().startswith("boosted explanation"):
        out = "Boosted Explanation:\n\n" + out
    return out, meta


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
    timeout = int(current_app.config.get("GEMINI_TIMEOUT_SEC", 60))
    g_temp = float(current_app.config.get("GEMINI_TEMPERATURE_BOOST", 0.4))
    g_max_tokens = int(current_app.config.get("GEMINI_MAX_OUTPUT_TOKENS", 2048))

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
            "generationConfig": {"temperature": g_temp, "maxOutputTokens": g_max_tokens},
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
        return None, {"provider": "gemini", "model": model}

    meta: dict[str, Any] = {"provider": "gemini", "model": model}
    um = data.get("usageMetadata")
    if isinstance(um, dict):
        meta["usage"] = um

    out = str(text).strip()
    if not out.lower().startswith("boosted explanation"):
        out = "Boosted Explanation:\n\n" + out
    return out, meta


# Design-doc name: secondary boost uses Gemini only (never the primary Course Answer).
generate_boosted_explanation = generate_gemini_boosted_explanation
