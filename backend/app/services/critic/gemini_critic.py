"""Gemini-powered critique of chatbot turns (admin eval tooling)."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from flask import current_app

logger = logging.getLogger(__name__)

# Enforce minimum spacing between critic HTTP calls (shared Gemini quota with boost, retries, etc.).
_CRITIC_REQ_LOCK = threading.Lock()
_CRITIC_REQ_LAST_MONO = 0.0


def _sleep_until_min_interval_between_requests() -> None:
    gap = float(current_app.config.get("CRITIC_MIN_INTERVAL_BETWEEN_REQUESTS_SEC", 0) or 0)
    if gap <= 0:
        return
    global _CRITIC_REQ_LAST_MONO
    with _CRITIC_REQ_LOCK:
        now = time.monotonic()
        wait = gap - (now - _CRITIC_REQ_LAST_MONO)
        if wait > 0:
            time.sleep(wait)
        _CRITIC_REQ_LAST_MONO = time.monotonic()


# Rubric when we send responseSchema (avoid duplicating JSON shape in prompt — Google recommends
# against duplicating schema text in the user prompt when using structured output).
_CRITIC_RUBRIC_SCHEMA = """You are an impartial judge for a LING 487 course-grounded tutor.

Score the CHATBOT_ANSWER using STUDENT_QUESTION, RETRIEVED_CHUNKS, STRUCTURED_PLAN_JSON,
EXPECTED_BEHAVIOR_JSON (may be empty), and EFFECTIVE_MODE from CRITIQUE_INPUT_JSON.

Rules (base only on chunks + answer text):
- If chunks are empty or too thin to support the answer, lower scores and explain in rationale.
- "accurate" = consistent with chunks, not outside knowledge.
- "complete" = reasonably answers the question for this mode.
- "mode_compliant" = shape matches EFFECTIVE_MODE.
- "no_hallucination" = penalize claims unsupported by chunks.
- error_categories: snake_case from: ungrounded_claim, thin_evidence, mode_shape_mismatch,
  incomplete_answer, inaccuracy_vs_chunks, vague_or_generic, other
- score = arithmetic mean of the five dimension floats. pass = true iff score >= PASS_THRESHOLD.
  Keep dimensions, score, pass, error_categories, and rationale mutually consistent.

Keep rationale to one short paragraph.
"""

# Fallback when structured output is off: full JSON template in prompt.
_CRITIC_RUBRIC_V1 = """You are an impartial judge for a LING 487 course-grounded tutor.

Task: rate the CHATBOT_ANSWER given the STUDENT_QUESTION, RETRIEVED_CHUNKS, STRUCTURED_PLAN_JSON, EXPECTED_BEHAVIOR_JSON (suite constraints; may be empty), and EFFECTIVE_MODE.

Rules:
- Base judgments ONLY on RETRIEVED_CHUNKS and what is explicit in CHATBOT_ANSWER. If chunks are empty or too thin to support the answer, lower scores and say so in rationale (do not invent lecture facts).
- "accurate" means consistent with the chunks, not with general world knowledge beyond them.
- "complete" means the answer reasonably addresses the question for this mode; do not require knowledge absent from chunks.
- "mode_compliant" means the shape matches EFFECTIVE_MODE (e.g. quiz has Quiz/Answer Key; summary avoids four-block Course Answer sections; compare separates entities).
- "no_hallucination" penalizes claims in the answer unsupported by chunks.
- allowed error_categories values (pick zero or more, use snake_case): ungrounded_claim, thin_evidence, mode_shape_mismatch, incomplete_answer, inaccuracy_vs_chunks, vague_or_generic, other

Output ONLY a single JSON object (no markdown fences, no prose before or after) with this exact shape:
{
  "dimensions": {
    "grounded": <float 0-1>,
    "accurate": <float 0-1>,
    "complete": <float 0-1>,
    "mode_compliant": <float 0-1>,
    "no_hallucination": <float 0-1>
  },
  "score": <float 0-1 arithmetic mean of the five dimensions>,
  "pass": <boolean true iff score >= PASS_THRESHOLD>,
  "error_categories": [<string>, ...],
  "rationale": <string one short paragraph>
}
"""

# Gemini JSON schema (types upper-case per REST Schema reference).
_CRITIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "dimensions": {
            "type": "OBJECT",
            "properties": {
                "grounded": {"type": "NUMBER"},
                "accurate": {"type": "NUMBER"},
                "complete": {"type": "NUMBER"},
                "mode_compliant": {"type": "NUMBER"},
                "no_hallucination": {"type": "NUMBER"},
            },
            "required": ["grounded", "accurate", "complete", "mode_compliant", "no_hallucination"],
        },
        "score": {"type": "NUMBER"},
        "pass": {"type": "BOOLEAN"},
        "error_categories": {"type": "ARRAY", "items": {"type": "STRING"}},
        "rationale": {"type": "STRING"},
    },
    "required": ["dimensions", "score", "pass", "error_categories", "rationale"],
}


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.I)
    if fence:
        raw = fence.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    return data if isinstance(data, dict) else None


def _candidate_text_and_meta(data: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Pull model text from a generateContent JSON body; surface block / finish hints."""
    meta: dict[str, Any] = {}
    pf = data.get("promptFeedback")
    if isinstance(pf, dict):
        br = pf.get("blockReason")
        if br:
            return None, {"error": "prompt_blocked", "block_reason": str(br), "prompt_feedback": pf}
    cands = data.get("candidates")
    if not isinstance(cands, list) or not cands:
        err = data.get("error")
        return None, {"error": "no_candidates", "api_error": err}
    c0 = cands[0] if isinstance(cands[0], dict) else {}
    fr = c0.get("finishReason")
    if fr:
        meta["finish_reason"] = str(fr)
    parts = (c0.get("content") or {}).get("parts") if isinstance(c0.get("content"), dict) else None
    if not isinstance(parts, list):
        parts = []
    texts: list[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("text"):
            texts.append(str(p["text"]))
    txt = "\n".join(texts).strip()
    if not txt:
        return None, {**meta, "error": "no_text"}
    return txt, meta


def _lower_headers(resp: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = resp.headers.items()
    except AttributeError:
        return out
    for k, v in items:
        out[str(k).lower()] = str(v)
    return out


def _post_generate(url: str, body_obj: dict[str, Any], timeout: int) -> tuple[int, str, dict[str, str]]:
    """Return status, body, response headers (lowercased keys; empty on failure to read)."""
    payload = json.dumps(body_obj).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            try:
                code = int(resp.getcode())
            except (AttributeError, TypeError, ValueError):
                code = 200
            return code, resp.read().decode("utf-8"), _lower_headers(resp)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return int(e.code), raw, _lower_headers(e)
    except OSError:
        raise


_TRANS_HTTP = frozenset({429, 500, 503})


def _retry_sleep_sec(
    *,
    status: int,
    headers: dict[str, str],
    retry_index: int,
    base_sec: float,
    max_delay_sec: float,
) -> float:
    ra = headers.get("retry-after", "").strip()
    if ra:
        try:
            return min(max_delay_sec, max(0.0, float(ra)))
        except ValueError:
            pass
    delay = base_sec * (2**retry_index)
    if status == 429:
        delay = max(delay, 1.0 + 0.5 * retry_index)
    return min(max_delay_sec, delay)


def _post_generate_resilient(
    url: str,
    body_obj: dict[str, Any],
    timeout: int,
    *,
    max_retries: int,
    base_sec: float,
    max_delay_sec: float,
) -> tuple[int, str, dict[str, str]]:
    """Retry on transient HTTP statuses (429 / 503 / 500) with exponential backoff."""
    last: tuple[int, str, dict[str, str]] | None = None
    for retry_i in range(max(0, max_retries) + 1):
        _sleep_until_min_interval_between_requests()
        status, raw, hdrs = _post_generate(url, body_obj, timeout)
        last = (status, raw, hdrs)
        if status == 200 or status not in _TRANS_HTTP or retry_i >= max_retries:
            return status, raw, hdrs
        delay = _retry_sleep_sec(
            status=status,
            headers=hdrs,
            retry_index=retry_i,
            base_sec=base_sec,
            max_delay_sec=max_delay_sec,
        )
        logger.warning(
            "Critic Gemini HTTP %s; retry %s/%s after %.1fs",
            status,
            retry_i + 1,
            max_retries,
            delay,
        )
        time.sleep(delay)
    assert last is not None
    return last


def _verdict_from_parsed(
    parsed: dict[str, Any],
    *,
    pass_threshold: float,
    model: str,
    prompt_version: str,
    api_usage: dict | None,
) -> tuple[CriticVerdict, dict[str, Any]]:
    dims_raw = parsed.get("dimensions") or {}
    if not isinstance(dims_raw, dict):
        dims_raw = {}
    keys = ("grounded", "accurate", "complete", "mode_compliant", "no_hallucination")
    dimensions = {k: _clamp01(dims_raw.get(k)) for k in keys}
    mean_score = sum(dimensions.values()) / max(1, len(keys))

    score = _clamp01(parsed.get("score", mean_score))
    passed_raw = parsed.get("pass")
    if isinstance(passed_raw, bool):
        passed = passed_raw
    else:
        passed = score >= pass_threshold

    errs = parsed.get("error_categories") or []
    if not isinstance(errs, list):
        errs = []
    error_categories = [str(x) for x in errs if str(x).strip()]

    rationale = str(parsed.get("rationale") or "").strip()[:8000]

    meta: dict[str, Any] = {"provider": "gemini", "model": model, "critic_prompt_version": prompt_version}
    if isinstance(api_usage, dict):
        meta["usage"] = api_usage
    verdict = CriticVerdict(
        score=score,
        passed=passed,
        dimensions=dimensions,
        error_categories=error_categories,
        rationale=rationale,
    )
    return verdict, meta


@dataclass(frozen=True)
class CriticVerdict:
    score: float
    passed: bool
    dimensions: dict[str, float]
    error_categories: list[str]
    rationale: str


def run_gemini_critic(
    *,
    user_question: str,
    course_answer: str,
    boosted_explanation: str | None,
    retrieved_chunks: list[dict],
    structured_plan: dict | None,
    expected_behavior: dict | None,
    mode: str,
) -> tuple[CriticVerdict | None, dict[str, Any]]:
    """Call Gemini; return ``(verdict, meta)``. Meta includes ``provider``, ``model``, or ``error``."""

    api_key = current_app.config.get("GEMINI_API_KEY") or current_app.config.get("GOOGLE_API_KEY")
    if not api_key:
        return None, {}

    model = str(current_app.config.get("CRITIC_MODEL") or current_app.config.get("GEMINI_MODEL") or "gemini-2.5-flash")
    timeout = int(current_app.config.get("CRITIC_TIMEOUT_SEC") or 60)
    temp = float(current_app.config.get("CRITIC_TEMPERATURE", 0.1))
    max_out = int(current_app.config.get("CRITIC_MAX_OUTPUT_TOKENS", 4096))
    pass_threshold = float(current_app.config.get("CRITIC_PASS_THRESHOLD", 0.7))
    prompt_version = str(current_app.config.get("CRITIC_PROMPT_VERSION", "v1"))
    use_schema = bool(current_app.config.get("CRITIC_USE_RESPONSE_SCHEMA", True))
    answer_cap = int(current_app.config.get("CRITIC_ANSWER_CHAR_CAP", 14_000))
    http_retries = int(current_app.config.get("CRITIC_HTTP_MAX_RETRIES", 8))
    retry_base = float(current_app.config.get("CRITIC_HTTP_RETRY_BASE_SEC", 2.0))
    retry_max_cap = float(current_app.config.get("CRITIC_HTTP_RETRY_MAX_DELAY_SEC", 120.0))

    chunks_in: list[Any] = []
    for c in list(retrieved_chunks or [])[:24]:
        if not isinstance(c, dict):
            continue
        c2 = dict(c)
        tx = c2.get("text")
        if isinstance(tx, str) and len(tx) > 6000:
            c2["text"] = tx[:6000] + "…"
        ex = c2.get("source_excerpt")
        if isinstance(ex, str) and len(ex) > 6000:
            c2["source_excerpt"] = ex[:6000] + "…"
        chunks_in.append(c2)
    payload = {
        "PASS_THRESHOLD": pass_threshold,
        "EFFECTIVE_MODE": mode or "auto",
        "STUDENT_QUESTION": (user_question or "")[:8000],
        "CHATBOT_ANSWER": (course_answer or "")[:answer_cap],
        "BOOSTED_EXPLANATION": (boosted_explanation or "")[:8000] if boosted_explanation else None,
        "RETRIEVED_CHUNKS": chunks_in,
        "STRUCTURED_PLAN_JSON": structured_plan or {},
        "EXPECTED_BEHAVIOR_JSON": expected_behavior or {},
    }
    critique_blob = json.dumps(payload, ensure_ascii=False)
    if len(critique_blob) > 100_000:
        critique_blob = critique_blob[:100_000] + "…"

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    base = {"temperature": temp, "maxOutputTokens": max_out}
    # JSON MIME first: widely supported. responseSchema can 400 or behave oddly on some keys/models;
    # keeping schema as a follow-up preserves strictness without blocking the common path.
    attempt_specs: list[tuple[str, dict[str, Any]]] = [
        ("json_mime", {**base, "responseMimeType": "application/json"}),
    ]
    if use_schema:
        attempt_specs.append(
            (
                "schema_json",
                {**base, "responseMimeType": "application/json", "responseSchema": _CRITIC_RESPONSE_SCHEMA},
            )
        )
    attempt_specs.append(("plain", dict(base)))

    last_meta: dict[str, Any] = {"provider": "gemini", "model": model, "critic_prompt_version": prompt_version}
    for attempt_name, gen_cfg in attempt_specs:
        rubric = _CRITIC_RUBRIC_SCHEMA if attempt_name == "schema_json" else _CRITIC_RUBRIC_V1
        user_text = f"{rubric}\n\nCRITIQUE_INPUT_JSON:\n{critique_blob}"
        body_obj = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": gen_cfg,
        }
        try:
            status, raw, _hdrs = _post_generate_resilient(
                url,
                body_obj,
                timeout,
                max_retries=http_retries,
                base_sec=retry_base,
                max_delay_sec=retry_max_cap,
            )
        except OSError as e:
            logger.warning("Critic Gemini request failed: %s", e)
            return None, {**last_meta, "error": str(e)}

        if status != 200:
            logger.info(
                "Critic Gemini HTTP %s attempt=%s preview=%s",
                status,
                attempt_name,
                raw[:400],
            )
            last_meta = {
                **last_meta,
                "error": f"http_{status}",
                "attempt": attempt_name,
                "body_preview": raw[:500],
            }
            try:
                ej = json.loads(raw)
                if isinstance(ej, dict):
                    last_meta["api_error"] = ej.get("error", ej)
            except json.JSONDecodeError:
                pass
            continue

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            last_meta = {**last_meta, "error": "invalid_json_body", "attempt": attempt_name}
            continue

        text, cand_meta = _candidate_text_and_meta(data)
        um = data.get("usageMetadata")
        if isinstance(um, dict):
            last_meta["usage"] = um

        if not text:
            last_meta = {**last_meta, **cand_meta, "attempt": attempt_name}
            continue

        parsed = _extract_json_object(text)
        if not parsed:
            fr = cand_meta.get("finish_reason", "")
            logger.info(
                "Critic malformed JSON attempt=%s finish=%s text_prefix=%s",
                attempt_name,
                fr,
                text[:120],
            )
            last_meta = {
                **last_meta,
                **cand_meta,
                "error": "critic_malformed_json",
                "attempt": attempt_name,
                "raw_preview": text[:400],
            }
            continue

        v, m = _verdict_from_parsed(
            parsed,
            pass_threshold=pass_threshold,
            model=model,
            prompt_version=prompt_version,
            api_usage=data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else None,
        )
        m["attempt"] = attempt_name
        return v, m

    return None, last_meta
