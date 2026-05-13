"""Tests for Gemini critic (mocked HTTP)."""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import patch

import pytest

from app import create_app
from app.config import TestConfig
from app.services.critic import gemini_critic as gemini_critic_mod
from app.services.critic.gemini_critic import run_gemini_critic


def test_critic_rubric_v2_guard_strings():
    """Rubric text must keep generous calibration + suite nonsense carve-out (regression guard)."""
    for blob in (gemini_critic_mod._CRITIC_RUBRIC_SCHEMA, gemini_critic_mod._CRITIC_RUBRIC_V1):
        low = blob.lower()
        assert "score generously" in low
        assert "error_tags" in low
        assert "nonsense" in low
        assert "off_topic" in low
        assert "adversarial" in low


def _gemini_response_payload(text: str) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"totalTokenCount": 42},
    }


def test_run_gemini_critic_no_api_key():
    app = create_app(TestConfig)
    with app.app_context():
        v, meta = run_gemini_critic(
            user_question="What is MFCC?",
            course_answer="MFCCs are …",
            boosted_explanation=None,
            retrieved_chunks=[{"id": 1, "text": "chunk"}],
            structured_plan=None,
            expected_behavior=None,
            mode="chat",
        )
    assert v is None
    assert meta == {}


def test_run_gemini_critic_success_json():
    verdict_json = json.dumps(
        {
            "dimensions": {
                "grounded": 0.8,
                "accurate": 0.9,
                "complete": 0.7,
                "mode_compliant": 1.0,
                "no_hallucination": 0.85,
            },
            "score": 0.85,
            "pass": True,
            "error_categories": [],
            "rationale": "Grounded in chunks.",
        }
    )

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(_gemini_response_payload(verdict_json)).encode()

    app = create_app(TestConfig)
    app.config["GEMINI_API_KEY"] = "test-key"
    app.config["CRITIC_PASS_THRESHOLD"] = 0.7

    with app.app_context():
        with patch("app.services.critic.gemini_critic.urllib.request.urlopen", return_value=_FakeResp()):
            v, meta = run_gemini_critic(
                user_question="q",
                course_answer="a",
                boosted_explanation=None,
                retrieved_chunks=[],
                structured_plan={},
                expected_behavior={},
                mode="chat",
            )
    assert v is not None
    assert v.passed is True
    assert v.score == pytest.approx(0.85)
    assert v.dimensions["grounded"] == pytest.approx(0.8)
    assert meta.get("provider") == "gemini"
    assert meta.get("usage", {}).get("totalTokenCount") == 42


def test_run_gemini_critic_code_fence():
    body = """```json
{"dimensions":{"grounded":1,"accurate":1,"complete":1,"mode_compliant":1,"no_hallucination":1},"score":1,"pass":true,"error_categories":[],"rationale":"ok"}
```"""
    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(_gemini_response_payload(body)).encode()

    app = create_app(TestConfig)
    app.config["GEMINI_API_KEY"] = "k"
    with app.app_context():
        with patch("app.services.critic.gemini_critic.urllib.request.urlopen", return_value=_FakeResp()):
            v, meta = run_gemini_critic(
                user_question="q",
                course_answer="a",
                boosted_explanation=None,
                retrieved_chunks=[{"text": "x"}],
                structured_plan=None,
                expected_behavior=None,
                mode="chat",
            )
    assert v is not None
    assert v.score == 1.0


def test_run_gemini_critic_http_error():
    import urllib.error

    class _FakeErr:
        def read(self):
            return b"bad"

    app = create_app(TestConfig)
    app.config["GEMINI_API_KEY"] = "k"
    with app.app_context():
        with patch(
            "app.services.critic.gemini_critic.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("url", 429, "Too Many", hdrs=None, fp=BytesIO(b"err")),
        ):
            v, meta = run_gemini_critic(
                user_question="q",
                course_answer="a",
                boosted_explanation=None,
                retrieved_chunks=[],
                structured_plan=None,
                expected_behavior=None,
                mode="chat",
            )
    assert v is None
    assert meta.get("error") == "http_429"


def test_run_gemini_critic_no_text_candidate():
    app = create_app(TestConfig)
    app.config["GEMINI_API_KEY"] = "k"

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"candidates": [{"content": {"parts": [{"text": ""}]}}]}).encode()

    with app.app_context():
        with patch("app.services.critic.gemini_critic.urllib.request.urlopen", return_value=_FakeResp()):
            v, meta = run_gemini_critic(
                user_question="q",
                course_answer="a",
                boosted_explanation=None,
                retrieved_chunks=[],
                structured_plan=None,
                expected_behavior=None,
                mode="chat",
            )
    assert v is None
    assert meta.get("provider") == "gemini"
    assert meta.get("error") == "no_text"
