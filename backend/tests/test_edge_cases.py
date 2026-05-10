"""Edge-case and stress-scenario tests.

Covers concept drift, alias resolution, typo tolerance, vague queries,
mode-override conflicts, out-of-scope queries, and boundary lengths.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.extensions import db
from app.services.knowledge.concept_kb import reset_kb_for_tests
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache
from tests.conftest import register_user

_DATA = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"
_PW = "Abcd1234!"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corpus(app):
    with app.app_context():
        db.drop_all()
        db.create_all()
        import_lecture_json(_DATA, upsert=False)
        invalidate_lecture_cache()
        load_lecture_cache()
    yield
    reset_kb_for_tests()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _login(client, email):
    register_user(client, email, _PW)
    client.post(
        "/api/auth/login",
        json={"email": email, "password": _PW},
        content_type="application/json",
    )


def _open_chat_session(client):
    return client.post(
        "/api/sessions",
        json={"title": "t"},
        content_type="application/json",
    ).get_json()["session"]["id"]


def _post_chat(client, sid, message, **extra):
    payload = {"session_id": sid, "message": message, **extra}
    response = client.post(
        "/api/chat",
        json=payload,
        content_type="application/json",
    )
    return response


def _chat(client, email, message, **extra):
    """Login, open session, post message, return raw response."""
    _login(client, email)
    sid = _open_chat_session(client)
    return _post_chat(client, sid, message, **extra)


def _chat_answer(client, email, message, **extra):
    """Convenience: return (response, answer_text)."""
    resp = _chat(client, email, message, **extra)
    body = resp.get_json()
    return resp, (body or {}).get("answer", "")


# ---------------------------------------------------------------------------
# TestConceptDrift
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConceptDrift:
    """Single-concept queries must not bleed unrelated concepts in the opening."""

    def test_cnn_no_transformer_bleed(self, client, app, corpus):
        resp, answer = _chat_answer(client, "drift-cnn@test.dev", "What is CNN?")
        assert resp.status_code == 200
        opening = answer[:400].lower()
        assert "self-attention" not in opening, f"transformer bleed in CNN opening: {opening}"
        assert "transformer" not in opening, f"transformer bleed in CNN opening: {opening}"

    def test_mfcc_no_softmax_bleed(self, client, app, corpus):
        resp, answer = _chat_answer(client, "drift-mfcc@test.dev", "What is MFCC?")
        assert resp.status_code == 200
        opening = answer[:400].lower()
        assert "softmax" not in opening, f"softmax bleed in MFCC opening: {opening}"

    def test_dp_no_neural_network_lead(self, client, app, corpus):
        resp, answer = _chat_answer(
            client, "drift-dp@test.dev", "What is dynamic programming?"
        )
        assert resp.status_code == 200
        opening = answer[:400].lower()
        assert not opening.lstrip().startswith("neural network"), (
            f"DP answer opens with neural network discussion: {opening}"
        )

    def test_softmax_no_hardmax_bleed(self, client, app, corpus):
        resp, answer = _chat_answer(client, "drift-soft@test.dev", "What is softmax?")
        assert resp.status_code == 200
        opening = answer[:400].lower()
        assert "hardmax" not in opening, f"hardmax bleed in softmax opening: {opening}"

    def test_dropout_no_normalization_primary(self, client, app, corpus):
        resp, answer = _chat_answer(client, "drift-drop@test.dev", "What is dropout?")
        assert resp.status_code == 200
        opening = answer[:400].lower()
        assert "layer norm" not in opening, f"layer norm bleed in dropout opening: {opening}"
        assert "normalization" not in opening, (
            f"normalization as primary topic in dropout opening: {opening}"
        )


# ---------------------------------------------------------------------------
# TestAliasResolution
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAliasResolution:
    """Queries using abbreviations / aliases should resolve to canonical concepts."""

    def test_sgd_resolves(self, client, app, corpus):
        _, answer = _chat_answer(client, "alias-sgd@test.dev", "What is SGD?")
        assert "gradient" in answer.lower()

    def test_backprop_resolves(self, client, app, corpus):
        _, answer = _chat_answer(client, "alias-bp@test.dev", "Explain backprop")
        low = answer.lower()
        assert "backpropagation" in low or "backprop" in low

    def test_autoencoder_resolves(self, client, app, corpus):
        _, answer = _chat_answer(
            client, "alias-ae@test.dev", "What is an auto-encoder?"
        )
        low = answer.lower()
        assert "autoencoder" in low or "auto-encoder" in low

    def test_asr_resolves(self, client, app, corpus):
        _, answer = _chat_answer(client, "alias-asr@test.dev", "What is ASR?")
        assert "speech recognition" in answer.lower()

    def test_tts_resolves(self, client, app, corpus):
        _, answer = _chat_answer(client, "alias-tts@test.dev", "Explain TTS")
        low = answer.lower()
        assert "speech" in low
        assert "synthesis" in low or "text" in low

    def test_mha_resolves(self, client, app, corpus):
        _, answer = _chat_answer(client, "alias-mha@test.dev", "What is MHA?")
        low = answer.lower()
        assert "attention" in low and "head" in low

    def test_vq_resolves(self, client, app, corpus):
        _, answer = _chat_answer(client, "alias-vq@test.dev", "What is VQ?")
        low = answer.lower()
        assert "vector quantization" in low or "quantiz" in low

    def test_rvq_resolves(self, client, app, corpus):
        _, answer = _chat_answer(client, "alias-rvq@test.dev", "Define RVQ")
        low = answer.lower()
        assert "residual" in low and "quantiz" in low

    def test_convnet_resolves(self, client, app, corpus):
        _, answer = _chat_answer(
            client, "alias-convnet@test.dev", "What is a convnet?"
        )
        low = answer.lower()
        assert "convolutional" in low or "cnn" in low


# ---------------------------------------------------------------------------
# TestTypoTolerance
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTypoTolerance:
    """Common typos should still produce useful responses."""

    def test_backpropogation_typo(self, client, app, corpus):
        resp, answer = _chat_answer(
            client, "typo-bp@test.dev", "What is backpropogation?"
        )
        assert resp.status_code == 200
        assert "backprop" in answer.lower()

    def test_gradiant_typo(self, client, app, corpus):
        resp, answer = _chat_answer(
            client, "typo-grad@test.dev", "Explan the gradiant"
        )
        assert resp.status_code == 200
        assert len(answer.strip()) > 0

    def test_sofmax_typo(self, client, app, corpus):
        resp, answer = _chat_answer(client, "typo-sm@test.dev", "What is sofmax?")
        assert resp.status_code == 200
        low = answer.lower()
        assert "softmax" in low or "probability" in low


# ---------------------------------------------------------------------------
# TestVagueQueries
# ---------------------------------------------------------------------------


_CLARIFICATION_RE = re.compile(
    r"clarif|could you|what|rephrase|specify|more detail|not sure",
    re.IGNORECASE,
)


@pytest.mark.slow
class TestVagueQueries:
    """Vague / nonsensical input should produce clarification, not hallucination."""

    def _assert_clarification(self, resp, answer):
        assert resp.status_code == 200
        assert (
            len(answer.strip()) < 300
            or _CLARIFICATION_RE.search(answer) is not None
        ), f"Expected clarification-style response, got: {answer[:400]}"

    def test_gibberish(self, client, app, corpus):
        resp, answer = _chat_answer(client, "vague-gib@test.dev", "asf")
        self._assert_clarification(resp, answer)

    def test_bare_what(self, client, app, corpus):
        resp, answer = _chat_answer(client, "vague-what@test.dev", "what")
        self._assert_clarification(resp, answer)

    def test_huh(self, client, app, corpus):
        resp, answer = _chat_answer(client, "vague-huh@test.dev", "huh")
        self._assert_clarification(resp, answer)

    def test_compare_no_entities(self, client, app, corpus):
        resp, answer = _chat_answer(
            client, "vague-cmp@test.dev", "compare these"
        )
        self._assert_clarification(resp, answer)

    def test_test_me_no_topic(self, client, app, corpus):
        resp = _chat(client, "vague-quiz@test.dev", "test me")
        assert resp.status_code == 200

    def test_empty_message_rejected(self, client, app, corpus):
        resp = _chat(client, "vague-empty@test.dev", "")
        assert resp.status_code in (400, 422), (
            f"Empty message should be rejected, got {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# TestModeOverrideConflicts
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestModeOverrideConflicts:
    """mode_override should always win over auto-detection or legacy mode key."""

    def test_quiz_override_on_compare_query(self, client, app, corpus):
        resp = _chat(
            client,
            "mo-quiz@test.dev",
            "Compare CNN and transformer",
            mode_override="quiz",
        )
        body = resp.get_json()
        assert body["mode"]["effective"] == "quiz"

    def test_summary_override_on_quiz_query(self, client, app, corpus):
        resp = _chat(
            client,
            "mo-sum@test.dev",
            "Quiz me on MFCCs",
            mode_override="summary",
        )
        body = resp.get_json()
        assert body["mode"]["effective"] == "summary"

    def test_invalid_override_falls_back(self, client, app, corpus):
        resp = _chat(
            client,
            "mo-inv@test.dev",
            "What is CNN?",
            mode_override="invalid_value",
        )
        body = resp.get_json()
        assert body["mode"]["effective"] != "invalid_value"

    def test_mode_override_beats_legacy_mode(self, client, app, corpus):
        resp = _chat(
            client,
            "mo-leg@test.dev",
            "What is CNN?",
            mode="compare",
            mode_override="quiz",
        )
        body = resp.get_json()
        assert body["mode"]["effective"] == "quiz"


# ---------------------------------------------------------------------------
# TestOutOfScope
# ---------------------------------------------------------------------------


def _is_out_of_scope_response(resp):
    """Heuristic: the system signals the query is off-syllabus."""
    body = resp.get_json() or {}
    answer = body.get("answer", "")
    low = answer.lower()
    if any(
        marker in low
        for marker in ("not covered", "outside", "course", "clarif", "don't have")
    ):
        return True
    if body.get("retrieval_confidence", 1.0) < 0.5:
        return True
    if len(answer.strip()) < 200:
        return True
    return False


@pytest.mark.slow
class TestOutOfScope:
    """Off-syllabus queries should not produce confident LING 487 answers."""

    def test_quantum_computing(self, client, app, corpus):
        resp = _chat(client, "oos-qc@test.dev", "What is quantum computing?")
        assert resp.status_code == 200
        assert _is_out_of_scope_response(resp)

    def test_french_revolution(self, client, app, corpus):
        resp = _chat(client, "oos-fr@test.dev", "Explain the French Revolution")
        assert resp.status_code == 200
        assert _is_out_of_scope_response(resp)

    def test_reactjs(self, client, app, corpus):
        resp = _chat(client, "oos-rjs@test.dev", "What is React.js?")
        assert resp.status_code == 200
        assert _is_out_of_scope_response(resp)

    def test_joke(self, client, app, corpus):
        resp = _chat(client, "oos-joke@test.dev", "Tell me a joke")
        assert resp.status_code == 200
        assert _is_out_of_scope_response(resp)

    def test_poem(self, client, app, corpus):
        resp = _chat(
            client, "oos-poem@test.dev", "Write me a poem about neural networks"
        )
        assert resp.status_code == 200
        assert _is_out_of_scope_response(resp)


# ---------------------------------------------------------------------------
# TestBoundaryLengths
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBoundaryLengths:
    """Very short and very long queries should still produce valid responses."""

    def test_single_word_query(self, client, app, corpus):
        resp, answer = _chat_answer(client, "blen-short@test.dev", "Softmax")
        assert resp.status_code == 200
        assert len(answer.strip()) > 0

    def test_very_long_query(self, client, app, corpus):
        long_msg = "What is softmax? " * 50
        resp, answer = _chat_answer(client, "blen-long@test.dev", long_msg)
        assert resp.status_code == 200
        assert len(answer.strip()) > 0
