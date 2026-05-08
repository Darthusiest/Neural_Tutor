"""POST /api/chat/boost/<message_id> (deferred constrained boost)."""

from __future__ import annotations

import json

from app.extensions import db
from app.models import ChatSession, LectureChunk, Message, RetrievalChunkHit, RetrievalLog, User
from app.services.answers.concept_constraints import build_concept_constraints
from app.services.generation.boost_provider import BoostAttempt
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import StructuredQuery
from app.services.query_understanding import QueryIntent, QueryType

from tests.conftest import register_user

_PW = "Abcd1234!"


def _boost_constraints_snapshot(concept_id: str) -> dict:
    intent = QueryIntent(
        query_type=QueryType.DEFINITION,
        original_query=f"what is {concept_id}",
        expanded_query=f"what is {concept_id}",
        query_tokens=["what"],
        expanded_tokens=["what"],
        lecture_numbers=[],
        detected_concepts=[concept_id],
        compare_concepts=None,
        compare_entities=[],
    )
    sq = StructuredQuery(
        intent=intent,
        concept_ids=[concept_id],
        answer_intent="direct_definition",
        sub_questions=[],
        retrieval_hints=[],
        lecture_scope=[],
        answer_style="teaching",
        decomposition_template=[],
    )
    return build_concept_constraints(sq, get_kb()).to_dict()


def _login(client, email: str) -> None:
    client.post(
        "/api/auth/login",
        json={"email": email, "password": _PW},
        content_type="application/json",
    )


def test_boost_endpoint_idempotent_skipped(client, app):
    register_user(client, "boost_skip@test.dev", _PW)
    _login(client, "boost_skip@test.dev")
    with app.app_context():
        u = User.query.filter_by(email="boost_skip@test.dev").first()
        s = ChatSession(user_id=u.id, title="t", mode="auto")
        db.session.add(s)
        db.session.flush()
        payload = {
            "course_answer": "Course Answer:\n\nok",
            "answer": "Course Answer:\n\nok",
            "boosted_explanation": None,
            "boost_status": "skipped",
            "boost_skip_reason": "no_retrieval_log",
            "mode": {"effective": "chat"},
            "mode_routing": {},
        }
        m = Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps(payload),
        )
        db.session.add(m)
        db.session.commit()
        mid = m.id

    r = client.post(f"/api/chat/boost/{mid}")
    assert r.status_code == 200
    data = r.get_json()
    assert data["boost_status"] == "skipped"


def test_boost_endpoint_forbidden_wrong_user(client, app):
    register_user(client, "boost_a@test.dev", _PW)
    register_user(client, "boost_b@test.dev", _PW)
    with app.app_context():
        ua = User.query.filter_by(email="boost_a@test.dev").first()
        ub = User.query.filter_by(email="boost_b@test.dev").first()
        s = ChatSession(user_id=ua.id, title="t", mode="auto")
        db.session.add(s)
        db.session.flush()
        m = Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps({"boost_status": "pending"}),
        )
        db.session.add(m)
        db.session.commit()
        mid = m.id

    _login(client, "boost_b@test.dev")
    r = client.post(f"/api/chat/boost/{mid}")
    assert r.status_code == 403


def test_widely_known_concept_with_shallow_draft_calls_extended_prompt(client, app, monkeypatch):
    captured: dict[str, bool | None] = {}

    def fake_openai_boost(*, allow_external_clarification=False, allowed_evidence_lines=None, **kwargs):
        captured["allow_external_clarification"] = bool(allow_external_clarification)
        lines = allowed_evidence_lines or []
        body = lines[0] if lines else "fallback softmax line."
        return (f"Boosted Explanation:\n\n{body}", {})

    monkeypatch.setattr(
        "app.services.boost_deferred.boost_provider_chain",
        lambda: [BoostAttempt(provider="openai", has_key=True)],
    )
    monkeypatch.setattr(
        "app.services.boost_deferred.generate_openai_constrained_boost",
        fake_openai_boost,
    )

    register_user(client, "boost_ext@test.dev", _PW)
    _login(client, "boost_ext@test.dev")
    with app.app_context():
        snap = _boost_constraints_snapshot("softmax")
        u = User.query.filter_by(email="boost_ext@test.dev").first()
        s = ChatSession(user_id=u.id, title="t", mode="auto")
        db.session.add(s)
        db.session.flush()
        lc = LectureChunk(
            chunk_key="pytest-softmax-chunk-ext",
            lecture_number=9,
            topic="softmax lecture softmax classification topic",
            keywords=json.dumps(["softmax"]),
            source_excerpt="Softmax converts logits.",
            clean_explanation=(
                "Softmax converts logits into a normalized probability distribution over classes.\n"
                "The softmax function exponentiates each logit before normalization.\n"
                "Students contrast softmax outputs with alternatives."
            ),
        )
        db.session.add(lc)
        db.session.flush()
        payload = {
            "course_answer": "short",
            "answer": "short",
            "boost_status": "pending",
            "boosted_explanation": None,
            "mode": {"effective": "chat"},
            "mode_routing": {},
            "boost_constraints": snap,
        }
        m = Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps(payload),
        )
        db.session.add(m)
        db.session.flush()
        rl = RetrievalLog(
            message_id=m.id,
            session_id=s.id,
            user_question="what is softmax",
        )
        db.session.add(rl)
        db.session.flush()
        db.session.add(
            RetrievalChunkHit(
                retrieval_log_id=rl.id,
                lecture_chunk_id=lc.id,
                rank=0,
                score=1.0,
                selected_for_answer=True,
            )
        )
        db.session.commit()
        mid = m.id

    r = client.post(f"/api/chat/boost/{mid}")
    assert r.status_code == 200
    assert captured.get("allow_external_clarification") is True


def test_unflagged_concept_uses_clarity_only_prompt(client, app, monkeypatch):
    captured: dict[str, bool | None] = {}

    def fake_openai_boost(*, allow_external_clarification=False, allowed_evidence_lines=None, **kwargs):
        captured["allow_external_clarification"] = bool(allow_external_clarification)
        lines = allowed_evidence_lines or []
        body = lines[0] if lines else "fallback hardmax line."
        return (f"Boosted Explanation:\n\n{body}", {})

    monkeypatch.setattr(
        "app.services.boost_deferred.boost_provider_chain",
        lambda: [BoostAttempt(provider="openai", has_key=True)],
    )
    monkeypatch.setattr(
        "app.services.boost_deferred.generate_openai_constrained_boost",
        fake_openai_boost,
    )

    register_user(client, "boost_plain@test.dev", _PW)
    _login(client, "boost_plain@test.dev")
    with app.app_context():
        snap = _boost_constraints_snapshot("hardmax")
        u = User.query.filter_by(email="boost_plain@test.dev").first()
        s = ChatSession(user_id=u.id, title="t", mode="auto")
        db.session.add(s)
        db.session.flush()
        lc = LectureChunk(
            chunk_key="pytest-hardmax-chunk-plain",
            lecture_number=9,
            topic="hardmax winner softmax classification comparison topic",
            keywords=json.dumps(["hardmax"]),
            source_excerpt="Hardmax picks winners.",
            clean_explanation=(
                "Hardmax picks only the top category from logits.\n"
                "Winner-take-all collapses uncertainty entirely.\n"
                "Contrasts are discussed versus softmax smoothing."
            ),
        )
        db.session.add(lc)
        db.session.flush()
        payload = {
            "course_answer": "short",
            "answer": "short",
            "boost_status": "pending",
            "boosted_explanation": None,
            "mode": {"effective": "chat"},
            "mode_routing": {},
            "boost_constraints": snap,
        }
        m = Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps(payload),
        )
        db.session.add(m)
        db.session.flush()
        rl = RetrievalLog(
            message_id=m.id,
            session_id=s.id,
            user_question="what is hardmax",
        )
        db.session.add(rl)
        db.session.flush()
        db.session.add(
            RetrievalChunkHit(
                retrieval_log_id=rl.id,
                lecture_chunk_id=lc.id,
                rank=0,
                score=1.0,
                selected_for_answer=True,
            )
        )
        db.session.commit()
        mid = m.id

    r = client.post(f"/api/chat/boost/{mid}")
    assert r.status_code == 200
    assert captured.get("allow_external_clarification") is False


def test_kb_flagged_concept_boost_when_no_chunk_evidence_uses_course_answer_fallback(
    client, app, monkeypatch
):
    """When retrieval yields zero pure evidence lines, external clarification still runs."""
    monkeypatch.setattr(
        "app.services.boost_deferred.collect_allowed_evidence_lines",
        lambda *_a, **_k: [],
    )

    captured: dict[str, bool | None] = {}

    def fake_openai_boost(*, allow_external_clarification=False, allowed_evidence_lines=None, **kwargs):
        captured["allow_external_clarification"] = bool(allow_external_clarification)
        lines = allowed_evidence_lines or []
        body = lines[0] if lines else "fallback."
        return (f"Boosted Explanation:\n\n{body}", {})

    monkeypatch.setattr(
        "app.services.boost_deferred.boost_provider_chain",
        lambda: [BoostAttempt(provider="openai", has_key=True)],
    )
    monkeypatch.setattr(
        "app.services.boost_deferred.generate_openai_constrained_boost",
        fake_openai_boost,
    )

    register_user(client, "boost_no_ev@test.dev", _PW)
    _login(client, "boost_no_ev@test.dev")
    with app.app_context():
        snap = _boost_constraints_snapshot("cnn")
        u = User.query.filter_by(email="boost_no_ev@test.dev").first()
        s = ChatSession(user_id=u.id, title="t", mode="auto")
        db.session.add(s)
        db.session.flush()
        lc = LectureChunk(
            chunk_key="pytest-cnn-chunk-no-evidence",
            lecture_number=16,
            topic="cnn speech topic",
            keywords=json.dumps(["cnn"]),
            source_excerpt="x",
            clean_explanation="y",
        )
        db.session.add(lc)
        db.session.flush()
        payload = {
            "course_answer": (
                "Course Answer:\n\n"
                "Convolutional neural networks extract local temporal patterns before pooling.\n"
                "Residual streams interact with CNN layers in lecture examples."
            ),
            "answer": "stub",
            "boost_status": "pending",
            "boosted_explanation": None,
            "mode": {"effective": "chat"},
            "mode_routing": {},
            "boost_constraints": snap,
        }
        m = Message(
            session_id=s.id,
            role="assistant",
            content_text=None,
            payload_json=json.dumps(payload),
        )
        db.session.add(m)
        db.session.flush()
        rl = RetrievalLog(
            message_id=m.id,
            session_id=s.id,
            user_question="what is cnn",
        )
        db.session.add(rl)
        db.session.flush()
        db.session.add(
            RetrievalChunkHit(
                retrieval_log_id=rl.id,
                lecture_chunk_id=lc.id,
                rank=0,
                score=1.0,
                selected_for_answer=True,
            )
        )
        db.session.commit()
        mid = m.id

    r = client.post(f"/api/chat/boost/{mid}")
    assert r.status_code == 200
    data = r.get_json()
    assert captured.get("allow_external_clarification") is True
    assert data["boost_status"] == "ready"
    assert "convolutional neural" in (data.get("boosted_explanation") or "").lower()
