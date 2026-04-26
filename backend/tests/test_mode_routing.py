"""End-to-end mode routing tests for ``POST /api/chat``.

Verify that ``effective_mode`` actually drives the renderer, not just retrieval:

- ``mode_override="quiz"``    -> quiz_render output (Quiz: + Answer Key, no Course Answer headings)
- ``mode_override="summary"`` -> summary_render output (Summary: + Main idea / Key topics)
- ``mode_override="compare"`` -> compare_render output (Course Answer with compare-specific
  headings such as "Contrast along course axes")
"""

from __future__ import annotations

import json

from app.extensions import db
from app.models import LectureChunk
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache

from tests.conftest import register_user

_PW = "Abcd1234!"


def _login(client, email: str) -> None:
    register_user(client, email, _PW)
    client.post(
        "/api/auth/login",
        json={"email": email, "password": _PW},
        content_type="application/json",
    )


def _seed_route_chunks(app) -> None:
    """Seed enough lecture chunks for retrieval to find evidence for routing tests."""
    with app.app_context():
        db.session.add_all(
            [
                LectureChunk(
                    chunk_key="route-mfcc-1",
                    lecture_number=10,
                    topic="MFCCs — Core Idea",
                    keywords=json.dumps(["mfcc", "speech", "spectrum"]),
                    source_excerpt="MFCCs summarize the spectrum of speech as a small vector.",
                    clean_explanation="MFCCs summarize the spectrum of speech as a small vector.",
                    sample_questions=json.dumps(["What do MFCCs summarize?"]),
                    sample_answer="The spectrum of speech.",
                ),
                LectureChunk(
                    chunk_key="route-mfcc-2",
                    lecture_number=10,
                    topic="MFCCs — Pipeline",
                    keywords=json.dumps(["mfcc", "filterbank", "log"]),
                    source_excerpt=(
                        "The MFCC pipeline applies a filterbank, takes logs, and runs a DCT."
                    ),
                    clean_explanation=(
                        "The MFCC pipeline applies a filterbank, takes logs, and runs a DCT."
                    ),
                    sample_questions="[]",
                    sample_answer=None,
                ),
                LectureChunk(
                    chunk_key="route-mfcc-3",
                    lecture_number=10,
                    topic="Formants — Core Idea",
                    keywords=json.dumps(["formant", "vowel", "spectrum"]),
                    source_excerpt="Formants are spectral peaks tied to vocal tract shape.",
                    clean_explanation="Formants are spectral peaks tied to vocal tract shape.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
                LectureChunk(
                    chunk_key="route-softmax-1",
                    lecture_number=14,
                    topic="Softmax — Core Idea",
                    keywords=json.dumps(["softmax", "probability", "logits"]),
                    source_excerpt="Softmax turns logits into a probability distribution.",
                    clean_explanation="Softmax turns logits into a probability distribution.",
                    sample_questions=json.dumps(["What does softmax produce?"]),
                    sample_answer="A probability distribution over classes.",
                ),
                LectureChunk(
                    chunk_key="route-hardmax-1",
                    lecture_number=14,
                    topic="Hardmax — Core Idea",
                    keywords=json.dumps(["hardmax", "argmax", "one-hot"]),
                    source_excerpt="Hardmax picks the argmax and returns a one-hot vector.",
                    clean_explanation="Hardmax picks the argmax and returns a one-hot vector.",
                    sample_questions="[]",
                    sample_answer=None,
                ),
            ]
        )
        db.session.commit()
        invalidate_lecture_cache()
        load_lecture_cache()


def _open_chat_session(client) -> int:
    return client.post(
        "/api/sessions",
        json={"title": "t"},
        content_type="application/json",
    ).get_json()["session"]["id"]


def _post_chat(client, sid: int, message: str, **extra) -> dict:
    payload = {"session_id": sid, "message": message, **extra}
    response = client.post(
        "/api/chat",
        json=payload,
        content_type="application/json",
    )
    assert response.status_code == 200, response.get_json()
    return response.get_json()


# ---------------------------------------------------------------------------
# Quiz routing
# ---------------------------------------------------------------------------

def test_quiz_mode_overrides_to_quiz_renderer(client, app):
    """``mode_override=quiz`` returns the quiz renderer output, never Course Answer headings."""
    _login(client, "route-quiz@test.dev")
    _seed_route_chunks(app)
    sid = _open_chat_session(client)
    body = _post_chat(client, sid, "Quiz me on MFCCs", mode_override="quiz")

    assert body["mode"]["effective"] == "quiz"
    answer = body["answer"]
    assert "Quiz:" in answer
    assert "Answer Key:" in answer
    assert "1." in answer
    # The four-block Course Answer headings must never appear in quiz output.
    forbidden_headings = (
        "### Direct Answer",
        "### Explanation",
        "### Example / Intuition",
        "### Why it matters",
        "Course Answer:",
    )
    for marker in forbidden_headings:
        assert marker not in answer, f"quiz output unexpectedly contains '{marker}'"


# ---------------------------------------------------------------------------
# Summary routing
# ---------------------------------------------------------------------------

def test_summary_mode_overrides_to_summary_renderer(client, app):
    """``mode_override=summary`` returns the summary renderer output, never Course Answer headings."""
    _login(client, "route-summary@test.dev")
    _seed_route_chunks(app)
    sid = _open_chat_session(client)
    body = _post_chat(client, sid, "Summarize Lecture 10", mode_override="summary")

    assert body["mode"]["effective"] == "summary"
    answer = body["answer"]
    assert "Summary: Lecture 10" in answer
    assert "### Main idea" in answer
    assert "### Key topics" in answer
    forbidden_headings = (
        "### Direct Answer",
        "### Explanation",
        "Course Answer:",
    )
    for marker in forbidden_headings:
        assert marker not in answer, f"summary output unexpectedly contains '{marker}'"


# ---------------------------------------------------------------------------
# Compare routing (regression: existing compare renderer is still used)
# ---------------------------------------------------------------------------

def test_compare_mode_uses_compare_renderer(client, app):
    """``mode_override=compare`` keeps using the deterministic compare renderer.

    The compare renderer emits the marker heading ``### Contrast along course axes``
    (see compare_render.format_two_entity_compare_markdown); chat mode never emits it.
    """
    _login(client, "route-compare@test.dev")
    _seed_route_chunks(app)
    sid = _open_chat_session(client)
    body = _post_chat(
        client,
        sid,
        "Compare softmax and hardmax",
        mode_override="compare",
    )

    assert body["mode"]["effective"] == "compare"
    answer = body["answer"]
    # Compare renderer's signature heading or the general "in one line" lead.
    assert (
        "### Contrast along course axes" in answer
        or "in one line:" in answer
        or "### Compared architectures" in answer
    ), f"compare output missing compare-renderer markers: {answer[:400]}"
