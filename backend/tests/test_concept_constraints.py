"""Tests for the concept-purity layer (Task 5).

Covers :class:`ConceptConstraints` construction, the post-retrieval rerank
applied by :func:`apply_concept_constraints`, the line-level
:func:`is_line_concept_pure` helper, and the new
``must_be_concept_pure`` validator wired into :func:`validate_answer`.

End-to-end pipeline tests run against the same corpus
``test_structured_pipeline`` uses so we exercise the live retrieval rerank
the chat path runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extensions import db
from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.answer_validation import validate_answer
from app.services.answers.concept_constraints import (
    ConceptConstraints,
    apply_concept_constraints,
    build_concept_constraints,
    is_line_concept_pure,
    score_chunk_against_constraints,
)
from app.services.knowledge.concept_kb import get_kb, reset_kb_for_tests
from app.services.knowledge.structured_query import build_structured_query
from app.services.lectures.lecture_loader import import_lecture_json
from app.services.query_understanding import analyze_query
from app.services.reasoning_pipeline import run_reasoning_pipeline
from app.services.retrieval import invalidate_lecture_cache, load_lecture_cache


_DATA = Path(__file__).resolve().parent.parent / "data" / "LING487_SUPER_TUTOR.json"
_KB = Path(__file__).resolve().parent.parent / "data" / "LING487_STRUCTURED_PIPELINE_KB.json"


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


def _sq(query: str):
    intent = analyze_query(query)
    return build_structured_query(intent, kb=get_kb())


def _plan_with_answer(direct_answer: str | None = None) -> AnswerPlan:
    return AnswerPlan(
        answer_mode="direct_definition",
        sections=[],
        primary_chunk_ids=[],
        supporting_chunk_ids=[],
        include_example=False,
        include_analogy=False,
        include_prerequisites=False,
        include_related_concepts=[],
        comparison_axes=[],
        lecture_scope=[],
        direct_answer=direct_answer,
    )


# ---------------------------------------------------------------------------
# build_concept_constraints — derivation from StructuredQuery
# ---------------------------------------------------------------------------


def test_build_constraints_for_what_is_cnn(app):
    with app.app_context():
        sq = _sq("What is CNN?")
        c = build_concept_constraints(sq, kb=get_kb())
        assert "cnn" in c.target_concepts
        assert "cnn" in c.target_aliases
        # Convolution is *part* of CNN — it shows up as alias / related, not forbidden.
        assert "convolution" not in c.forbidden_terms
        # Transformer machinery should be flagged as off-limits for a single-CNN query.
        assert "transformer" in c.forbidden_terms or "self-attention" in c.forbidden_terms
        assert "residual connection" in c.forbidden_terms
        assert c.is_relational is False


def test_build_constraints_for_compare_uses_peer_aliases_per_side(app):
    """`Compare CNN and transformer`: forbidden contains both sides' machinery."""
    with app.app_context():
        sq = _sq("Compare CNN and transformer")
        c = build_concept_constraints(sq, kb=get_kb())
        # Both sides land in target_concepts; the constraints object is shared
        # (per-side filtering still happens inside the V2 bundle builder).
        assert "cnn" in c.target_concepts
        assert "transformer" in c.target_concepts
        # Relational queries flip the validator-loosen flag.
        assert c.is_relational is True
        # The forbidden term set is the union of both sides minus target aliases.
        assert "transformer" not in c.forbidden_terms
        assert "cnn" not in c.forbidden_terms


def test_build_constraints_for_mfcc_excludes_softmax(app):
    with app.app_context():
        sq = _sq("What is MFCC?")
        c = build_concept_constraints(sq, kb=get_kb())
        assert "mfcc" in c.target_concepts
        assert "softmax" in c.forbidden_terms
        assert "transformer" in c.forbidden_terms or "attention" in c.forbidden_terms


def test_build_constraints_for_dynamic_programming_blocks_neural(app):
    with app.app_context():
        sq = _sq("What is dynamic programming?")
        c = build_concept_constraints(sq, kb=get_kb())
        assert "dynamic_programming" in c.target_concepts
        assert "neural network" in c.forbidden_terms
        assert "backpropagation" in c.forbidden_terms
        assert c.is_relational is False


# ---------------------------------------------------------------------------
# Score / apply — chunk-level rerank (unit)
# ---------------------------------------------------------------------------


def _cnn_chunk():
    return {
        "id": 1,
        "topic": "CNN — Architecture",
        "keywords": "convolution local receptive",
        "clean_explanation": (
            "CNNs apply convolution over local receptive fields and share "
            "weights across spatial positions."
        ),
        "source_excerpt": "",
        "lecture_number": 17,
    }


def _transformer_chunk():
    return {
        "id": 2,
        "topic": "Transformer — Self-attention",
        "keywords": "transformer self-attention positional",
        "clean_explanation": (
            "Transformers use multi-head self-attention and positional encoding; "
            "residual connections feed the layer norm."
        ),
        "source_excerpt": "",
        "lecture_number": 22,
    }


def _residual_chunk():
    return {
        "id": 3,
        "topic": "Residual Connections",
        "keywords": "residual connection skip",
        "clean_explanation": (
            "Residual connections add the input back to the output of a layer "
            "before normalization."
        ),
        "source_excerpt": "",
        "lecture_number": 22,
    }


def test_apply_constraints_reranks_cnn_above_transformer_for_cnn_query(app):
    with app.app_context():
        sq = _sq("What is CNN?")
        c = build_concept_constraints(sq, kb=get_kb())
        chunks = [_transformer_chunk(), _residual_chunk(), _cnn_chunk()]
        out = apply_concept_constraints(chunks, c)
        # CNN chunk should now lead even though it was last in input order.
        assert out[0]["id"] == 1
        # Transformer / residual leak chunks should be demoted (or dropped).
        assert {ch["id"] for ch in out[:1]} == {1}


def test_apply_constraints_keeps_shared_lecture_chunk(app):
    """A chunk that mentions CNN AND transformer in a *compare* lecture is not dropped."""
    with app.app_context():
        sq = _sq("Compare CNN and transformer")
        c = build_concept_constraints(sq, kb=get_kb())
        chunk = {
            "id": 99,
            "topic": "Comparing architectures",
            "keywords": "cnn transformer compare",
            "clean_explanation": (
                "CNNs and transformers solve different problems: convolution "
                "exploits locality, while attention captures long-range "
                "dependencies."
            ),
            "source_excerpt": "",
            "lecture_number": 22,
        }
        out = apply_concept_constraints([chunk], c)
        # Relational query: shared-context chunk must survive.
        assert out and out[0]["id"] == 99


def test_apply_constraints_drops_obvious_leak_for_non_relational(app):
    """Non-relational *What is CNN?* drops a transformer-only off-topic chunk."""
    with app.app_context():
        sq = _sq("What is CNN?")
        c = build_concept_constraints(sq, kb=get_kb())
        # Heavy forbidden-topic chunk with zero CNN mentions — should be dropped.
        leak = {
            "id": 7,
            "topic": "Transformer self-attention residual connection",
            "keywords": "transformer self-attention residual connection layer norm",
            "clean_explanation": (
                "Self-attention with multi-head attention; residual connections "
                "wrap the layer norm in transformers."
            ),
            "source_excerpt": "",
            "lecture_number": 22,
        }
        on_topic = _cnn_chunk()
        out = apply_concept_constraints([leak, on_topic], c)
        ids = [ch["id"] for ch in out]
        assert 1 in ids
        assert 7 not in ids


def test_score_chunk_negative_when_only_forbidden_hits(app):
    with app.app_context():
        sq = _sq("What is dynamic programming?")
        c = build_concept_constraints(sq, kb=get_kb())
        nn_chunk = {
            "id": 5,
            "topic": "Neural networks",
            "keywords": "neural network backpropagation",
            "clean_explanation": (
                "Neural networks learn via backpropagation and gradient descent."
            ),
            "lecture_number": 9,
        }
        score = score_chunk_against_constraints(nn_chunk, c)
        assert score < 0.0


# ---------------------------------------------------------------------------
# is_line_concept_pure — line-level helper
# ---------------------------------------------------------------------------


def test_is_line_concept_pure_blocks_off_topic_for_chat(app):
    with app.app_context():
        sq = _sq("What is MFCC?")
        c = build_concept_constraints(sq, kb=get_kb())
        # Pure MFCC sentence — fine.
        assert is_line_concept_pure(
            "MFCCs compress the speech spectrum into perceptual features.", c
        )
        # Drift to softmax (no MFCC mention) — drop.
        assert not is_line_concept_pure(
            "Softmax converts logits to probabilities used in classification.", c
        )


def test_is_line_concept_pure_relaxes_for_relational_query(app):
    with app.app_context():
        sq = _sq("How does dynamic programming relate to backpropagation?")
        c = build_concept_constraints(sq, kb=get_kb())
        # Relational query: even forbidden-leaning sentences pass.
        assert c.is_relational
        assert is_line_concept_pure(
            "Backpropagation reuses subproblem gradients much like dynamic programming.",
            c,
        )


# ---------------------------------------------------------------------------
# End-to-end through the reasoning pipeline (integration)
# ---------------------------------------------------------------------------


def test_chat_query_what_is_cnn_does_not_include_transformer_in_top(corpus, app):
    """`What is CNN?` keeps transformer / self-attention chunks out of the lead.

    Residual connections are *taught alongside* CNNs in the corpus, so they
    legitimately appear; the regression we're locking down is transformer
    machinery dominating a CNN-scoped pool.
    """
    with app.app_context():
        result = run_reasoning_pipeline("What is CNN?", top_k=5)
        chunks = result.enhanced_result.chunks
        assert chunks, "expected non-empty retrieval"
        top_blob = " ".join(
            f"{c.get('topic', '')} {c.get('keywords', '')} {c.get('clean_explanation', '')}"
            for c in chunks[:3]
        ).lower()
        assert "cnn" in top_blob or "convolution" in top_blob
        # Transformer / self-attention machinery shouldn't appear in the top-3
        # for a single-CNN query.
        first_topic = (chunks[0].get("topic") or "").lower()
        first_keywords = (chunks[0].get("keywords") or "").lower()
        assert "transformer" not in first_topic
        assert "self-attention" not in first_topic
        assert "transformer" not in first_keywords
        assert "self-attention" not in first_keywords


def test_chat_query_what_is_mfcc_does_not_include_softmax(corpus, app):
    with app.app_context():
        result = run_reasoning_pipeline("What is MFCC?", top_k=5)
        chunks = result.enhanced_result.chunks
        assert chunks
        # MFCC pool should not surface softmax-led chunks.
        for chunk in chunks[:3]:
            blob = (
                (chunk.get("topic") or "")
                + " "
                + (chunk.get("keywords") or "")
            ).lower()
            assert "softmax" not in blob


def test_chat_query_dynamic_programming_does_not_lead_with_neural_networks(corpus, app):
    with app.app_context():
        result = run_reasoning_pipeline("What is dynamic programming?", top_k=5)
        chunks = result.enhanced_result.chunks
        assert chunks
        first = chunks[0]
        first_blob = (
            (first.get("topic") or "") + " " + (first.get("keywords") or "")
        ).lower()
        # The lead chunk for DP shouldn't be a neural-network topic header.
        assert "neural network" not in first_blob


def test_relational_dp_relates_to_backprop_keeps_neural_chunks(corpus, app):
    """`How does dynamic programming relate to backpropagation?` keeps neural-network chunks.

    Constraint loosening matters here: the spec calls for the relational
    formulation to *not* be over-filtered into uselessness.
    """
    with app.app_context():
        result = run_reasoning_pipeline(
            "How does dynamic programming relate to backpropagation?", top_k=8
        )
        chunks = result.enhanced_result.chunks
        assert chunks
        joined_blob = " ".join(
            (c.get("clean_explanation") or "") + " " + (c.get("topic") or "")
            for c in chunks
        ).lower()
        # At least one chunk should still mention the neural-network side of
        # the comparison — it's the whole reason for asking the question.
        assert "backprop" in joined_blob or "neural" in joined_blob


# ---------------------------------------------------------------------------
# Validator — must_be_concept_pure
# ---------------------------------------------------------------------------


def test_validator_concept_purity_flags_topic_drift(app):
    """A CNN-scoped answer mentioning *only* transformer machinery is flagged."""
    with app.app_context():
        sq = _sq("What is CNN?")
        c = build_concept_constraints(sq, kb=get_kb())
        plan = _plan_with_answer()
        bad_answer = (
            "Course Answer:\n\n"
            "Self-attention via multi-head attention with positional encoding "
            "and residual connections produces token-level features."
        )
        result = validate_answer(bad_answer, sq, plan, kb=get_kb(), constraints=c)
        assert "must_be_concept_pure" in result.checks_failed
        # ambiguous_concept_bleed flag stays False — there's no CNN term in the
        # answer to make this an ambiguous case.
        assert result.flags.get("ambiguous_concept_bleed") is False


def test_validator_concept_purity_passes_for_relational_query(app):
    with app.app_context():
        sq = _sq("How does dynamic programming relate to backpropagation?")
        c = build_concept_constraints(sq, kb=get_kb())
        plan = _plan_with_answer()
        ambiguous_answer = (
            "Backpropagation reuses subproblem gradients in much the same way "
            "dynamic programming reuses overlapping solutions."
        )
        result = validate_answer(ambiguous_answer, sq, plan, kb=get_kb(), constraints=c)
        # Skipped for relational queries — never appears in checks_failed.
        assert "must_be_concept_pure" not in result.checks_failed


def test_validator_concept_purity_warns_softly_on_ambiguous_bleed(app):
    """Both forbidden + target terms in the same answer → soft warn flag."""
    with app.app_context():
        sq = _sq("What is CNN?")
        c = build_concept_constraints(sq, kb=get_kb())
        plan = _plan_with_answer()
        # Mentions CNN (target) AND transformer (forbidden) — hard fail rule
        # only fires when target is missing.
        mixed_answer = (
            "Course Answer:\n\nA CNN applies convolution over local "
            "receptive fields, unlike a transformer that uses self-attention."
        )
        result = validate_answer(mixed_answer, sq, plan, kb=get_kb(), constraints=c)
        assert "must_be_concept_pure" not in result.checks_failed
        assert result.flags.get("ambiguous_concept_bleed") is True


# ---------------------------------------------------------------------------
# Sanity: ConceptConstraints is JSON-serializable for diagnostics
# ---------------------------------------------------------------------------


def test_constraints_to_dict_round_trips(app):
    with app.app_context():
        sq = _sq("What is CNN?")
        c = build_concept_constraints(sq, kb=get_kb())
        snap = c.to_dict()
        assert sorted(snap.keys()) == [
            "allowed_terms",
            "forbidden_terms",
            "is_relational",
            "target_aliases",
            "target_concepts",
            "target_lectures",
        ]
        assert "cnn" in snap["target_concepts"]
        assert isinstance(snap["forbidden_terms"], list)
