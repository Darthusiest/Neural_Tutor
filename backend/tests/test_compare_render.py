"""Unit tests for deterministic compare evidence extraction and rendering.

Covers:

- Existing :func:`scoped_lines_from_chunks` peer filter (CNN vs transformer
  leak).
- Existing multi-entity table renderer (axis columns + V1 bundle support).
- V2 :class:`ConceptEvidenceBundleV2` per-line cross-entity filtering for
  compare queries: ``Compare CNN and MLP``, ``CNN vs transformer``,
  ``Difference between MFCCs and formants``, ``Contrast softmax and hardmax``.
- Shared-line bucket: ``Bias versus variance`` produces a non-empty
  ``shared_lines`` collection that the renderer surfaces as
  ``### What they share``.
- :meth:`ConceptEvidenceBundleV2.from_legacy_bundle` /
  :meth:`ConceptEvidenceBundleV2.to_legacy_bundle` round-trip preserves the
  legacy four-field accessors.
"""

from __future__ import annotations

from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.compare_evidence import scoped_lines_from_chunks
from app.services.answers.compare_render import (
    format_multi_entity_compare_markdown,
    format_two_entity_compare_markdown,
)
from app.services.answers.entity_retrieval import (
    ConceptEvidenceBundle,
    ConceptEvidenceBundleV2,
    build_bundles_for_compare_v2,
    classify_line_for_compare,
)
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import StructuredQuery, build_structured_query
from app.services.query_understanding import analyze_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_plan(comparison_axes: list[str] | None = None) -> AnswerPlan:
    return AnswerPlan(
        answer_mode="compare",
        sections=[],
        primary_chunk_ids=[],
        supporting_chunk_ids=[],
        include_example=False,
        include_analogy=False,
        include_prerequisites=False,
        include_related_concepts=[],
        comparison_axes=list(comparison_axes or ["role", "computation", "typical use"]),
        lecture_scope=[],
    )


def _structured_query_for(query: str):
    intent = analyze_query(query)
    return build_structured_query(intent, kb=get_kb())


# ---------------------------------------------------------------------------
# Existing behaviour (kept green under V2)
# ---------------------------------------------------------------------------

def test_scoped_extraction_drops_peer_terms(app):
    """CNN-scoped pool must not surface a transformer-only line when peer is transformer."""
    with app.app_context():
        kb = get_kb()
        chunk = {
            "id": 99,
            "topic": "Architectures",
            "keywords": "",
            "clean_explanation": (
                "CNNs apply convolution over local neighborhoods.\n"
                "Transformers use self-attention across all positions."
            ),
            "source_excerpt": "",
        }
        lines, _prov = scoped_lines_from_chunks(
            [chunk],
            "cnn",
            ["transformer"],
            kb,
            None,
            max_lines=8,
        )
        joined = " ".join(lines).lower()
        assert "convolution" in joined or "local" in joined
        assert "self-attention" not in joined
        assert "transformers use" not in joined


def test_compare_multi_table_has_axis_columns(app):
    """Matrix header includes Architecture plus one column per comparison axis."""
    with app.app_context():
        kb = get_kb()
        bundles = [
            ConceptEvidenceBundle("cnn", "CNN", 1.0, [1]),
            ConceptEvidenceBundle("transformer", "Transformer", 1.0, [2]),
        ]
        chunks = [
            {
                "id": 1,
                "topic": "CNN",
                "keywords": "",
                "clean_explanation": "Convolutional layers scan local patches with shared weights.",
                "source_excerpt": "",
            },
            {
                "id": 2,
                "topic": "Transformer",
                "keywords": "",
                "clean_explanation": "Attention mixes information across all token positions.",
                "source_excerpt": "",
            },
        ]
        plan = AnswerPlan(
            answer_mode="compare_multi",
            sections=[],
            primary_chunk_ids=[1, 2],
            supporting_chunk_ids=[],
            include_example=False,
            include_analogy=False,
            include_prerequisites=False,
            include_related_concepts=[],
            comparison_axes=["role", "computation"],
            lecture_scope=[],
        )
        intent = analyze_query("Compare CNN vs transformer")
        sq = build_structured_query(intent, kb=kb)
        assert isinstance(sq, StructuredQuery)
        text = format_multi_entity_compare_markdown(bundles, chunks, sq, plan=plan, kb=kb)
        assert "| Architecture |" in text
        assert "| role |" in text
        assert "| computation |" in text
        assert "| **CNN** |" in text
        assert "| **Transformer** |" in text


# ---------------------------------------------------------------------------
# V2 bundle builder — entity-separated evidence
# ---------------------------------------------------------------------------

def test_compare_cnn_and_mlp_disjoint_evidence(app):
    """*Compare CNN and MLP* — CNN core_lines never leak into MLP and vice versa.

    MLP isn't a KB concept on its own (closest is ``feedforward``); the test
    uses ``feedforward`` as the second concept id, but the spec query is the
    plain English form. Aliases are overridden so cross-entity matching has
    a tight ``mlp`` / ``fully connected`` term set to work with.
    """
    with app.app_context():
        kb = get_kb()
        chunks = [
            {
                "id": 1,
                "topic": "CNN — Architecture",
                "keywords": "convolution local",
                "clean_explanation": (
                    "CNNs apply convolution over local receptive fields.\n"
                    "Convolutional layers share weights spatially."
                ),
                "source_excerpt": "",
                "lecture_number": 17,
            },
            {
                "id": 2,
                "topic": "MLP — Architecture",
                "keywords": "fully connected mlp feedforward",
                "clean_explanation": (
                    "MLPs are fully connected feedforward networks.\n"
                    "Each MLP layer multiplies inputs by a dense weight matrix."
                ),
                "source_excerpt": "",
                "lecture_number": 1,
            },
        ]
        bundle_cnn, bundle_mlp = build_bundles_for_compare_v2(
            chunks,
            "cnn",
            "feedforward",
            kb,
            top_per_side=2,
            aliases_override={
                "feedforward": ["mlp", "fully connected", "feedforward network"],
            },
            label_override={"feedforward": "MLP"},
        )

        cnn_blob = " ".join(bundle_cnn.core_lines).lower()
        mlp_blob = " ".join(bundle_mlp.core_lines).lower()
        assert "convolution" in cnn_blob or "local" in cnn_blob
        assert "convolution" not in mlp_blob
        assert "fully connected" in mlp_blob or "mlp" in mlp_blob
        assert "fully connected" not in cnn_blob
        # core_lines on each side must be disjoint — same invariant as the
        # softmax/hardmax test. Top-N chunks may overlap when only two are
        # available; the per-line classifier is what enforces purity.
        cnn_lines = {line.strip().lower() for line in bundle_cnn.core_lines}
        mlp_lines = {line.strip().lower() for line in bundle_mlp.core_lines}
        assert cnn_lines.isdisjoint(mlp_lines)


def test_compare_cnn_vs_transformer_no_shared_attention_leak(app):
    """*CNN vs transformer* — CNN core_lines must not mention self-attention; transformer
    core_lines must not mention convolution.
    """
    with app.app_context():
        kb = get_kb()
        chunks = [
            {
                "id": 10,
                "topic": "CNN",
                "keywords": "convolution",
                "clean_explanation": (
                    "CNNs use convolutional kernels over local windows.\n"
                    "Receptive fields grow with depth."
                ),
                "source_excerpt": "",
                "lecture_number": 17,
            },
            {
                "id": 11,
                "topic": "Transformer",
                "keywords": "attention",
                "clean_explanation": (
                    "Transformers rely on self-attention over the full sequence.\n"
                    "Multi-head attention mixes token representations."
                ),
                "source_excerpt": "",
                "lecture_number": 18,
            },
        ]
        bundle_cnn, bundle_tx = build_bundles_for_compare_v2(
            chunks, "cnn", "transformer", kb, top_per_side=2
        )

        cnn_blob = " ".join(bundle_cnn.core_lines).lower()
        tx_blob = " ".join(bundle_tx.core_lines).lower()
        assert "self-attention" not in cnn_blob
        assert "multi-head" not in cnn_blob
        assert "convolution" not in tx_blob
        assert "receptive field" not in tx_blob


def test_compare_mfccs_and_formants(app):
    """*Difference between MFCCs and formants* — each side keeps its distinctive vocabulary."""
    with app.app_context():
        kb = get_kb()
        chunks = [
            {
                "id": 21,
                "topic": "MFCCs — Pipeline",
                "keywords": "cepstrum filterbank dct",
                "clean_explanation": (
                    "MFCCs apply a filterbank, log compression, and DCT to produce cepstral coefficients.\n"
                    "Cepstrum captures the spectral envelope shape."
                ),
                "source_excerpt": "",
                "lecture_number": 10,
            },
            {
                "id": 22,
                "topic": "Formants — Core Idea",
                "keywords": "vocal tract resonance",
                "clean_explanation": (
                    "Formants are spectral peaks that come from vocal tract resonances.\n"
                    "Different vowels produce different formant patterns."
                ),
                "source_excerpt": "",
                "lecture_number": 10,
            },
        ]
        bundle_mfcc, bundle_form = build_bundles_for_compare_v2(
            chunks, "mfcc", "formants", kb, top_per_side=2
        )

        mfcc_blob = " ".join(bundle_mfcc.core_lines).lower()
        form_blob = " ".join(bundle_form.core_lines).lower()
        assert "cepstr" in mfcc_blob or "filterbank" in mfcc_blob or "dct" in mfcc_blob
        assert "vocal tract" not in mfcc_blob
        assert "vocal tract" in form_blob or "formant" in form_blob
        assert "cepstr" not in form_blob
        assert "filterbank" not in form_blob


def test_compare_softmax_and_hardmax_separate_bundles(app):
    """*Contrast softmax and hardmax* — each side has disjoint distinctive vocabulary."""
    with app.app_context():
        kb = get_kb()
        chunks = [
            {
                "id": 31,
                "topic": "Softmax — Core",
                "keywords": "probability distribution",
                "clean_explanation": (
                    "Softmax converts logits into a probability distribution that sums to one.\n"
                    "Softmax is differentiable, which makes gradient-based training possible."
                ),
                "source_excerpt": "",
                "lecture_number": 14,
            },
            {
                "id": 32,
                "topic": "Hardmax — Core",
                "keywords": "argmax one-hot",
                "clean_explanation": (
                    "Hardmax picks the argmax index and emits a one-hot vector.\n"
                    "Hardmax discards the relative magnitudes of the other logits."
                ),
                "source_excerpt": "",
                "lecture_number": 14,
            },
        ]
        bundle_sm, bundle_hm = build_bundles_for_compare_v2(
            chunks, "softmax", "hardmax", kb, top_per_side=2
        )

        soft_blob = " ".join(bundle_sm.core_lines).lower()
        hard_blob = " ".join(bundle_hm.core_lines).lower()
        # Softmax-specific phrasing must not appear in hardmax core_lines.
        assert "probability distribution" in soft_blob
        assert "probability distribution" not in hard_blob
        # Hardmax-specific phrasing must not appear in softmax core_lines.
        assert "one-hot" in hard_blob or "argmax" in hard_blob
        assert "one-hot" not in soft_blob
        # core_lines on each side must be disjoint — that's the entity-purity
        # invariant the V2 builder enforces. (chunk_ids may legitimately
        # overlap when only two chunks total are available; line-level
        # filtering is what guarantees separation.)
        soft_lines = {line.strip().lower() for line in bundle_sm.core_lines}
        hard_lines = {line.strip().lower() for line in bundle_hm.core_lines}
        assert soft_lines.isdisjoint(hard_lines)


# ---------------------------------------------------------------------------
# Shared bucket — "What they share" section
# ---------------------------------------------------------------------------

def test_compare_bias_versus_variance_shared_section(app):
    """*Bias versus variance* — a sentence mentioning both terms ends up in
    ``shared_lines`` and renders under ``### What they share``.

    Bias and variance aren't separate KB concepts (the LING487 KB packages
    them as a single ``bias_variance`` concept); the test exercises the
    classifier directly with explicit aliases for each side, then verifies
    the renderer surfaces the shared bucket.
    """
    with app.app_context():
        kb = get_kb()
        # Confirm the classifier sees the cross-entity sentence as shared.
        label, _scores = classify_line_for_compare(
            "The bias-variance tradeoff balances bias and variance.",
            entity_a_terms=["bias"],
            entity_b_terms=["variance"],
        )
        assert label == "shared", (label, _scores)

        # Construct V2 bundles directly so we don't depend on the analyzer
        # picking out two separate KB concepts (see docstring above).
        shared_line = "The bias-variance tradeoff balances bias and variance."
        bundle_bias = ConceptEvidenceBundleV2(
            concept="bias",
            aliases=["high bias"],
            evidence_chunks=[
                {"id": 1, "topic": "Bias", "lecture_number": 13},
            ],
            core_lines=[
                "High bias means the model is too simple and underfits the data.",
            ],
            support_score=0.6,
            shared_lines=[shared_line],
            confidence=0.6,
            label_override="Bias",
        )
        bundle_variance = ConceptEvidenceBundleV2(
            concept="variance",
            aliases=["high variance"],
            evidence_chunks=[
                {"id": 2, "topic": "Variance", "lecture_number": 13},
            ],
            core_lines=[
                "High variance means the model overfits and is sensitive to small data changes.",
            ],
            support_score=0.6,
            shared_lines=[shared_line],
            confidence=0.6,
            label_override="Variance",
        )

        chunks = [
            {
                "id": 1,
                "topic": "Bias",
                "keywords": "bias underfitting",
                "clean_explanation": "High bias means the model is too simple and underfits the data.",
                "source_excerpt": "",
                "lecture_number": 13,
            },
            {
                "id": 2,
                "topic": "Variance",
                "keywords": "variance overfitting",
                "clean_explanation": (
                    "High variance means the model overfits and is sensitive to small data changes."
                ),
                "source_excerpt": "",
                "lecture_number": 13,
            },
        ]
        plan = _empty_plan(comparison_axes=["role", "typical failure mode"])
        sq = _structured_query_for("Bias versus variance")
        text = format_two_entity_compare_markdown(
            plan, chunks, sq, bundle_bias, bundle_variance, kb=kb
        )
        assert "### What they share" in text
        assert "bias-variance tradeoff" in text.lower()


def test_compare_two_entity_no_shared_section_when_disjoint(app):
    """Two-way compare with no shared evidence must not render the ``### What they share`` section."""
    with app.app_context():
        kb = get_kb()
        chunks = [
            {
                "id": 1,
                "topic": "CNN",
                "keywords": "convolution",
                "clean_explanation": "CNNs use convolution over local windows.",
                "source_excerpt": "",
                "lecture_number": 17,
            },
            {
                "id": 2,
                "topic": "Transformer",
                "keywords": "attention",
                "clean_explanation": "Transformers use self-attention across the sequence.",
                "source_excerpt": "",
                "lecture_number": 18,
            },
        ]
        bundle_cnn, bundle_tx = build_bundles_for_compare_v2(
            chunks, "cnn", "transformer", kb, top_per_side=2
        )
        plan = _empty_plan()
        sq = _structured_query_for("Compare CNN and transformer")
        text = format_two_entity_compare_markdown(
            plan, chunks, sq, bundle_cnn, bundle_tx, kb=kb
        )
        assert "### What they share" not in text


# ---------------------------------------------------------------------------
# Adapter round-trip
# ---------------------------------------------------------------------------

def test_concept_evidence_bundle_v2_legacy_adapter_roundtrip(app):
    """``from_legacy_bundle`` then ``to_legacy_bundle`` preserves the legacy 4-field surface."""
    with app.app_context():
        kb = get_kb()
        legacy = ConceptEvidenceBundle(
            concept_id="mfcc",
            label="MFCC",
            support_score=0.42,
            chunk_ids=[101, 202],
            gap_flags=["low_support"],
        )
        evidence_chunks = [
            {"id": 101, "topic": "MFCC — Pipeline", "lecture_number": 10},
            {"id": 202, "topic": "MFCC — Filterbank", "lecture_number": 10},
        ]
        v2 = ConceptEvidenceBundleV2.from_legacy_bundle(
            legacy,
            kb=kb,
            evidence_chunks=evidence_chunks,
            core_lines=["MFCCs summarize the spectrum."],
            shared_lines=[],
            forbidden_hits=[],
        )
        # Legacy attribute surface still works on V2.
        assert v2.concept_id == "mfcc"
        assert v2.label == "MFCC"
        assert v2.chunk_ids == [101, 202]
        assert v2.gap_flags == ["low_support"]
        assert v2.aliases  # KB lookup populates aliases for the concept

        roundtripped = v2.to_legacy_bundle()
        assert isinstance(roundtripped, ConceptEvidenceBundle)
        assert roundtripped.concept_id == legacy.concept_id
        assert roundtripped.label == legacy.label
        assert roundtripped.support_score == legacy.support_score
        assert roundtripped.chunk_ids == legacy.chunk_ids
        assert roundtripped.gap_flags == legacy.gap_flags
