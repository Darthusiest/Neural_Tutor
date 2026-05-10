"""Entity-scored retrieval and compare bundle helpers."""

from app.services.answers.entity_retrieval import (
    ConceptEvidenceBundleV2,
    build_bundles_for_compare,
    chunk_text_blob,
    display_heading_for_compare,
    forbidden_terms_for_concept,
    score_chunk_for_entity,
)
from app.services.knowledge.concept_kb import get_kb
from app.services.query_understanding import extract_compare_entities


def test_extract_compare_entities_multi_vs():
    q = "Compare CNN vs MLP vs transformer vs residual networks"
    ent = extract_compare_entities(q)
    assert ent is not None
    assert len(ent) >= 3


def test_score_chunk_prefers_entity_over_peer():
    kb = get_kb()
    chunk = {
        "id": 1,
        "topic": "CNN",
        "keywords": "convolution",
        "clean_explanation": "Convolutional networks use local receptive fields.",
        "source_excerpt": "",
    }
    s, parts = score_chunk_for_entity(chunk, "cnn", kb, peer_concept_ids=["transformer"])
    assert s > 0
    assert parts["entity"] >= 1.0


def test_forbidden_terms_include_peer_names():
    kb = get_kb()
    terms = forbidden_terms_for_concept("mfcc", ["softmax"], kb)
    assert any("softmax" in t for t in terms)


def test_build_bundles_splits_sides():
    kb = get_kb()
    chunks = [
        {
            "id": 10,
            "topic": "MFCC",
            "keywords": "cepstrum",
            "clean_explanation": "MFCC features summarize spectral shape.",
            "source_excerpt": "",
        },
        {
            "id": 11,
            "topic": "Formants",
            "keywords": "resonance",
            "clean_explanation": "Formants are spectral peaks from vocal tract resonance.",
            "source_excerpt": "",
        },
    ]
    if kb.get_concept_by_id("mfcc") and kb.get_concept_by_id("formants"):
        a, b = build_bundles_for_compare(chunks, "mfcc", "formants", kb, top_per_side=2)
        assert a.chunk_ids or b.chunk_ids
        blob_a = chunk_text_blob(chunks[0])
        assert "mfcc" in blob_a or "mfcc" in blob_a.lower()


def test_display_heading_adds_alias_when_label_omits_concept_id(app):
    with app.app_context():
        kb = get_kb()
        bundle = ConceptEvidenceBundleV2(
            concept="cnn",
            aliases=["cnn"],
            evidence_chunks=[],
            core_lines=[],
            support_score=0.5,
            label_override="convolutional neural network",
        )
        heading = display_heading_for_compare(bundle, kb)
        assert "convolutional neural network" in heading.lower()
        assert "cnn" in heading.lower()

