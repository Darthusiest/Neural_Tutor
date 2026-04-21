"""Unit tests for deterministic compare evidence extraction and rendering."""

from __future__ import annotations

from app.services.answers.answer_planning import AnswerPlan
from app.services.answers.compare_evidence import scoped_lines_from_chunks
from app.services.answers.compare_render import format_multi_entity_compare_markdown
from app.services.answers.entity_retrieval import ConceptEvidenceBundle
from app.services.knowledge.concept_kb import get_kb
from app.services.knowledge.structured_query import StructuredQuery, build_structured_query
from app.services.query_understanding import analyze_query


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
