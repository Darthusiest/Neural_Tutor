from __future__ import annotations

from app.services.knowledge.lecture_kb_audit import audit_lecture_kb_payload


def test_audit_strips_export_stubs_from_text_fields():
    payload = {
        "lecture_id": 20,
        "concepts": [
            {
                "concept": "Probing",
                "professor_definition": (
                    "A probe maps hidden states to labels "
                    ":contentReference[oaicite:0]{index=0}"
                ),
                "key_points": [
                    "Probe accuracy is decodability, not causality.",
                    "Keep language careful :contentReference[oaicite:1]{index=1}",
                ],
                "forbidden_drift": ["Do not equate probe score with causal use."],
            }
        ],
    }

    result = audit_lecture_kb_payload(payload)
    cleaned = result.cleaned_payload
    concept = cleaned["concepts"][0]

    assert ":contentReference[oaicite:" not in concept["professor_definition"]
    assert ":contentReference[oaicite:" not in concept["key_points"][1]
    assert any(issue.code == "export_stub_removed" for issue in result.issues)


def test_audit_flags_common_overclaim_and_technical_slip_patterns():
    payload = {
        "lecture_id": "L22",
        "concepts": [
            {
                "concept_name": "Diagnostics",
                "professor_definition": (
                    "Probe accuracy proves the concept is causally located in one head."
                ),
                "key_points": [
                    "Multimodal systems must joint-train and must have aligned data.",
                    "A p-value is the probability the null hypothesis is true.",
                ],
                "forbidden_drift": [],
            }
        ],
    }

    result = audit_lecture_kb_payload(payload)
    codes = {issue.code for issue in result.issues}

    assert "probing_causality_overclaim" in codes
    assert "multimodal_joint_training_overclaim" in codes
    assert "pvalue_probability_hypothesis_true" in codes


def test_audit_flags_forbidden_drift_blanket_bans():
    payload = {
        "lecture_id": 21,
        "concepts": [
            {
                "concept_id": "statistics",
                "professor_definition": "Association language should stay scoped.",
                "forbidden_drift": [
                    "Never trust correlation.",
                    "Never ablate multiple things.",
                    "Counterfactual steering is always more causal.",
                ],
            }
        ],
    }

    result = audit_lecture_kb_payload(payload)
    codes = {issue.code for issue in result.issues}

    assert "forbidden_drift_blanket_correlation_ban" in codes
    assert "forbidden_drift_blanket_ablation_ban" in codes
    assert "forbidden_drift_always_claim" in codes


def test_audit_accepts_collection_payload_shape():
    payload = {
        "packs": [
            {"lecture_id": 1, "concepts": [{"concept": "x", "professor_definition": "ok"}]},
            {"lecture_id": 2, "concepts": [{"concept": "y", "professor_definition": "ok"}]},
        ]
    }

    result = audit_lecture_kb_payload(payload)

    assert result.error_count == 0
    assert isinstance(result.cleaned_payload, dict)
    assert len(result.cleaned_payload["packs"]) == 2
