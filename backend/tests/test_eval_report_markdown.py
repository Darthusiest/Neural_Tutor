"""Tests for eval CLI markdown report helpers."""

from __future__ import annotations

from pathlib import Path

from app.eval.report_markdown import (
    aggregate_canonical_tags,
    aggregate_forbidden_leakage,
    aggregate_scoring_categories,
    mode_score_stats,
    write_error_analysis_md,
    write_examples_md,
    _recommendations,
)


def test_aggregate_scoring_categories_top5():
    rows = [
        {"pass": "false", "errors": "forbidden_leak;must_include_failed"},
        {"pass": "false", "errors": "forbidden_leak"},
        {"pass": "true", "errors": ""},
    ]
    top = aggregate_scoring_categories(rows, only_failed=True)
    assert top[0][0] == "forbidden_leak"
    assert top[0][1] == 2
    assert len(top) <= 5


def test_aggregate_canonical_tags():
    details = {
        "a": {"pass": False, "canonical_tags": ["retrieval_leakage", "compare_entity_collapse"]},
        "b": {"pass": False, "canonical_tags": ["retrieval_leakage"]},
        "c": {"pass": True, "canonical_tags": ["should_ignore"]},
    }
    top = aggregate_canonical_tags(details, only_failed=True)
    assert top[0] == ("retrieval_leakage", 2)


def test_aggregate_forbidden_leakage_answer_vs_retrieval():
    details = {
        "x": {
            "pass": False,
            "actual_response": "hello softmax world",
            "retrieval_blob_lower": "mfcc formant softmax",
            "expected_behavior": {"must_not_include": ["softmax", "mfcc"]},
        }
    }
    ans, ret = aggregate_forbidden_leakage(details)
    assert any("softmax" in t for t, _ in ans)
    assert any("mfcc" in t for t, _ in ret)


def test_mode_score_stats_sorted_by_mean():
    rows = [
        {"expected_mode": "compare", "score": 0.5},
        {"expected_mode": "compare", "score": 0.5},
        {"expected_mode": "chat", "score": 1.0},
    ]
    stats = mode_score_stats(rows)
    assert stats[0][0] == "compare"
    assert stats[0][1] == 0.5
    assert stats[-1][0] == "chat"


def test_recommendations_includes_compare_entity_collapse():
    lines = _recommendations([("compare_entity_collapse", 2)], [])
    assert lines
    assert any("compare_entity_collapse" in x.lower() for x in lines)


def test_write_examples_md_sections(tmp_path: Path):
    row_results = [
        {
            "id": "pass_a",
            "query": "ok query",
            "expected_mode": "chat",
            "pass": "true",
            "score": 1.0,
            "detected": "chat",
            "effective": "chat",
            "errors": "",
        },
        {
            "id": "fail_b",
            "query": "bad query",
            "expected_mode": "chat",
            "pass": "false",
            "score": 0.25,
            "detected": "chat",
            "effective": "chat",
            "errors": "must_include_failed",
        },
    ]
    case_details = {
        "pass_a": {
            "query": "ok query",
            "actual_response": "has answer",
            "expected_behavior": {"expected_mode": "chat", "must_include": ["x"]},
            "detected": "chat",
            "effective": "chat",
            "canonical_tags": [],
            "scoring_errors": [],
            "pass": True,
            "score": 1.0,
        },
        "fail_b": {
            "query": "bad query",
            "actual_response": "short",
            "expected_behavior": {"expected_mode": "chat"},
            "detected": "chat",
            "effective": "chat",
            "canonical_tags": ["missing_required_concept"],
            "scoring_errors": ["must_include_failed"],
            "pass": False,
            "score": 0.25,
        },
    }
    p = tmp_path / "examples.md"
    write_examples_md(
        p,
        row_results=row_results,
        case_details=case_details,
        prev_run=None,
        prev_by_id=None,
        current_run_id=99,
    )
    text = p.read_text()
    assert "# Eval examples" in text
    assert "Run id: 99" in text
    assert "## Best passing examples" in text
    assert "## Worst failing examples" in text
    assert "## Representative examples by expected mode" in text
    assert "## Before / after" in text
    assert "No previous run found" in text
    assert "==================================================" in text


def test_write_error_analysis_md_with_app(app, tmp_path: Path):
    p = tmp_path / "err.md"
    row_results = [
        {
            "id": "f1",
            "expected_mode": "chat",
            "pass": "false",
            "score": 0.5,
            "errors": "mode_mismatch",
        }
    ]
    case_details = {
        "f1": {
            "pass": False,
            "canonical_tags": ["mode_routing_failure"],
            "actual_response": "",
            "retrieval_blob_lower": "",
            "expected_behavior": {},
        }
    }
    with app.app_context():
        write_error_analysis_md(
            p,
            row_results=row_results,
            case_details=case_details,
            dataset_name="toy@1",
        )
    text = p.read_text()
    assert "# Error analysis" in text
    assert "## Top scoring error categories" in text
    assert "mode_mismatch" in text
    assert "## Top canonical failure tags" in text
    assert "## Recommended next engineering fixes" in text
