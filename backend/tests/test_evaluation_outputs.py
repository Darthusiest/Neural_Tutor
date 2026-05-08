from __future__ import annotations

import csv
import json
from pathlib import Path

from app.eval.evaluation_outputs import (
    _low_n_warning,
    _flatten_answer_for_csv,
    derive_case_scores,
    generate_evaluation_outputs,
    write_coverage_by_concept_chart,
    write_evaluation_summary_chart,
    write_error_analysis_csv,
    write_example_answers_csv,
    write_failure_modes_chart,
    write_pipeline_diagram,
    write_question_type_chart,
    write_regression_comparison_chart,
    write_retrieval_accuracy_chart,
)
from app.extensions import db
from app.models import EvaluationCaseResult, EvaluationRun
from app.models.content import LectureChunk


_RICH_RESPONSE = (
    "Course Answer:\n\n"
    "The key idea: softmax maps scores to probabilities. "
    "For example, given logits [2.0, 1.0], softmax yields [0.73, 0.27]. "
    "This mechanism normalizes via exponentiation."
)


def _run(
    *,
    run_name: str = "eo",
    dataset_name: str = "ds",
    overall_score: float = 0.0,
) -> EvaluationRun:
    run = EvaluationRun(
        run_name=run_name,
        dataset_name=dataset_name,
        total_cases=0,
        passed_cases=0,
        failed_cases=0,
        overall_score=overall_score,
        notes_json="{}",
    )
    db.session.add(run)
    db.session.flush()
    return run


def _case(
    run: EvaluationRun,
    *,
    test_id: str,
    intent: str = "definition",
    pass_bool: bool = True,
    actual_response: str = _RICH_RESPONSE,
    must_include: list[str] | None = None,
    errors: list[str] | None = None,
    primary_error_type: str | None = None,
    chunk_ids: list[int] | None = None,
    score: float | None = None,
    query_text: str | None = None,
) -> EvaluationCaseResult:
    row = EvaluationCaseResult(
        evaluation_run_id=run.id,
        test_id=test_id,
        query_text=query_text or f"query {test_id}",
        expected_mode="chat",
        detected_mode="chat",
        effective_mode="chat",
        expected_behavior_json=json.dumps(
            {
                "category": "definitions",
                "intent": intent,
                "must_include": must_include or ["softmax"],
            }
        ),
        actual_response=actual_response,
        pass_bool=pass_bool,
        score=score if score is not None else (1.0 if pass_bool else 0.5),
        error_categories_json=json.dumps(errors or []),
        primary_error_type=primary_error_type,
        validation_failures_json=json.dumps({"checks_failed": [], "flags": {}}),
        retrieval_chunk_ids_json=json.dumps(chunk_ids or []),
        boost_metrics_json=None,
        latency_ms=10,
    )
    db.session.add(row)
    db.session.flush()
    return row


def _png_header_ok(path: Path) -> bool:
    with path.open("rb") as fh:
        return fh.read(4) == b"\x89PNG"


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


def test_derive_scores_perfect_case(app) -> None:
    with app.app_context():
        run = _run()
        row = _case(run, test_id="ok", pass_bool=True)
        scores = derive_case_scores(row)
        assert scores["grounding"] == 1
        assert scores["explanation_quality"] == 2
        assert scores["depth"] == 2
        assert scores["question_handling"] == 1
        assert scores["retrieval_quality"] in (0, 1)


def test_derive_scores_empty_response(app) -> None:
    with app.app_context():
        run = _run()
        row = _case(
            run,
            test_id="empty",
            pass_bool=False,
            actual_response="",
            primary_error_type="shallow_explanation",
        )
        scores = derive_case_scores(row)
        assert scores["grounding"] == 1
        assert scores["explanation_quality"] == 0
        assert scores["depth"] == 0
        assert scores["question_handling"] == 1


def test_derive_scores_hallucination(app) -> None:
    with app.app_context():
        run = _run()
        row = _case(
            run,
            test_id="hallu",
            pass_bool=False,
            primary_error_type="hallucination",
        )
        scores = derive_case_scores(row)
        assert scores["grounding"] == 0


def test_derive_scores_forbidden_leak(app) -> None:
    with app.app_context():
        run = _run()
        row = _case(
            run,
            test_id="leak",
            pass_bool=False,
            errors=["forbidden_leak"],
            primary_error_type="hallucination",
        )
        scores = derive_case_scores(row)
        assert scores["grounding"] == 0


# ---------------------------------------------------------------------------
# Chart tests
# ---------------------------------------------------------------------------


def test_retrieval_accuracy_chart(app, tmp_path) -> None:
    with app.app_context():
        chunk = LectureChunk(
            chunk_key="softmax-test",
            lecture_number=1,
            topic="Softmax",
            keywords=json.dumps(["softmax", "probability"]),
            source_excerpt="Softmax converts scores to probabilities.",
            clean_explanation="Softmax converts scores to probabilities.",
        )
        db.session.add(chunk)
        db.session.flush()
        run = _run()
        _case(run, test_id="a", intent="definition", chunk_ids=[chunk.id])
        _case(run, test_id="b", intent="step_by_step", chunk_ids=[chunk.id])
        cases = (
            EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id)
            .order_by(EvaluationCaseResult.test_id)
            .all()
        )

        write_retrieval_accuracy_chart(cases, tmp_path)
        out = tmp_path / "retrieval_accuracy.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)


def test_question_type_chart(app, tmp_path) -> None:
    with app.app_context():
        run = _run()
        _case(run, test_id="a", intent="definition", pass_bool=True)
        _case(run, test_id="b", intent="compare", pass_bool=False)
        _case(run, test_id="c", intent="step_by_step", pass_bool=True)
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_question_type_chart(cases, tmp_path)
        out = tmp_path / "question_type_breakdown.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)


def test_question_type_chart_low_n_warning(app, tmp_path) -> None:
    with app.app_context():
        run = _run()
        _case(run, test_id="a", intent="definition", pass_bool=True)
        _case(run, test_id="b", intent="compare", pass_bool=False)
        _case(run, test_id="c", intent="step_by_step", pass_bool=True)
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_question_type_chart(cases, tmp_path)
        out = tmp_path / "question_type_breakdown.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)
        assert _low_n_warning(1) == "n too small"


def test_retrieval_accuracy_split_views(app, tmp_path) -> None:
    with app.app_context():
        chunk = LectureChunk(
            chunk_key="retrieval-split-test",
            lecture_number=2,
            topic="CNN",
            keywords=json.dumps(["cnn", "convolution"]),
            source_excerpt="CNNs use local receptive fields and shared filters.",
            clean_explanation="CNNs use local receptive fields and shared filters.",
        )
        db.session.add(chunk)
        db.session.flush()
        run = _run()
        _case(
            run,
            test_id="n1",
            intent="definition",
            query_text="Define CNN from lecture material",
            chunk_ids=[chunk.id],
        )
        _case(
            run,
            test_id="m1",
            intent="compare",
            query_text="Compare these",
            chunk_ids=[chunk.id],
        )
        _case(
            run,
            test_id="m2",
            intent="step_by_step",
            query_text="Summarize this",
            chunk_ids=[chunk.id],
        )
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_retrieval_accuracy_chart(cases, tmp_path)
        out = tmp_path / "retrieval_accuracy.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)


def test_pipeline_diagram(tmp_path) -> None:
    write_pipeline_diagram(tmp_path)
    out = tmp_path / "pipeline_diagram.png"
    assert out.exists()
    assert out.stat().st_size > 0
    assert _png_header_ok(out)


# ---------------------------------------------------------------------------
# CSV tests
# ---------------------------------------------------------------------------


def test_example_answers_csv(app, tmp_path) -> None:
    with app.app_context():
        run = _run()
        for i in range(12):
            _case(
                run,
                test_id=f"c{i:02d}",
                pass_bool=(i % 2 == 0),
                score=1.0 - i * 0.05,
            )
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_example_answers_csv(cases, tmp_path)
        out = tmp_path / "example_answers.csv"
        assert out.exists()
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "test_id",
                "score_group",
                "rank_in_group",
                "passed_scoring",
                "query_type_label",
                "grounding",
                "notes_or_error_type",
                "user_query",
                "course_answer_one_line",
            ]
            rows = list(reader)
        assert len(rows) == 10
        assert rows[0]["test_id"] == "c00"
        assert rows[0]["score_group"] == "top_5_by_score"
        assert rows[-1]["test_id"] == "c11"
        assert rows[-1]["score_group"] == "bottom_5_by_score"
        # One physical row per example: no embedded newlines in the flat answer
        assert "\n" not in rows[0]["course_answer_one_line"]


def test_flatten_answer_for_csv_collapses_newlines() -> None:
    text = "Line one.\n\nSecond paragraph.\nStill second."
    flat = _flatten_answer_for_csv(text)
    assert "\n" not in flat
    assert " || " in flat
    assert "Second paragraph." in flat


def test_error_analysis_csv(app, tmp_path) -> None:
    with app.app_context():
        run = _run()
        _case(
            run,
            test_id="f1",
            pass_bool=False,
            primary_error_type="hallucination",
            errors=["forbidden_leak"],
            actual_response="leaked content",
        )
        _case(
            run,
            test_id="f2",
            pass_bool=False,
            primary_error_type="structure_failure",
            errors=["structure_summary_no_header"],
            actual_response="no header here",
        )
        _case(
            run,
            test_id="f3",
            pass_bool=False,
            primary_error_type="hallucination",
            errors=["forbidden_leak"],
            actual_response="leaked twice",
        )
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_error_analysis_csv(cases, tmp_path)
        out = tmp_path / "error_analysis.csv"
        assert out.exists()
        with out.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "error_type",
                "count",
                "percentage",
                "example_query",
            ]
            rows = list(reader)
        assert rows
        for row in rows:
            assert row["example_query"]


# ---------------------------------------------------------------------------
# Orchestrator + module-removal tests
# ---------------------------------------------------------------------------


def test_generate_evaluation_outputs_smoke(app, tmp_path) -> None:
    with app.app_context():
        run = _run()
        _case(run, test_id="ok1", pass_bool=True)
        _case(
            run,
            test_id="bad1",
            pass_bool=False,
            primary_error_type="hallucination",
            errors=["forbidden_leak"],
        )
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        generate_evaluation_outputs(cases, tmp_path)

        for name in (
            "retrieval_accuracy.png",
            "question_type_breakdown.png",
            "pipeline_diagram.png",
            "report_dashboard.png",
            "coverage_by_concept.png",
            "failure_modes.png",
            "example_answers.csv",
            "error_analysis.csv",
        ):
            assert (tmp_path / name).exists(), f"missing {name}"


def test_legacy_plot_module_removed() -> None:
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("app.eval.plot_eval_metrics")


def test_evaluation_summary_small_dataset_banner(app, tmp_path) -> None:
    with app.app_context():
        run = _run(run_name="summary-small-n", overall_score=0.667)
        _case(run, test_id="s1", intent="definition", pass_bool=True)
        _case(run, test_id="s2", intent="compare", pass_bool=False)
        _case(run, test_id="s3", intent="synthesis", pass_bool=True)
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_evaluation_summary_chart(run, cases, tmp_path)
        out = tmp_path / "evaluation_summary.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)


def test_regression_comparison_no_movement_notice(app, tmp_path) -> None:
    with app.app_context():
        prev = _run(run_name="prev", dataset_name="same_ds", overall_score=0.5)
        _case(
            prev,
            test_id="r1",
            intent="definition",
            pass_bool=True,
            chunk_ids=[],
        )
        curr = _run(run_name="curr", dataset_name="same_ds", overall_score=0.5)
        _case(
            curr,
            test_id="r1",
            intent="definition",
            pass_bool=True,
            chunk_ids=[],
        )
        prev_cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=prev.id).all()
        curr_cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=curr.id).all()

        write_regression_comparison_chart(prev, prev_cases, curr, curr_cases, tmp_path)
        out = tmp_path / "regression_comparison.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)


def test_coverage_by_concept_chart(app, tmp_path) -> None:
    with app.app_context():
        run = _run()
        _case(run, test_id="c1", intent="definition", pass_bool=True, must_include=["softmax"])
        _case(run, test_id="c2", intent="definition", pass_bool=False, must_include=["softmax"])
        _case(run, test_id="c3", intent="compare", pass_bool=False, must_include=["cnn"])
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_coverage_by_concept_chart(cases, tmp_path)
        out = tmp_path / "coverage_by_concept.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)


def test_failure_modes_chart(app, tmp_path) -> None:
    with app.app_context():
        run = _run()
        _case(
            run,
            test_id="f10",
            pass_bool=False,
            intent="compare",
            primary_error_type="validation_missed_error",
            errors=["missing_required_concept"],
        )
        _case(
            run,
            test_id="f11",
            pass_bool=False,
            intent="compare",
            primary_error_type="compare_asymmetry",
            errors=["compare_entity_collapse"],
        )
        cases = EvaluationCaseResult.query.filter_by(evaluation_run_id=run.id).all()

        write_failure_modes_chart(cases, tmp_path)
        out = tmp_path / "failure_modes.png"
        assert out.exists()
        assert out.stat().st_size > 0
        assert _png_header_ok(out)
