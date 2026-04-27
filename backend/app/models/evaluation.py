"""
Batch evaluation run storage for rule-based pipeline quality tracking.

``evaluation_runs`` aggregates a suite run (git commit, pass/fail counts);
``evaluation_case_results`` store one row per test case for drill-down and regression.
"""

from __future__ import annotations

from app.extensions import db


class EvaluationRun(db.Model):
    """One row per execution of a static eval dataset (e.g. after a commit or CI job)."""

    __tablename__ = "evaluation_runs"

    id = db.Column(db.Integer, primary_key=True)
    run_name = db.Column(db.String(256), nullable=False)
    git_commit = db.Column(db.String(64), nullable=True, index=True)
    branch_name = db.Column(db.String(256), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now(), index=True)
    dataset_name = db.Column(db.String(256), nullable=False, index=True)
    total_cases = db.Column(db.Integer, nullable=False, default=0)
    passed_cases = db.Column(db.Integer, nullable=False, default=0)
    failed_cases = db.Column(db.Integer, nullable=False, default=0)
    overall_score = db.Column(db.Float, nullable=True)
    notes_json = db.Column(db.Text, nullable=True)

    case_results = db.relationship(
        "EvaluationCaseResult",
        backref="run",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )


class EvaluationCaseResult(db.Model):
    """Result for a single test id within an :class:`EvaluationRun`."""

    __tablename__ = "evaluation_case_results"

    id = db.Column(db.Integer, primary_key=True)
    evaluation_run_id = db.Column(
        db.Integer,
        db.ForeignKey("evaluation_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    test_id = db.Column(db.String(256), nullable=False)
    # Not named `query` — that shadows ``Model.query`` on the SQLAlchemy class.
    query_text = db.Column(db.Text, nullable=False)
    expected_mode = db.Column(db.String(32), nullable=True)
    detected_mode = db.Column(db.String(32), nullable=True)
    effective_mode = db.Column(db.String(32), nullable=True)
    expected_behavior_json = db.Column(db.Text, nullable=True)
    actual_response = db.Column(db.Text, nullable=True)
    pass_bool = db.Column(db.Boolean, nullable=False, default=False)
    score = db.Column(db.Float, nullable=True)
    error_categories_json = db.Column(db.Text, nullable=True)
    validation_failures_json = db.Column(db.Text, nullable=True)
    retrieval_chunk_ids_json = db.Column(db.Text, nullable=True)
    latency_ms = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

    __table_args__ = (
        db.UniqueConstraint("evaluation_run_id", "test_id", name="uq_eval_run_test_id"),
    )
