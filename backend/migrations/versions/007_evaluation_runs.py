"""Evaluation runs and per-case results for batch pipeline quality tracking.

Revision ID: 007
Revises: 006
"""

from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_name", sa.String(256), nullable=False),
        sa.Column("git_commit", sa.String(64), nullable=True),
        sa.Column("branch_name", sa.String(256), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.Column("dataset_name", sa.String(256), nullable=False),
        sa.Column("total_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("passed_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_cases", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("notes_json", sa.Text(), nullable=True),
    )
    op.create_index("ix_evaluation_runs_created_at", "evaluation_runs", ["created_at"])
    op.create_index("ix_evaluation_runs_dataset_name", "evaluation_runs", ["dataset_name"])
    op.create_index("ix_evaluation_runs_git_commit", "evaluation_runs", ["git_commit"])

    op.create_table(
        "evaluation_case_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("evaluation_run_id", sa.Integer(), nullable=False),
        sa.Column("test_id", sa.String(256), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("expected_mode", sa.String(32), nullable=True),
        sa.Column("detected_mode", sa.String(32), nullable=True),
        sa.Column("effective_mode", sa.String(32), nullable=True),
        sa.Column("expected_behavior_json", sa.Text(), nullable=True),
        sa.Column("actual_response", sa.Text(), nullable=True),
        sa.Column("pass_bool", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("error_categories_json", sa.Text(), nullable=True),
        sa.Column("validation_failures_json", sa.Text(), nullable=True),
        sa.Column("retrieval_chunk_ids_json", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(
            ["evaluation_run_id"],
            ["evaluation_runs.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("evaluation_run_id", "test_id", name="uq_eval_run_test_id"),
    )
    op.create_index(
        "ix_evaluation_case_results_evaluation_run_id",
        "evaluation_case_results",
        ["evaluation_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_evaluation_case_results_evaluation_run_id", table_name="evaluation_case_results"
    )
    op.drop_table("evaluation_case_results")
    op.drop_index("ix_evaluation_runs_git_commit", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_dataset_name", table_name="evaluation_runs")
    op.drop_index("ix_evaluation_runs_created_at", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
