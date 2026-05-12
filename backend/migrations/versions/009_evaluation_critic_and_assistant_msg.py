"""Assistant message id on eval cases + Gemini critic verdict rows.

Revision ID: 009
Revises: 008
"""

from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluation_critic_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("evaluation_run_id", sa.Integer(), nullable=False),
        sa.Column("case_result_id", sa.Integer(), nullable=False),
        sa.Column("critic_batch_id", sa.String(64), nullable=False),
        sa.Column("critic_score", sa.Float(), nullable=True),
        sa.Column("critic_pass", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("dimension_scores_json", sa.Text(), nullable=True),
        sa.Column("error_categories_json", sa.Text(), nullable=True),
        sa.Column("rationale_text", sa.Text(), nullable=True),
        sa.Column("category", sa.String(128), nullable=True),
        sa.Column("query_type_v2", sa.String(64), nullable=True),
        sa.Column("answer_mode", sa.String(64), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("tokens_estimated", sa.Integer(), nullable=True),
        sa.Column("critic_prompt_version", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(
            ["case_result_id"],
            ["evaluation_case_results.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["evaluation_run_id"],
            ["evaluation_runs.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "critic_batch_id",
            "case_result_id",
            name="uq_critic_batch_case",
        ),
    )
    op.create_index(
        "ix_evaluation_critic_results_evaluation_run_id",
        "evaluation_critic_results",
        ["evaluation_run_id"],
    )
    op.create_index(
        "ix_evaluation_critic_results_case_result_id",
        "evaluation_critic_results",
        ["case_result_id"],
    )
    op.create_index(
        "ix_evaluation_critic_results_critic_batch_id",
        "evaluation_critic_results",
        ["critic_batch_id"],
    )
    op.create_index(
        "ix_evaluation_critic_results_category",
        "evaluation_critic_results",
        ["category"],
    )
    op.create_index(
        "ix_evaluation_critic_results_query_type_v2",
        "evaluation_critic_results",
        ["query_type_v2"],
    )
    op.create_index(
        "ix_evaluation_critic_results_answer_mode",
        "evaluation_critic_results",
        ["answer_mode"],
    )

    with op.batch_alter_table("evaluation_case_results") as batch_op:
        batch_op.add_column(
            sa.Column(
                "assistant_message_id",
                sa.Integer(),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            "fk_evaluation_case_results_assistant_message_id",
            "messages",
            ["assistant_message_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_evaluation_case_results_assistant_message_id",
            ["assistant_message_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("evaluation_case_results") as batch_op:
        batch_op.drop_index("ix_evaluation_case_results_assistant_message_id")
        batch_op.drop_constraint(
            "fk_evaluation_case_results_assistant_message_id",
            type_="foreignkey",
        )
        batch_op.drop_column("assistant_message_id")

    op.drop_index("ix_evaluation_critic_results_answer_mode", "evaluation_critic_results")
    op.drop_index("ix_evaluation_critic_results_query_type_v2", "evaluation_critic_results")
    op.drop_index("ix_evaluation_critic_results_category", "evaluation_critic_results")
    op.drop_index("ix_evaluation_critic_results_critic_batch_id", "evaluation_critic_results")
    op.drop_index("ix_evaluation_critic_results_case_result_id", "evaluation_critic_results")
    op.drop_index("ix_evaluation_critic_results_evaluation_run_id", "evaluation_critic_results")
    op.drop_table("evaluation_critic_results")
