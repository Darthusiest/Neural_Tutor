"""Capability analytics columns for evaluation case results.

Revision ID: 008
Revises: 007
"""

from alembic import op
import sqlalchemy as sa

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("evaluation_case_results") as batch_op:
        batch_op.add_column(sa.Column("primary_error_type", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("boost_metrics_json", sa.Text(), nullable=True))
        batch_op.create_index(
            "ix_evaluation_case_results_primary_error_type",
            ["primary_error_type"],
        )


def downgrade() -> None:
    with op.batch_alter_table("evaluation_case_results") as batch_op:
        batch_op.drop_index("ix_evaluation_case_results_primary_error_type")
        batch_op.drop_column("boost_metrics_json")
        batch_op.drop_column("primary_error_type")
