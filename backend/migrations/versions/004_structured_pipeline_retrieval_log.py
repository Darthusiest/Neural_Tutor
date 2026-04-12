"""Structured reasoning pipeline columns on retrieval_logs.

Revision ID: 004
Revises: 003
"""

from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("retrieval_logs") as batch_op:
        batch_op.add_column(sa.Column("query_type_v2", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("sub_questions_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("answer_mode", sa.String(64), nullable=True))
        batch_op.add_column(sa.Column("validation_passed", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("validation_checks_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("generic_answer_flag", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("missing_comparison_side_flag", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("answer_plan_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("retrieval_logs") as batch_op:
        batch_op.drop_column("answer_plan_json")
        batch_op.drop_column("missing_comparison_side_flag")
        batch_op.drop_column("generic_answer_flag")
        batch_op.drop_column("validation_checks_json")
        batch_op.drop_column("validation_passed")
        batch_op.drop_column("answer_mode")
        batch_op.drop_column("sub_questions_json")
        batch_op.drop_column("query_type_v2")
