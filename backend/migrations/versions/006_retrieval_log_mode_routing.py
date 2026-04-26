"""Mode routing columns on retrieval_logs.

Revision ID: 006
Revises: 005
"""

from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("retrieval_logs") as batch_op:
        batch_op.add_column(sa.Column("mode_detected", sa.String(16), nullable=True))
        batch_op.add_column(sa.Column("mode_effective", sa.String(16), nullable=True))
        batch_op.add_column(sa.Column("mode_overridden", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("mode_confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("mode_ambiguous", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("mode_signals_json", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("mode_request_source", sa.String(16), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("retrieval_logs") as batch_op:
        batch_op.drop_column("mode_request_source")
        batch_op.drop_column("mode_signals_json")
        batch_op.drop_column("mode_ambiguous")
        batch_op.drop_column("mode_confidence")
        batch_op.drop_column("mode_overridden")
        batch_op.drop_column("mode_effective")
        batch_op.drop_column("mode_detected")
