"""Add chunk_type and concept_family columns to lecture_chunks.

Revision ID: 003
Revises: 002_chunk_key
"""

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002_chunk_key"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("lecture_chunks") as batch_op:
        batch_op.add_column(sa.Column("chunk_type", sa.String(32), nullable=True))
        batch_op.add_column(sa.Column("concept_family", sa.String(64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("lecture_chunks") as batch_op:
        batch_op.drop_column("concept_family")
        batch_op.drop_column("chunk_type")
