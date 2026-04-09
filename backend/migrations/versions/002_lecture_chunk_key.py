"""Add stable chunk_key to lecture_chunks.

Revision ID: 002_chunk_key
Revises: 001_analytics
Create Date: 2026-04-09
"""

import sqlalchemy as sa
from alembic import op

revision = "002_chunk_key"
down_revision = "001_analytics"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "lecture_chunks",
        sa.Column("chunk_key", sa.String(length=128), nullable=True),
    )
    op.execute(
        "UPDATE lecture_chunks SET chunk_key = 'migrated-' || CAST(id AS VARCHAR) "
        "WHERE chunk_key IS NULL"
    )
    op.alter_column("lecture_chunks", "chunk_key", nullable=False)
    op.create_index(
        "ix_lecture_chunks_chunk_key",
        "lecture_chunks",
        ["chunk_key"],
        unique=True,
    )


def downgrade():
    op.drop_index("ix_lecture_chunks_chunk_key", table_name="lecture_chunks")
    op.drop_column("lecture_chunks", "chunk_key")
