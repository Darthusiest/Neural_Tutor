"""Analytics layer redesign: enriched RetrievalLog, new RetrievalChunkHit,
enriched ResponseVariant, enriched Feedback, new MessageOutcome.

Revision ID: 001_analytics
Revises: (initial)
Create Date: 2026-04-09
"""

from alembic import op
import sqlalchemy as sa

revision = "001_analytics"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # --- New tables ---

    op.create_table(
        "retrieval_chunk_hits",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "retrieval_log_id",
            sa.Integer,
            sa.ForeignKey("retrieval_logs.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "lecture_chunk_id",
            sa.Integer,
            sa.ForeignKey("lecture_chunks.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("rank", sa.SmallInteger, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("selected_for_answer", sa.Boolean, nullable=False, server_default="1"),
        sa.Column("token_score", sa.Float, nullable=True),
        sa.Column("phrase_score", sa.Float, nullable=True),
        sa.Column("lecture_bonus", sa.Float, nullable=True),
        sa.Column("strong_field_token_score", sa.Float, nullable=True),
        sa.Column("matched_query_terms", sa.SmallInteger, nullable=True),
        sa.Column("phrase_events", sa.SmallInteger, nullable=True),
        sa.Column("field_scores_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_chunk_hit_log_rank",
        "retrieval_chunk_hits",
        ["retrieval_log_id", "rank"],
    )

    op.create_table(
        "message_outcomes",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "message_id",
            sa.Integer,
            sa.ForeignKey("messages.id"),
            unique=True,
            nullable=False,
            index=True,
        ),
        sa.Column("had_follow_up", sa.Boolean, nullable=True),
        sa.Column("follow_up_count", sa.SmallInteger, nullable=True),
        sa.Column("follow_up_type", sa.String(32), nullable=True),
        sa.Column("was_rephrased", sa.Boolean, nullable=True),
        sa.Column("user_changed_topic_after", sa.Boolean, nullable=True),
        sa.Column("answer_resolved", sa.Boolean, nullable=True),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # --- Enrich retrieval_logs ---

    with op.batch_alter_table("retrieval_logs") as batch_op:
        batch_op.add_column(sa.Column("normalized_query", sa.Text, nullable=True))
        batch_op.add_column(sa.Column("query_tokens_json", sa.Text, nullable=True))
        batch_op.add_column(
            sa.Column("lecture_numbers_detected_json", sa.Text, nullable=True)
        )
        batch_op.add_column(
            sa.Column("retrieval_backend", sa.String(32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("top_k_requested", sa.SmallInteger, nullable=True)
        )
        batch_op.add_column(sa.Column("num_chunks_scored", sa.Integer, nullable=True))
        batch_op.add_column(sa.Column("num_chunks_hit", sa.Integer, nullable=True))
        batch_op.add_column(sa.Column("top_score", sa.Float, nullable=True))
        batch_op.add_column(sa.Column("second_score", sa.Float, nullable=True))
        batch_op.add_column(sa.Column("score_margin", sa.Float, nullable=True))
        batch_op.add_column(sa.Column("query_coverage", sa.Float, nullable=True))
        batch_op.add_column(sa.Column("is_low_confidence", sa.Boolean, nullable=True))
        batch_op.add_column(sa.Column("is_off_topic", sa.Boolean, nullable=True))
        # Make message_id unique (one retrieval per assistant message).
        # batch_alter_table handles SQLite's lack of ALTER COLUMN.
        batch_op.create_unique_constraint("uq_retrieval_logs_message_id", ["message_id"])

    # --- Enrich response_variants ---

    with op.batch_alter_table("response_variants") as batch_op:
        batch_op.add_column(
            sa.Column(
                "retrieval_log_id",
                sa.Integer,
                sa.ForeignKey("retrieval_logs.id"),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("boost_used", sa.Boolean, nullable=True))
        batch_op.add_column(
            sa.Column("boost_auto_triggered", sa.Boolean, nullable=True)
        )
        batch_op.add_column(
            sa.Column("boost_toggle_user_selected", sa.Boolean, nullable=True)
        )
        batch_op.add_column(
            sa.Column("provider_name", sa.String(64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("course_answer_prompt_version", sa.String(32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("boost_prompt_version", sa.String(32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("course_answer_length", sa.Integer, nullable=True)
        )
        batch_op.add_column(
            sa.Column("boosted_answer_length", sa.Integer, nullable=True)
        )
        batch_op.add_column(
            sa.Column("response_fingerprint", sa.String(40), nullable=True)
        )
        batch_op.create_index(
            "ix_response_variants_retrieval_log_id", ["retrieval_log_id"]
        )
        batch_op.create_index(
            "ix_response_variants_fingerprint", ["response_fingerprint"]
        )

    # --- Enrich feedback ---

    with op.batch_alter_table("feedback") as batch_op:
        batch_op.add_column(
            sa.Column("helpfulness_rating", sa.SmallInteger, nullable=True)
        )
        batch_op.add_column(sa.Column("resolved", sa.Boolean, nullable=True))
        batch_op.add_column(
            sa.Column("follow_up_required", sa.Boolean, nullable=True)
        )
        batch_op.add_column(
            sa.Column("follow_up_type", sa.String(32), nullable=True)
        )
        batch_op.add_column(
            sa.Column("explicit_confusion_flag", sa.Boolean, nullable=True)
        )
        batch_op.add_column(sa.Column("feedback_note", sa.Text, nullable=True))
        batch_op.add_column(
            sa.Column("preference_strength", sa.String(16), nullable=True)
        )


def downgrade():
    # --- Revert feedback ---
    with op.batch_alter_table("feedback") as batch_op:
        batch_op.drop_column("preference_strength")
        batch_op.drop_column("feedback_note")
        batch_op.drop_column("explicit_confusion_flag")
        batch_op.drop_column("follow_up_type")
        batch_op.drop_column("follow_up_required")
        batch_op.drop_column("resolved")
        batch_op.drop_column("helpfulness_rating")

    # --- Revert response_variants ---
    with op.batch_alter_table("response_variants") as batch_op:
        batch_op.drop_index("ix_response_variants_fingerprint")
        batch_op.drop_index("ix_response_variants_retrieval_log_id")
        batch_op.drop_column("response_fingerprint")
        batch_op.drop_column("boosted_answer_length")
        batch_op.drop_column("course_answer_length")
        batch_op.drop_column("boost_prompt_version")
        batch_op.drop_column("course_answer_prompt_version")
        batch_op.drop_column("provider_name")
        batch_op.drop_column("boost_toggle_user_selected")
        batch_op.drop_column("boost_auto_triggered")
        batch_op.drop_column("boost_used")
        batch_op.drop_column("retrieval_log_id")

    # --- Revert retrieval_logs ---
    with op.batch_alter_table("retrieval_logs") as batch_op:
        batch_op.drop_constraint("uq_retrieval_logs_message_id", type_="unique")
        batch_op.drop_column("is_off_topic")
        batch_op.drop_column("is_low_confidence")
        batch_op.drop_column("query_coverage")
        batch_op.drop_column("score_margin")
        batch_op.drop_column("second_score")
        batch_op.drop_column("top_score")
        batch_op.drop_column("num_chunks_hit")
        batch_op.drop_column("num_chunks_scored")
        batch_op.drop_column("top_k_requested")
        batch_op.drop_column("retrieval_backend")
        batch_op.drop_column("lecture_numbers_detected_json")
        batch_op.drop_column("query_tokens_json")
        batch_op.drop_column("normalized_query")

    # --- Drop new tables ---
    op.drop_table("message_outcomes")
    op.drop_index("ix_chunk_hit_log_rank", table_name="retrieval_chunk_hits")
    op.drop_table("retrieval_chunk_hits")
