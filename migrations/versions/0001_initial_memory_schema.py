"""initial_memory_schema

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("external_project_id", sa.Text(), nullable=False),
        sa.Column("remote_identity", sa.Text(), nullable=True),
        sa.Column("root_label", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_project_id"),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("opencode_session_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "id", name="uq_session_project_id"),
        sa.UniqueConstraint("project_id", "opencode_session_id", name="uq_session_external"),
    )
    op.create_table(
        "conversation_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("sequence", sa.BigInteger(), nullable=False),
        sa.Column("source_message_id", sa.Text(), nullable=True),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model", sa.Text(), nullable=True),
        sa.Column("completed", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("length(content_hash) > 0", name="ck_event_hash"),
        sa.CheckConstraint("sequence > 0", name="ck_event_sequence"),
        sa.ForeignKeyConstraint(
            ["project_id", "session_id"],
            ["sessions.project_id", "sessions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "session_id", "id", name="uq_event_scope_id"),
        sa.UniqueConstraint("session_id", "content_hash", name="uq_event_content"),
        sa.UniqueConstraint("session_id", "sequence", name="uq_event_sequence"),
    )
    op.create_index(
        "ix_event_project_created",
        "conversation_events",
        ["project_id", "created_at"],
        unique=False,
    )
    op.create_table(
        "memory_items",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("source_session_id", sa.UUID(), nullable=False),
        sa.Column("source_event_id", sa.UUID(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("importance", sa.Float(), nullable=False),
        sa.Column("state", sa.Text(), server_default="active", nullable=False),
        sa.Column("superseded_by_id", sa.UUID(), nullable=True),
        sa.Column("curator_model", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=True),
        sa.Column("embedding_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(state = 'superseded') = (superseded_by_id IS NOT NULL)", name="ck_memory_supersession"
        ),
        sa.CheckConstraint(
            "kind IN ('requirement','decision','constraint','preference','error','fix','outcome')",
            name="ck_memory_kind",
        ),
        sa.CheckConstraint(
            "state IN ('active','superseded','rejected','deleted')", name="ck_memory_state"
        ),
        sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_memory_confidence"),
        sa.CheckConstraint("embedding_version > 0", name="ck_memory_embedding_version"),
        sa.CheckConstraint("importance BETWEEN 0 AND 1", name="ck_memory_importance"),
        sa.CheckConstraint("length(trim(text)) > 0", name="ck_memory_text"),
        sa.CheckConstraint(
            "superseded_by_id IS NULL OR superseded_by_id != id",
            name="ck_memory_not_self_superseded",
        ),
        sa.ForeignKeyConstraint(
            ["project_id", "source_session_id", "source_event_id"],
            [
                "conversation_events.project_id",
                "conversation_events.session_id",
                "conversation_events.id",
            ],
        ),
        sa.ForeignKeyConstraint(
            ["project_id", "superseded_by_id"],
            ["memory_items.project_id", "memory_items.id"],
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", "id", name="uq_memory_project_id"),
        sa.UniqueConstraint("source_event_id", "kind", "text", name="uq_memory_source_content"),
    )
    op.create_index(
        "ix_memory_project_state", "memory_items", ["project_id", "state"], unique=False
    )
    op.create_index("ix_memory_source_event", "memory_items", ["source_event_id"], unique=False)
    op.create_table(
        "memory_jobs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("project_id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("source_event_id", sa.UUID(), nullable=False),
        sa.Column("job_kind", sa.Text(), server_default="curate", nullable=False),
        sa.Column("deduplication_key", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("lease_owner", sa.Text(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_category", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status = 'running') = (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_job_running_lease",
        ),
        sa.CheckConstraint(
            "(status IN ('completed','failed')) = (completed_at IS NOT NULL)",
            name="ck_job_terminal_completion",
        ),
        sa.CheckConstraint(
            "status IN ('pending','running','retry','completed','failed')", name="ck_job_status"
        ),
        sa.CheckConstraint(
            "(lease_owner IS NULL) = (lease_expires_at IS NULL)", name="ck_job_lease_pair"
        ),
        sa.CheckConstraint("attempt_count >= 0", name="ck_job_attempt_count"),
        sa.ForeignKeyConstraint(
            ["project_id", "session_id", "source_event_id"],
            [
                "conversation_events.project_id",
                "conversation_events.session_id",
                "conversation_events.id",
            ],
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("deduplication_key"),
        sa.UniqueConstraint("source_event_id", "job_kind", name="uq_job_source_kind"),
    )
    op.create_index("ix_job_lease", "memory_jobs", ["status", "lease_expires_at"], unique=False)
    op.create_index("ix_job_ready", "memory_jobs", ["status", "next_attempt_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_job_ready", table_name="memory_jobs")
    op.drop_index("ix_job_lease", table_name="memory_jobs")
    op.drop_table("memory_jobs")
    op.drop_index("ix_memory_source_event", table_name="memory_items")
    op.drop_index("ix_memory_project_state", table_name="memory_items")
    op.drop_table("memory_items")
    op.drop_index("ix_event_project_created", table_name="conversation_events")
    op.drop_table("conversation_events")
    op.drop_table("sessions")
    op.drop_table("projects")
