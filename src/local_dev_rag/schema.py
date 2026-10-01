"""SQLAlchemy Core table mappings. Alembic owns schema creation."""

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = sa.MetaData()


def timestamp(name: str) -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


projects = sa.Table(
    "projects",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("external_project_id", sa.Text, nullable=False, unique=True),
    sa.Column("remote_identity", sa.Text),
    sa.Column("root_label", sa.Text),
    timestamp("created_at"),
    timestamp("updated_at"),
)
sessions = sa.Table(
    "sessions",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
    sa.Column("opencode_session_id", sa.Text, nullable=False),
    sa.Column("title", sa.Text),
    timestamp("created_at"),
    timestamp("last_seen_at"),
    sa.UniqueConstraint("project_id", "opencode_session_id", name="uq_session_external"),
    sa.UniqueConstraint("project_id", "id", name="uq_session_project_id"),
)
conversation_events = sa.Table(
    "conversation_events",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
    sa.Column("session_id", UUID(as_uuid=True), nullable=False),
    sa.Column("sequence", sa.BigInteger, nullable=False),
    sa.Column("source_message_id", sa.Text),
    sa.Column("request_id", sa.Text, nullable=False),
    sa.Column("content_hash", sa.Text, nullable=False),
    sa.Column("event_type", sa.Text, nullable=False),
    sa.Column("role", sa.Text, nullable=False),
    sa.Column("payload", JSONB[dict[str, object]](), nullable=False),
    sa.Column("model", sa.Text),
    sa.Column("completed", sa.Boolean, nullable=False, server_default=sa.true()),
    timestamp("created_at"),
    sa.ForeignKeyConstraint(["project_id", "session_id"], ["sessions.project_id", "sessions.id"]),
    sa.UniqueConstraint("session_id", "content_hash", name="uq_event_content"),
    sa.UniqueConstraint("session_id", "sequence", name="uq_event_sequence"),
    sa.UniqueConstraint("project_id", "session_id", "id", name="uq_event_scope_id"),
    sa.CheckConstraint("sequence > 0", name="ck_event_sequence"),
    sa.CheckConstraint("length(content_hash) > 0", name="ck_event_hash"),
    sa.Index("ix_event_project_created", "project_id", "created_at"),
)
memory_items = sa.Table(
    "memory_items",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
    sa.Column("source_session_id", UUID(as_uuid=True), nullable=False),
    sa.Column("source_event_id", UUID(as_uuid=True), nullable=False),
    sa.Column("kind", sa.Text, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("confidence", sa.Float(asdecimal=False), nullable=False),
    sa.Column("importance", sa.Float(asdecimal=False), nullable=False),
    sa.Column("state", sa.Text, nullable=False, server_default="active"),
    sa.Column("superseded_by_id", UUID(as_uuid=True)),
    sa.Column("curator_model", sa.Text, nullable=False),
    sa.Column("embedding_model", sa.Text),
    sa.Column("embedding_version", sa.Integer, nullable=False, server_default="1"),
    timestamp("created_at"),
    timestamp("updated_at"),
    sa.ForeignKeyConstraint(
        ["project_id", "source_session_id", "source_event_id"],
        [
            "conversation_events.project_id",
            "conversation_events.session_id",
            "conversation_events.id",
        ],
    ),
    sa.UniqueConstraint("project_id", "id", name="uq_memory_project_id"),
    sa.UniqueConstraint("source_event_id", "kind", "text", name="uq_memory_source_content"),
    sa.ForeignKeyConstraint(
        ["project_id", "superseded_by_id"], ["memory_items.project_id", "memory_items.id"]
    ),
    sa.CheckConstraint(
        "kind IN ('requirement','decision','constraint','preference','error','fix','outcome')",
        name="ck_memory_kind",
    ),
    sa.CheckConstraint(
        "state IN ('active','superseded','rejected','deleted')", name="ck_memory_state"
    ),
    sa.CheckConstraint("confidence BETWEEN 0 AND 1", name="ck_memory_confidence"),
    sa.CheckConstraint("importance BETWEEN 0 AND 1", name="ck_memory_importance"),
    sa.CheckConstraint("length(trim(text)) > 0", name="ck_memory_text"),
    sa.CheckConstraint("embedding_version > 0", name="ck_memory_embedding_version"),
    sa.CheckConstraint(
        "(state = 'superseded') = (superseded_by_id IS NOT NULL)", name="ck_memory_supersession"
    ),
    sa.CheckConstraint(
        "superseded_by_id IS NULL OR superseded_by_id != id", name="ck_memory_not_self_superseded"
    ),
    sa.Index("ix_memory_project_state", "project_id", "state"),
    sa.Index("ix_memory_source_event", "source_event_id"),
)
memory_jobs = sa.Table(
    "memory_jobs",
    metadata,
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("project_id", UUID(as_uuid=True), sa.ForeignKey("projects.id"), nullable=False),
    sa.Column("session_id", UUID(as_uuid=True), nullable=False),
    sa.Column("source_event_id", UUID(as_uuid=True), nullable=False),
    sa.Column("job_kind", sa.Text, nullable=False, server_default="curate"),
    sa.Column("deduplication_key", sa.Text, nullable=False, unique=True),
    sa.Column("status", sa.Text, nullable=False, server_default="pending"),
    sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
    timestamp("next_attempt_at"),
    sa.Column("lease_owner", sa.Text),
    sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
    sa.Column("error_category", sa.String(100)),
    sa.Column("error_message", sa.String(500)),
    timestamp("created_at"),
    sa.Column("started_at", sa.DateTime(timezone=True)),
    sa.Column("completed_at", sa.DateTime(timezone=True)),
    sa.ForeignKeyConstraint(
        ["project_id", "session_id", "source_event_id"],
        [
            "conversation_events.project_id",
            "conversation_events.session_id",
            "conversation_events.id",
        ],
    ),
    sa.UniqueConstraint("source_event_id", "job_kind", name="uq_job_source_kind"),
    sa.CheckConstraint(
        "status IN ('pending','running','retry','completed','failed')", name="ck_job_status"
    ),
    sa.CheckConstraint("attempt_count >= 0", name="ck_job_attempt_count"),
    sa.CheckConstraint(
        "(status = 'running') = (lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL)",
        name="ck_job_running_lease",
    ),
    sa.CheckConstraint(
        "(lease_owner IS NULL) = (lease_expires_at IS NULL)", name="ck_job_lease_pair"
    ),
    sa.CheckConstraint(
        "(status IN ('completed','failed')) = (completed_at IS NOT NULL)",
        name="ck_job_terminal_completion",
    ),
    sa.Index("ix_job_ready", "status", "next_attempt_at"),
    sa.Index("ix_job_lease", "status", "lease_expires_at"),
)
