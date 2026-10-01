"""Request identity: stable IDs are authoritative; roots are diagnostic metadata."""

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal, Self, cast
from urllib.parse import unquote
from uuid import UUID


class InvalidRequestError(ValueError):
    def __init__(self, message: str, param: str):
        super().__init__(message)
        self.param = param


@dataclass(frozen=True)
class RequestIdentity:
    session_id: str
    project_id: str
    project_root: str | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> Self:
        normalized = {key.lower(): value for key, value in headers.items()}
        for header in ("x-opencode-session-id", "x-opencode-project-id"):
            if not normalized.get(header, "").strip():
                raise InvalidRequestError(f"Missing required header: {header}", param=header)
        encoded_root = normalized.get("x-opencode-project-root")
        return cls(
            session_id=normalized["x-opencode-session-id"],
            project_id=normalized["x-opencode-project-id"],
            project_root=unquote(encoded_root) if encoded_root is not None else None,
        )


def freeze_json(value: object) -> object:
    """Own an immutable snapshot, including nested JSON content."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return MappingProxyType({key: freeze_json(item) for key, item in mapping.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in cast(list[object] | tuple[object, ...], value))
    return value


def _freeze_payload(payload: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: freeze_json(value) for key, value in payload.items()})


MemoryKind = Literal[
    "requirement", "decision", "constraint", "preference", "error", "fix", "outcome"
]
MemoryState = Literal["active", "superseded", "rejected", "deleted"]
JobStatus = Literal["pending", "running", "retry", "completed", "failed"]


@dataclass(frozen=True)
class Scope:
    project_id: UUID
    session_id: UUID


@dataclass(frozen=True)
class ConversationEventInput:
    event_type: str
    role: str
    payload: Mapping[str, object]
    content_hash: str
    request_id: str
    source_message_id: str | None = None
    model: str | None = None
    completed: bool = True
    # Normalizers retain the preceding occurrence identity so hashes can be recomputed.
    parent_hash: str = ""
    completion_request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True)
class StoredEvent:
    id: UUID
    project_id: UUID
    session_id: UUID
    sequence: int
    event_type: str
    role: str
    payload: Mapping[str, object]
    content_hash: str
    request_id: str
    created_at: datetime
    source_message_id: str | None = None
    model: str | None = None
    completed: bool = True
    completion_request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True)
class AssistantCompletion:
    payload: Mapping[str, object]
    content_hash: str
    request_id: str
    model: str | None = None
    source_message_id: str | None = None
    completed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True)
class UpstreamResponse:
    status_code: int
    headers: Mapping[str, str]
    body: AsyncIterator[bytes]


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    state: Literal["healthy", "degraded", "unavailable"]
    detail: str | None = None


@dataclass(frozen=True)
class MemoryItem:
    id: UUID
    project_id: UUID
    source_session_id: UUID
    source_event_id: UUID
    kind: MemoryKind
    text: str
    confidence: float
    importance: float
    state: MemoryState
    curator_model: str
    created_at: datetime
    updated_at: datetime
    superseded_by_id: UUID | None = None
    embedding_model: str | None = None
    embedding_version: int = 1


@dataclass(frozen=True)
class MemoryCandidate:
    memory: MemoryItem
    score: float = 0.0
    score_components: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.score_components is not None:
            object.__setattr__(
                self, "score_components", MappingProxyType(dict(self.score_components))
            )


@dataclass(frozen=True)
class EmbeddedMemory:
    memory: MemoryItem
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(self.vector))


@dataclass(frozen=True)
class VectorHit:
    memory: MemoryItem
    distance: float


@dataclass(frozen=True)
class CuratorSource:
    project_id: UUID
    session_id: UUID
    source_event_id: UUID
    events: tuple[StoredEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True)
class MemoryDraft:
    kind: MemoryKind
    text: str
    confidence: float
    importance: float


@dataclass(frozen=True)
class MemoryJob:
    id: UUID
    project_id: UUID
    session_id: UUID
    source_event_id: UUID
    job_kind: str
    deduplication_key: str
    status: JobStatus
    attempt_count: int
    next_attempt_at: datetime
    created_at: datetime
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    error_category: str | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
