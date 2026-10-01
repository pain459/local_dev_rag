"""Request identity: stable IDs are authoritative; roots are diagnostic metadata."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self


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
        return cls(
            session_id=normalized["x-opencode-session-id"],
            project_id=normalized["x-opencode-project-id"],
            project_root=normalized.get("x-opencode-project-root"),
        )
