"""Content-free request diagnostics for the local OpenCode status notice."""

from collections import OrderedDict
from time import monotonic
from uuid import UUID

from local_dev_rag.domain import RequestIdentity


def valid_observation_id(value: str) -> bool:
    try:
        identifier = UUID(value)
        return identifier.version == 4 and str(identifier) == value
    except ValueError:
        return False


class ObservationStore:
    """Five-minute observations, bounded to 1024 requests per proxy process."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, tuple[str, str, int, float]] = OrderedDict()

    def _prune(self) -> None:
        now = monotonic()
        while self._entries:
            entry = next(iter(self._entries.values()))
            if now - entry[3] < 300:
                break
            self._entries.popitem(last=False)

    def put(self, identifier: str, identity: RequestIdentity, tokens: int) -> None:
        self._prune()
        # First writer wins, including when a caller reuses an ID across scopes.
        self._entries.setdefault(
            identifier, (identity.project_id, identity.session_id, tokens, monotonic())
        )
        while len(self._entries) > 1024:
            self._entries.popitem(last=False)

    def get(self, identifier: str, identity: RequestIdentity) -> int | None:
        self._prune()
        entry = self._entries.get(identifier)
        if entry is None or entry[:2] != (identity.project_id, identity.session_id):
            return None
        return entry[2]
