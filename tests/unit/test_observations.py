from uuid import uuid4

from local_dev_rag import observations
from local_dev_rag.domain import RequestIdentity
from local_dev_rag.observations import ObservationStore

IDENTITY = RequestIdentity("session", "project")


def test_observation_expires_after_five_minutes_without_refresh_on_read(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(observations, "monotonic", lambda: now[0], raising=False)
    store = ObservationStore()
    identifier = str(uuid4())
    store.put(identifier, IDENTITY, 137)
    now[0] = 299.0
    assert store.get(identifier, IDENTITY) == 137
    now[0] = 300.0
    assert store.get(identifier, IDENTITY) is None


def test_observation_store_evicts_oldest_above_capacity():
    store = ObservationStore()
    identifiers = [str(uuid4()) for _ in range(1025)]
    for identifier in identifiers:
        store.put(identifier, IDENTITY, 137)
    assert store.get(identifiers[0], IDENTITY) is None
    assert store.get(identifiers[1], IDENTITY) == 137
    assert store.get(identifiers[-1], IDENTITY) == 137


def test_reused_identifier_cannot_overwrite_or_disclose_another_scope():
    store = ObservationStore()
    identifier = str(uuid4())
    store.put(identifier, IDENTITY, 137)
    foreign = RequestIdentity("other-session", "other-project")
    store.put(identifier, foreign, 999)
    assert store.get(identifier, foreign) is None
    assert store.get(identifier, IDENTITY) == 137


def test_expired_identifier_can_be_reused_without_retaining_old_scope(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(observations, "monotonic", lambda: now[0], raising=False)
    store = ObservationStore()
    identifier = str(uuid4())
    store.put(identifier, IDENTITY, 137)
    now[0] = 301.0
    foreign = RequestIdentity("other-session", "other-project")
    store.put(identifier, foreign, 999)
    assert store.get(identifier, IDENTITY) is None
    assert store.get(identifier, foreign) == 999
