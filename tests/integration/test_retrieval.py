from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from local_dev_rag.config import Settings
from local_dev_rag.domain import EmbeddedMemory, MemoryItem


def memory(project, text="shared error load_widget"):
    now = datetime.now(UTC)
    return MemoryItem(
        uuid4(),
        project,
        uuid4(),
        uuid4(),
        "fix",
        text,
        0.9,
        0.8,
        "active",
        "curator",
        now,
        now,
        embedding_model="nomic-embed-text:latest",
        embedding_version=2,
    )


async def test_real_chroma_filters_before_top_k_and_roundtrips_all_provenance(chroma_url):
    from local_dev_rag.vector_store import VectorStore

    store = VectorStore(
        Settings(chromadb_url=chroma_url, _env_file=None), collection_name=f"test-{uuid4()}"
    )
    project, other = uuid4(), uuid4()
    own = memory(project)
    foreign = memory(other)
    await store.upsert([EmbeddedMemory(own, (0.8, 0.2)), EmbeddedMemory(foreign, (1, 0))])
    hits = await store.query(project, (1, 0), 1)
    assert [hit.memory for hit in hits] == [own]
    assert hits[0].distance == pytest.approx(0.0298575, abs=1e-6)
    assert [hit.memory for hit in await store.query(other, (1, 0), 1)] == [foreign]


async def test_real_chroma_upsert_updates_one_id_and_project_delete_is_scoped(chroma_url):
    from local_dev_rag.vector_store import VectorStore

    store = VectorStore(
        Settings(chromadb_url=chroma_url, _env_file=None), collection_name=f"test-{uuid4()}"
    )
    project, other = uuid4(), uuid4()
    first, foreign = memory(project), memory(other)
    await store.upsert([EmbeddedMemory(first, (1, 0)), EmbeddedMemory(foreign, (1, 0))])
    updated = replace(
        first,
        text="updated durable evidence",
        state="superseded",
        superseded_by_id=uuid4(),
        embedding_version=3,
    )
    await store.upsert([EmbeddedMemory(updated, (0, 1))])
    await store.upsert([EmbeddedMemory(updated, (0, 1))])
    assert [h.memory for h in await store.query(project, (0, 1), 10)] == [updated]
    await store.delete_project(project)
    assert await store.query(project, (1, 0), 10) == []
    assert [h.memory for h in await store.query(other, (1, 0), 10)] == [foreign]


async def test_unavailable_chroma_raises_domain_error():
    from local_dev_rag.vector_store import VectorStore, VectorStoreUnavailable

    store = VectorStore(Settings(chromadb_url="http://127.0.0.1:1", _env_file=None))
    with pytest.raises(VectorStoreUnavailable):
        await store.query(uuid4(), (1, 0), 1)


async def test_query_emits_exact_project_filter_at_chroma_boundary():
    from local_dev_rag.vector_store import VectorStore

    project = uuid4()
    queries = []

    def handler(request):
        import json

        body = json.loads(request.content)
        if request.url.path.endswith("/query"):
            queries.append(body)
            return httpx.Response(
                200, json={"metadatas": [[]], "documents": [[]], "distances": [[]], "ids": [[]]}
            )
        return httpx.Response(200, json={"id": str(uuid4())})

    store = VectorStore(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    assert await store.query(project, (1, 0), 7) == []
    assert queries == [
        {
            "query_embeddings": [[1, 0]],
            "n_results": 7,
            "where": {"project_id": {"$eq": str(project)}},
            "include": ["metadatas", "documents", "distances"],
        }
    ]
