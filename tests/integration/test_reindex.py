"""Project-only index rebuilding from active PostgreSQL memories, including the CLI."""

import importlib
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from sqlalchemy import update

from local_dev_rag.config import Settings
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.schema import memory_items

from .test_worker import envelope, memories, seed, vector_store, worker


def cli():
    assert importlib.util.find_spec("local_dev_rag.cli") is not None, "Reindex CLI is missing"
    return importlib.import_module("local_dev_rag.cli")


def handler(request):
    if request.url.path == "/api/embed":
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[1, 0]] * len(body["input"])})
    return httpx.Response(200, json=envelope())


async def populate(database, store):
    scope, _ = await seed(database)
    await worker(database, store, handler).run_once()
    own = await memories(database)
    other, _ = await seed(database, project="other", session_id="foreign")
    await worker(database, store, handler).run_once()
    async with database.session() as session:
        for item, state in zip(own[:3], ["superseded", "rejected", "deleted"], strict=True):
            await session.execute(
                update(memory_items)
                .where(memory_items.c.id == item["id"])
                .values(
                    state=state, superseded_by_id=own[3]["id"] if state == "superseded" else None
                )
            )
    return scope, other, {item["id"] for item in own[3:]}


async def test_rebuild_deletes_stale_records_and_only_indexes_active_project_memories(
    database, chroma_url
):
    module = cli()
    store = vector_store(chroma_url)
    scope, other, active_ids = await populate(database, store)
    foreign_before = await store.query(other.project_id, (1, 0), 10)
    settings = Settings(_env_file=None, embedding_version=7, embedding_model="new-embedding-model")
    batches = []

    def embedding(request):
        body = json.loads(request.content)
        assert body["model"] == "new-embedding-model"
        batches.append(len(body["input"]))
        return httpx.Response(200, json={"embeddings": [[0, 1]] * len(body["input"])})

    count = await module.reindex(
        database,
        settings,
        "project",
        embedder=OllamaClient(settings, transport=httpx.MockTransport(embedding)),
        vector_store=store,
        batch_size=1,
    )
    assert count == 2 and batches == [1, 1]
    hits = await store.query(scope.project_id, (0, 1), 10)
    assert {hit.memory.id for hit in hits} == active_ids
    assert all(
        hit.memory.state == "active"
        and hit.memory.embedding_version == 7
        and hit.memory.embedding_model == "new-embedding-model"
        for hit in hits
    )
    assert {
        hit.memory.id: hit.memory for hit in await store.query(other.project_id, (1, 0), 10)
    } == {hit.memory.id: hit.memory for hit in foreign_before}
    stored = await memories(database)
    assert all(item["embedding_version"] == 7 for item in stored if item["id"] in active_ids)
    assert all(item["embedding_version"] == 3 for item in stored if item["id"] not in active_ids)
    # Loss of the entire project's index converges again from authoritative rows.
    await store.delete_project(scope.project_id)
    assert await store.query(scope.project_id, (0, 1), 10) == []
    assert (
        await module.reindex(
            database,
            settings,
            "project",
            embedder=OllamaClient(settings, transport=httpx.MockTransport(embedding)),
            vector_store=store,
            batch_size=1,
        )
        == 2
    )
    assert {hit.memory.id for hit in await store.query(scope.project_id, (0, 1), 10)} == active_ids


@pytest.mark.parametrize("external_id", ["", " ", "unknown"])
async def test_missing_or_unknown_project_cannot_delete_any_vectors(
    database, chroma_url, external_id
):
    module = cli()
    store = vector_store(chroma_url)
    scope, _, _ = await populate(database, store)
    before = await store.query(scope.project_id, (1, 0), 10)
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError):
        await module.reindex(database, settings, external_id, vector_store=store)
    assert {
        hit.memory.id: hit.memory for hit in await store.query(scope.project_id, (1, 0), 10)
    } == {hit.memory.id: hit.memory for hit in before}


async def test_empty_project_rebuild_does_not_call_embedder(database, chroma_url):
    module = cli()
    await seed(database)

    def forbidden(request):
        pytest.fail("Empty rebuild called Ollama")

    settings = Settings(_env_file=None)
    assert (
        await module.reindex(
            database,
            settings,
            "project",
            embedder=OllamaClient(settings, transport=httpx.MockTransport(forbidden)),
            vector_store=vector_store(chroma_url),
        )
        == 0
    )


async def test_memory_retired_during_embedding_is_never_upserted(database, chroma_url):
    module = cli()
    store = vector_store(chroma_url)
    scope, _, active_ids = await populate(database, store)
    settings = Settings(_env_file=None, embedding_version=8)

    async def embedding(request):
        async with database.session() as session:
            await session.execute(
                update(memory_items)
                .where(memory_items.c.id.in_(active_ids))
                .values(state="deleted")
            )
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[1, 0]] * len(body["input"])})

    assert (
        await module.reindex(
            database,
            settings,
            "project",
            vector_store=store,
            embedder=OllamaClient(settings, transport=httpx.MockTransport(embedding)),
        )
        == 0
    )
    assert await store.query(scope.project_id, (1, 0), 10) == []


def test_cli_requires_project_and_never_defaults_to_all_projects():
    with pytest.raises(SystemExit) as error:
        cli().main(["reindex"])
    assert error.value.code == 2


async def test_installed_cli_rebuilds_project_and_prints_only_counts(database, chroma_url):
    cli()
    store = vector_store(chroma_url)
    scope, _, active_ids = await populate(database, store)

    class EmbedHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            result = json.dumps({"embeddings": [[0, 1]] * len(body["input"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), EmbedHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = ["uv", "run", "local-dev-rag", "reindex", "--project", "project"]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            env={
                **os.environ,
                "DATABASE_URL": str(database.engine.url.render_as_string(hide_password=False)),
                "CHROMADB_URL": chroma_url,
                "OLLAMA_URL": f"http://127.0.0.1:{server.server_port}",
                "EMBEDDING_VERSION": "9",
            },
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "reindexed=2\n"
        assert "Use PostgreSQL" not in result.stdout + result.stderr
        from local_dev_rag.vector_store import VectorStore

        default_store = VectorStore(
            Settings(_env_file=None, chromadb_url=chroma_url, embedding_version=9)
        )
        hits = await default_store.query(scope.project_id, (0, 1), 10)
        assert {hit.memory.id for hit in hits} == active_ids
        assert all(hit.memory.embedding_version == 9 for hit in hits)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_new_embedding_version_can_rebuild_with_a_different_vector_dimension(
    database, chroma_url
):
    from local_dev_rag.vector_store import VectorStore

    module = cli()
    settings = Settings(_env_file=None, chromadb_url=chroma_url)
    scope, _ = await seed(database)
    old_store = VectorStore(settings)
    await worker(database, old_store, handler, settings=settings).run_once()
    old_hits = await old_store.query(scope.project_id, (1, 0), 10)
    foreign, _ = await seed(database, project="foreign-version")
    await worker(database, old_store, handler, settings=settings).run_once()
    next_settings = Settings(
        _env_file=None, chromadb_url=chroma_url, embedding_version=2, embedding_model="new-model"
    )

    def embedding(request):
        size = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[0, 1, 0]] * size})

    assert (
        await module.reindex(
            database,
            next_settings,
            "project",
            embedder=OllamaClient(next_settings, transport=httpx.MockTransport(embedding)),
        )
        == 5
    )
    new_store = VectorStore(next_settings)
    hits = await new_store.query(scope.project_id, (0, 1, 0), 10)
    assert {hit.memory.id for hit in hits} == {hit.memory.id for hit in old_hits}
    assert all(hit.memory.embedding_version == 2 for hit in hits)
    assert len(await old_store.query(foreign.project_id, (1, 0), 10)) == 5
