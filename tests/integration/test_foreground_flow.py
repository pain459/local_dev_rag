import json
from dataclasses import replace
from uuid import uuid4

import anyio
import httpx
import pytest
from sqlalchemy import select

from local_dev_rag.api import create_app
from local_dev_rag.config import ModelBudget, Settings
from local_dev_rag.db import Database
from local_dev_rag.domain import EmbeddedMemory, RequestIdentity
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.repository import ConversationRepository
from local_dev_rag.schema import conversation_events, memory_jobs

from .capture_utils import Fragments, invoke_stream
from .test_retrieval import memory

MODEL = "qwen3-coder:30b"
RESULT = {
    "id": "reply",
    "model": MODEL,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}
    ],
}
SSE = (
    b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
)


def settings(**changes):
    return Settings(
        _env_file=None,
        memory_token_budget=600,
        retrieval_candidate_limit=7,
        retrieval_result_limit=1,
        retrieval_min_score=0.6,
        model_budgets={
            MODEL: ModelBudget(context_tokens=1600, output_tokens=200, safety_tokens=100)
        },
        **changes,
    )


def payload(stream=False):
    return {
        "model": MODEL,
        "stream": stream,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": "Current project instructions"},
            {"role": "user", "content": "old " * 2000},
            {"role": "assistant", "content": "old answer " * 500},
            {"role": "user", "content": "load_widget ECONNREFUSED"},
        ],
    }


async def deliver(app, request_payload):
    bodies = []

    async def send(message):
        if message["type"] == "http.response.body":
            bodies.append(message.get("body", b""))

    scope = await invoke_stream(app, send, payload=request_payload)
    return b"".join(bodies), scope["state"]


@pytest.mark.parametrize("stream", [False, True])
async def test_real_foreground_scope_rerank_bounded_injection_and_durable_capture(
    database,
    chroma_url,
    stream,
):
    from local_dev_rag.vector_store import VectorStore

    config = settings(chromadb_url=chroma_url)
    store = VectorStore(config, collection_name=f"flow-{uuid4()}")
    async with database.session() as session:
        repository = ConversationRepository(session)
        own_scope = await repository.ensure_scope(RequestIdentity("old", "project"))
        foreign_scope = await repository.ensure_scope(RequestIdentity("old", "other"))
    own = memory(own_scope.project_id, "load_widget ECONNREFUSED: retry the socket")
    low_importance = replace(
        memory(own_scope.project_id, "load_widget issue: weaker evidence"), importance=0
    )
    foreign = memory(foreign_scope.project_id, "FOREIGN SECRET load_widget ECONNREFUSED")
    rejected = replace(
        memory(own_scope.project_id, "REJECTED load_widget ECONNREFUSED"), state="rejected"
    )
    await store.upsert(
        [
            EmbeddedMemory(own, (0.9, 0.1)),
            EmbeddedMemory(low_importance, (1, 0)),
            EmbeddedMemory(foreign, (1, 0)),
            EmbeddedMemory(rejected, (1, 0)),
        ]
    )
    calls = []

    async def handler(request):
        body = json.loads(request.content)
        calls.append(request.url.path)
        if request.url.path == "/api/embed":
            assert body == {"model": config.embedding_model, "input": ["load_widget ECONNREFUSED"]}
            return httpx.Response(200, json={"embeddings": [[1, 0]]})
        async with database.session() as session:
            inbound = (await session.execute(select(conversation_events))).mappings().all()
        assert len(inbound) == 4
        assert inbound[1]["payload"]["content"] == "old " * 2000
        text = json.dumps(body["messages"])
        assert own.text in text
        assert "weaker evidence" not in text
        assert "FOREIGN SECRET" not in text
        assert "REJECTED" not in text
        assert body["messages"][0] == request_payload["messages"][0]
        assert body["messages"][-1] == request_payload["messages"][-1]
        assert len(body["messages"]) < 5
        assert body["temperature"] == 0.1
        return (
            httpx.Response(
                200,
                stream=Fragments([SSE[:25], SSE[25:]]),
                headers={"content-type": "text/event-stream"},
            )
            if stream
            else httpx.Response(200, json=RESULT)
        )

    request_payload = payload(stream)
    app = create_app(
        config,
        database=database,
        vector_store=store,
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(handler)),
    )
    body, state = await deliver(app, request_payload)
    assert body == SSE if stream else json.loads(body) == RESULT
    assert calls == ["/api/embed", "/v1/chat/completions"]
    diagnostics = state["proxy_diagnostics"]
    assert diagnostics.retrieval_count == 1
    assert 0 < diagnostics.injected_memory_tokens <= 600
    assert diagnostics.degraded_dependencies == ()
    assert state["capture_status"] == "completed"
    async with database.session() as session:
        assert len((await session.execute(select(memory_jobs))).all()) == 1
        assert len((await session.execute(select(conversation_events))).all()) == 5


@pytest.mark.parametrize("failure", ["chromadb", "embedder"])
async def test_optional_dependency_failure_keeps_recent_context_and_durable_capture(
    database, failure
):
    from local_dev_rag.vector_store import VectorStore

    config = settings(chromadb_url="http://127.0.0.1:1")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/embed":
            return (
                httpx.Response(503, json={"error": "embedder offline"})
                if failure == "embedder"
                else httpx.Response(200, json={"embeddings": [[1, 0]]})
            )
        outbound = json.loads(request.content)
        assert outbound["messages"] == [payload()["messages"][0], payload()["messages"][-1]]
        return httpx.Response(200, json=RESULT)

    def chroma_handler(request):
        pytest.fail("Failed embedder must never reach the vector store")

    store = (
        VectorStore(config, transport=httpx.MockTransport(chroma_handler))
        if (failure == "embedder")
        else VectorStore(config)
    )
    app = create_app(
        config,
        database=database,
        vector_store=store,
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(handler)),
    )
    body, state = await deliver(app, payload())
    assert json.loads(body) == RESULT
    assert state["proxy_diagnostics"].degraded_dependencies == (failure,)
    assert state["proxy_diagnostics"].injected_memory_tokens == 0
    assert state["capture_status"] == "completed"
    assert calls == ["/api/embed", "/v1/chat/completions"]
    async with database.session() as session:
        assert len((await session.execute(select(conversation_events))).all()) == 5
        assert len((await session.execute(select(memory_jobs))).all()) == 1


async def test_postgres_down_is_untrimmed_passthrough_without_embedding_or_memory_claim():
    config = settings(database_url="postgresql+asyncpg://unused:unused@127.0.0.1:1/unused")
    database = Database.create(config)
    request_payload = payload()

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content) == request_payload
        return httpx.Response(200, json=RESULT)

    try:
        app = create_app(
            config,
            database=database,
            ollama_client=OllamaClient(config, transport=httpx.MockTransport(handler)),
        )
        body, state = await deliver(app, request_payload)
        assert json.loads(body) == RESULT
        assert state["capture_status"] == "unavailable"
        assert state["proxy_diagnostics"].degraded_dependencies == ("postgres",)
        assert state["proxy_diagnostics"].retrieval_count == 0
        assert state["proxy_diagnostics"].injected_memory_tokens == 0
    finally:
        await database.engine.dispose()


async def test_cancelled_query_embedding_closes_transport_and_records_incomplete_attempt(database):
    waiting = anyio.Event()
    cleanup = []

    class BlockingEmbedding(httpx.AsyncByteStream):
        async def __aiter__(self):
            waiting.set()
            await anyio.sleep_forever()
            yield b"unreachable"

        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            cleanup.append("closed")

    config = settings()

    def handler(request):
        assert request.url.path == "/api/embed"
        return httpx.Response(200, stream=BlockingEmbedding())

    app = create_app(
        config,
        database=database,
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(handler)),
    )
    with anyio.fail_after(2):
        async with anyio.create_task_group() as group:
            group.start_soon(deliver, app, payload())
            await waiting.wait()
            group.cancel_scope.cancel()
    assert cleanup == ["closed"]
    async with database.session() as session:
        rows = (await session.execute(select(conversation_events))).mappings().all()
        assert [row["role"] for row in rows] == ["system", "user", "assistant", "user", "proxy"]
        assert rows[-1]["completed"] is False
        assert (await session.execute(select(memory_jobs))).all() == []


async def test_misbehaving_search_cannot_pass_a_foreign_candidate_to_context_builder(database):
    from local_dev_rag.context import ContextBuilder
    from local_dev_rag.domain import VectorHit
    from local_dev_rag.models import ModelRegistry
    from local_dev_rag.proxy import ProxyService
    from local_dev_rag.repository import PostgresCaptureStore

    config = settings()
    built = []

    class ForeignSearch:
        async def query(self, project_id, vector, limit):
            assert limit == 7
            return [VectorHit(memory(uuid4(), "FOREIGN SECRET"), 0)]

    class CheckedBuilder(ContextBuilder):
        def build(self, request, memories, history_durable):
            assert memories == []
            built.append(history_durable)
            return super().build(request, memories, history_durable)

    def handler(request):
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 0]]})
        assert "FOREIGN SECRET" not in request.content.decode()
        return httpx.Response(200, json=RESULT)

    service = ProxyService(
        config,
        registry=ModelRegistry(config),
        ollama=OllamaClient(config, transport=httpx.MockTransport(handler)),
        capture_store=PostgresCaptureStore(database),
        vector_store=ForeignSearch(),
        context_builder_factory=lambda spec: CheckedBuilder(
            spec, safety_tokens=100, memory_token_budget=600
        ),
    )
    result = await service.complete(RequestIdentity("session", "project"), payload())
    assert json.loads(result.response.body) == RESULT
    assert result.diagnostics.degraded_dependencies == ("chromadb",)
    assert result.diagnostics.retrieval_count == 0
    assert built == [True]


async def test_low_semantic_score_omits_irrelevant_active_memory(database, chroma_url):
    from local_dev_rag.vector_store import VectorStore

    config = settings(chromadb_url=chroma_url)
    store = VectorStore(config, collection_name=f"irrelevant-{uuid4()}")
    async with database.session() as session:
        scope = await ConversationRepository(session).ensure_scope(
            RequestIdentity("session", "project")
        )
    irrelevant = replace(memory(scope.project_id, "IRRELEVANT cosmetic preference"), importance=0)
    await store.upsert([EmbeddedMemory(irrelevant, (-1, 0))])

    def handler(request):
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 0]]})
        assert "IRRELEVANT" not in request.content.decode()
        return httpx.Response(200, json=RESULT)

    app = create_app(
        config,
        database=database,
        vector_store=store,
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(handler)),
    )
    body, state = await deliver(app, payload())
    assert json.loads(body) == RESULT
    assert state["proxy_diagnostics"].retrieval_count == 0
    assert state["proxy_diagnostics"].injected_memory_tokens == 0
    assert state["proxy_diagnostics"].degraded_dependencies == ()


async def test_cancelled_chroma_query_closes_transport_and_records_incomplete_attempt(database):
    from local_dev_rag.vector_store import VectorStore

    waiting = anyio.Event()
    cleanup = []

    class BlockingQuery(httpx.AsyncByteStream):
        async def __aiter__(self):
            waiting.set()
            await anyio.sleep_forever()
            yield b"unreachable"

        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            cleanup.append("closed")

    config = settings()

    def chroma_handler(request):
        if request.url.path.endswith("/query"):
            return httpx.Response(200, stream=BlockingQuery())
        return httpx.Response(200, json={"id": str(uuid4())})

    def ollama_handler(request):
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json={"embeddings": [[1, 0]]})

    app = create_app(
        config,
        database=database,
        vector_store=VectorStore(config, transport=httpx.MockTransport(chroma_handler)),
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(ollama_handler)),
    )
    with anyio.fail_after(2):
        async with anyio.create_task_group() as group:
            group.start_soon(deliver, app, payload())
            await waiting.wait()
            group.cancel_scope.cancel()
    assert cleanup == ["closed"]
    async with database.session() as session:
        rows = (await session.execute(select(conversation_events))).mappings().all()
        assert rows[-1]["role"] == "proxy"
        assert rows[-1]["completed"] is False
        assert (await session.execute(select(memory_jobs))).all() == []


@pytest.mark.parametrize("floor", [None, 0], ids=["default-rejects", "explicitly-disabled"])
async def test_default_relevance_excludes_orthogonal_cosmetic_memory(database, chroma_url, floor):
    from local_dev_rag.vector_store import VectorStore

    overrides = {} if floor is None else {"ranking_weights": {"min_semantic_similarity": floor}}
    config = Settings(_env_file=None, chromadb_url=chroma_url, **overrides)
    store = VectorStore(config, collection_name=f"default-relevance-{uuid4()}")
    async with database.session() as session:
        scope = await ConversationRepository(session).ensure_scope(
            RequestIdentity("session", "project")
        )
    cosmetic = replace(memory(scope.project_id, "Prefer a blue sidebar"), importance=1)
    await store.upsert([EmbeddedMemory(cosmetic, (0, 1))])
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[1, 0]]})
        if floor is None:
            assert "Prefer a blue sidebar" not in request.content.decode()
        else:
            assert "Prefer a blue sidebar" in request.content.decode()
        return httpx.Response(200, json=RESULT)

    app = create_app(
        config,
        database=database,
        vector_store=store,
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(handler)),
    )
    request_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "ECONNREFUSED load_widget"}],
    }
    body, state = await deliver(app, request_payload)
    assert json.loads(body) == RESULT
    assert state["proxy_diagnostics"].retrieval_count == (0 if floor is None else 1)
    assert bool(state["proxy_diagnostics"].injected_memory_tokens) == (floor is not None)
    assert state["proxy_diagnostics"].degraded_dependencies == ()
    assert state["capture_status"] == "completed"
    assert calls == ["/api/embed", "/v1/chat/completions"]


@pytest.mark.parametrize("embedding_payload", [[], None, "invalid", 42])
async def test_nonobject_embedding_json_uses_recent_context_and_preserves_capture(
    database, embedding_payload
):
    from local_dev_rag.vector_store import VectorStore

    config = settings()
    calls = []

    def ollama_handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/embed":
            return httpx.Response(
                200,
                content=json.dumps(embedding_payload),
                headers={"content-type": "application/json"},
            )
        outbound = json.loads(request.content)
        assert outbound["messages"] == [payload()["messages"][0], payload()["messages"][-1]]
        return httpx.Response(200, json=RESULT)

    def chroma_handler(request):
        pytest.fail("Malformed embedding JSON must never reach Chroma")

    app = create_app(
        config,
        database=database,
        vector_store=VectorStore(config, transport=httpx.MockTransport(chroma_handler)),
        ollama_client=OllamaClient(config, transport=httpx.MockTransport(ollama_handler)),
    )
    body, state = await deliver(app, payload())
    assert json.loads(body) == RESULT
    assert state["proxy_diagnostics"].degraded_dependencies == ("embedder",)
    assert state["proxy_diagnostics"].retrieval_count == 0
    assert state["proxy_diagnostics"].injected_memory_tokens == 0
    assert state["capture_status"] == "completed"
    assert calls == ["/api/embed", "/v1/chat/completions"]
    async with database.session() as session:
        events = (await session.execute(select(conversation_events))).mappings().all()
        assert len(events) == 5
        assert events[1]["payload"]["content"] == "old " * 2000
        assert events[-1]["role"] == "assistant"
        assert events[-1]["completed"] is True
        assert len((await session.execute(select(memory_jobs))).all()) == 1
