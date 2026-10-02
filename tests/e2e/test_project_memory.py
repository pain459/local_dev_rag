"""Real HTTP proxy/worker/PostgreSQL/Chroma with only model inference faked.

Breaks caught: missing job/capture/indexing, session-scoped rather than project-scoped
recall, cross-project leakage, model-dependent memory, or a split tool chain.
"""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.curator import Curator
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.schema import conversation_events, memory_items, memory_jobs, sessions
from local_dev_rag.vector_store import VectorStore
from local_dev_rag.worker import Worker

DECISION = "We selected PostgreSQL for durable memory."
QUERY = "Which database did we select for durable memory?"
MODELS = (
    "qwen3-coder:30b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "qwen2.5:7b",
)
TOOL = {
    "id": "call_database",
    "type": "function",
    "function": {"name": "read_config", "arguments": '{"path":"config.json"}'},
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_config",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]


@asynccontextmanager
async def http_server(app):
    """Start on an OS-assigned loopback port; close even on assertion failure."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(5):
                while not server.started:
                    if task.done():
                        await task
                        pytest.fail("HTTP fixture exited before startup")
                    await asyncio.sleep(0.01)
            yield f"http://127.0.0.1:{listener.getsockname()[1]}"
        finally:
            server.should_exit = True
            async with asyncio.timeout(5):
                await task


def fake_ollama():
    app = FastAPI()

    @app.post("/api/embed")
    async def embed(request: Request):
        body = await request.json()
        assert body["model"] == "nomic-embed-text:latest"
        return {"embeddings": [[1.0, 0.0, 0.0] for _ in body["input"]]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        # Only inference is fake; its answer depends on the *actual outbound* context.
        if body.get("response_format", {}).get("type") == "json_schema":
            assert body["model"] == "qwen2.5-coder:1.5b" and not body["stream"]
            content = json.dumps(
                {
                    "memories": [
                        {
                            "kind": "decision",
                            "text": "Use PostgreSQL for durable memory.",
                            "confidence": 0.95,
                            "importance": 0.9,
                        }
                    ]
                }
            )
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ]
            }
        messages = body["messages"]
        last = messages[-1]
        finish = "stop"
        if last["role"] == "tool":
            assert body["tools"] == TOOLS and body["tool_choice"] == "auto"
            assert messages[-2] == {"role": "assistant", "content": None, "tool_calls": [TOOL]}
            assert last == {"role": "tool", "tool_call_id": TOOL["id"], "content": "postgres"}
            message = {"role": "assistant", "content": "tool round trip complete"}
        elif body.get("tool_choice") == "required":
            assert body["tools"] == TOOLS
            message = {"role": "assistant", "content": None, "tool_calls": [TOOL]}
            finish = "tool_calls"
        else:
            historical = "\n".join(
                m.get("content", "") or "" for m in messages if m["role"] == "system"
            )
            content = (
                DECISION
                if last["content"] == DECISION
                else (
                    "PostgreSQL"
                    if "Use PostgreSQL for durable memory." in historical
                    else "unknown"
                )
            )
            message = {"role": "assistant", "content": content}
        if body.get("stream"):
            delta = dict(message)
            if "tool_calls" in message:
                delta["tool_calls"] = [{"index": 0, **TOOL}]
            chunk = {
                "id": "fake-stream",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish,
                    }
                ],
            }
            data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()

            async def fragments():
                for offset in range(0, len(data), 17):
                    yield data[offset : offset + 17]
                    await asyncio.sleep(0)

            return StreamingResponse(fragments(), media_type="text/event-stream")
        return {
            "id": f"fake-{uuid4()}",
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                }
            ],
        }

    return app


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_cross_session_memory_model_switch_isolation_and_tool_round_trip(
    database,
    chroma_url,
    stream,
):
    async with http_server(fake_ollama()) as ollama_url:
        settings = Settings(_env_file=None, ollama_url=ollama_url, chromadb_url=chroma_url)
        store = VectorStore(settings, collection_name=f"e2e-{uuid4()}")
        app = create_app(settings, database=database, vector_store=store)
        async with (
            http_server(app) as proxy_url,
            httpx.AsyncClient(
                base_url=proxy_url,
                timeout=10,
            ) as client,
        ):

            async def complete(project, session, model, messages, **extra):
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "x-opencode-project-id": project,
                        "x-opencode-session-id": session,
                    },
                    json={"model": model, "messages": messages, "stream": stream, **extra},
                )
                assert response.status_code == 200, response.text
                if not stream:
                    return response.json()["choices"][0]["message"]
                assert response.headers["content-type"].startswith("text/event-stream")
                assert response.text.endswith("data: [DONE]\n\n")
                chunks = [
                    json.loads(line[6:])
                    for line in response.text.splitlines()
                    if line.startswith("data: {")
                ]
                assert chunks[0]["model"] == model
                message = chunks[0]["choices"][0]["delta"]
                for tool in message.get("tool_calls", []):
                    assert tool.pop("index") == 0
                return message

            first = [{"role": "user", "content": DECISION}]
            assert (await complete("project-a", "first", MODELS[0], first))["content"] == DECISION

            # HTTP delivery ends before the server's shielded stream finalization.
            # Wait for the durable effect, rather than racing post-delivery cleanup.
            async def wait_jobs(count):
                async with asyncio.timeout(5):
                    while True:
                        async with database.session() as session:
                            jobs = (await session.execute(select(memory_jobs))).mappings().all()
                        if len(jobs) == count:
                            return jobs
                        await asyncio.sleep(0.01)

            jobs = await wait_jobs(1)
            assert jobs[0]["status"] == "pending"
            # A fresh Worker uses the durable source, not any request-local state.
            inference = OllamaClient(settings)
            result = await Worker(
                database,
                settings,
                curator=Curator(inference, settings),
                embedder=inference,
                vector_store=store,
            ).run_once()
            assert (result.state, result.memory_count) == ("completed", 1)
            async with database.session() as session:
                [item] = (await session.execute(select(memory_items))).mappings().all()
                [job] = (await session.execute(select(memory_jobs))).mappings().all()
                assert job["status"] == "completed" and job["attempt_count"] == 1
                assert item["source_event_id"] == job["source_event_id"]
                assert item["embedding_model"] == "nomic-embed-text:latest"
                assert item["state"] == "active"
            assert [
                hit.memory.id for hit in await store.query(item["project_id"], [1, 0, 0], 5)
            ] == [item["id"]]
            query = [{"role": "user", "content": QUERY}]
            # All five foreground models recall across fresh sessions.
            for index, model in enumerate(MODELS):
                assert (await complete("project-a", f"recall-{index}", model, query))[
                    "content"
                ] == "PostgreSQL"
                # Identical query and session ID must not cross the project boundary.
                assert (await complete("project-b", f"recall-{index}", model, query))[
                    "content"
                ] == "unknown"
            tool_prompt = [{"role": "user", "content": "Read the database configuration"}]
            call = await complete(
                "project-a", "tools", MODELS[2], tool_prompt, tools=TOOLS, tool_choice="required"
            )
            assert call == {"role": "assistant", "content": None, "tool_calls": [TOOL]}
            chain = [
                *tool_prompt,
                call,
                {"role": "tool", "tool_call_id": TOOL["id"], "content": "postgres"},
            ]
            assert (
                await complete(
                    "project-a", "tools", MODELS[2], chain, tools=TOOLS, tool_choice="auto"
                )
            )["content"] == "tool round trip complete"
            await wait_jobs(13)
            async with database.session() as session:
                events = (await session.execute(select(conversation_events))).mappings().all()
                jobs = (await session.execute(select(memory_jobs))).mappings().all()
                scopes = (await session.execute(select(sessions))).mappings().all()
                assert len(jobs) == 13  # initial + ten recall + two tool completions
                assert len(scopes) == 12
                assert all(event["completed"] for event in events)
                tool_events = [e for e in events if e["event_type"] == "tool_result"]
                assert (
                    len(tool_events) == 1
                    and tool_events[0]["payload"]["tool_call_id"] == TOOL["id"]
                )
            # Missing identity is rejected before any recall can occur.
            rejected = await client.post(
                "/v1/chat/completions", json={"model": MODELS[0], "messages": query}
            )
            assert rejected.status_code == 400
