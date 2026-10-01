import asyncio
import json

import anyio
import httpx
import pytest
from starlette.requests import ClientDisconnect

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.streaming import StreamAccumulator, relay_stream

HEADERS = {"x-opencode-session-id": "session", "x-opencode-project-id": "project"}
MODEL = "qwen3-coder:30b"


class Fragments(httpx.AsyncByteStream):
    def __init__(self, chunks, failure=None):
        self.chunks = chunks
        self.failure = failure
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.failure:
            raise self.failure

    async def aclose(self):
        self.closed = True


def app_for(handler):
    settings = Settings(_env_file=None)
    return create_app(
        settings, ollama_client=OllamaClient(settings, transport=httpx.MockTransport(handler))
    )


async def post(app, payload, headers=HEADERS):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post("/v1/chat/completions", json=payload, headers=headers)


async def test_nonstream_preserves_model_tools_sampling_and_structured_response():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "read file"}],
        "stream": False,
        "temperature": 0.2,
        "stop": ["END"],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        "response_format": {"type": "json_object"},
    }
    result = {
        "id": "chat-a",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content) == payload
        return httpx.Response(
            201, json=result, headers={"x-request-id": "upstream-a", "connection": "close"}
        )

    response = await post(app_for(handler), payload)
    assert response.status_code == 201
    assert response.json() == result
    assert response.headers["x-request-id"] == "upstream-a"
    assert "connection" not in response.headers


async def test_stream_preserves_raw_byte_order_and_closes_upstream():
    fragments = Fragments(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"',
            b"\xe2",
            b'\x82\xac"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    response = await post(
        app_for(
            lambda request: httpx.Response(
                200, stream=fragments, headers={"content-type": "text/event-stream"}
            )
        ),
        {"model": MODEL, "messages": [], "stream": True},
    )
    assert response.content == (
        b'data: {"choices":[{"index":0,"delta":{"content":"'
        b'\xe2\x82\xac"},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    assert response.headers["content-type"].startswith("text/event-stream")
    assert fragments.closed


@pytest.mark.parametrize(
    "payload,headers,param",
    [
        ({"model": "unknown", "messages": []}, HEADERS, "model"),
        ({"messages": []}, HEADERS, "model"),
        ({"model": MODEL, "messages": []}, {}, "x-opencode-session-id"),
        ({"model": MODEL}, HEADERS, "messages"),
    ],
)
async def test_invalid_requests_are_rejected_before_upstream(payload, headers, param):
    def handler(request):
        pytest.fail("invalid request reached upstream")

    response = await post(app_for(handler), payload, headers)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == param


@pytest.mark.parametrize("failure", [httpx.ConnectError("offline"), httpx.ReadTimeout("slow")])
async def test_upstream_connection_failures_use_openai_502_envelope(failure):
    def handler(request):
        raise failure

    response = await post(app_for(handler), {"model": MODEL, "messages": []})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"
    assert response.json()["error"]["message"]


async def test_ollama_model_error_preserves_status_and_wraps_native_error():
    response = await post(
        app_for(lambda request: httpx.Response(404, json={"error": "model missing"})),
        {"model": MODEL, "messages": []},
    )
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "model missing"
    assert response.json()["error"]["type"] == "upstream_error"


async def test_upstream_error_preserves_safe_headers_and_completes_error_envelope():
    response = await post(
        app_for(
            lambda request: httpx.Response(
                429,
                json={"error": {"message": "busy", "code": "rate_limit"}},
                headers={"retry-after": "5", "connection": "x-private", "x-private": "hidden"},
            )
        ),
        {"model": MODEL, "messages": []},
    )
    assert response.status_code == 429
    assert response.headers["retry-after"] == "5"
    assert "x-private" not in response.headers
    assert response.json()["error"] == {
        "message": "busy",
        "type": "upstream_error",
        "param": None,
        "code": "rate_limit",
    }


@pytest.mark.parametrize("failure", [httpx.ReadError("broken"), asyncio.CancelledError()])
async def test_broken_or_cancelled_relay_never_produces_completed_record(failure):
    accumulator = StreamAccumulator(request_id="failed")
    fragments = Fragments(
        [
            b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ],
        failure,
    )
    with pytest.raises(type(failure)):
        async for _ in relay_stream(fragments.__aiter__(), accumulator):
            pass
    assert accumulator.completion() is None


async def test_abandoned_relay_does_not_produce_completed_record():
    accumulator = StreamAccumulator(request_id="abandoned")
    fragments = Fragments(
        [
            b'data: {"choices":[{"delta":{"content":"answer"},'
            b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            b": later\n\n",
        ]
    )
    relay = relay_stream(fragments.__aiter__(), accumulator)
    await anext(relay)
    await relay.aclose()
    assert accumulator.completion() is None


async def test_embed_preserves_input_order_and_uses_native_batch_endpoint():
    def handler(request):
        assert request.url.path == "/api/embed"
        assert json.loads(request.content) == {
            "model": "nomic-embed-text:latest",
            "input": ["first", "second"],
        }
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    client = OllamaClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    assert await client.embed("nomic-embed-text:latest", ["first", "second"]) == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


@pytest.mark.parametrize("status,want", [(200, "healthy"), (503, "unavailable")])
async def test_health_reports_upstream_availability(status, want):
    def handler(request):
        assert request.url.path == "/api/tags"
        return httpx.Response(status, json={"models": []})

    client = OllamaClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    result = await client.health()
    assert result.name == "ollama"
    assert result.state == want


async def test_failure_before_first_stream_fragment_returns_502():
    fragments = Fragments([], httpx.ReadError("closed before response body"))
    response = await post(
        app_for(
            lambda request: httpx.Response(
                200, stream=fragments, headers={"content-type": "text/event-stream"}
            )
        ),
        {"model": MODEL, "messages": [], "stream": True},
    )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"
    assert fragments.closed


@pytest.mark.parametrize("stream", [True, False], ids=["stream-prefetch", "nonstream-read"])
async def test_cancelled_body_read_finishes_checkpointed_upstream_cleanup(stream):
    # Without shielding the route's exit stack, cancellation interrupts aclose().
    waiting = anyio.Event()
    cleanup = []

    class Checkpointed(httpx.AsyncByteStream):
        async def __aiter__(self):
            waiting.set()
            await anyio.sleep_forever()
            yield b"unreachable"

        async def aclose(self):
            cleanup.append("started")
            await anyio.lowlevel.checkpoint()
            cleanup.append("finished")

    app = app_for(
        lambda request: httpx.Response(
            200, stream=Checkpointed(), headers={"content-type": "text/event-stream"}
        )
    )
    with anyio.fail_after(1):
        async with anyio.create_task_group() as group:
            group.start_soon(post, app, {"model": MODEL, "messages": [], "stream": stream})
            await waiting.wait()
            group.cancel_scope.cancel()

    assert cleanup == ["started", "finished"]


async def invoke_stream(app, send, state=None, *, asgi_version="2.4", disconnect=None):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": asgi_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "server": ("proxy", 80),
        "client": ("test", 123),
        "state": {} if state is None else state,
        "headers": [(key.encode(), value.encode()) for key, value in HEADERS.items()]
        + [(b"content-type", b"application/json")],
    }

    request_read = False

    async def receive():
        nonlocal request_read
        if request_read:
            if disconnect is not None:
                await disconnect.wait()
                return {"type": "http.disconnect"}
            await asyncio.Future()
        request_read = True
        return {
            "type": "http.request",
            "body": json.dumps({"model": MODEL, "messages": [], "stream": True}).encode(),
            "more_body": False,
        }

    await app(scope, receive, send)
    return scope


async def test_stream_delivers_first_fragment_before_waiting_for_upstream_remainder():
    delivered = asyncio.Event()

    class Gated(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"early"}}]}\n\n'
            await asyncio.wait_for(delivered.wait(), 1)
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

    bodies = []

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            bodies.append(message["body"])
            delivered.set()

    scope = await invoke_stream(
        app_for(
            lambda request: httpx.Response(
                200, stream=Gated(), headers={"content-type": "text/event-stream"}
            )
        ),
        send,
    )
    assert bodies[0] == b'data: {"choices":[{"delta":{"content":"early"}}]}\n\n'
    assert scope["state"]["stream_accumulator"].completion().payload["content"] == "early"


async def test_client_disconnect_during_done_fragment_invalidates_completion_and_closes_upstream():
    fragments = Fragments(
        [
            b'data: {"choices":[{"delta":{"content":"answer"},'
            b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        ]
    )
    app = app_for(
        lambda request: httpx.Response(
            200, stream=fragments, headers={"content-type": "text/event-stream"}
        )
    )
    state = {}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("client disconnected")

    with pytest.raises(ClientDisconnect):
        await invoke_stream(app, send, state)
    assert fragments.closed
    assert state["stream_accumulator"].completion() is None


async def test_asgi_disconnect_cancels_upstream_and_invalidates_completion():
    disconnect = asyncio.Event()

    class Waiting(Fragments):
        async def __aiter__(self):
            yield (
                b'data: {"choices":[{"delta":{"content":"answer"},'
                b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
            )
            await asyncio.Future()

    fragments = Waiting([])

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            disconnect.set()

    state = await asyncio.wait_for(
        invoke_stream(
            app_for(
                lambda request: httpx.Response(
                    200, stream=fragments, headers={"content-type": "text/event-stream"}
                )
            ),
            send,
            asgi_version="2.3",
            disconnect=disconnect,
        ),
        1,
    )
    assert fragments.closed
    assert state["state"]["stream_accumulator"].completion() is None


async def test_actual_broken_response_closes_upstream_and_invalidates_completion():
    fragments = Fragments(
        [
            b'data: {"choices":[{"delta":{"content":"answer"},'
            b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        ],
        httpx.ReadError("socket closed"),
    )
    state = {}

    async def send(message):
        pass

    with pytest.raises(httpx.ReadError):
        await invoke_stream(
            app_for(
                lambda request: httpx.Response(
                    200, stream=fragments, headers={"content-type": "text/event-stream"}
                )
            ),
            send,
            state,
        )
    assert fragments.closed
    assert state["stream_accumulator"].completion() is None


@pytest.mark.parametrize("vectors", [[], [[True]], [["text"]], [[]], [[float("nan")]]])
async def test_invalid_embedding_batch_is_rejected(vectors):
    client = OllamaClient(
        Settings(_env_file=None),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                content=json.dumps({"embeddings": vectors}),
                headers={"content-type": "application/json"},
            )
        ),
    )
    with pytest.raises(ValueError):
        await client.embed("nomic-embed-text:latest", ["input"])


async def test_health_connection_failure_reports_unavailable_detail():
    def handler(request):
        raise httpx.ConnectError("offline")

    client = OllamaClient(Settings(_env_file=None), transport=httpx.MockTransport(handler))
    result = await client.health()
    assert result.state == "unavailable"
    assert result.detail
