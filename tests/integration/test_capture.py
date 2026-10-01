import asyncio
import json

import anyio
import httpx
import pytest
from sqlalchemy import select, text
from starlette.requests import ClientDisconnect

from local_dev_rag.api import create_app
from local_dev_rag.config import Settings
from local_dev_rag.db import Database
from local_dev_rag.ollama import OllamaClient
from local_dev_rag.schema import conversation_events, memory_jobs, sessions

from .capture_utils import Fragments, invoke_stream

HEADERS = {"x-opencode-session-id": "capture-session", "x-opencode-project-id": "capture-project"}
MODEL = "qwen3-coder:30b"
MESSAGE = {"role": "assistant", "content": "answer"}
RESULT = {
    "id": "upstream-1",
    "model": MODEL,
    "object": "chat.completion",
    "choices": [{"index": 0, "message": MESSAGE, "finish_reason": "stop"}],
}
SSE = (
    b'data: {"id":"upstream-1","choices":[{"index":0,"delta":{"role":"assistant",'
    b'"content":"answer"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
)


def capture_app(database, handler):
    settings = Settings(_env_file=None)
    app = create_app(
        settings,
        database=database,
        ollama_client=OllamaClient(settings, transport=httpx.MockTransport(handler)),
    )
    return app


async def post(app, messages=None, stream=False):
    payload = {
        "model": MODEL,
        "messages": messages if messages is not None else [{"role": "user", "content": "question"}],
        "stream": stream,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        return await client.post("/v1/chat/completions", json=payload, headers=HEADERS)


async def rows(database):
    async with database.session() as session:
        events = (
            (
                await session.execute(
                    select(conversation_events).order_by(conversation_events.c.sequence)
                )
            )
            .mappings()
            .all()
        )
        jobs = (await session.execute(select(memory_jobs))).mappings().all()
        return events, jobs


async def test_inbound_is_durable_before_upstream_and_success_is_enqueued_once(database):
    async def handler(request):
        events, jobs = await rows(database)
        assert [event["payload"] for event in events] == [{"role": "user", "content": "question"}]
        assert jobs == []
        return httpx.Response(200, json=RESULT)

    response = await post(capture_app(database, handler))
    assert response.json() == RESULT
    events, jobs = await rows(database)
    assert [event["role"] for event in events] == ["user", "assistant"]
    assert events[1]["payload"] == MESSAGE
    assert jobs[0]["source_event_id"] == events[1]["id"]


@pytest.mark.parametrize("stream", [False, True])
async def test_identical_retries_create_one_completion_and_job(database, stream):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        upstream_id = f"upstream-{attempts}"
        answer = f"answer {attempts}"
        result = {
            **RESULT,
            "id": upstream_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            content=(
                SSE.replace(b"upstream-1", upstream_id.encode()).replace(
                    b'"answer"', json.dumps(answer).encode()
                )
                if stream
                else json.dumps(result).encode()
            ),
            headers={"content-type": "text/event-stream" if stream else "application/json"},
        )

    app = capture_app(database, handler)
    await post(app, stream=stream)
    response = await post(app, stream=stream)
    if stream:
        assert b'"content":"answer 2"' in response.content
    else:
        assert response.json()["choices"][0]["message"]["content"] == "answer 2"
    events, jobs = await rows(database)
    assert [event["role"] for event in events] == ["user", "assistant"]
    assert events[1]["payload"]["content"] == "answer 1"
    assert len(jobs) == 1
    question = {"role": "user", "content": "question"}
    first_answer = {"role": "assistant", "content": "answer 1"}
    await post(app, [question, first_answer, question], stream=stream)
    events, jobs = await rows(database)
    assert [event["payload"] for event in events] == [
        question,
        first_answer,
        question,
        {"role": "assistant", "content": "answer 3"},
    ]
    assert len(jobs) == 2


async def test_growing_history_deduplicates_completion_echo_and_retains_repeated_turn(database):
    app = capture_app(database, lambda request: httpx.Response(200, json=RESULT))
    question = {"role": "user", "content": "question"}
    await post(app, [question])
    await post(app, [question, MESSAGE, question])
    events, jobs = await rows(database)
    assert [event["payload"] for event in events] == [question, MESSAGE, question, MESSAGE]
    assert len(jobs) == 2


@pytest.mark.parametrize(
    "chunks", [[SSE.split(b"data: [DONE]")[0]], [SSE, b"event: error\ndata: {}\n\n"]]
)
async def test_incomplete_or_failed_stream_records_attempt_without_completion(database, chunks):
    response = await post(
        capture_app(
            database,
            lambda request: httpx.Response(
                200, stream=Fragments(chunks), headers={"content-type": "text/event-stream"}
            ),
        ),
        stream=True,
    )
    assert response.content == b"".join(chunks)
    events, jobs = await rows(database)
    assert [event["role"] for event in events] == ["user", "proxy"]
    assert events[1]["event_type"] == "attempt"
    assert events[1]["payload"]["status"] == "incomplete"
    assert events[1]["completed"] is False
    assert jobs == []


@pytest.mark.parametrize("failure", ["send", "read", "disconnect"])
async def test_interrupted_delivery_records_diagnostic_without_completed_event(database, failure):
    disconnect = asyncio.Event()

    class Waiting(Fragments):
        async def __aiter__(self):
            yield SSE
            await asyncio.Future()

    fragments = (
        Waiting([])
        if failure == "disconnect"
        else Fragments([SSE], httpx.ReadError("broken") if failure == "read" else None)
    )
    app = capture_app(
        database,
        lambda request: httpx.Response(
            200, stream=fragments, headers={"content-type": "text/event-stream"}
        ),
    )

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            if failure == "send":
                raise OSError("disconnected")
            disconnect.set()

    if failure == "disconnect":
        await asyncio.wait_for(
            invoke_stream(app, send, asgi_version="2.3", disconnect=disconnect), 2
        )
    else:
        with pytest.raises(ClientDisconnect if failure == "send" else httpx.ReadError):
            await invoke_stream(app, send)
    events, jobs = await rows(database)
    assert [event["role"] for event in events] == ["proxy"]
    assert events[0]["completed"] is False
    assert events[0]["payload"]["status"] == "incomplete"
    assert jobs == []
    assert fragments.closed


@pytest.mark.parametrize("stream", [True, False])
async def test_prefetch_cancellation_persists_diagnostic_after_cleanup(database, stream):
    waiting = anyio.Event()
    cleanup = []

    class Blocking(httpx.AsyncByteStream):
        async def __aiter__(self):
            waiting.set()
            await anyio.sleep_forever()
            yield b"unreachable"

        async def aclose(self):
            await anyio.lowlevel.checkpoint()
            cleanup.append("closed")

    app = capture_app(database, lambda request: httpx.Response(200, stream=Blocking()))
    with anyio.fail_after(2):
        async with anyio.create_task_group() as group:
            group.start_soon(post, app, None, stream)
            await waiting.wait()
            group.cancel_scope.cancel()
    events, jobs = await rows(database)
    assert [event["role"] for event in events] == ["user", "proxy"]
    assert not events[1]["completed"]
    assert jobs == []
    assert cleanup == ["closed"]


async def test_postgres_connection_failure_forwards_the_entire_original_request():
    database = Database.create(
        Settings(
            database_url="postgresql+asyncpg://unused:unused@127.0.0.1:1/unused", _env_file=None
        )
    )
    messages = [{"role": "system", "content": "all instructions"}] + [
        {"role": "user", "content": f"turn {index}"} for index in range(20)
    ]

    def handler(request):
        assert json.loads(request.content)["messages"] == messages
        return httpx.Response(200, json=RESULT)

    try:
        delivered = []
        statuses = []

        async def send(message):
            if message["type"] == "http.response.body":
                delivered.append(message.get("body", b""))
            elif message["type"] == "http.response.start":
                statuses.append(message["status"])

        state = await invoke_stream(
            capture_app(database, handler),
            send,
            payload={"model": MODEL, "messages": messages, "stream": False},
        )
        assert statuses == [200]
        assert json.loads(b"".join(delivered)) == RESULT
        assert state["state"]["capture_status"] == "unavailable"
    finally:
        await database.engine.dispose()


async def test_cancelled_nonstream_finalization_records_attempt_after_transaction_rollback(
    database,
):
    locked = anyio.Event()
    async with database.engine.connect() as blocker:
        transaction = await blocker.begin()

        async def handler(request):
            await blocker.execute(select(sessions).with_for_update())
            locked.set()
            return httpx.Response(200, json=RESULT)

        app = capture_app(database, handler)
        with anyio.fail_after(3):
            async with anyio.create_task_group() as group:
                group.start_soon(post, app)
                await locked.wait()
                while True:
                    async with database.session() as session:
                        waiting = await session.scalar(
                            text(
                                "SELECT count(*) FROM pg_stat_activity "
                                "WHERE wait_event_type = 'Lock' "
                                "AND query LIKE '%sessions%FOR UPDATE%'"
                            )
                        )
                    if waiting:
                        break
                    await anyio.sleep(0.01)
                group.cancel_scope.cancel()
                with anyio.CancelScope(shield=True):
                    await transaction.rollback()
    events, jobs = await rows(database)
    assert [event["role"] for event in events] == ["user", "proxy"]
    assert jobs == []


@pytest.mark.parametrize("stream", [True, False])
async def test_upstream_context_cleanup_failure_records_incomplete_attempt(database, stream):
    class BrokenClose(httpx.MockTransport):
        async def aclose(self):
            raise httpx.CloseError("cleanup failed")

    settings = Settings(_env_file=None)
    app = create_app(
        settings,
        database=database,
        ollama_client=OllamaClient(
            settings,
            transport=BrokenClose(
                lambda request: httpx.Response(
                    200,
                    stream=Fragments([SSE if stream else json.dumps(RESULT).encode()]),
                    headers={"content-type": "text/event-stream" if stream else "application/json"},
                )
            ),
        ),
    )
    if stream:
        with pytest.raises(httpx.CloseError):
            await post(app, stream=True)
    else:
        response = await post(app)
        assert response.status_code == 502
    events, jobs = await rows(database)
    assert [event["role"] for event in events] == ["user", "proxy"]
    assert jobs == []


async def test_successful_assistant_tool_calls_are_classified_and_retained(database):
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-z",
                "type": "function",
                "function": {"name": "read", "arguments": '{ "path": "a" }'},
            },
            {
                "id": "call-a",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path":"b"}'},
            },
        ],
    }
    result = {
        **RESULT,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
    }
    await post(capture_app(database, lambda request: httpx.Response(200, json=result)))
    events, jobs = await rows(database)
    assert events[1]["event_type"] == "tool_call"
    assert events[1]["payload"] == message
    assert len(jobs) == 1


async def test_completion_database_failure_rolls_back_event_and_job_without_claiming_capture(
    database,
):
    async with database.session() as session:
        await session.execute(
            text(
                "CREATE FUNCTION reject_capture_job() RETURNS trigger LANGUAGE plpgsql "
                "AS $$ BEGIN RAISE EXCEPTION 'test storage failure'; END $$"
            )
        )
        await session.execute(
            text(
                "CREATE TRIGGER reject_capture_job BEFORE INSERT ON memory_jobs "
                "FOR EACH ROW EXECUTE FUNCTION reject_capture_job()"
            )
        )
    try:
        app = capture_app(
            database,
            lambda request: httpx.Response(
                200, stream=Fragments([SSE]), headers={"content-type": "text/event-stream"}
            ),
        )
        delivered = []

        async def send(message):
            if message["type"] == "http.response.body":
                delivered.append(message.get("body", b""))

        question = {"role": "user", "content": "question"}
        state = await invoke_stream(
            app, send, payload={"model": MODEL, "messages": [question], "stream": True}
        )
        assert b"".join(delivered) == SSE
        assert state["state"]["capture_status"] == "unavailable"
        events, jobs = await rows(database)
        assert [event["payload"] for event in events] == [question]
        assert jobs == []
    finally:
        async with database.session() as session:
            await session.execute(text("DROP TRIGGER reject_capture_job ON memory_jobs"))
            await session.execute(text("DROP FUNCTION reject_capture_job()"))
