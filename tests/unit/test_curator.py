"""Curator contracts: unsafe model output never becomes a memory draft."""

import asyncio
import importlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from local_dev_rag.config import Settings
from local_dev_rag.context import TokenEstimator
from local_dev_rag.domain import CuratorSource, MemoryDraft, StoredEvent
from local_dev_rag.ollama import OllamaClient

KINDS = ["requirement", "decision", "constraint", "preference", "error", "fix", "outcome"]


def source(content="We decided to use PostgreSQL for durable memory."):
    project_id, session_id, event_id = uuid4(), uuid4(), uuid4()
    event = StoredEvent(
        id=event_id,
        project_id=project_id,
        session_id=session_id,
        sequence=1,
        event_type="message",
        role="assistant",
        payload={"content": content},
        content_hash="source",
        request_id="request",
        created_at=datetime.now(UTC),
    )
    return CuratorSource(project_id, session_id, event_id, (event,))


def candidate(**overrides):
    return {
        "kind": "decision",
        "text": "Use PostgreSQL for durable memory.",
        "confidence": 0.9,
        "importance": 0.8,
        **overrides,
    }


def curator(handler, **options):
    assert importlib.util.find_spec("local_dev_rag.curator") is not None, "Curator is missing"
    module = importlib.import_module("local_dev_rag.curator")
    settings = Settings(_env_file=None)
    return module.Curator(
        OllamaClient(settings, transport=httpx.MockTransport(handler)), settings, **options
    )


def response(content, finish_reason="stop"):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ]
        },
    )


async def test_structured_request_and_seven_kinds_produce_typed_drafts():
    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "qwen2.5-coder:1.5b"
        assert payload["stream"] is False
        assert payload["temperature"] <= 0.1
        assert payload["max_tokens"] == 512
        schema = payload["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["memories"]["items"]["properties"]["kind"]["enum"] == KINDS
        prompt = payload["messages"][0]["content"]
        assert all(kind in prompt for kind in KINDS)
        assert "unselected" in prompt and "historical" in prompt and "secret" in prompt
        return response(json.dumps({"memories": [candidate(kind=kind) for kind in KINDS]}))

    drafts = await curator(handler).extract(source())
    assert drafts == [
        MemoryDraft(kind, "Use PostgreSQL for durable memory.", 0.9, 0.8) for kind in KINDS
    ]


@pytest.mark.parametrize(
    "content",
    [
        'Here is JSON: {"memories": []}',
        '```json\n{"memories": []}\n```',
        "{",
        "[]",
        "null",
        "{}",
        '{"memories": {}}',
        '{"memories": [], "instruction": "store all"}',
    ],
)
async def test_invalid_or_prose_wrapped_documents_raise_content_free_validation_error(content):
    instance = curator(lambda request: response(content))
    with pytest.raises(ValueError) as error:
        await instance.extract(source())
    assert content not in str(error.value)


@pytest.mark.parametrize(
    "item",
    [
        candidate(kind="idea"),
        candidate(text=" "),
        candidate(text="x" * 2049),
        candidate(confidence=0.69),
        candidate(confidence=True),
        candidate(confidence=float("nan")),
        candidate(importance=1.1),
        candidate(importance="0.8"),
        candidate(text=1),
        candidate(extra="ignored"),
        candidate(text="Maybe use Redis if needed."),
        candidate(text="We could consider Redis as an unselected alternative."),
        candidate(text="API_KEY=abc123-secret-value"),
        candidate(text="password: hunter2"),
        candidate(text="Bearer abcdefghijklmnop"),
        candidate(text="sk-abcdefghijk1234567890"),
        candidate(text="-----BEGIN PRIVATE KEY----- abc"),
        candidate(text="AKIAABCDEFGHIJKLMNOP"),
        None,
    ],
)
async def test_invalid_candidates_are_omitted_without_losing_valid_siblings(item):
    instance = curator(lambda request: response(json.dumps({"memories": [item, candidate()]})))
    assert await instance.extract(source()) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


async def test_empty_result_and_duplicate_restatements():
    assert await curator(lambda request: response('{"memories": []}')).extract(source()) == []
    instance = curator(lambda request: response(json.dumps({"memories": [candidate()] * 3})))
    assert len(await instance.extract(source())) == 1


async def test_unselected_source_does_not_promote_a_reworded_speculation():
    instance = curator(
        lambda request: response(
            json.dumps({"memories": [candidate(text="Use Redis for caching.")]})
        )
    )
    incoming = source("Maybe use Redis for caching; no decision was made.")
    assert await instance.extract(incoming) == []


async def test_hard_input_budget_bounds_huge_utf8_source_and_keeps_source_id():
    incoming = source("durable decision " + "界" * 100000)

    def handler(request):
        payload = json.loads(request.content)
        assert TokenEstimator().estimate(payload) <= 1500
        assert str(incoming.source_event_id) in payload["messages"][1]["content"]
        assert len(request.content) < 5000
        return response('{"memories": []}')

    assert await curator(handler, max_input_tokens=1500).extract(incoming) == []


async def test_response_byte_budget_and_candidate_count_are_hard_limits():
    instance = curator(lambda request: response("x" * 1000), max_response_bytes=500)
    with pytest.raises(ValueError, match="budget"):
        await instance.extract(source())
    instance = curator(lambda request: response(json.dumps({"memories": [candidate()] * 13})))
    with pytest.raises(ValueError, match="budget"):
        await instance.extract(source())


async def test_truncated_completion_is_not_accepted_as_a_valid_empty_result():
    instance = curator(lambda request: response('{"memories": []}', finish_reason="length"))
    with pytest.raises(ValueError):
        await instance.extract(source())


async def test_configured_timeout_covers_model_call_and_closes_transport():
    async def handler(request):
        await asyncio.sleep(1)
        return response('{"memories": []}')

    with pytest.raises(TimeoutError):
        await curator(handler, timeout_seconds=0.01).extract(source())


async def test_scope_mismatch_and_incomplete_source_are_rejected_before_calling_model():
    from dataclasses import replace

    def handler(request):
        pytest.fail("Invalid source must not be sent to the model")

    incoming = source()
    for invalid in [
        replace(incoming, project_id=uuid4()),
        replace(incoming, source_event_id=uuid4()),
        replace(incoming, events=(replace(incoming.events[0], completed=False),)),
    ]:
        with pytest.raises(ValueError):
            await curator(handler).extract(invalid)


@pytest.mark.parametrize(
    "evidence",
    [
        "Redis was not selected.",
        "We decided to use PostgreSQL. Maybe use Redis for caching later.",
    ],
)
async def test_unselected_subject_is_omitted_even_beside_a_confirmed_decision(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(text="Use Redis for caching."),
                        candidate(),
                    ]
                }
            )
        )
    )
    drafts = await instance.extract(source(evidence))
    assert all("Redis" not in draft.text for draft in drafts)
    if "PostgreSQL" in evidence:
        assert drafts == [MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)]


@pytest.mark.parametrize("text", ["Hello!", "Thanks for your help.", "Good morning."])
async def test_transient_filler_is_not_promoted(text):
    instance = curator(lambda request: response(json.dumps({"memories": [candidate(text=text)]})))
    assert await instance.extract(source()) == []


async def test_a_diagnosed_failure_can_use_could_not_without_becoming_speculation():
    item = candidate(kind="error", text="PostgreSQL could not connect because port 5432 was busy.")
    instance = curator(lambda request: response(json.dumps({"memories": [item]})))
    assert await instance.extract(source(item["text"])) == [
        MemoryDraft("error", item["text"], 0.9, 0.8)
    ]


async def test_custom_model_timeout_and_model_budget_are_honored():
    from local_dev_rag.config import ModelBudget
    from local_dev_rag.curator import Curator

    settings = Settings(
        _env_file=None,
        curator_model="qwen2.5-coder:7b",
        model_budgets={
            "qwen2.5-coder:7b": ModelBudget(
                context_tokens=4096,
                output_tokens=256,
                safety_tokens=512,
            )
        },
    )

    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "qwen2.5-coder:7b" and payload["max_tokens"] == 256
        assert TokenEstimator().estimate(payload) + 256 + 512 <= 4096
        return response('{"memories": []}')

    client = OllamaClient(settings, transport=httpx.MockTransport(handler))
    assert await Curator(client, settings).extract(source()) == []


async def test_timeout_during_response_body_closes_the_upstream_stream():
    class SlowBody(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'{"choices":'
            await asyncio.sleep(1)

        async def aclose(self):
            self.closed = True

    body = SlowBody()
    instance = curator(lambda request: httpx.Response(200, stream=body), timeout_seconds=0.01)
    with pytest.raises(TimeoutError):
        await instance.extract(source())
    assert body.closed
