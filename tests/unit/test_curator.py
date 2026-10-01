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


@pytest.mark.parametrize(
    "unsafe",
    [
        'Configuration uses {"password": "hunter2"}.',
        "Configuration uses {'api_key': 'private-value'}.",
        'Settings use {"access_token": "private-value"}.',
        'Nested settings use {"credentials": {"client_secret": "private-value"}}.',
        "The database URL is postgresql://postgres:hunter2@localhost:5432/db.",
        "The cache URL is redis://:hunter2@localhost:6379/0.",
        "The URL is https://user:pass%40word@example.test/api.",
    ],
)
async def test_structured_credentials_and_authenticated_urls_reject_only_unsafe_draft(unsafe):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text=unsafe), candidate()],
                }
            )
        )
    )
    assert await instance.extract(source()) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        "Redis is available. Maybe use Redis for caching later; no decision was made.",
        "We evaluated Redis but decided not to use it.",
        "Redis was evaluated and rejected.",
        "We decided to use Redis. We later decided not to use Redis.",
    ],
)
async def test_mentions_and_explicit_rejection_do_not_confirm_an_option(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis."), candidate()],
                }
            )
        )
    )
    incoming = source(evidence + " We decided to use PostgreSQL for durable memory.")
    assert await instance.extract(incoming) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        "Maybe use Redis for caching. We decided to use Redis.",
        "Redis is available. We selected Redis for caching.",
        "We evaluated Redis and chose Redis for caching.",
    ],
)
async def test_explicit_confirmation_allows_an_evaluated_or_speculative_option(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis.")],
                }
            )
        )
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use Redis.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "mention",
    [
        "Redis is available.",
        "We evaluated Redis.",
        "Redis was evaluated for durable memory.",
    ],
)
@pytest.mark.parametrize("decision_first", [False, True])
async def test_neutral_option_is_not_promoted_beside_a_confirmed_sibling(mention, decision_first):
    confirmed = "We decided to use PostgreSQL for durable memory."
    evidence = (confirmed + " " + mention) if decision_first else (mention + " " + confirmed)
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis."), candidate()],
                }
            )
        )
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        "We decided not to use Redis and will use PostgreSQL for durable memory.",
        "We will use PostgreSQL for durable memory and decided not to use Redis.",
        "We will use PostgreSQL for durable memory but rejected Redis for durable memory.",
        "We evaluated Redis but chose PostgreSQL for durable memory.",
        "Redis is available, but PostgreSQL was selected for durable memory.",
        "We chose PostgreSQL for durable memory while Redis was only evaluated.",
        "We evaluated Redis and will use PostgreSQL instead of Redis for durable memory.",
        "We selected PostgreSQL rather than Redis for durable memory.",
        "We decided against Redis and will use PostgreSQL for durable memory.",
        "We didn't choose Redis and will use PostgreSQL for durable memory.",
    ],
)
async def test_positive_and_negative_clauses_apply_only_to_their_own_option(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis."), candidate()],
                }
            )
        )
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        "We evaluated Redis and later selected Redis.",
        "Redis is available; we decided to use Redis.",
        "We selected Redis. Redis is available.",
        "We selected PostgreSQL and Redis for durable memory.",
    ],
)
async def test_explicit_selection_is_preserved_across_mentions_and_selected_option_lists(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis.")],
                }
            )
        )
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use Redis.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        "For durable memory we evaluated Redis. We decided to use PostgreSQL for durable memory.",
        "For durable memory, we evaluated Redis. We decided to use PostgreSQL for durable memory.",
        "We will use PostgreSQL for durable memory, not Redis.",
        "We decided not to use Redis and to use PostgreSQL for durable memory.",
        "We decided not to use Redis and use PostgreSQL for durable memory.",
        "We evaluated Redis and selected PostgreSQL as the best option for durable memory.",
        "We selected PostgreSQL as the best option for durable memory and rejected Redis.",
        "Redis was evaluated. "
        "PostgreSQL was selected as the best available option for durable memory.",
        "We must not use Redis and will use PostgreSQL for durable memory.",
        "We decided to not use Redis and to use PostgreSQL for durable memory.",
    ],
)
async def test_predicate_subject_scope_preserves_only_the_confirmed_sibling(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis."), candidate()],
                }
            )
        )
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        "Redis is available. We selected Redis as the best option.",
        "We evaluated Redis. We selected Redis as the best available option.",
        "We evaluated Redis and selected Redis as the best option.",
    ],
)
async def test_explicit_predicate_restores_the_option_despite_neutral_descriptive_words(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis.")],
                }
            )
        )
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use Redis.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "evidence",
    [
        "We decided to use PostgreSQL for durable memory.",  # Redis has no source evidence.
        "We haven't decided to use Redis.",
        "We selected neither Redis nor PostgreSQL.",
        "We evaluated Redis and maybe selected Redis as an option.",
    ],
)
async def test_unknown_or_ambiguous_selection_cannot_support_an_invented_decision(evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [candidate(text="Use Redis.")],
                }
            )
        )
    )
    assert await instance.extract(source(evidence)) == []


@pytest.mark.parametrize(
    "evidence",
    [
        "We decided to evaluate Redis for durable memory.",
        "We decided to benchmark Redis for durable memory.",
        "We agreed to investigate Redis for durable memory.",
        "We must evaluate Redis for durable memory.",
        "We may have selected Redis for durable memory.",
        "We probably selected Redis for durable memory.",
        "We would have selected Redis for durable memory.",
        "We will use either Redis or PostgreSQL for durable memory.",
        "We will use Redis or PostgreSQL for durable memory.",
        "We will use Redis, PostgreSQL or MongoDB for durable memory.",
        "We will use Redis, PostgreSQL, or MongoDB for durable memory.",
        "We selected Redis, if the benchmark succeeds.",
        "We decided to use Redis if the benchmark succeeds.",
        "We decided to evaluate MongoDB and use Redis for durable memory.",
        "We may have selected MongoDB and use Redis for durable memory.",
    ],
)
async def test_uncertain_or_exploratory_actions_cannot_affirm_option_subjects(evidence):
    instance = curator(
        lambda request: response(json.dumps({"memories": [candidate(text="Use Redis.")]}))
    )
    assert await instance.extract(source(evidence)) == []


@pytest.mark.parametrize(
    "evidence",
    [
        "We decided to evaluate Redis for durable memory.",
        "We may have selected Redis for durable memory.",
        "We will use either Redis or MongoDB for durable memory.",
    ],
)
@pytest.mark.parametrize("confirmed_first", [False, True])
async def test_ambiguous_option_does_not_suppress_an_independent_confirmation(
    evidence, confirmed_first
):
    confirmed = "We decided to use PostgreSQL for durable memory."
    content = f"{confirmed} {evidence}" if confirmed_first else f"{evidence} {confirmed}"
    instance = curator(
        lambda request: response(
            json.dumps({"memories": [candidate(text="Use Redis."), candidate()]})
        )
    )
    assert await instance.extract(source(content)) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Redis will replace PostgreSQL for durable memory.",
        "Redis and PostgreSQL will be used for durable memory.",
        "We selected PostgreSQL and Redis for durable memory.",
        "For durable memory, Redis replaces PostgreSQL.",
        "PostgreSQL is selected alongside Redis for durable memory.",
        "PostgreSQL will be replaced by Redis for durable memory.",
        "PostgreSQL is the database and Redis is the cache.",
        "PostgreSQL and Redis for durable memory.",
        "Use PostgreSQL with Redis for durable memory.",
    ],
)
async def test_every_decision_draft_requires_evidence_for_all_option_subjects(text):
    instance = curator(
        lambda request: response(json.dumps({"memories": [candidate(text=text), candidate()]}))
    )
    assert await instance.extract(source()) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
async def test_declarative_decision_subject_checks_do_not_depend_on_model_kind(kind):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(
                            kind=kind,
                            text="Redis will replace PostgreSQL for durable memory.",
                        ),
                        candidate(),
                    ]
                }
            )
        )
    )
    assert await instance.extract(source()) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "PostgreSQL was selected for durable memory.",
        "We chose PostgreSQL for durable memory.",
        "PostgreSQL will be used for durable memory.",
        "Durable memory will use PostgreSQL.",
        "PostgreSQL is the selected durable memory store.",
        "Choose PostgreSQL for durable memory.",
    ],
)
async def test_confirmed_subjects_allow_bounded_declarative_paraphrases(text):
    instance = curator(lambda request: response(json.dumps({"memories": [candidate(text=text)]})))
    assert await instance.extract(source()) == [MemoryDraft("decision", text, 0.9, 0.8)]


@pytest.mark.parametrize(
    "evidence",
    [
        "We selected PostgreSQL and may have selected Redis for durable memory.",
        "We will use PostgreSQL and will use either Redis or MongoDB for caching.",
        "We decided to evaluate Redis and will use PostgreSQL for durable memory.",
        "We selected PostgreSQL, Redis and MongoDB for durable memory.",
    ],
)
async def test_an_independent_confirmed_clause_or_selected_list_remains_usable(evidence):
    instance = curator(lambda request: response(json.dumps({"memories": [candidate()]})))
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "conditional",
    [
        "We decided to use PostgreSQL and use Redis for caching if the benchmark succeeds.",
        "We decided to use PostgreSQL and to use Redis if the benchmark succeeds.",
        "We selected PostgreSQL and use Redis, if the benchmark succeeds.",
        "We selected PostgreSQL and, if the benchmark succeeds, use Redis for caching.",
        "We selected PostgreSQL and if the benchmark succeeds, we will use Redis.",
        "If the benchmark succeeds, we will use Redis for durable memory.",
        "If the benchmark succeeds, we decided to use MongoDB and use Redis.",
        "Unless the benchmark fails, Redis was selected for durable memory.",
        "Provided that the benchmark succeeds, we will use Redis.",
        "Assuming the benchmark succeeds, we decided to use Redis.",
        "Once the benchmark succeeds, we will use Redis.",
        "We will use Redis provided that the benchmark succeeds.",
        "We decided to use PostgreSQL and use Redis unless the benchmark fails.",
        "We selected PostgreSQL and use Redis when the benchmark succeeds.",
        "We selected PostgreSQL and use Redis as long as the benchmark succeeds.",
        "Redis was selected, if the benchmark succeeds.",
    ],
)
async def test_conditional_scope_never_affirms_an_option_for_any_memory_kind(kind, conditional):
    instance = curator(
        lambda request: response(
            json.dumps({"memories": [candidate(kind=kind, text="Use Redis."), candidate()]})
        )
    )
    # An independent certainty must survive whichever clause path carries uncertainty.
    evidence = "We decided to use PostgreSQL for durable memory. " + conditional
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "text",
    [
        "Redis is used alongside PostgreSQL for durable memory.",
        "PostgreSQL is the database and Redis is the cache.",
        "PostgreSQL runs with Valkey.",
        "The PostgreSQL service depends on nats.",
        "PostgreSQL recovered after restarting MongoDB.",
    ],
)
async def test_every_kind_requires_grounding_for_every_content_subject(kind, text):
    instance = curator(
        lambda request: response(
            json.dumps({"memories": [candidate(kind=kind, text=text), candidate()]})
        )
    )
    assert await instance.extract(source()) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize(
    ("kind", "evidence", "text"),
    [
        (
            "requirement",
            "The project requires PostgreSQL to support transactional writes.",
            "PostgreSQL requires support for transactional writes.",
        ),
        ("constraint", "Redis has a 256 MB limit.", "Redis has a 256 MB limit."),
        ("preference", "The team prefers Ruff for linting.", "The team prefers Ruff for linting."),
        (
            "error",
            "PostgreSQL could not connect because port 5432 was busy.",
            "PostgreSQL could not connect because port 5432 was busy.",
        ),
        (
            "fix",
            "Restarting PostgreSQL fixed the connection failure.",
            "Restarting PostgreSQL fixed the connection failure.",
        ),
        (
            "outcome",
            "The PostgreSQL migration completed successfully.",
            "The PostgreSQL migration completed successfully.",
        ),
        (
            "decision",
            "We decided to use PostgreSQL for durable memory.",
            "PostgreSQL is the selected durable memory store.",
        ),
    ],
)
async def test_grounded_durable_facts_do_not_require_an_option_selection(kind, evidence, text):
    instance = curator(
        lambda request: response(json.dumps({"memories": [candidate(kind=kind, text=text)]}))
    )
    assert await instance.extract(source(evidence)) == [MemoryDraft(kind, text, 0.9, 0.8)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "evidence",
    [
        "PostgreSQL failed during startup.",
        "If the benchmark succeeds, Redis will recover PostgreSQL.",
        "Redis may recover PostgreSQL.",
    ],
)
async def test_factual_grounding_does_not_turn_mentions_or_conditions_into_usage(kind, evidence):
    instance = curator(
        lambda request: response(
            json.dumps({"memories": [candidate(kind=kind, text="Use Redis with PostgreSQL.")]})
        )
    )
    assert await instance.extract(source(evidence)) == []


@pytest.mark.parametrize(
    "evidence",
    [
        "We decided to use PostgreSQL and use Redis for caching if the benchmark succeeds.",
        "We selected PostgreSQL and, if the benchmark succeeds, use Redis for caching.",
        "We selected PostgreSQL, and if the benchmark succeeds, we will use Redis.",
        "We selected PostgreSQL and use Redis when the benchmark succeeds.",
        "If the benchmark succeeds, we will use Redis, but we selected PostgreSQL.",
        "We selected PostgreSQL and use Redis provided that the benchmark succeeds.",
    ],
)
async def test_a_certain_sibling_survives_without_an_earlier_confirmation(evidence):
    instance = curator(
        lambda request: response(
            json.dumps({"memories": [candidate(text="Use Redis."), candidate()]})
        )
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL for durable memory.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "evidence",
    [
        "If the benchmark succeeds, we will use PostgreSQL and Redis.",
        "We selected PostgreSQL and Redis if the benchmark succeeds.",
        "Provided that the benchmark succeeds, PostgreSQL and Redis are used.",
    ],
)
async def test_a_condition_on_a_shared_predicate_governs_every_option(kind, evidence):
    instance = curator(
        lambda request: response(
            json.dumps(
                {"memories": [candidate(kind=kind, text="Use Redis."), candidate(kind=kind)]}
            )
        )
    )
    assert await instance.extract(source(evidence)) == []


@pytest.mark.parametrize("kind", KINDS)
async def test_supported_facts_cannot_be_combined_into_an_unsupported_usage_claim(kind):
    instance = curator(
        lambda request: response(
            json.dumps({"memories": [candidate(kind=kind, text="Use Redis.")]})
        )
    )
    evidence = "We decided to use PostgreSQL. Redis restarted successfully."
    assert await instance.extract(source(evidence)) == []


@pytest.mark.parametrize("kind", KINDS)
async def test_fronted_condition_survives_nested_while(kind):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        "We selected PostgreSQL. If the benchmark succeeds while latency stays low, "
        "we will use Redis."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    ("evidence", "text"),
    [
        ("Redis is not used for caching.", "Redis is used for caching."),
        ("Redis is used for caching.", "Redis is not used for caching."),
    ],
)
async def test_factual_support_cannot_reverse_explicit_polarity(kind, evidence, text):
    instance = curator(
        lambda request: response(
            json.dumps(
                {"memories": [candidate(kind=kind, text=text), candidate(text="Use PostgreSQL.")]}
            )
        )
    )
    assert await instance.extract(source("We selected PostgreSQL. " + evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("text", ["Redis is used for caching.", "Redis is not used for caching."])
async def test_factual_support_retains_matching_positive_and_negative_polarity(kind, text):
    instance = curator(
        lambda request: response(json.dumps({"memories": [candidate(kind=kind, text=text)]}))
    )
    assert await instance.extract(source(text)) == [MemoryDraft(kind, text, 0.9, 0.8)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("connector", ["but", "while", "whereas", "however"])
async def test_fronted_condition_survives_nested_context_connectors(kind, connector):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        f"We selected PostgreSQL. If the benchmark succeeds {connector} latency remains high, "
        "we will use Redis."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("connector", ["but", "while", "whereas", "however"])
async def test_certain_sibling_after_governed_conditional_action_remains_usable(connector):
    instance = curator(
        lambda request: response(
            json.dumps(
                {"memories": [candidate(text="Use Redis."), candidate(text="Use PostgreSQL.")]}
            )
        )
    )
    evidence = f"If the benchmark succeeds, we will use Redis {connector} we selected PostgreSQL."
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
async def test_new_fronted_condition_resets_preceding_governed_action_boundary(kind):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        "We selected PostgreSQL. If the benchmark succeeds, we will use PostgreSQL "
        "and if throughput improves but latency remains high, we will use Redis."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


CONTRACTIONS = [
    ("isn't used", "is not used", "is used"),
    ("aren't used", "are not used", "are used"),
    ("wasn't used", "was not used", "was used"),
    ("weren't used", "were not used", "were used"),
    ("doesn't restart", "does not restart", "does restart"),
    ("don't restart", "do not restart", "do restart"),
    ("didn't restart", "did not restart", "did restart"),
    ("can't be used", "can not be used", "can be used"),
    ("won't be used", "will not be used", "will be used"),
    ("hasn't been used", "has not been used", "has been used"),
    ("haven't been used", "have not been used", "have been used"),
    ("hadn't been used", "had not been used", "had been used"),
    ("ISN’T used", "is not used", "is used"),
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("contracted", "negative", "positive"), CONTRACTIONS)
async def test_contracted_negation_never_supports_a_positive_fact(
    kind, contracted, negative, positive
):
    text = f"Redis {positive} for caching."
    instance = curator(
        lambda request: response(
            json.dumps(
                {"memories": [candidate(kind=kind, text=text), candidate(text="Use PostgreSQL.")]}
            )
        )
    )
    assert await instance.extract(
        source(f"We selected PostgreSQL. Redis {contracted} for caching.")
    ) == [MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("contracted", "negative", "positive"), CONTRACTIONS)
@pytest.mark.parametrize("contract_candidate", [False, True])
async def test_contracted_and_explicit_negative_facts_preserve_polarity(
    kind, contracted, negative, positive, contract_candidate
):
    text = f"Redis {contracted if contract_candidate else negative} for caching."
    evidence = f"Redis {negative if contract_candidate else contracted} for caching."
    instance = curator(
        lambda request: response(json.dumps({"memories": [candidate(kind=kind, text=text)]}))
    )
    assert await instance.extract(source(evidence)) == [MemoryDraft(kind, text, 0.9, 0.8)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(("contracted", "negative", "positive"), CONTRACTIONS)
async def test_positive_evidence_never_supports_a_contracted_negative_fact(
    kind, contracted, negative, positive
):
    text = f"Redis {contracted} for caching."
    instance = curator(
        lambda request: response(json.dumps({"memories": [candidate(kind=kind, text=text)]}))
    )
    assert await instance.extract(source(f"Redis {positive} for caching.")) == []


ANTECEDENT_PREDICATES = [
    "selected",
    "chose",
    "confirmed",
    "evaluated",
    "considered",
    "rejected",
    "did not use",
    "didn't use",
    "didn’t use",
]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("predicate", ANTECEDENT_PREDICATES)
@pytest.mark.parametrize("connector", ["and", "but", "while", "whereas", "however"])
async def test_predicates_inside_antecedent_cannot_consume_fronted_condition(
    kind, predicate, connector
):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        f"We selected PostgreSQL. If we {predicate} MongoDB {connector} latency remains high, "
        "we will use Redis."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("predicate", ANTECEDENT_PREDICATES)
async def test_second_antecedent_predicate_stays_inside_condition_until_consequent(kind, predicate):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(kind=kind, text="Use MongoDB."),
                        candidate(kind=kind, text="Use NATS."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        f"We selected PostgreSQL. If we {predicate} MongoDB and we confirmed NATS "
        "but latency remains high, we will use Redis."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("predicate", ["rejected", "did not use", "didn't use", "didn’t use"])
@pytest.mark.parametrize("connector", ["but", "while", "whereas", "however"])
async def test_antecedent_predicate_does_not_hide_certain_sibling_after_consequent(
    kind, predicate, connector
):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        f"If we {predicate} MongoDB but latency remains high, we will use Redis "
        f"{connector} we selected PostgreSQL."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("predicate", ANTECEDENT_PREDICATES)
async def test_unconditional_governed_action_remains_certain(kind, predicate):
    text = "Use Redis."
    instance = curator(
        lambda request: response(
            json.dumps(
                {"memories": [candidate(kind=kind, text=text), candidate(text="Use PostgreSQL.")]}
            )
        )
    )
    evidence = (
        f"We {predicate} MongoDB but latency remains high, and we will use Redis. "
        "We selected PostgreSQL."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft(kind, text, 0.9, 0.8),
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8),
    ]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("predicate", ["rejected", "did not use", "didn't use", "didn’t use"])
@pytest.mark.parametrize("context", ["For caching, ", "In this project, "])
async def test_context_before_fronted_condition_does_not_make_antecedent_a_consequent(
    kind, predicate, context
):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        f"We selected PostgreSQL. {context}if we {predicate} MongoDB but latency remains high, "
        "we will use Redis."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8)
    ]


@pytest.mark.parametrize("kind", KINDS)
async def test_trailing_condition_on_infinitive_keeps_independent_certain_sibling(kind):
    instance = curator(
        lambda request: response(
            json.dumps(
                {
                    "memories": [
                        candidate(kind=kind, text="Use Redis."),
                        candidate(kind=kind, text="Use NATS."),
                        candidate(text="Use PostgreSQL."),
                    ]
                }
            )
        )
    )
    evidence = (
        "We selected PostgreSQL and use Redis if the benchmark succeeds, but we selected NATS."
    )
    assert await instance.extract(source(evidence)) == [
        MemoryDraft(kind, "Use NATS.", 0.9, 0.8),
        MemoryDraft("decision", "Use PostgreSQL.", 0.9, 0.8),
    ]
