from dataclasses import replace

import pytest

from local_dev_rag.events import content_hash, normalize_messages


def test_mapping_key_order_does_not_change_event_identity():
    first = normalize_messages([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
    second = normalize_messages([{"content": [{"text": "hi", "type": "text"}], "role": "user"}])
    assert first[0].content_hash == second[0].content_hash
    assert content_hash(first[0]) == first[0].content_hash


def test_ordered_multimodal_parts_remain_exact_and_affect_hash():
    parts = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "high"}},
    ]
    event = normalize_messages([{"role": "user", "content": parts}])[0]
    assert event.payload["content"][0]["text"] == "describe"
    assert event.payload["content"][1]["image_url"]["url"] == "data:image/png;base64,AA"
    reversed_event = normalize_messages([{"role": "user", "content": list(reversed(parts))}])[0]
    assert event.content_hash != reversed_event.content_hash


def test_tool_calls_retain_order_ids_and_literal_arguments():
    calls = [
        {
            "id": "second",
            "type": "function",
            "function": {"name": "read", "arguments": '{ "b":2,"a":1 }'},
        },
        {"id": "first", "type": "function", "function": {"name": "read", "arguments": "{}"}},
    ]
    event = normalize_messages([{"role": "assistant", "content": None, "tool_calls": calls}])[0]
    assert event.event_type == "tool_call"
    assert event.payload["tool_calls"][0]["id"] == "second"
    assert event.payload["tool_calls"][0]["function"]["arguments"] == '{ "b":2,"a":1 }'
    assert event.payload["tool_calls"][1]["id"] == "first"
    reversed_event = normalize_messages(
        [{"role": "assistant", "content": None, "tool_calls": list(reversed(calls))}]
    )[0]
    assert event.content_hash != reversed_event.content_hash


@pytest.mark.parametrize(
    "role,kind",
    [
        ("system", "message"),
        ("developer", "message"),
        ("user", "message"),
        ("assistant", "message"),
        ("tool", "tool_result"),
        ("function", "tool_result"),
    ],
)
def test_role_classification_and_source_identity(role, kind):
    event = normalize_messages(
        [{"id": "message-1", "role": role, "content": "value", "tool_call_id": "call-1"}]
    )[0]
    assert (event.role, event.event_type, event.source_message_id) == (role, kind, "message-1")
    assert event.payload["tool_call_id"] == "call-1"


def test_repeated_occurrences_are_distinct_but_prefix_replays_are_stable():
    messages = [
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]
    events = normalize_messages(messages)
    assert events[0].content_hash != events[2].content_hash
    assert events[0].content_hash == normalize_messages(messages[:1])[0].content_hash
    assert [event.role for event in events] == ["user", "assistant", "user"]
    assert [event.content_hash for event in events] == [
        event.content_hash for event in normalize_messages(messages)
    ]
    swapped = normalize_messages([messages[1], messages[0], messages[2]])
    assert events[2].content_hash != swapped[2].content_hash


def test_transport_request_metadata_does_not_change_content_identity():
    event = normalize_messages([{"role": "user", "content": "hello"}])[0]
    assert content_hash(replace(event, request_id="retry", model="other")) == event.content_hash
