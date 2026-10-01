"""Lossless message normalization with ordered-prefix occurrence identities."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

from local_dev_rag.domain import ConversationEventInput, InvalidRequestError


def json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in cast(Mapping[str, object], value).items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in cast(Sequence[object], value)]
    return value


def content_hash(event: ConversationEventInput) -> str:
    """Key order is representation; array order and the preceding history are identity."""
    value = {"parent": event.parent_hash, "payload": json_value(event.payload)}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def normalize_messages(
    messages: Sequence[Mapping[str, object]],
) -> list[ConversationEventInput]:
    events: list[ConversationEventInput] = []
    parent = ""
    for value in cast(Sequence[object], messages):
        if not isinstance(value, Mapping):
            raise InvalidRequestError("Each message must be an object with a role", "messages")
        message = cast(Mapping[str, object], value)
        if not isinstance(message.get("role"), str):
            raise InvalidRequestError("Each message must be an object with a role", "messages")
        role = cast(str, message["role"])
        kind = "message"
        if role in {"tool", "function"}:
            kind = "tool_result"
        elif role == "assistant" and (message.get("tool_calls") or message.get("function_call")):
            kind = "tool_call"
        source = message.get("id")
        event = ConversationEventInput(
            event_type=kind,
            role=role,
            payload=message,
            content_hash="",
            request_id="",
            source_message_id=source if isinstance(source, str) else None,
            parent_hash=parent,
        )
        parent = content_hash(event)
        events.append(replace(event, content_hash=parent, request_id=parent))
    return events
