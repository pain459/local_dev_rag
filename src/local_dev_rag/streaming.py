"""Incremental SSE observation; returned fragments are always the original bytes."""

import codecs
import hashlib
import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast
from uuid import uuid4

from local_dev_rag.domain import AssistantCompletion


async def relay_stream(
    body: AsyncIterator[bytes], accumulator: "StreamAccumulator"
) -> AsyncGenerator[bytes]:
    exhausted = False
    try:
        async for chunk in body:
            yield accumulator.feed(chunk)
        exhausted = True
    finally:
        if not exhausted:
            accumulator.abort()


class StreamAccumulator:
    def __init__(self, request_id: str | None = None, model: str | None = None):
        self.request_id = request_id or str(uuid4())
        self.model = model
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._line = ""
        self._after_cr = False
        self._data: list[str] = []
        self._event = ""
        self._content: list[str] = []
        self._tools: dict[int, dict[str, Any]] = {}
        self._source_id: str | None = None
        self._finished = False
        self._done = False
        self._failed = False

    def abort(self) -> None:
        """An interrupted transport must never become a completed memory source."""
        self._failed = True

    def feed(self, chunk: bytes) -> bytes:
        if self._failed:
            return chunk
        try:
            text = self._decoder.decode(chunk)
            for character in text:
                if self._after_cr and character == "\n":
                    self._after_cr = False
                    continue
                self._after_cr = character == "\r"
                if character in "\r\n":
                    self._consume_line(self._line)
                    self._line = ""
                else:
                    self._line += character
        except (UnicodeError, ValueError, TypeError, KeyError, AttributeError):
            self.abort()
        return chunk

    def _consume_line(self, line: str) -> None:
        if not line:
            self._consume_event()
            self._data = []
            self._event = ""
        elif not line.startswith(":"):
            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "data":
                self._data.append(value)
            elif field == "event":
                self._event = value

    def _consume_event(self) -> None:
        if self._event == "error":
            self.abort()
        if not self._data or self._failed:
            return
        data = "\n".join(self._data)
        if data == "[DONE]":
            self._done = True
            return
        parsed: object = json.loads(data)
        if not isinstance(parsed, dict) or "error" in parsed:
            self.abort()
            return
        payload = cast(dict[str, Any], parsed)
        if self._done:
            self.abort()
            return
        self._source_id = payload.get("id", self._source_id)
        self.model = payload.get("model", self.model)
        for choice in payload.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("Invalid content delta")
                self._content.append(content)
            for fragment in delta.get("tool_calls", []):
                index = fragment["index"]
                if not isinstance(index, int) or index < 0:
                    raise ValueError("Invalid tool index")
                tool = self._tools.setdefault(index, {})
                for key, value in fragment.items():
                    if key == "index":
                        continue
                    if key == "function":
                        function = tool.setdefault("function", {})
                        for field, text in value.items():
                            if field in ("name", "arguments"):
                                function[field] = function.get(field, "") + text
                            else:
                                function[field] = text
                    elif key == "id":
                        tool[key] = tool.get(key, "") + value
                    else:
                        tool[key] = value
            if choice.get("finish_reason") is not None:
                self._finished = True

    def completion(self) -> AssistantCompletion | None:
        if self._failed or not self._finished or not self._done:
            return None
        if self._line.strip() or self._data or self._decoder.getstate()[0]:
            return None
        payload: dict[str, object] = {
            "role": "assistant",
            "content": "".join(self._content) if self._content else None,
        }
        if self._tools:
            payload["tool_calls"] = [self._tools[index] for index in sorted(self._tools)]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        return AssistantCompletion(
            payload=payload,
            content_hash=digest,
            request_id=self.request_id,
            model=self.model,
            source_message_id=self._source_id,
        )
