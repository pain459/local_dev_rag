"""Transport fixtures for exercising real ASGI stream delivery failures."""

import asyncio
import json

import httpx


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


async def invoke_stream(app, send, *, asgi_version="2.4", disconnect=None, payload=None):
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
        "state": {},
        "headers": [
            (b"x-opencode-project-id", b"project"),
            (b"x-opencode-session-id", b"session"),
            (b"content-type", b"application/json"),
        ],
    }
    read = False

    async def receive():
        nonlocal read
        if read:
            if disconnect is not None:
                await disconnect.wait()
                return {"type": "http.disconnect"}
            await asyncio.Future()
        read = True
        return {
            "type": "http.request",
            "body": json.dumps(
                payload
                if payload is not None
                else {"model": "qwen3-coder:30b", "messages": [], "stream": True}
            ).encode(),
            "more_body": False,
        }

    await app(scope, receive, send)
    return scope
