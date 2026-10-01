import json

import pytest

from local_dev_rag.streaming import StreamAccumulator


def event(payload):
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\r\n\r\n").encode()


def delta(content=None, *, finish=None, tools=None):
    message = {} if content is None else {"content": content}
    if tools is not None:
        message["tool_calls"] = tools
    return event(
        {
            "id": "chat-123",
            "model": "qwen3-coder:30b",
            "choices": [{"index": 0, "delta": message, "finish_reason": finish}],
        }
    )


def test_arbitrary_byte_boundaries_relay_exact_bytes_and_accumulate_unicode_content():
    # Dropping fragments or decoding each byte independently loses the euro sign.
    chunks = delta("Hi €") + delta("!", finish="stop") + b"data: [DONE]\n\n"
    accumulator = StreamAccumulator(request_id="request-1")
    relayed = [accumulator.feed(bytes([byte])) for byte in chunks]
    assert b"".join(relayed) == chunks
    completion = accumulator.completion()
    assert completion is not None
    assert dict(completion.payload) == {"role": "assistant", "content": "Hi €!"}
    assert completion.model == "qwen3-coder:30b"
    assert completion.source_message_id == "chat-123"
    assert completion.request_id == "request-1"
    assert len(completion.content_hash) == 64


def test_tool_call_fragments_accumulate_by_index_without_losing_fields():
    accumulator = StreamAccumulator(request_id="request-2")
    accumulator.feed(
        delta(
            tools=[
                {
                    "index": 1,
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "other", "arguments": "{}"},
                },
                {
                    "index": 0,
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"pa'},
                },
            ]
        )
    )
    accumulator.feed(
        delta(
            tools=[{"index": 0, "function": {"name": "_file", "arguments": 'th":"a.py"}'}}],
            finish="tool_calls",
        )
    )
    accumulator.feed(b"data: [DONE]\n\n")
    completion = accumulator.completion()
    assert completion is not None
    assert completion.payload["content"] is None
    calls = completion.payload["tool_calls"]
    assert calls[0]["id"] == "call_a"
    assert calls[0]["type"] == "function"
    assert dict(calls[0]["function"]) == {"name": "read_file", "arguments": '{"path":"a.py"}'}
    assert calls[1]["id"] == "call_b"


@pytest.mark.parametrize("suffix", [b"", b"data: [DONE]\n\n", delta(finish="stop")])
def test_truncated_stream_does_not_produce_completed_record(suffix):
    # DONE alone or a finish delta alone cannot prove a complete transport stream.
    accumulator = StreamAccumulator(request_id="truncated")
    accumulator.feed(delta("partial") + suffix)
    assert accumulator.completion() is None


@pytest.mark.parametrize(
    "error",
    [
        event({"error": {"message": "upstream failed", "type": "server_error"}}),
        b"event: error\ndata: {}\n\n",
        b"data: {not json}\n\n",
        b"data: \xff\n\n",
    ],
)
def test_upstream_errors_invalidate_completion_without_changing_relay(error):
    accumulator = StreamAccumulator(request_id="failed")
    accumulator.feed(delta("partial", finish="stop"))
    assert accumulator.feed(error) == error
    accumulator.feed(b"data: [DONE]\n\n")
    assert accumulator.completion() is None


def test_abort_invalidates_even_a_finished_stream():
    accumulator = StreamAccumulator(request_id="cancelled")
    accumulator.feed(delta("finished", finish="stop") + b"data: [DONE]\n\n")
    accumulator.abort()
    assert accumulator.completion() is None


def test_comments_usage_events_and_multiline_data_do_not_erase_message():
    accumulator = StreamAccumulator(request_id="metadata")
    accumulator.feed(b': keepalive\n\ndata: {"choices":\ndata: []}\n\n')
    accumulator.feed(delta("answer", finish="stop"))
    accumulator.feed(event({"choices": [], "usage": {"total_tokens": 10}}))
    accumulator.feed(b"data: [DONE]\n\n")
    assert accumulator.completion().payload["content"] == "answer"
