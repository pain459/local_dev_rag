import json
import subprocess
from pathlib import Path
from uuid import UUID

import pytest

ROOT = Path(__file__).resolve().parents[2]


def notification_run(tmp_path, scenario):
    script = """
        const { RagMemoryPlugin } = await import(process.argv[1]);
        const { createOpencodeClient } = await import(process.argv[2]);
        const scenario = JSON.parse(process.argv[4]);
        const notices = [], lookups = [], headers = [];
        let now = 0;
        Date.now = () => now;
        const client = createOpencodeClient({
            baseUrl: 'http://opencode',
            fetch: async (request) => {
                notices.push({url: request.url, body: await request.json()});
                if (scenario.promptFailure) throw new Error('unavailable');
                return Response.json({info: {}, parts: []});
            },
        });
        globalThis.fetch = async (url, options) => {
            lookups.push({url: String(url), headers: options.headers});
            if (scenario.failure === 'network') throw new Error('offline');
            if (scenario.failure === 'http') return new Response('', {status: 404});
            if (scenario.failure === 'json') return new Response('invalid');
            return Response.json({injected_memory_tokens:
                Object.hasOwn(scenario, 'tokens') ? scenario.tokens : 137});
        };
        const hooks = await RagMemoryPlugin({
            project: {id: 'project'}, directory: process.argv[3],
            worktree: process.argv[3], client,
        });
        const queue = async (sessionID, id = 'user-1') => {
            const output = {headers: {}};
            await hooks['chat.headers']({
                sessionID, message: {id}, model: {providerID: 'local-rag'},
                provider: {info: {id: 'local-rag'}, options: {
                    baseURL: 'http://127.0.0.1:9876/custom/v1/'
                }},
            }, output);
            headers.push(output.headers);
        };
        const completed = async (sessionID, id, extra = {}) => {
            await hooks.event?.({event: {type: 'message.updated', properties: {info: {
                id, sessionID, parentID: 'user-1', role: 'assistant',
                providerID: 'local-rag', time: {created: 0, completed: 1},
                modelID: 'qwen3-coder:30b', finish: 'stop', ...extra,
            }}}});
        };
        await queue('s-1');
        if (scenario.aggregate) await queue('s-1');
        if (scenario.otherSession) await queue('s-2');
        if (scenario.expired) now = 301_000;
        if (scenario.evicted) {
            for (let i = 0; i < 512; i++) await queue('unused-' + i);
        }
        if (scenario.duplicateWithNewTurn) {
            await completed('s-1', 'assistant-1');
            await queue('s-1', 'user-2');
            await completed('s-1', 'assistant-1');
            await completed('s-1', 'assistant-2', {parentID: 'user-2'});
        } else {
            await completed('s-1', 'assistant-1', scenario.eventExtra);
            if (scenario.repeat) {
                await Promise.all([
                    completed('s-1', 'assistant-1'), completed('s-1', 'assistant-1'),
                ]);
            }
        }
        if (scenario.otherSession) await completed('s-2', 'assistant-2');
        console.log(JSON.stringify({notices, lookups, headers}));
    """
    result = subprocess.run(
        [
            "node", "--input-type=module", "-e", script,
            (ROOT / ".opencode/plugins/rag-memory.js").as_uri(),
            (ROOT / ".opencode/node_modules/@opencode-ai/sdk/dist/client.js").as_uri(),
            str(tmp_path), json.dumps(scenario),
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_hit_appends_visible_ignored_no_reply_notice_via_installed_sdk(tmp_path):
    result = notification_run(tmp_path, {"aggregate": True, "repeat": True})
    assert len(result["notices"]) == 1
    notice = result["notices"][0]
    assert notice["body"] == {
        "noReply": True,
        "parts": [{"type": "text", "text": "🧠 RAG memory applied · 274 context tokens",
                   "ignored": True}],
    }
    assert notice["url"].startswith("http://opencode/session/s-1/message?")
    assert len(result["lookups"]) == 2
    for lookup, headers in zip(result["lookups"], result["headers"], strict=True):
        identifier = headers["x-opencode-rag-observation-id"]
        assert str(UUID(identifier)) == identifier
        assert lookup["url"] == f"http://127.0.0.1:9876/custom/v1/rag/observations/{identifier}"
        assert lookup["headers"]["x-opencode-session-id"] == "s-1"
        assert lookup["headers"]["x-opencode-project-id"] == headers["x-opencode-project-id"]


def test_pending_notices_are_isolated_between_sessions(tmp_path):
    result = notification_run(tmp_path, {"otherSession": True})
    assert len(result["notices"]) == 2
    assert [lookup["headers"]["x-opencode-session-id"] for lookup in result["lookups"]] == [
        "s-1", "s-2"
    ]


def test_duplicate_old_event_cannot_consume_next_turn_observation(tmp_path):
    result = notification_run(tmp_path, {"duplicateWithNewTurn": True})
    assert len(result["notices"]) == len(result["lookups"]) == 2


@pytest.mark.parametrize("scenario", [
    {"tokens": 0}, {"tokens": -1}, {"tokens": "137"}, {"tokens": None},
    {"failure": "network"}, {"failure": "http"}, {"failure": "json"},
    {"expired": True}, {"evicted": True},
    {"eventExtra": {"providerID": "other"}},
    {"eventExtra": {"time": {"created": 0}}},
    {"eventExtra": {"role": "user"}},
    {"eventExtra": {"error": {"name": "UnknownError"}}},
])
def test_misses_and_diagnostic_failures_are_silent(tmp_path, scenario):
    result = notification_run(tmp_path, scenario)
    assert result["notices"] == []


def test_notice_append_failure_is_silent_and_not_retried(tmp_path):
    result = notification_run(tmp_path, {"promptFailure": True, "repeat": True})
    assert len(result["notices"]) == len(result["lookups"]) == 1
