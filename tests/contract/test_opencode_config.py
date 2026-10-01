import json
import subprocess
from pathlib import Path

import pytest

from local_dev_rag.domain import RequestIdentity

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "qwen3-coder:30b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "qwen2.5:7b",
}


def test_opencode_provider_routes_only_generation_models_to_proxy():
    config = json.loads((ROOT / "opencode.json").read_text())
    provider = config["provider"]["local-rag"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://localhost:8080/v1"
    assert set(provider["models"]) == MODELS
    assert config["model"] == "local-rag/qwen3-coder:30b"
    for model in provider["models"].values():
        assert model["limit"] == {"context": 8192, "output": 2048}


def plugin_headers(directory, provider="local-rag", project_id="repo-identity", session="s-1"):
    script = """
        const { RagMemoryPlugin } = await import(process.argv[1]);
        const hooks = await RagMemoryPlugin({
            project: { id: process.argv[4] }, directory: process.argv[2],
            worktree: process.argv[2],
        });
        const output = { headers: { existing: 'kept' } };
        await hooks['chat.headers']({ sessionID: process.argv[5],
            provider: { info: { id: process.argv[3] } } }, output);
        const request = new Request('http://localhost:8080/v1/chat/completions', {
            method: 'POST', headers: output.headers,
        });
        console.log(JSON.stringify(Object.fromEntries(request.headers)));
    """
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            script,
            (ROOT / ".opencode/plugins/rag-memory.js").as_uri(),
            str(directory),
            provider,
            project_id,
            session,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_plugin_export_and_required_hook_headers_are_present():
    source = (ROOT / ".opencode/plugins/rag-memory.js").read_text()
    assert "export const RagMemoryPlugin" in source
    assert '"chat.headers"' in source
    for header in ["x-opencode-session-id", "x-opencode-project-id", "x-opencode-project-root"]:
        assert header in source


def test_plugin_adds_identity_and_preserves_existing_headers(tmp_path):
    headers = plugin_headers(tmp_path)
    assert headers["existing"] == "kept"
    assert headers["x-opencode-session-id"] == "s-1"
    assert RequestIdentity.from_headers(headers).project_root == str(tmp_path.resolve())
    assert len(headers["x-opencode-project-id"]) == 64
    assert all(char in "0123456789abcdef" for char in headers["x-opencode-project-id"])
    assert (
        plugin_headers(tmp_path, session="s-2")["x-opencode-project-id"]
        == headers["x-opencode-project-id"]
    )


def test_plugin_leaves_other_providers_untouched(tmp_path):
    assert plugin_headers(tmp_path, provider="other") == {"existing": "kept"}


@pytest.mark.parametrize("name", ["项目", "percent%2F space\nfolder"])
def test_diagnostic_root_round_trips_through_actual_request_headers(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    headers = plugin_headers(directory)
    assert headers["x-opencode-project-root"].isascii()
    identity = RequestIdentity.from_headers(headers)
    assert identity.project_root == str(directory.resolve())
    assert identity.project_id == headers["x-opencode-project-id"]


def test_local_remote_paths_normalize_absolute_and_relative_addresses(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    first = tmp_path / "a"
    second = tmp_path / "b"
    for directory, address in [(first, str(remote)), (second, "../remote")]:
        directory.mkdir()
        subprocess.run(["git", "init", "--quiet", str(directory)], check=True)
        subprocess.run(
            ["git", "-C", str(directory), "remote", "add", "origin", address], check=True
        )
    assert (
        plugin_headers(first)["x-opencode-project-id"]
        == plugin_headers(second)["x-opencode-project-id"]
    )


def test_fallback_identity_is_scoped_to_project_and_normalized_root(tmp_path):
    first = plugin_headers(tmp_path)["x-opencode-project-id"]
    assert plugin_headers(tmp_path / ".")["x-opencode-project-id"] == first
    assert plugin_headers(tmp_path, project_id="other")["x-opencode-project-id"] != first
    other = tmp_path / "other"
    other.mkdir()
    assert plugin_headers(other)["x-opencode-project-id"] != first


@pytest.mark.parametrize(
    "other_remote",
    [
        "https://github.com/example/repo.git",
        "ssh://git@github.com/example/repo.git",
    ],
)
def test_remote_identity_survives_checkout_location_and_transport(tmp_path, other_remote):
    directories = [tmp_path / "a", tmp_path / "b"]
    for directory, remote in zip(
        directories, ["git@GitHub.com:example/repo.git", other_remote], strict=True
    ):
        directory.mkdir()
        subprocess.run(["git", "init", "--quiet", str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "remote", "add", "origin", remote], check=True)
    first = plugin_headers(directories[0])["x-opencode-project-id"]
    assert plugin_headers(directories[1])["x-opencode-project-id"] == first
    subprocess.run(
        [
            "git",
            "-C",
            str(directories[1]),
            "remote",
            "set-url",
            "origin",
            "https://github.com/example/different.git",
        ],
        check=True,
    )
    assert plugin_headers(directories[1])["x-opencode-project-id"] != first
