"""Check the launch contract against OpenCode's real, version-pinned resolver."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("selector", ["", "local-rag/qwen2.5-coder:7b"])
def test_launch_replaces_target_provider_before_opencode_model_resolution(tmp_path, selector):
    # Without whole-provider replacement, recursive merging preserves the target's
    # API alias, model implementation/endpoint, input limit, options and headers.
    executable = shutil.which("opencode")
    if executable is None:
        pytest.skip("Effective config regression requires installed OpenCode 1.18.30")
    if not (ROOT / ".opencode/node_modules/@opencode-ai/plugin").is_dir():
        pytest.skip("Effective config regression requires existing .opencode plugin dependencies")
    version = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True)
    assert version.stdout.strip() == "1.18.30", "Revalidate resolver compatibility on upgrade"
    checkout = tmp_path / "RAG checkout"
    checkout.mkdir()
    shutil.copy(ROOT / "opencode.json", checkout / "opencode.json")
    shutil.copytree(ROOT / "scripts", checkout / "scripts")
    shutil.copytree(ROOT / ".opencode/plugins", checkout / ".opencode/plugins")
    repo = tmp_path / "coding project"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    target = {
        "$schema": "https://opencode.ai/config.json",
        "model": "other/kept",
        "instructions": ["KEEP_TARGET_INSTRUCTIONS.md"],
        "permission": {"bash": "ask"},
        "provider": {
            "other": {"models": {"kept": {"limit": {"context": 1000, "output": 100}}}},
            "local-rag": {
                "npm": "@ai-sdk/anthropic",
                "api": "http://invalid.example/provider",
                "options": {"baseURL": "http://invalid.example/v1", "headers": {"bad": "yes"}},
                "models": {
                    "qwen3-coder:30b": {
                        "id": "llama3.1:8b",
                        "provider": {
                            "npm": "@ai-sdk/anthropic", "api": "http://invalid.example/model",
                        },
                        "limit": {"context": 512, "input": 64, "output": 32},
                        "options": {"baseURL": "http://invalid.example/model-options"},
                        "headers": {"bad": "yes"},
                    },
                    "target-only": {"limit": {"context": 1000, "output": 100}},
                },
            },
        },
    }
    if selector:
        target["enabled_providers"] = ["other"]
        target["disabled_providers"] = ["local-rag"]
    target_path = repo / "opencode.json"
    target_path.write_text(json.dumps(target))
    # Adapt only the interactive entry point: preserve the launch environment and
    # project argv, and run the installed resolver in that project without inference.
    adapter = tmp_path / "inspect opencode"
    adapter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "os.chdir(sys.argv[1])\n"
        "outputs = []\n"
        "for args in [['models', '--verbose'], ['debug', 'config']]:\n"
        "    result = subprocess.run([os.environ['REAL_OPENCODE'], *args], check=True,\n"
        "                            capture_output=True, text=True, timeout=20)\n"
        "    outputs.append(result.stdout)\n"
        "print(json.dumps(outputs))\n"
    )
    adapter.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith("OPENCODE_")}
    for name in ["HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"]:
        directory = tmp_path / name
        directory.mkdir()
        env[name] = str(directory)
    # Reuse the project's already installed plugin API without any npm downloads.
    for directory in [checkout / ".opencode", Path(env["XDG_CONFIG_HOME"]) / "opencode"]:
        directory.mkdir(exist_ok=True)
        for name in ["package.json", "package-lock.json"]:
            shutil.copy(ROOT / ".opencode" / name, directory / name)
        (directory / "node_modules").symlink_to(ROOT / ".opencode/node_modules")
    env.update({
        "REPO": str(repo), "MODEL": selector, "OPENCODE": str(adapter),
        "REAL_OPENCODE": executable, "OPENCODE_DISABLE_MODELS_FETCH": "1",
        "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_PURE": "1",
        "npm_config_offline": "true",
    })
    result = subprocess.run(
        ["/bin/sh", str(checkout / "scripts/launch-opencode.sh")], cwd=checkout,
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    models, config_text = json.loads(result.stdout)
    config = json.loads(config_text)
    decoder = json.JSONDecoder()
    selected = models.split("local-rag/qwen3-coder:30b\n", 1)[1]
    model, _ = decoder.raw_decode(selected)
    assert model["api"] == {
        "id": "qwen3-coder:30b", "npm": "@ai-sdk/openai-compatible", "url": "",
    }
    assert model["providerID"] == "local-rag"
    assert model["limit"] == {"context": 65536, "output": 8192}
    assert model["options"] == {}
    assert model["headers"] == {}
    assert "local-rag/target-only\n" not in models
    assert "other/kept\n" in models
    assert config["instructions"] == ["KEEP_TARGET_INSTRUCTIONS.md"]
    assert config["permission"]["bash"] == "ask"
    assert config["model"] == (selector or "local-rag/qwen3-coder:30b")
    assert config["provider"]["local-rag"] == json.loads(
        (checkout / "opencode.json").read_text()
    )["provider"]["local-rag"]
    assert json.loads(target_path.read_text()) == target
