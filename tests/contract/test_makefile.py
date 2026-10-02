"""Exercise operator boundaries with fake host tools; never contact the real stack."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = [
    "qwen3-coder:30b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "qwen2.5:7b",
    "nomic-embed-text:latest",
]

FAKE = r"""
import json, os, pathlib, signal, subprocess, sys, time
name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ["TOOL_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name == "docker" and args and args[0] == "compose":
    with open(os.environ["PROJECT_LOG"], "a") as log:
        log.write(os.environ.get("COMPOSE_PROJECT_NAME", "") + "\n")
if os.environ.get("HANG_CALL") == json.dumps([name, *args]):
    child = subprocess.Popen([sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"])
    pathlib.Path(os.environ["CHILD_PID"]).write_text(str(child.pid))
    pathlib.Path(os.environ["PARENT_PID"]).write_text(str(os.getpid()))
    time.sleep(60)
if name == "python3.12":
    if args == ["--version"]:
        print(os.environ.get("PYTHON_VERSION", "Python 3.12.9"))
    else:
        # Network I/O is an external boundary just like Docker/model downloads.
        # Run the real helper and its parser/decisions with controlled HTTP/socket I/O.
        import io, runpy, socket, urllib.error, urllib.request
        class Connection:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        def connect(*args, **kwargs):
            if os.environ.get("HTTP_MODE"):
                return Connection()
            raise ConnectionRefusedError
        class Opener:
            def open(self, url, **kwargs):
                if not os.environ.get("HTTP_MODE"):
                    raise urllib.error.URLError("controlled unreachable fixture")
                payload = {"status": "ok"} if url.endswith("/healthz") else {
                    "status": os.environ["HTTP_MODE"], "dependencies": dict.fromkeys(
                        ["postgres", "chromadb", "ollama", "curator", "embedder", "memory_jobs"],
                        "healthy",
                    ),
                }
                return io.BytesIO(json.dumps(payload).encode())
        socket.create_connection = connect
        urllib.request.build_opener = lambda *args: Opener()
        sys.argv = args
        runpy.run_path(args[0], run_name="__main__")
elif name == "uname":
    print(os.environ.get("PLATFORM", "Darwin"))
elif name in {"docker", "chosen-docker"}:
    if args == ["--version"]: print("Docker version 29.8.1")
    elif args == ["info"]: sys.exit(int(os.environ.get("DAEMON_FAIL", "0")))
    elif args[:2] == ["compose", "version"]:
        if os.environ.get("COMPOSE_VERSION_FAIL"): sys.exit(1)
        print("Docker Compose version v5.5.1")
    elif args[:2] == ["compose", "config"]:
        if os.environ.get("COMPOSE_FAIL"): sys.exit(1)
        if "--format" in args:
            config = json.loads(os.environ["COMPOSE_CONFIG"])
            config["name"] = os.environ.get("COMPOSE_PROJECT_NAME", "test-rag")
            print(json.dumps(config))
    elif args[:2] == ["compose", "ps"]:
        print(os.environ.get("RUNNING", "proxy\nworker\npostgres\nchromadb"))
    elif args[:2] == ["compose", "port"]: print("127.0.0.1:8080")
    elif "alembic" in args and "current" in args:
        print(os.environ.get("REVISION", "0001 (head)"))
    elif "--healthcheck" in args: sys.exit(int(os.environ.get("WORKER_FAIL", "0")))
elif name == "ollama":
    if args == ["--version"]: print("ollama version is 0.35.0")
    elif args == ["list"]:
        if os.environ.get("OLLAMA_FAIL"): sys.exit(1)
        print("NAME ID SIZE MODIFIED")
        for model in json.loads(os.environ["INSTALLED"]): print(model + " abc 1GB now")
elif name == "opencode":
    if args == ["--version"]: print("1.18.30")
    elif args == ["debug", "config"]:
        config = json.loads(pathlib.Path("opencode.json").read_text())
        config["plugin"] = [] if os.environ.get("PLUGIN_FAIL") else [
            "file:///fixture/.opencode/plugins/rag-memory.js"]
        print(json.dumps(config))
    elif args == ["models", "local-rag"]:
        for model in json.loads(os.environ["INSTALLED"])[:5]: print("local-rag/" + model)
elif name == "node": print("v24.21.0")
elif name == "uv":
    if "sync" in args: pathlib.Path(".venv").mkdir(exist_ok=True)
    else: print("uv 0.9.0")
else:
    print("Forbidden host tool invoked", file=sys.stderr)
    sys.exit(99)
"""


@pytest.fixture
def operator(tmp_path):
    for name in ["Makefile", ".env.example", "compose.yaml", "opencode.json"]:
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    shutil.copytree(ROOT / ".opencode/plugins", tmp_path / ".opencode/plugins")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for utility in ["awk", "printf"]:
        executable = shutil.which(utility)
        assert executable is not None
        (bin_dir / utility).symlink_to(executable)
    (bin_dir / "python3").symlink_to(sys.executable)
    for name in [
        "docker",
        "ollama",
        "opencode",
        "uv",
        "python3.12",
        "node",
        "uname",
        "sudo",
        "brew",
        "apt",
        "apt-get",
        "npm",
        "pip",
    ]:
        tool = bin_dir / name
        tool.write_text(f"#!{sys.executable}\n" + FAKE)
        tool.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir),
        "TOOL_LOG": str(tmp_path / "tools.jsonl"),
        "PROJECT_LOG": str(tmp_path / "projects.log"),
        "CHILD_PID": str(tmp_path / "child.pid"),
        "PARENT_PID": str(tmp_path / "parent.pid"),
        "INSTALLED": json.dumps(MODELS),
        "COMPOSE_CONFIG": json.dumps(
            {
                "name": "test-rag",
                "services": {
                    "proxy": {
                        "ports": [{"host_ip": "127.0.0.1", "published": "8080", "target": 8080}],
                        "environment": {"OLLAMA_URL": "http://host.docker.internal:11434"},
                    },
                    "worker": {},
                    "postgres": {},
                    "chromadb": {},
                },
                "volumes": {
                    "postgres_data": {"name": "test-rag_postgres_data"},
                    "chroma_data": {"name": "test-rag_chroma_data"},
                },
            }
        ),
    }
    for key in [
        "COMPOSE_PROJECT_NAME",
        "COMPOSE_FILE",
        "COMPOSE",
        "DOCKER",
        "PROJECT",
        "CONFIRM",
        "OLLAMA_HOST",
        "UV",
        "PYTHON",
        "NODE",
        "OLLAMA",
        "OPENCODE",
        "HTTP_MODE",
    ]:
        env.pop(key, None)

    class Operator:
        root = tmp_path

        def run(self, target, *variables, **overrides):
            test_timeout = overrides.pop("test_timeout", 30)
            return subprocess.run(
                ["/usr/bin/make", target, *variables],
                cwd=tmp_path,
                env={**env, **overrides},
                capture_output=True,
                text=True,
                timeout=test_timeout,
                stdin=subprocess.DEVNULL,
            )

        def calls(self):
            log = tmp_path / "tools.jsonl"
            return (
                [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            )

        def remove(self, name):
            (bin_dir / name).unlink()

        def rename(self, name, new_name):
            (bin_dir / name).rename(bin_dir / new_name)

        def spaced_tool(self, name):
            directory = tmp_path / "space in path"
            directory.mkdir(exist_ok=True)
            tool = directory / name
            shutil.copy(bin_dir / name, tool)
            return str(tool)

        def assert_child_stopped(self):
            pid = int((tmp_path / "child.pid").read_text())
            for _ in range(30):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return
                # Linux may briefly retain an init-owned terminated child as a zombie.
                stat = Path(f"/proc/{pid}/stat")
                if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    return
                time.sleep(0.05)
            raise AssertionError(f"Child {pid} survived deadline cleanup")

        def cleanup_fake_processes(self):
            for name in ["child.pid", "parent.pid"]:
                pid = tmp_path / name
                if pid.exists():
                    try:
                        os.kill(int(pid.read_text()), 9)
                    except ProcessLookupError:
                        pass

        def config(self, **values):
            config = json.loads(env["COMPOSE_CONFIG"])
            config.update(values)
            return json.dumps(config)

        def interactive_reset(self, confirmation):
            master, slave = os.openpty()
            try:
                child = subprocess.Popen(
                    ["/usr/bin/make", "reset"],
                    cwd=tmp_path,
                    env=env,
                    stdin=slave,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                os.write(master, (confirmation + "\n").encode())
                stdout, stderr = child.communicate(timeout=30)
                return subprocess.CompletedProcess(child.args, child.returncode, stdout, stderr)
            finally:
                os.close(master)
                os.close(slave)

    return Operator()


def successful(result):
    assert result.returncode == 0, result.stdout + result.stderr


def mutations(operator):
    return [
        call
        for call in operator.calls()
        if (
            call[0] == "uv"
            and "sync" in call
            or call[0] == "ollama"
            and "pull" in call
            or call[0] == "docker"
            and any(
                value in call for value in ["up", "down", "restart", "pull", "build", "upgrade"]
            )
            or call[0] in ["sudo", "brew", "apt", "apt-get", "npm", "pip"]
        )
    ]


def test_help_discovers_commands_and_safety(operator):
    result = operator.run("help")
    successful(result)
    for target in ["precheck", "essentials", "doctor-fix", "reset", "reindex", "ready", "check"]:
        assert target in result.stdout
    assert "RESET" in result.stdout and "volumes" in result.stdout
    assert "host tools" in result.stdout
    assert operator.calls() == []


@pytest.mark.parametrize("platform", ["Darwin", "Linux"])
@pytest.mark.parametrize("tool", ["docker", "ollama", "opencode", "uv", "python3.12", "node"])
def test_missing_tools_recommend_without_installing(operator, tool, platform):
    operator.remove(tool)
    result = operator.run("precheck", PLATFORM=platform)
    assert result.returncode != 0
    assert tool in result.stdout + result.stderr
    assert ("macOS" if platform == "Darwin" else "Linux") in result.stdout + result.stderr
    assert not mutations(operator)
    assert not (operator.root / ".env").exists()


def test_precheck_reports_versions_and_is_read_only(operator):
    result = operator.run("precheck")
    successful(result)
    assert "Docker Compose version v5.5.1" in result.stdout
    assert not mutations(operator)
    assert not (operator.root / ".env").exists()
    assert ["docker", "compose", "config", "--format", "json"] in operator.calls()


def test_wrong_python_and_daemon_fail_early_before_downloads(operator):
    result = operator.run("essentials", PYTHON_VERSION="Python 3.13.2", DAEMON_FAIL="1")
    assert result.returncode != 0
    assert "3.12" in result.stdout + result.stderr
    assert "daemon" in result.stdout + result.stderr
    assert not mutations(operator)
    assert not (operator.root / ".env").exists()


def test_essentials_creates_env_syncs_assets_and_all_six_models(operator):
    result = operator.run("essentials")
    successful(result)
    assert (operator.root / ".env").read_bytes() == (operator.root / ".env.example").read_bytes()
    assert ["uv", "sync", "--all-groups"] in operator.calls()
    assert ["docker", "compose", "pull", "--ignore-buildable"] in operator.calls()
    assert ["docker", "compose", "build"] in operator.calls()
    assert [call[2] for call in operator.calls() if call[:2] == ["ollama", "pull"]] == MODELS
    assert "large" in result.stdout and "qwen3-coder:30b" in result.stdout
    assert not any(
        call[0] in ["sudo", "brew", "apt", "apt-get", "npm", "pip"]
        or "prune" in call
        or "--volumes" in call
        for call in operator.calls()
    )


def test_essentials_preserves_existing_env_bytes(operator):
    secret = b"POSTGRES_PASSWORD='literal $(touch overwritten)'\nPROXY_PORT=8080\n"
    (operator.root / ".env").write_bytes(secret)
    successful(operator.run("essentials"))
    assert (operator.root / ".env").read_bytes() == secret
    assert not (operator.root / "overwritten").exists()


def test_doctor_is_read_only_reports_env_models_plugin_and_stopped_stack(operator):
    result = operator.run("doctor", RUNNING="", INSTALLED=json.dumps([MODELS[0]]), PLUGIN_FAIL="1")
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert ".env" in output and "make doctor-fix" in output
    assert "ollama pull nomic-embed-text:latest" in output
    assert "rag-memory.js" in output and "make up" in output
    assert not mutations(operator)
    assert not (operator.root / ".env").exists()


def test_doctor_checks_migrations_and_worker_only_when_running(operator):
    (operator.root / ".env").write_text("PROXY_PORT=8080\n")
    operator.run("doctor")
    assert ["docker", "compose", "exec", "-T", "proxy", "alembic", "current"] in operator.calls()
    assert [
        "docker",
        "compose",
        "exec",
        "-T",
        "worker",
        "python",
        "-m",
        "local_dev_rag.worker",
        "--healthcheck",
    ] in operator.calls()
    assert not mutations(operator)


def test_doctor_fix_pulls_only_missing_models_then_diagnoses_without_startup(operator):
    result = operator.run("doctor-fix", INSTALLED=json.dumps(MODELS[:5]), RUNNING="")
    assert result.returncode != 0  # Still stopped: doctor must honestly report it.
    assert (operator.root / ".env").exists()
    assert ["uv", "sync", "--all-groups"] in operator.calls()
    assert [
        "docker",
        "compose",
        "pull",
        "--ignore-buildable",
        "--policy",
        "missing",
    ] in operator.calls()
    pulls = [call for call in operator.calls() if call[:2] == ["ollama", "pull"]]
    assert pulls == [["ollama", "pull", "nomic-embed-text:latest"]]
    assert ["opencode", "debug", "config"] in operator.calls()
    assert not any("up" in call or "down" in call or "prune" in call for call in operator.calls())


@pytest.mark.parametrize(
    ("target", "args"),
    [
        ("up", ["up", "-d", "--wait", "--wait-timeout", "120"]),
        ("down", ["down"]),
        ("restart", ["restart"]),
        (
            "recreate",
            ["up", "--build", "--force-recreate", "-d", "--wait", "--wait-timeout", "120"],
        ),
        ("status", ["ps"]),
        ("logs", ["logs", "--tail", "100", "--follow"]),
        ("migrate", ["exec", "-T", "proxy", "alembic", "upgrade", "head"]),
    ],
)
def test_stack_command_construction_preserves_project(operator, target, args):
    successful(operator.run(target, "COMPOSE_PROJECT_NAME=operator-fixture"))
    assert ["docker", "compose", *args] in operator.calls()
    assert not any("--volumes" in call for call in operator.calls())
    assert set((operator.root / "projects.log").read_text().splitlines()) == {"operator-fixture"}


def test_maintenance_delegates_to_real_tools(operator):
    successful(operator.run("test"))
    assert ["uv", "run", "pytest"] in operator.calls()
    successful(operator.run("check"))
    for args in [["ruff", "check", "."], ["pyright"], ["pytest"]]:
        assert ["uv", "run", *args] in operator.calls()
    assert ["docker", "compose", "config", "--quiet"] in operator.calls()
    successful(operator.run("reindex", "PROJECT=project:exact-id"))
    assert [
        "docker",
        "compose",
        "exec",
        "-T",
        "proxy",
        "python",
        "-m",
        "local_dev_rag.cli",
        "reindex",
        "--project",
        "project:exact-id",
    ] in operator.calls()


@pytest.mark.parametrize("project", ["", " ", "-all", "a;b", "a\nb", "a b", "a" * 257])
def test_reindex_refuses_invalid_project_before_io(operator, project):
    result = operator.run("reindex", "PROJECT=" + project)
    assert result.returncode != 0
    assert "PROJECT" in result.stdout + result.stderr
    assert not operator.calls()


def test_reset_without_exact_confirmation_fails_safely_without_tty(operator):
    for variables in [[], ["CONFIRM=yes"], ["CONFIRM=reset"]]:
        result = operator.run("reset", *variables)
        assert result.returncode != 0
        assert "RESET" in result.stdout + result.stderr
    assert not mutations(operator)


def test_confirmed_reset_scopes_project_and_preserves_env_models_images(operator):
    (operator.root / ".env").write_text("POSTGRES_PASSWORD=keep\n")
    result = operator.run("reset", "CONFIRM=RESET", "COMPOSE_PROJECT_NAME=chosen-rag")
    successful(result)
    assert "chosen-rag" in result.stdout
    assert "test-rag_postgres_data" in result.stdout and "test-rag_chroma_data" in result.stdout
    assert mutations(operator) == [["docker", "compose", "down", "--volumes", "--remove-orphans"]]
    assert (operator.root / ".env").read_text() == "POSTGRES_PASSWORD=keep\n"


def test_reset_rejects_unresolved_config_before_deleting(operator):
    result = operator.run("reset", "CONFIRM=RESET", COMPOSE_FAIL="1")
    assert result.returncode != 0
    assert "config" in result.stdout + result.stderr
    assert not mutations(operator)


def test_doctor_and_ready_validate_http_contracts(operator):
    (operator.root / ".env").write_text("PROXY_PORT=8080\n")
    (operator.root / ".venv").mkdir()
    successful(operator.run("doctor", HTTP_MODE="ready"))
    successful(operator.run("ready", HTTP_MODE="ready"))
    result = operator.run("ready", HTTP_MODE="degraded")
    assert result.returncode != 0
    assert "/readyz" in result.stdout + result.stderr
    assert not mutations(operator)


def test_config_refuses_remote_publication_and_invalid_port_without_echoing_secrets(operator):
    config = json.loads(operator.config())
    config["services"]["proxy"]["ports"][0].update(host_ip="0.0.0.0", published="70000")
    config["services"]["proxy"]["environment"]["POSTGRES_PASSWORD"] = "sensitive-never-echo"
    result = operator.run("precheck", COMPOSE_CONFIG=json.dumps(config))
    assert result.returncode != 0
    assert "port" in result.stdout + result.stderr
    assert "sensitive-never-echo" not in result.stdout + result.stderr
    assert not mutations(operator)


def test_linux_precheck_explains_container_ollama_reachability(operator):
    result = operator.run("precheck", PLATFORM="Linux")
    successful(result)
    assert "OLLAMA_HOST" in result.stdout and "firewall" in result.stdout


@pytest.mark.parametrize("confirmation", ["RESET", "reset", "no"])
def test_reset_interactive_requires_exact_word(operator, confirmation):
    result = operator.interactive_reset(confirmation)
    if confirmation == "RESET":
        successful(result)
        assert mutations(operator) == [
            ["docker", "compose", "down", "--volumes", "--remove-orphans"]
        ]
    else:
        assert result.returncode != 0
        assert not mutations(operator)


def test_command_override_passes_arguments_without_evaluating_shell(operator):
    successful(operator.run("up", "COMPOSE=docker compose --ansi never"))
    assert [
        "docker",
        "compose",
        "--ansi",
        "never",
        "up",
        "-d",
        "--wait",
        "--wait-timeout",
        "120",
    ] in operator.calls()
    result = operator.run("up", "COMPOSE=docker compose ; touch injected")
    successful(result)  # Literal arguments, never shell syntax.
    assert not (operator.root / "injected").exists()


def test_doctor_reports_migration_and_worker_failures_without_fixing(operator):
    (operator.root / ".env").write_text("PROXY_PORT=8080\n")
    result = operator.run("doctor", REVISION="0000", WORKER_FAIL="1", HTTP_MODE="ready")
    assert result.returncode != 0
    assert "make migrate" in result.stdout + result.stderr
    assert "Worker health failed" in result.stdout + result.stderr
    assert not mutations(operator)


def test_custom_curator_and_embedder_are_downloaded_once(operator):
    config = json.loads(operator.config())
    config["services"]["proxy"]["environment"].update(
        CURATOR_MODEL="custom-curator:latest",
        EMBEDDING_MODEL="custom-embed:latest",
    )
    successful(operator.run("essentials", COMPOSE_CONFIG=json.dumps(config)))
    pulls = [call[2] for call in operator.calls() if call[:2] == ["ollama", "pull"]]
    assert pulls == [*MODELS[:5], "custom-curator:latest", "custom-embed:latest"]


def test_doctor_fix_keeps_env_and_refuses_missing_host_tools(operator):
    original = b"POSTGRES_PASSWORD=keep\n"
    (operator.root / ".env").write_bytes(original)
    operator.remove("node")
    result = operator.run("doctor-fix")
    assert result.returncode != 0
    assert (operator.root / ".env").read_bytes() == original
    assert not mutations(operator)


def test_compose_plugin_missing_has_platform_remediation(operator):
    result = operator.run("precheck", COMPOSE_VERSION_FAIL="1", PLATFORM="Linux")
    assert result.returncode != 0
    assert "Compose" in result.stdout + result.stderr
    assert "Linux" in result.stdout + result.stderr
    assert not mutations(operator)


def test_assets_refuse_unreachable_ollama_before_env_or_downloads(operator):
    result = operator.run("essentials", OLLAMA_FAIL="1")
    assert result.returncode != 0
    assert "ollama serve" in result.stdout + result.stderr
    assert not mutations(operator)
    assert not (operator.root / ".env").exists()


def test_opencode_port_mismatch_reports_remedy_without_printing_config(operator):
    source = operator.root / "opencode.json"
    config = json.loads(source.read_text())
    config["provider"]["local-rag"]["options"].update(
        baseURL="http://localhost:9999/v1",
        apiKey="secret-not-output",
    )
    source.write_text(json.dumps(config))
    result = operator.run("doctor")
    assert result.returncode != 0
    assert "baseURL mismatch" in result.stdout + result.stderr
    assert "secret-not-output" not in result.stdout + result.stderr
    assert not mutations(operator)


def test_missing_env_template_fails_without_leaving_empty_env(operator):
    (operator.root / ".env.example").unlink()
    result = operator.run("essentials")
    assert result.returncode != 0
    assert ".env.example" in result.stdout + result.stderr
    assert not (operator.root / ".env").exists()
    assert not mutations(operator)


def test_precheck_validates_local_provider_port_and_plugin_file(operator):
    source = operator.root / "opencode.json"
    config = json.loads(source.read_text())
    config["provider"]["local-rag"]["options"]["baseURL"] = "http://localhost:9999/v1"
    source.write_text(json.dumps(config))
    (operator.root / ".opencode/plugins/rag-memory.js").unlink()
    result = operator.run("precheck")
    assert result.returncode != 0
    assert "baseURL" in result.stdout + result.stderr
    assert "rag-memory.js" in result.stdout + result.stderr
    assert not mutations(operator)


def test_smoke_honors_docker_and_compose_overrides(operator):
    operator.rename("docker", "chosen-docker")
    result = operator.run("smoke", "DOCKER=chosen-docker", "COMPOSE=chosen-docker compose")
    successful(result)
    assert ["chosen-docker", "info"] in operator.calls()
    assert ["chosen-docker", "compose", "config", "--quiet"] in operator.calls()
    assert any(call[:4] == ["chosen-docker", "compose", "exec", "-T"] for call in operator.calls())


@pytest.mark.parametrize(
    "variable",
    [
        "PROJECT",
        "CONFIRM",
        "COMPOSE_PROJECT_NAME",
        "COMPOSE_FILE",
        "LOG_TAIL",
        "DOCKER",
        "COMPOSE",
        "UV",
        "PYTHON",
        "NODE",
        "OLLAMA",
        "OPENCODE",
        "OLLAMA_HOST",
        "DIAGNOSTIC_TIMEOUT_SECONDS",
        "COMPOSE_TIMEOUT_SECONDS",
        "STARTUP_TIMEOUT_SECONDS",
        "DOWNLOAD_TIMEOUT_SECONDS",
        "SMOKE_MODEL",
        "SMOKE_TIMEOUT_SECONDS",
        "CUSTOM_OPAQUE",
        "name",
        "operator_user_variables",
        "operator_literal_iterator",
    ],
)
def test_make_function_overrides_are_never_evaluated(operator, variable):
    result = operator.run(
        "help",
        variable + "=$(shell printf OPERATOR_REVIEW_SENTINEL >&2)",
    )
    successful(result)
    assert "OPERATOR_REVIEW_SENTINEL" not in result.stdout + result.stderr


def test_project_make_expression_is_rejected_literally_before_io(operator):
    result = operator.run(
        "reindex",
        "PROJECT=$(shell printf OPERATOR_REVIEW_SENTINEL >&2)",
        "COMPOSE=/usr/bin/false",
    )
    assert result.returncode != 0
    assert "OPERATOR_REVIEW_SENTINEL" not in result.stdout + result.stderr
    assert not operator.calls()


@pytest.mark.parametrize(
    "confirmation",
    [
        "$(shell printf OPERATOR_REVIEW_SENTINEL >&2; printf RESET)",
        "$$RESET",
        "$RESET",
    ],
)
def test_reset_does_not_expand_confirmation_into_authorization(operator, confirmation):
    result = operator.run("reset", "CONFIRM=" + confirmation, RESET="RESET")
    assert result.returncode != 0
    assert "OPERATOR_REVIEW_SENTINEL" not in result.stdout + result.stderr
    assert not mutations(operator)


def test_data_dollar_signs_survive_to_command_boundary(operator):
    successful(operator.run("logs", "LOG_TAIL=$$KEEP_$literal"))
    assert [
        "docker",
        "compose",
        "logs",
        "--tail",
        "$$KEEP_$literal",
        "--follow",
    ] in operator.calls()


@pytest.mark.parametrize(
    "value",
    [
        'uv"; printf OPERATOR_UV_SENTINEL; #',
        "uv$(printf OPERATOR_UV_SENTINEL)",
        "uv`printf OPERATOR_UV_SENTINEL`",
    ],
)
def test_uv_override_is_executable_data_not_recipe_source(operator, value):
    result = operator.run("test", "UV=" + value)
    assert result.returncode != 0
    assert not operator.calls()
    assert "\nOPERATOR_UV_SENTINEL" not in result.stdout + result.stderr
    dry = operator.run("-n", "test", "UV=" + value)
    successful(dry)
    assert value not in dry.stdout


def test_uv_executable_path_with_spaces_is_preserved(operator):
    successful(operator.run("test", "UV=" + operator.spaced_tool("uv")))
    assert ["uv", "run", "pytest"] in operator.calls()


def test_default_compose_preserves_docker_executable_path_with_spaces(operator):
    successful(operator.run("status", "DOCKER=" + operator.spaced_tool("docker")))
    assert ["docker", "compose", "ps"] in operator.calls()


def test_smoke_default_compose_preserves_docker_path_with_spaces(operator):
    successful(operator.run("smoke", "DOCKER=" + operator.spaced_tool("docker")))
    assert ["docker", "compose", "config", "--quiet"] in operator.calls()


@pytest.mark.parametrize(
    ("target", "call"),
    [
        ("precheck", ["uname", "-s"]),
        ("precheck", ["node", "--version"]),
        ("precheck", ["python3.12", "--version"]),
        ("precheck", ["docker", "info"]),
        ("doctor", ["docker", "compose", "config", "--format", "json"]),
        ("doctor", ["opencode", "debug", "config"]),
        ("doctor", ["opencode", "models", "local-rag"]),
        (
            "doctor",
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "worker",
                "python",
                "-m",
                "local_dev_rag.worker",
                "--healthcheck",
            ],
        ),
        ("migrate", ["docker", "compose", "exec", "-T", "proxy", "alembic", "upgrade", "head"]),
    ],
)
def test_hanging_subprocesses_fail_with_deadline_and_kill_descendants(operator, target, call):
    started = time.monotonic()
    try:
        try:
            result = operator.run(
                target,
                "DIAGNOSTIC_TIMEOUT_SECONDS=1",
                "COMPOSE_TIMEOUT_SECONDS=1",
                HANG_CALL=json.dumps(call),
                test_timeout=5,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Operator subprocess exceeded its configured outer deadline")
        assert result.returncode != 0
        assert time.monotonic() - started < 5
        operator.assert_child_stopped()
    finally:
        operator.cleanup_fake_processes()


@pytest.mark.parametrize("target", ["up", "recreate"])
def test_startup_wait_has_configured_inner_and_outer_bound(operator, target):
    successful(operator.run(target, "STARTUP_TIMEOUT_SECONDS=17"))
    assert any(call[-2:] == ["--wait-timeout", "17"] for call in operator.calls())
    call = ["docker", "compose", "up"]
    if target == "recreate":
        call += ["--build", "--force-recreate"]
    call += ["-d", "--wait", "--wait-timeout", "17"]
    try:
        started = time.monotonic()
        try:
            result = operator.run(
                target,
                "STARTUP_TIMEOUT_SECONDS=17",
                "COMPOSE_TIMEOUT_SECONDS=1",
                HANG_CALL=json.dumps(call),
                test_timeout=4,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("Compose startup exceeded its configured outer deadline")
        assert result.returncode != 0
        assert time.monotonic() - started < 4
        operator.assert_child_stopped()
    finally:
        operator.cleanup_fake_processes()


@pytest.mark.parametrize("role", ["CURATOR_MODEL", "EMBEDDING_MODEL"])
def test_model_roles_require_explicit_tags_before_downloads(operator, role):
    config = json.loads(operator.config())
    config["services"]["proxy"]["environment"][role] = "untagged-model"
    result = operator.run("essentials", COMPOSE_CONFIG=json.dumps(config))
    assert result.returncode != 0
    assert role in result.stdout + result.stderr and "tag" in result.stdout + result.stderr
    assert not mutations(operator)


@pytest.mark.parametrize("timeout", ["0", "-1", "forever", "1.5", "90000"])
def test_invalid_startup_timeout_is_rejected_before_docker_io(operator, timeout):
    result = operator.run("up", "STARTUP_TIMEOUT_SECONDS=" + timeout)
    assert result.returncode != 0
    assert "STARTUP_TIMEOUT_SECONDS" in result.stdout + result.stderr
    assert not operator.calls()


def test_command_line_data_override_wins_over_environment_literally(operator):
    result = operator.run(
        "status",
        "COMPOSE_PROJECT_NAME=literal-cli",
        COMPOSE_PROJECT_NAME="different-environment",
    )
    successful(result)
    assert set((operator.root / "projects.log").read_text().splitlines()) == {"literal-cli"}


def test_logs_follow_is_the_explicit_unbounded_exception(operator):
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            operator.run(
                "logs",
                "COMPOSE_TIMEOUT_SECONDS=1",
                test_timeout=2.5,
                HANG_CALL=json.dumps(["docker", "compose", "logs", "--tail", "100", "--follow"]),
            )
    finally:
        operator.cleanup_fake_processes()
