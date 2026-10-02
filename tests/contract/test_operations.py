"""Execute the rendered startup contract; failed migrations must never start apps."""

import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("explicit_url", [False, True])
def test_compose_special_character_password_reaches_both_apps_intact(explicit_url, monkeypatch):
    from sqlalchemy.engine import make_url

    from local_dev_rag.config import Settings

    password = "synthetic@password:with/slash"
    encoded = (
        "postgresql+asyncpg://local_rag:synthetic%40password%3Awith%2Fslash@postgres:5432/local_rag"
    )
    config = json.loads(
        subprocess.check_output(
            ["docker", "compose", "--env-file", "/dev/null", "config", "--format", "json"],
            cwd=ROOT,
            env={
                **os.environ,
                "POSTGRES_PASSWORD": password,
                "DATABASE_URL": encoded if explicit_url else "",
            },
        )
    )
    assert config["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"] == password
    for name in ("proxy", "worker"):
        container_environment = {
            key: str(value) for key, value in config["services"][name]["environment"].items()
        }
        container_environment.setdefault(
            "MODEL_BUDGETS",
            json.dumps(
                {
                    "contract-model": {
                        "context_tokens": 4096,
                        "output_tokens": 512,
                        "safety_tokens": 128,
                    }
                }
            ),
        )
        container_environment.setdefault(
            "RANKING_WEIGHTS",
            json.dumps(
                {
                    "semantic": 0.8,
                    "importance": 0.2,
                    "recency": 0,
                    "overlap": 0,
                    "diversity": 0,
                    "recency_half_life_days": 30,
                    "min_semantic_similarity": 0.2,
                }
            ),
        )
        with monkeypatch.context() as environment:
            environment.setattr(os, "environ", container_environment)
            settings = Settings(_env_file=None)
        url = make_url(settings.database_url)
        assert url.password == password
        assert url.host == "postgres" and url.username == "local_rag"
        assert url.database == "local_rag"
        rendered_budgets = json.loads(container_environment["MODEL_BUDGETS"])
        for model_id, budget in rendered_budgets.items():
            assert settings.model_budgets[model_id].model_dump() == budget
        assert settings.ranking_weights.model_dump() == json.loads(
            container_environment["RANKING_WEIGHTS"]
        )


def test_worker_healthcheck_is_bounded_and_does_not_start_the_job_loop():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "local_dev_rag.worker", "--healthcheck"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Worker healthcheck started the job loop instead of exiting")
    # This host test process is not the Compose worker at PID 1.
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"status": "not_running"}


@pytest.fixture(scope="module")
def compose():
    return json.loads(
        subprocess.check_output(["docker", "compose", "config", "--format", "json"], cwd=ROOT)
    )


def test_compose_runs_real_worker_with_internal_dependencies_and_restart(compose):
    services = compose["services"]
    assert set(services) == {"proxy", "worker", "postgres", "chromadb"}
    assert services["worker"]["command"] == ["python", "-m", "local_dev_rag.worker"]
    for name, service in services.items():
        assert service["restart"] == "unless-stopped"
        assert service.get("healthcheck", {}).get("test")
        if name != "proxy":
            assert not service.get("ports")
    assert services["proxy"]["ports"][0]["host_ip"] == "127.0.0.1"
    for name in ("proxy", "worker"):
        for dependency in ("postgres", "chromadb"):
            assert services[name]["depends_on"][dependency]["condition"] == "service_healthy"
    assert services["worker"]["depends_on"]["proxy"]["condition"] == "service_healthy"
    assert services["postgres"]["volumes"] and services["chromadb"]["volumes"]


@pytest.mark.parametrize("ready_status", ["ready", "degraded", "not_ready"])
def test_proxy_healthcheck_covers_readiness_without_blocking_degraded_startup(
    compose, monkeypatch, ready_status
):
    checked = []

    def urlopen(url, timeout):
        checked.append(url.rsplit("/", 1)[-1])
        if url.endswith("/healthz"):
            return io.BytesIO(b'{"status":"ok"}')
        content = io.BytesIO(json.dumps({"status": ready_status, "dependencies": {}}).encode())
        if ready_status == "not_ready":
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, content)
        return content

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    command = compose["services"]["proxy"]["healthcheck"]["test"]
    assert command[:3] == ["CMD", "python", "-c"]
    exec(command[3], {})
    assert checked == ["healthz", "readyz"]


@pytest.mark.parametrize("service_name", ["proxy", "worker"])
@pytest.mark.parametrize("migration_status", [0, 17])
def test_resolved_entrypoint_migrates_before_exec_and_stops_on_failure(
    compose, tmp_path, service_name, migration_status
):
    service = compose["services"][service_name]
    entrypoint = service.get("entrypoint")
    assert entrypoint, "Migration-before-start entrypoint is missing"
    script = ROOT / entrypoint[-1].removeprefix("/app/")
    assert script.is_file()
    events = tmp_path / "events"
    migration = tmp_path / "alembic"
    migration.write_text(
        '#!/bin/sh\nprintf "migration:$*\\n" >> "$TASK_EVENTS"\nexit "$TASK_STATUS"\n'
    )
    migration.chmod(0o755)
    application = tmp_path / service["command"][0]
    application.write_text('#!/bin/sh\nprintf "app:$*\\n" >> "$TASK_EVENTS"\n')
    application.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", str(script), *service["command"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "TASK_EVENTS": str(events),
            "TASK_STATUS": str(migration_status),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == migration_status
    lines = events.read_text().splitlines()
    assert lines[0] == "migration:upgrade head"
    assert len(lines) == (2 if migration_status == 0 else 1)
    if migration_status == 0:
        assert lines[1] == "app:" + " ".join(service["command"][1:])
