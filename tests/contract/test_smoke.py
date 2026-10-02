"""The optional operator smoke exits with actionable prerequisite remedies."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/smoke.sh"


@pytest.mark.parametrize(
    ("docker_body", "remedy"),
    [
        (None, "Install Docker"),
        ("exit 1", "Start Docker"),
        (
            'case "$*" in "info") exit 0;; "compose version") exit 1;; esac',
            "Install Docker Compose",
        ),
        (
            'case "$*" in "compose ps --status running --services") echo postgres;; esac',
            "docker compose up --build -d --wait",
        ),
        (
            'case "$*" in "compose ps --status running --services") '
            'printf "proxy\\nworker\\npostgres\\nchromadb\\n";; '
            '"compose port proxy 8080") echo 0.0.0.0:8080;; esac',
            "127.0.0.1",
        ),
    ],
)
def test_prerequisite_failure_has_precise_remedy(tmp_path, docker_body, remedy):
    if docker_body is not None:
        docker = tmp_path / "docker"
        docker.write_text(f"#!/bin/sh\n{docker_body}\n")
        docker.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", str(SCRIPT)],
        env={**os.environ, "PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert remedy in result.stderr
    assert "Traceback" not in result.stderr
