"""Real PostgreSQL service, isolated from developer data and scoped to this test run."""

import os
import subprocess
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text

from local_dev_rag.config import Settings

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def chroma_url() -> Iterator[str]:
    container = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            "127.0.0.1::8000",
            "-e",
            "ANONYMIZED_TELEMETRY=FALSE",
            "chromadb/chroma:0.6.3",
        ],
        text=True,
    ).strip()
    try:
        port = (
            subprocess.check_output(
                ["docker", "port", container, "8000/tcp"],
                text=True,
            )
            .strip()
            .rsplit(":", 1)[1]
        )
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                if httpx.get(f"{url}/api/v1/heartbeat", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        else:
            pytest.fail("Temporary ChromaDB did not become ready")
        yield url
    finally:
        subprocess.run(["docker", "stop", container], check=True, capture_output=True)


@pytest.fixture(scope="session")
def postgres_url() -> Iterator[str]:
    container = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            "127.0.0.1::5432",
            "-e",
            "POSTGRES_USER=local_rag_test",
            "-e",
            "POSTGRES_PASSWORD=test_password",
            "-e",
            "POSTGRES_DB=local_rag_test",
            "postgres:16-bookworm",
        ],
        text=True,
    ).strip()
    try:
        for _ in range(120):
            ready = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "pg_isready",
                    "-h",
                    "127.0.0.1",
                    "-U",
                    "local_rag_test",
                ],
                capture_output=True,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.25)
        else:
            pytest.fail("Temporary PostgreSQL did not become ready")
        port = (
            subprocess.check_output(
                ["docker", "port", container, "5432/tcp"],
                text=True,
            )
            .strip()
            .rsplit(":", 1)[1]
        )
        yield f"postgresql+asyncpg://local_rag_test:test_password@127.0.0.1:{port}/local_rag_test"
    finally:
        subprocess.run(["docker", "stop", container], check=True, capture_output=True)


def migrate(url: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["uv", "run", "alembic", *arguments],
        cwd=ROOT,
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="session")
def migrated_postgres_url(postgres_url: str) -> str:
    migrate(postgres_url, "upgrade", "head")
    return postgres_url


@pytest.fixture
async def database(migrated_postgres_url: str) -> AsyncIterator[object]:
    from local_dev_rag.db import Database

    database = Database.create(Settings(database_url=migrated_postgres_url, _env_file=None))
    async with database.session() as session:
        await session.execute(text("TRUNCATE projects CASCADE"))
    try:
        yield database
    finally:
        await database.engine.dispose()
