import pytest
from pydantic import ValidationError

from local_dev_rag.config import Settings, get_settings


def test_defaults_target_local_services_and_memory_models():
    settings = Settings(_env_file=None)
    assert settings.proxy_port == 8080
    assert settings.proxy_host == "127.0.0.1"
    assert settings.ollama_url == "http://host.docker.internal:11434"
    assert settings.curator_model == "qwen2.5-coder:1.5b"
    assert settings.embedding_model == "nomic-embed-text:latest"
    assert settings.default_model == "qwen3-coder:30b"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.20", "example.com"])
def test_remote_bind_requires_explicit_opt_in(host):
    with pytest.raises(ValidationError, match="ALLOW_REMOTE_BIND"):
        Settings(proxy_host=host, _env_file=None)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_loopback_bind_is_allowed(host):
    assert Settings(proxy_host=host, _env_file=None).proxy_host == host


def test_environment_can_explicitly_enable_container_bind(monkeypatch):
    monkeypatch.setenv("PROXY_HOST", "0.0.0.0")
    monkeypatch.setenv("ALLOW_REMOTE_BIND", "true")
    assert get_settings().proxy_host == "0.0.0.0"


def test_model_budgets_are_explicit_and_usable():
    settings = Settings(_env_file=None)
    assert set(settings.model_budgets) == {
        "qwen3-coder:30b",
        "qwen2.5-coder:1.5b",
        "qwen2.5-coder:7b",
        "llama3.1:8b",
        "qwen2.5:7b",
    }
    for budget in settings.model_budgets.values():
        assert budget.context_tokens > budget.output_tokens + budget.safety_tokens


@pytest.mark.parametrize(
    "values",
    [
        {"proxy_port": 0},
        {"proxy_port": 65536},
        {"retrieval_candidate_limit": 0},
        {"retry_max_attempts": 0},
        {"retry_initial_seconds": -1},
        {"log_level": "INVALID"},
        {"model_budgets": {"qwen3-coder:30b": {"context_tokens": 100, "output_tokens": 100}}},
    ],
)
def test_invalid_operational_limits_are_rejected(values):
    with pytest.raises(ValidationError):
        Settings(**values, _env_file=None)


def test_environment_overrides_budgets_and_service_settings(monkeypatch):
    monkeypatch.setenv("MODEL_BUDGETS__QWEN3-CODER:30B__CONTEXT_TOKENS", "16384")
    monkeypatch.setenv("PROXY_PORT", "8081")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_LIMIT", "12")
    settings = Settings(_env_file=None)
    assert settings.model_budgets["qwen3-coder:30b"].context_tokens == 16384
    assert settings.model_budgets["qwen2.5-coder:1.5b"].context_tokens == 8192
    assert len(settings.model_budgets) == 5
    assert settings.proxy_port == 8081
    assert settings.retrieval_candidate_limit == 12
