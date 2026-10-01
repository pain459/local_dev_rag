import pytest

from local_dev_rag.config import ModelBudget, Settings
from local_dev_rag.models import ModelRegistry, UnknownModelError

GENERATORS = {
    "qwen3-coder:30b",
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "qwen2.5:7b",
}


def test_catalog_exposes_only_the_five_generation_models():
    registry = ModelRegistry(Settings(_env_file=None))
    payload = registry.as_openai_models()
    assert payload["object"] == "list"
    assert {row["id"] for row in payload["data"]} == GENERATORS
    assert all(
        row["object"] == "model" and row["owned_by"] == "local-rag" for row in payload["data"]
    )
    assert registry.default_model == "qwen3-coder:30b"


@pytest.mark.parametrize("model_id", ["nomic-embed-text:latest", "unknown:latest"])
def test_unconfigured_generation_model_is_rejected(model_id):
    with pytest.raises(UnknownModelError):
        ModelRegistry(Settings(_env_file=None)).get(model_id)


def test_catalog_consumes_explicit_settings_budgets():
    settings = Settings(
        _env_file=None,
        model_budgets={
            "qwen3-coder:30b": ModelBudget(context_tokens=16384, output_tokens=4096),
        },
    )
    model = ModelRegistry(settings).get("qwen3-coder:30b")
    assert model.id == "qwen3-coder:30b"
    assert model.display_name == "Qwen3 Coder 30B"
    assert model.context_tokens == 16384
    assert model.output_tokens == 4096


def test_embedder_budget_cannot_add_embedder_to_generation_catalog():
    registry = ModelRegistry(
        Settings(
            _env_file=None,
            model_budgets={
                "nomic-embed-text:latest": ModelBudget(),
            },
        )
    )
    with pytest.raises(UnknownModelError):
        registry.get("nomic-embed-text:latest")


def test_default_model_must_be_a_generation_model():
    with pytest.raises(UnknownModelError):
        ModelRegistry(Settings(_env_file=None, default_model="nomic-embed-text:latest"))
