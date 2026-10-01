"""Explicit generation catalog shared by model discovery and request validation."""

from dataclasses import dataclass

from local_dev_rag.config import Settings


class UnknownModelError(ValueError):
    """The requested model is not an enabled generation model."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    context_tokens: int
    output_tokens: int


class ModelRegistry:
    def __init__(self, settings: Settings):
        names = {
            "qwen3-coder:30b": "Qwen3 Coder 30B",
            "qwen2.5-coder:1.5b": "Qwen2.5 Coder 1.5B",
            "qwen2.5-coder:7b": "Qwen2.5 Coder 7B",
            "llama3.1:8b": "Llama 3.1 8B",
            "qwen2.5:7b": "Qwen2.5 7B",
        }
        self._models = {
            model_id: ModelSpec(
                model_id,
                name,
                settings.model_budgets[model_id].context_tokens,
                settings.model_budgets[model_id].output_tokens,
            )
            for model_id, name in names.items()
        }
        self.default_model = self.get(settings.default_model).id

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError:
            raise UnknownModelError(f"Unknown generation model: {model_id}") from None

    def as_openai_models(self) -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {"id": model.id, "object": "model", "created": 0, "owned_by": "local-rag"}
                for model in self._models.values()
            ],
        }
