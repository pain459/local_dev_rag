"""Validated environment configuration shared by the proxy and worker."""

from ipaddress import ip_address
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelBudget(BaseModel):
    """Conservative token limits until local model metadata is verified."""

    context_tokens: int = Field(default=8192, gt=0)
    output_tokens: int = Field(default=2048, gt=0)
    safety_tokens: int = Field(default=512, ge=0)

    @model_validator(mode="after")
    def validate_capacity(self) -> Self:
        if self.output_tokens + self.safety_tokens >= self.context_tokens:
            raise ValueError("context_tokens must exceed output_tokens plus safety_tokens")
        return self


def _default_budgets() -> dict[str, ModelBudget]:
    return {
        model_id: ModelBudget()
        for model_id in (
            "qwen3-coder:30b",
            "qwen2.5-coder:1.5b",
            "qwen2.5-coder:7b",
            "llama3.1:8b",
            "qwen2.5:7b",
        )
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    proxy_host: str = "127.0.0.1"
    proxy_port: int = Field(default=8080, ge=1, le=65535)
    allow_remote_bind: bool = False
    ollama_url: str = "http://host.docker.internal:11434"
    chromadb_url: str = "http://chromadb:8000"
    database_url: str = "postgresql+asyncpg://local_rag:local_rag@postgres:5432/local_rag"
    default_model: str = "qwen3-coder:30b"
    curator_model: str = "qwen2.5-coder:1.5b"
    embedding_model: str = "nomic-embed-text:latest"
    model_budgets: dict[str, ModelBudget] = Field(default_factory=_default_budgets)
    memory_token_budget: int = Field(default=1024, gt=0)
    retrieval_candidate_limit: int = Field(default=20, gt=0)
    retrieval_result_limit: int = Field(default=6, gt=0)
    retrieval_min_score: float = Field(default=0.35, ge=0, le=1)
    retry_max_attempts: int = Field(default=5, gt=0)
    retry_initial_seconds: float = Field(default=2, gt=0)
    retry_max_seconds: float = Field(default=300, gt=0)
    upstream_timeout_seconds: float = Field(default=120, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("model_budgets")
    @classmethod
    def preserve_unmodified_budgets(cls, value: dict[str, ModelBudget]) -> dict[str, ModelBudget]:
        return _default_budgets() | value

    @model_validator(mode="after")
    def validate_bind(self) -> Self:
        try:
            loopback = ip_address(self.proxy_host).is_loopback
        except ValueError:
            loopback = self.proxy_host == "localhost"
        if not loopback and not self.allow_remote_bind:
            raise ValueError("Non-loopback PROXY_HOST requires ALLOW_REMOTE_BIND=true")
        return self


def get_settings() -> Settings:
    return Settings()
