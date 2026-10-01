FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM python:3.12-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable

FROM python:3.12-slim-bookworm AS production
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8080
CMD ["uvicorn", "local_dev_rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
