FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY ./src ./src

EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]