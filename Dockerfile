FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-install-project

COPY README.md ./
COPY config.yaml ./
COPY src/ src/
COPY static/ static/
RUN uv sync --no-dev --frozen

EXPOSE 8000
CMD ["uv", "run", "app"]
