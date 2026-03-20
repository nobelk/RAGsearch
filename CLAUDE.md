# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-based document search application. Users upload PDFs which get chunked, embedded, and stored in Qdrant. Queries are semantically matched against stored documents and an LLM generates answers grounded in the retrieved context. The backend is FastAPI (Python), the frontend is Flutter Web, and both Qdrant and Ollama run as Docker services.

## Commands

```bash
# Dependencies
make install              # uv sync + flutter pub get + flutter build web

# Run
make run                  # Start FastAPI server (port 8000)
make services-up          # Start Qdrant + Ollama via docker compose
make ui-run               # Run Flutter UI in Chrome (port 3000)

# Test
make test                 # Unit tests only (excludes @pytest.mark.integration)
make test-all             # All tests including integration (requires Qdrant + Ollama running)
make test-file FILE=tests/llm_tests.py  # Single test file

# Format
make format               # black src/ tests/
make format-check         # Check only

# Ingest
make ingest               # Ingest PDFs from data/ into Qdrant
uv run app-ingest <dir>   # Ingest from specific directory

# Deploy
make deploy               # Full: build UI + Docker image, push, start on remote
```

## Architecture

**Backend (`src/app/`):**
- `api.py` — FastAPI app. Three endpoints: `POST /search` (sync), `POST /search/stream` (SSE streaming), `GET /health`. Serves Flutter static files at `/`.
- `vectorstore.py` — Async Qdrant client (singleton). Embeds text via Ollama's `nomic-embed-text` model into 768-dim vectors. Uses `"search_query: "` / `"search_document: "` prefixes. Batch size 64, max 30k chars per embedding.
- `llm.py` — Calls Ollama for LLM generation (streaming and non-streaming). Contains jailbreak detection via regex patterns. Orchestrates the full RAG pipeline: classify query → search vectors → generate answer.
- `classifier.py` — Lightweight query classifier (uses `llama3.2:1b`). Classifies queries as on_topic/off_topic/adversarial. Fails open on errors.
- `text/converter.py` — PDF→Markdown via pymupdf4llm, then paragraph-aware chunking (1500 chars, 200 overlap).
- `ingest.py` — CLI entry point (`app-ingest`) for batch PDF ingestion.

**Frontend (`app_ui/`):**
- Flutter Web app using Provider for state management.
- `ChatNotifier` manages message state; `SearchStreamService` handles SSE streaming from the backend.
- Built output goes to `static/` which FastAPI serves.

**Configuration (`config.yaml` + env vars):**
Config resolution order: environment variables > config.yaml > hardcoded defaults.
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `QDRANT_URL` (default: `http://localhost:6333`)
- `OLLAMA_EMBEDDING_MODEL`, `OLLAMA_GENERATION_MODEL`, `OLLAMA_CLASSIFIER_MODEL`
- `APP_SYSTEM_PROMPT`, `APP_CLASSIFIER_ENABLED`

## Key Patterns

- **Async throughout**: Backend uses async/await with httpx for Ollama calls and async Qdrant client.
- **SSE streaming**: `/search/stream` sends `sources` event first, then `token` events as LLM generates, then `done`.
- **Deterministic IDs**: Document chunks get UUID5 IDs from `{source_name}-{chunk_index}` for idempotent ingestion.
- **Integration tests** are marked with `@pytest.mark.integration` and require running Qdrant and Ollama services.
- **pytest-asyncio** with `asyncio_mode = "auto"` — async test functions are automatically detected.
- **Test file naming**: `*_tests.py` pattern (not `test_*.py`).

## Infrastructure

- **Python 3.13+**, managed with `uv`
- **Qdrant v1.14.1** — vector database
- **Ollama v0.6.2** — local LLM inference (models: `nomic-embed-text`, `llama3.2`, `llama3.2:1b`)
- **Docker Compose** for local services; separate `deploy/docker-compose.deploy.yml` for production
