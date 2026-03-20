# RAGsearch

[![codecov](https://codecov.io/gh/nobelk/RAGsearch/branch/main/graph/badge.svg)](https://codecov.io/gh/nobelk/RAGsearch)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)

A personal search assistant that lets you search your documents using RAG (Retrieval-Augmented Generation). PDFs are chunked, embedded, and indexed in a vector database via a CLI ingestion tool. Queries are semantically matched against your documents and answered by a local LLM.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Flutter](https://flutter.dev/) SDK (for UI development)
- Docker and Docker Compose (for Qdrant and Ollama services)

## Setup

Install Python and Flutter dependencies:

```bash
make install
```

Start the backing services (Qdrant vector database + Ollama LLM):

```bash
make services-up
```

Pull the required Ollama models:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
docker compose exec ollama ollama pull llama3.2:1b
```

## Ingesting Documents

Place PDF files in a `data/` directory, then run:

```bash
make ingest
```

Or specify a custom directory:

```bash
uv run app-ingest path/to/pdfs/
```

## Running

Start the API server (serves both the API and the Flutter web UI):

```bash
make run
```

The app is available at `http://localhost:8000`.

To run the Flutter UI in development mode with hot reload:

```bash
make ui-run
```

## Testing

```bash
make test                              # Unit tests only
make test-all                          # All tests (requires Qdrant + Ollama running)
make test-file FILE=tests/llm_tests.py # Single test file
```

## Formatting

```bash
make format        # Format with black
make format-check  # Check only
```

## Configuration

Models and behavior are configured in `config.yaml`. Environment variables take precedence over `config.yaml`, which takes precedence over hardcoded defaults:

| Environment Variable        | Description                  | Default                       |
|-----------------------------|------------------------------|-------------------------------|
| `OLLAMA_EMBEDDING_MODEL`    | Embedding model              | `nomic-embed-text`            |
| `OLLAMA_GENERATION_MODEL`   | Generation model             | `llama3.2`                    |
| `OLLAMA_CLASSIFIER_MODEL`   | Query classifier model       | `llama3.2:1b`                 |
| `APP_SYSTEM_PROMPT`         | Custom system prompt         | See `config.yaml`             |
| `APP_CLASSIFIER_ENABLED`    | Enable/disable classifier    | `true`                        |

Service URLs are configured via environment variables only (not in `config.yaml`):

| Environment Variable        | Description                  | Default                       |
|-----------------------------|------------------------------|-------------------------------|
| `OLLAMA_BASE_URL`           | Ollama service URL           | `http://localhost:11434`      |
| `QDRANT_URL`                | Qdrant service URL           | `http://localhost:6333`       |

## Deployment

Full deploy (build Flutter UI + Docker image, push to registry, start on remote host):

```bash
make deploy
```

First-time setup (includes data transfer and model pulling):

```bash
make deploy-setup
```
