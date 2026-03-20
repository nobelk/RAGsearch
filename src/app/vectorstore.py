import os
import uuid

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import EMBEDDING_MODEL
from app.text.converter import DocumentChunk

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_BATCH_SIZE = 64
EMBEDDING_DIM = 768
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "documents"

# Task prefixes for nomic-embed-text (model-specific)
EMBED_PREFIX_DOCUMENT = "search_document: "
EMBED_PREFIX_QUERY = "search_query: "

# Conservative character limit (~4 chars/token * 8192 tokens = 32k, use 30k)
MAX_EMBED_CHARS = 30_000

# Application-specific UUID namespace for deterministic section IDs (RFC 4122)
APP_NAMESPACE = uuid.UUID("b6e0f37c-4a3a-4e8d-9f1b-2c5d7e8a9b0c")


def _chunk_to_point_id(section_id: str) -> str:
    """Deterministic UUID5 from section_id for idempotent upserts."""
    return str(uuid.uuid5(APP_NAMESPACE, section_id))


def _truncate_for_embedding(text: str) -> str:
    """Truncate text to stay within the embedding model's context window."""
    if len(text) <= MAX_EMBED_CHARS:
        return text
    return text[:MAX_EMBED_CHARS]


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts via Ollama's /api/embed endpoint.
    Processes in batches of EMBED_BATCH_SIZE.
    """
    all_embeddings: list[list[float]] = []
    async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=120.0) as client:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            resp = await client.post(
                "/api/embed", json={"model": EMBEDDING_MODEL, "input": batch}
            )
            resp.raise_for_status()
            all_embeddings.extend(resp.json()["embeddings"])
    return all_embeddings


# --- Shared Qdrant client (singleton) ---

_qdrant_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    """Return a shared Qdrant client, creating it on first use."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(url=QDRANT_URL)
    return _qdrant_client


async def close_qdrant_client() -> None:
    """Close the shared Qdrant client if open."""
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


async def ensure_collection() -> None:
    """Create the vector collection if it doesn't exist."""
    client = await get_qdrant_client()
    if not await client.collection_exists(COLLECTION_NAME):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


async def upsert_chunks(chunks: list[DocumentChunk]) -> None:
    """Embed and upsert document chunks into Qdrant.
    Uses deterministic UUIDs for idempotent re-ingestion.
    """
    if not chunks:
        return

    await ensure_collection()

    # Embed with document prefix + title for richer embedding
    texts = [
        _truncate_for_embedding(f"{EMBED_PREFIX_DOCUMENT}{c.title} - {c.text}")
        for c in chunks
    ]
    embeddings = await get_embeddings(texts)

    points = [
        PointStruct(
            id=_chunk_to_point_id(chunk.section_id),
            vector=embedding,
            payload={
                "section_id": chunk.section_id,
                "title": chunk.title,
                "subpart": chunk.subpart,
                "text": chunk.text,
            },
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    client = await get_qdrant_client()
    for i in range(0, len(points), 100):
        await client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + 100],
        )


async def search(query: str, limit: int = 5) -> list[dict]:
    """Semantic search: embed query, find nearest vectors, return results."""
    embeddings = await get_embeddings([f"{EMBED_PREFIX_QUERY}{query}"])
    query_vector = embeddings[0]

    client = await get_qdrant_client()
    results = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    )
    return [
        {
            "section_id": point.payload["section_id"],
            "title": point.payload["title"],
            "subpart": point.payload["subpart"],
            "text": point.payload["text"],
            "score": point.score,
        }
        for point in results.points
    ]
