import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.text.converter import DocumentChunk
from app.vectorstore import (
    APP_NAMESPACE,
    EMBED_PREFIX_DOCUMENT,
    EMBED_PREFIX_QUERY,
    MAX_EMBED_CHARS,
    _chunk_to_point_id,
    _truncate_for_embedding,
    ensure_collection,
    get_embeddings,
    search,
    upsert_chunks,
)


class TestChunkToPointId:
    def test_deterministic(self):
        id1 = _chunk_to_point_id("section-1")
        id2 = _chunk_to_point_id("section-1")
        assert id1 == id2

    def test_different_sections_different_ids(self):
        id1 = _chunk_to_point_id("section-1")
        id2 = _chunk_to_point_id("section-2")
        assert id1 != id2

    def test_valid_uuid(self):
        result = _chunk_to_point_id("test")
        uuid.UUID(result)  # Should not raise

    def test_uses_app_namespace(self):
        result = _chunk_to_point_id("test")
        expected = str(uuid.uuid5(APP_NAMESPACE, "test"))
        assert result == expected


class TestTruncateForEmbedding:
    def test_short_text_unchanged(self):
        text = "short text"
        assert _truncate_for_embedding(text) == text

    def test_long_text_truncated(self):
        text = "a" * (MAX_EMBED_CHARS + 1000)
        result = _truncate_for_embedding(text)
        assert len(result) == MAX_EMBED_CHARS

    def test_exact_limit_unchanged(self):
        text = "a" * MAX_EMBED_CHARS
        assert _truncate_for_embedding(text) == text


class TestGetEmbeddings:
    async def test_returns_embeddings(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.vectorstore.httpx.AsyncClient", return_value=mock_client):
            result = await get_embeddings(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    async def test_batches_requests(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1]]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        # Create 65 texts to trigger 2 batches (batch size = 64)
        texts = [f"text{i}" for i in range(65)]

        with patch("app.vectorstore.httpx.AsyncClient", return_value=mock_client):
            with patch("app.vectorstore.EMBED_BATCH_SIZE", 64):
                await get_embeddings(texts)
                assert mock_client.post.call_count == 2


class TestEnsureCollection:
    async def test_creates_collection_if_not_exists(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists = AsyncMock(return_value=False)
        mock_qdrant.create_collection = AsyncMock()

        with patch("app.vectorstore.get_qdrant_client", return_value=mock_qdrant):
            await ensure_collection()
            mock_qdrant.create_collection.assert_called_once()

    async def test_skips_if_collection_exists(self):
        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists = AsyncMock(return_value=True)

        with patch("app.vectorstore.get_qdrant_client", return_value=mock_qdrant):
            await ensure_collection()
            mock_qdrant.create_collection.assert_not_called()


class TestUpsertChunks:
    async def test_empty_chunks_noop(self):
        # Should return without calling anything
        await upsert_chunks([])

    async def test_upserts_with_embeddings(self):
        chunks = [
            DocumentChunk(
                section_id="s1", title="Title 1", subpart="Part A", text="Text 1"
            ),
            DocumentChunk(
                section_id="s2", title="Title 2", subpart="Part A", text="Text 2"
            ),
        ]

        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists = AsyncMock(return_value=True)
        mock_qdrant.upsert = AsyncMock()

        with patch("app.vectorstore.get_qdrant_client", return_value=mock_qdrant):
            with patch(
                "app.vectorstore.get_embeddings",
                return_value=[[0.1] * 768, [0.2] * 768],
            ):
                await upsert_chunks(chunks)
                mock_qdrant.upsert.assert_called_once()

    async def test_uses_document_prefix(self):
        chunks = [
            DocumentChunk(
                section_id="s1", title="Title", subpart="Part", text="Body"
            ),
        ]

        captured_texts = []

        async def mock_get_embeddings(texts):
            captured_texts.extend(texts)
            return [[0.1] * 768]

        mock_qdrant = AsyncMock()
        mock_qdrant.collection_exists = AsyncMock(return_value=True)
        mock_qdrant.upsert = AsyncMock()

        with patch("app.vectorstore.get_qdrant_client", return_value=mock_qdrant):
            with patch(
                "app.vectorstore.get_embeddings", side_effect=mock_get_embeddings
            ):
                await upsert_chunks(chunks)
                assert captured_texts[0].startswith(EMBED_PREFIX_DOCUMENT)


class TestSearch:
    async def test_returns_formatted_results(self):
        mock_point = MagicMock()
        mock_point.payload = {
            "section_id": "s1",
            "title": "Title 1",
            "subpart": "Part A",
            "text": "Content here",
        }
        mock_point.score = 0.95

        mock_results = MagicMock()
        mock_results.points = [mock_point]

        mock_qdrant = AsyncMock()
        mock_qdrant.query_points = AsyncMock(return_value=mock_results)

        with patch("app.vectorstore.get_qdrant_client", return_value=mock_qdrant):
            with patch(
                "app.vectorstore.get_embeddings", return_value=[[0.1] * 768]
            ):
                results = await search("test query")
                assert len(results) == 1
                assert results[0]["section_id"] == "s1"
                assert results[0]["title"] == "Title 1"
                assert results[0]["score"] == 0.95

    async def test_uses_query_prefix(self):
        captured_texts = []

        async def mock_get_embeddings(texts):
            captured_texts.extend(texts)
            return [[0.1] * 768]

        mock_qdrant = AsyncMock()
        mock_qdrant.query_points = AsyncMock(
            return_value=MagicMock(points=[])
        )

        with patch("app.vectorstore.get_qdrant_client", return_value=mock_qdrant):
            with patch(
                "app.vectorstore.get_embeddings", side_effect=mock_get_embeddings
            ):
                await search("my query")
                assert captured_texts[0] == f"{EMBED_PREFIX_QUERY}my query"
