import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import app


def parse_sse_events(body: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data = line[len("data:") :].strip()
        if event_type and data:
            events.append({"event": event_type, "data": json.loads(data)})
    return events


SAMPLE_SEARCH_RESULTS = [
    {
        "section_id": "s1",
        "title": "DNS Overview",
        "subpart": "Introduction",
        "text": "DNS is a distributed database.",
        "score": 0.9,
    },
]


@pytest.fixture
def test_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestSearchEndpoint:
    async def test_successful_search(self, test_client):
        with patch(
            "app.api.vectorstore.search",
            new_callable=AsyncMock,
            return_value=SAMPLE_SEARCH_RESULTS,
        ):
            with patch(
                "app.api.llm.generate_rag_answer",
                new_callable=AsyncMock,
                return_value="DNS is the Domain Name System.",
            ):
                response = await test_client.post(
                    "/search", json={"query": "What is DNS?"}
                )
                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "DNS is the Domain Name System."
                assert len(data["sources"]) == 1
                assert data["query"] == "What is DNS?"

    async def test_empty_results(self, test_client):
        with patch(
            "app.api.vectorstore.search",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = await test_client.post(
                "/search", json={"query": "unknown topic"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "No relevant documents" in data["answer"]
            assert data["sources"] == []

    async def test_empty_query_rejected(self, test_client):
        response = await test_client.post("/search", json={"query": ""})
        assert response.status_code == 422

    async def test_limit_parameter(self, test_client):
        with patch(
            "app.api.vectorstore.search",
            new_callable=AsyncMock,
            return_value=SAMPLE_SEARCH_RESULTS,
        ) as mock_search:
            with patch(
                "app.api.llm.generate_rag_answer",
                new_callable=AsyncMock,
                return_value="Answer",
            ):
                await test_client.post(
                    "/search", json={"query": "test", "limit": 10}
                )
                mock_search.assert_called_once_with("test", limit=10)


class TestSearchStreamEndpoint:
    async def test_stream_with_results(self, test_client):
        async def fake_stream(query, chunks, model=None):
            for token in ["Hello", " ", "world"]:
                yield token

        with patch(
            "app.api.vectorstore.search",
            new_callable=AsyncMock,
            return_value=SAMPLE_SEARCH_RESULTS,
        ):
            with patch(
                "app.api.llm.generate_rag_answer_stream", side_effect=fake_stream
            ):
                response = await test_client.post(
                    "/search/stream", json={"query": "What is DNS?"}
                )
                assert response.status_code == 200
                assert "text/event-stream" in response.headers["content-type"]

                events = parse_sse_events(response.text)
                event_types = [e["event"] for e in events]
                assert event_types[0] == "sources"
                assert "done" in event_types
                # Token events should be present
                token_events = [
                    e for e in events if e["event"] == "token"
                ]
                assert len(token_events) >= 1

    async def test_stream_empty_results(self, test_client):
        with patch(
            "app.api.vectorstore.search",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = await test_client.post(
                "/search/stream", json={"query": "unknown"}
            )
            assert response.status_code == 200
            events = parse_sse_events(response.text)
            assert events[0]["event"] == "sources"
            # Should have a token event with "No relevant documents"
            token_events = [e for e in events if e["event"] == "token"]
            assert any(
                "No relevant documents" in e["data"]["content"]
                for e in token_events
            )

    async def test_stream_sources_first(self, test_client):
        async def fake_stream(query, chunks, model=None):
            yield "token"

        with patch(
            "app.api.vectorstore.search",
            new_callable=AsyncMock,
            return_value=SAMPLE_SEARCH_RESULTS,
        ):
            with patch(
                "app.api.llm.generate_rag_answer_stream", side_effect=fake_stream
            ):
                response = await test_client.post(
                    "/search/stream", json={"query": "test"}
                )
                events = parse_sse_events(response.text)
                assert events[0]["event"] == "sources"
                assert len(events[0]["data"]["sources"]) == 1


class TestHealthEndpoint:
    async def test_healthy(self, test_client):
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock()

        mock_ollama_response = MagicMock()
        mock_ollama_response.raise_for_status = MagicMock()

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_ollama_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "app.api.vectorstore.get_qdrant_client",
            new_callable=AsyncMock,
            return_value=mock_qdrant,
        ):
            with patch(
                "app.api.httpx.AsyncClient", return_value=mock_http_client
            ):
                response = await test_client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["qdrant"] is True
                assert data["ollama"] is True

    async def test_degraded_when_qdrant_down(self, test_client):
        mock_http_client = AsyncMock()
        mock_ollama_response = MagicMock()
        mock_ollama_response.raise_for_status = MagicMock()
        mock_http_client.get = AsyncMock(return_value=mock_ollama_response)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "app.api.vectorstore.get_qdrant_client",
            new_callable=AsyncMock,
            side_effect=Exception("Connection refused"),
        ):
            with patch(
                "app.api.httpx.AsyncClient", return_value=mock_http_client
            ):
                response = await test_client.get("/health")
                data = response.json()
                assert data["status"] == "degraded"
                assert data["qdrant"] is False
