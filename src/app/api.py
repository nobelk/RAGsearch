import importlib.metadata
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config as cfg
from app import llm, vectorstore

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = cfg._resolve("generation_model", "OLLAMA_BASE_URL")


# --- Request/Response Models ---


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)
    model: str | None = None


class SearchResult(BaseModel):
    section_id: str
    title: str
    subpart: str
    text: str
    score: float


class SearchResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    query: str


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    ollama: bool


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await vectorstore.ensure_collection()
        yield
        await vectorstore.close_qdrant_client()

    try:
        version = importlib.metadata.version("app")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"

    application = FastAPI(
        title="Document Search",
        version=version,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    @application.post("/search", response_model=SearchResponse)
    async def search_endpoint(request: SearchRequest) -> SearchResponse:
        try:
            chunks = await vectorstore.search(request.query, limit=request.limit)

            if not chunks:
                return SearchResponse(
                    answer="No relevant documents found for your query.",
                    sources=[],
                    query=request.query,
                )

            model = request.model or cfg.GENERATION_MODEL
            answer = await llm.generate_rag_answer(request.query, chunks, model=model)
            sources = [SearchResult(**chunk) for chunk in chunks]
            return SearchResponse(answer=answer, sources=sources, query=request.query)
        except Exception:
            logger.exception("Search request failed")
            raise HTTPException(status_code=500, detail="Internal server error")

    @application.post("/search/stream")
    async def search_stream(request: SearchRequest):
        try:
            results = await vectorstore.search(request.query, limit=request.limit)
        except Exception:
            logger.exception("Streaming search failed during vector search")
            raise HTTPException(status_code=500, detail="Search failed")

        async def event_generator():
            # 1. Emit sources event first
            sources = [
                {
                    "section_id": r["section_id"],
                    "title": r["title"],
                    "subpart": r["subpart"],
                    "text": r["text"],
                    "score": r["score"],
                }
                for r in results
            ]
            yield f"event: sources\ndata: {json.dumps({'sources': sources})}\n\n"

            # 2. Handle empty results
            if not results:
                yield f"event: token\ndata: {json.dumps({'content': 'No relevant documents found.'})}\n\n"
                yield "event: done\ndata: {}\n\n"
                return

            # 3. Stream LLM tokens
            try:
                async for token in llm.generate_rag_answer_stream(
                    request.query, results, request.model
                ):
                    yield f"event: token\ndata: {json.dumps({'content': token})}\n\n"
                yield "event: done\ndata: {}\n\n"
            except Exception:
                logger.exception("LLM streaming failed mid-generation")
                yield f"event: error\ndata: {json.dumps({'message': 'LLM generation failed'})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @application.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        import os

        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        qdrant_ok = False
        try:
            client = await vectorstore.get_qdrant_client()
            await client.get_collections()
            qdrant_ok = True
        except Exception:
            logger.warning("Qdrant health check failed", exc_info=True)

        ollama_ok = False
        try:
            async with httpx.AsyncClient(base_url=ollama_url, timeout=5.0) as client:
                resp = await client.get("/")
                resp.raise_for_status()
                ollama_ok = True
        except Exception:
            logger.warning("Ollama health check failed", exc_info=True)

        status = "healthy" if (qdrant_ok and ollama_ok) else "degraded"
        return HealthResponse(status=status, qdrant=qdrant_ok, ollama=ollama_ok)

    # Serve Flutter UI as static files
    static_dir = Path(__file__).resolve().parent.parent.parent / "static"
    if static_dir.is_dir():
        application.mount(
            "/", StaticFiles(directory=static_dir, html=True), name="static"
        )

    return application


app = create_app()
