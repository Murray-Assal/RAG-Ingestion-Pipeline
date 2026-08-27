"""Thin FastAPI retrieval layer; generation is deliberately outside the core pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rag_ingestion.config import Settings, get_settings
from rag_ingestion.database import PgVectorStore
from rag_ingestion.embeddings import EmbeddingProvider, SentenceTransformerEmbedder
from rag_ingestion.models import SearchMatch


class SearchRequest(BaseModel):
    """Request body for vector retrieval."""

    question: str = Field(min_length=3, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SearchResult(BaseModel):
    """A citation-ready result that a UI or optional LLM can consume."""

    content: str
    source_url: str
    repository: str
    path: str
    header_path: list[str]
    score: float

    @classmethod
    def from_match(cls, match: SearchMatch) -> "SearchResult":
        return cls(
            content=match.content,
            source_url=match.source_url,
            repository=match.repository,
            path=match.path,
            header_path=list(match.header_path),
            score=match.score,
        )


class SearchResponse(BaseModel):
    """Retrieved context, intentionally not an ungrounded generated answer."""

    question: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    status: str


class RetrievalService:
    """Small synchronous application service, easy to replace with an LLM adapter later."""

    def __init__(self, store: PgVectorStore, embedder: EmbeddingProvider, default_top_k: int) -> None:
        self.store = store
        self.embedder = embedder
        self.default_top_k = default_top_k

    def search(self, question: str, top_k: int | None = None) -> list[SearchMatch]:
        embedding = self.embedder.embed([question])[0]
        return self.store.search(embedding, top_k or self.default_top_k)


@lru_cache
def _build_service() -> RetrievalService:
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_batch_size)
    if embedder.dimension != settings.embedding_dimension:
        raise ValueError("Configured embedding dimension does not match the selected model")
    return RetrievalService(
        store=PgVectorStore(settings.database_url, settings.embedding_dimension),
        embedder=embedder,
        default_top_k=settings.top_k,
    )


def create_app(service: RetrievalService | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the API with dependency injection for a cheap test and deployment surface."""

    resolved_settings = settings or get_settings()
    resolved_service = service

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Do not create embeddings on startup; the model loads only when /search is used.
        active_service = resolved_service or _build_service()
        active_service.store.ensure_schema()
        yield

    app = FastAPI(
        title="GitHub Docs Retrieval API",
        version="0.1.0",
        description="Returns grounded documentation chunks from the incremental pgvector index.",
        lifespan=lifespan,
    )

    def get_service() -> RetrievalService:
        return resolved_service or _build_service()

    @app.get("/health", response_model=HealthResponse)
    def health(current_service: Annotated[RetrievalService, Depends(get_service)]) -> HealthResponse:
        if not current_service.store.healthcheck():
            raise HTTPException(status_code=503, detail="database unavailable")
        return HealthResponse(status="ok")

    @app.post("/search", response_model=SearchResponse)
    def search(
        request: SearchRequest,
        current_service: Annotated[RetrievalService, Depends(get_service)],
        top_k: Annotated[int | None, Query(ge=1, le=50)] = None,
    ) -> SearchResponse:
        limit = top_k or request.top_k or resolved_settings.top_k
        matches = current_service.search(request.question, limit)
        return SearchResponse(question=request.question, results=[SearchResult.from_match(match) for match in matches])

    return app


app = create_app()
