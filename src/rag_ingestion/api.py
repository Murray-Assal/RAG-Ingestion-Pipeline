"""Thin FastAPI retrieval layer; generation is deliberately outside the core pipeline."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal, Protocol

from fastapi import Depends, FastAPI, HTTPException, Query
from psycopg import Error as PsycopgError
from pydantic import BaseModel, Field

from rag_ingestion.config import Settings, get_settings
from rag_ingestion.database import PgVectorStore
from rag_ingestion.embeddings import EmbeddingProvider, SentenceTransformerEmbedder
from rag_ingestion.models import SearchMatch

logger = logging.getLogger(__name__)


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
    """Retrieved context with an optional, source-grounded answer generated from it."""

    question: str
    results: list[SearchResult]
    answer: str | None = None
    answer_status: Literal["disabled", "generated", "unavailable", "no_context"] = "disabled"


class HealthResponse(BaseModel):
    status: str


class EmptyVectorIndexError(RuntimeError):
    """The database is reachable but no active chunk is available to retrieve."""


class VectorStoreUnavailableError(RuntimeError):
    """A pgvector request failed and retrieval cannot safely produce a response."""


class AnswerGenerator(Protocol):
    """Optional, narrow contract for a source-grounded answer provider."""

    def generate(self, question: str, matches: Sequence[SearchMatch]) -> str: ...


class RetrievalService:
    """Small synchronous application service, easy to replace with an LLM adapter later."""

    def __init__(
        self,
        store: PgVectorStore,
        embedder: EmbeddingProvider,
        default_top_k: int,
        similarity_threshold: float,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.default_top_k = default_top_k
        self.similarity_threshold = similarity_threshold

    def query(self, question: str, top_k: int | None = None) -> list[SearchMatch]:
        """Retrieve relevant chunks or raise a meaningful error for unusable index state."""

        try:
            if self.store.active_chunk_count() == 0:
                raise EmptyVectorIndexError("No active documentation chunks have been indexed")
            embedding = self.embedder.embed([question])[0]
            matches = self.store.search(embedding, top_k or self.default_top_k)
        except PsycopgError as exc:
            raise VectorStoreUnavailableError("The vector store is unavailable") from exc
        return [match for match in matches if match.score >= self.similarity_threshold]


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
        similarity_threshold=settings.similarity_threshold,
    )


def create_app(
    service: RetrievalService | None = None,
    settings: Settings | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> FastAPI:
    """Build the API with dependency injection for a cheap test and deployment surface."""

    resolved_settings = settings or get_settings()
    resolved_service = service
    resolved_answer_generator = answer_generator

    def get_answer_generator() -> AnswerGenerator | None:
        if not resolved_settings.answer_generation_enabled:
            return None
        if resolved_answer_generator is not None:
            return resolved_answer_generator
        # Importing and initializing the provider happens only when explicitly enabled.
        from rag_ingestion.answer_generation import build_answer_generator

        return build_answer_generator(
            model=resolved_settings.answer_generation_model,
            base_url=resolved_settings.answer_generation_url,
            timeout_seconds=resolved_settings.answer_generation_timeout_seconds,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Retrieval never mutates the index. The ingestion pipeline owns schema creation.
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
    def health(current_service: RetrievalService = Depends(get_service)) -> HealthResponse:
        if not current_service.store.healthcheck():
            raise HTTPException(status_code=503, detail="database unavailable")
        return HealthResponse(status="ok")

    @app.post("/query", response_model=SearchResponse)
    def query(
        request: SearchRequest,
        current_service: RetrievalService = Depends(get_service),
        top_k: int | None = Query(default=None, ge=1, le=50),
    ) -> SearchResponse:
        limit = top_k or request.top_k or resolved_settings.top_k
        try:
            matches = current_service.query(request.question, limit)
        except EmptyVectorIndexError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "empty_index", "message": str(exc)},
            ) from exc
        except VectorStoreUnavailableError as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "vector_store_unavailable", "message": str(exc)},
            ) from exc
        if not matches:
            return SearchResponse(question=request.question, results=[], answer_status="no_context")

        answer = None
        answer_status: Literal["disabled", "generated", "unavailable"] = "disabled"
        generator = get_answer_generator()
        if generator is not None:
            try:
                answer = generator.generate(request.question, matches)
                answer_status = "generated"
            except Exception:
                # Answer generation is optional: retrieval remains usable if the local model is offline.
                logger.exception("Optional answer generation failed; returning retrieved chunks")
                answer_status = "unavailable"
        return SearchResponse(
            question=request.question,
            results=[SearchResult.from_match(match) for match in matches],
            answer=answer,
            answer_status=answer_status,
        )

    return app


app = create_app()
