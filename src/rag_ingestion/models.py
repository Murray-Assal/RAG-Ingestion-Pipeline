"""Shared data structures for the ingestion and retrieval layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A documentation file fetched from a source repository."""

    repository: str
    path: str
    source_url: str
    content: str
    content_hash: str
    source_sha: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """The persisted document identity and latest content fingerprint."""

    id: str
    source_url: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A header-scoped, embedding-ready document fragment."""

    id: str
    ordinal: int
    content: str
    header_path: tuple[str, ...]
    token_count: int


@dataclass(frozen=True, slots=True)
class SearchMatch:
    """A vector-search result returned by pgvector."""

    content: str
    source_url: str
    repository: str
    path: str
    header_path: tuple[str, ...]
    score: float


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Counters from one deterministic ingestion run."""

    repositories: tuple[str, ...]
    discovered_documents: int
    indexed_documents: int
    skipped_documents: int
    embedded_chunks: int
    deleted_documents: int
    finished_at: datetime
