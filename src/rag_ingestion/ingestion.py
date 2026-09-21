"""The incremental ingestion job used directly and by the Airflow DAG."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from rag_ingestion.change_detector import ChangeDetector
from rag_ingestion.chunking import MarkdownChunker
from rag_ingestion.config import Settings, get_settings
from rag_ingestion.database import PgVectorStore
from rag_ingestion.embeddings import EmbeddingProvider, SentenceTransformerEmbedder
from rag_ingestion.models import DocumentRecord, IngestionSummary, SourceDocument
from rag_ingestion.source_connector import build_document_source

logger = logging.getLogger(__name__)


class DocumentSource(Protocol):
    def list_repositories(self) -> list[str]: ...

    def crawl(self, repositories: Sequence[str]) -> list[SourceDocument]: ...


class IngestionStore(Protocol):
    def ensure_schema(self) -> None: ...

    def list_active_documents(self, repositories: Sequence[str]) -> list[DocumentRecord]: ...

    def mark_seen(self, source_url: str, source_sha: str | None) -> None: ...

    def replace_document(self, document, chunks, embeddings, embedding_model: str) -> None: ...

    def mark_documents_deleted(self, documents: Sequence[DocumentRecord]) -> int: ...


class IngestionPipeline:
    """Coordinates source snapshotting, structure-aware chunking, and idempotent writes."""

    def __init__(
        self,
        settings: Settings,
        source: DocumentSource,
        store: IngestionStore,
        embedder: EmbeddingProvider,
        chunker: MarkdownChunker | None = None,
    ) -> None:
        self.settings = settings
        self.source = source
        self.store = store
        self.embedder = embedder
        self.chunker = chunker or MarkdownChunker(
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

    def run(self, repositories: Sequence[str] | None = None) -> IngestionSummary:
        """Index only documents whose source content hash has changed since the last run."""

        resolved_repositories = tuple(repositories or self.settings.github_repositories)
        if not resolved_repositories:
            resolved_repositories = tuple(self.source.list_repositories())
        if not resolved_repositories:
            raise RuntimeError("No repositories found for the configured GitHub owner")

        documents = self.source.crawl(resolved_repositories)
        # Fetch the full snapshot before any data write, so an unreachable source
        # cannot corrupt the previously indexed state.
        self.store.ensure_schema()
        change_set = ChangeDetector().detect(
            documents,
            self.store.list_active_documents(resolved_repositories),
        )
        indexed_documents = 0
        skipped_documents = len(change_set.unchanged)
        embedded_chunks = 0

        for document in change_set.unchanged:
            self.store.mark_seen(document.source_url, document.source_sha)

        for document in change_set.changed:
            chunks = self.chunker.chunk(document)
            embedding_result = self.embedder.embed_resilient([chunk.content for chunk in chunks])
            indexed_chunks = [chunks[index] for index, _ in embedding_result.vectors]
            vectors = [vector for _, vector in embedding_result.vectors]
            if embedding_result.failed_indices:
                logger.error(
                    "Skipped %s failed chunk(s) from %s/%s",
                    len(embedding_result.failed_indices),
                    document.repository,
                    document.path,
                )
            if chunks and not indexed_chunks:
                logger.error("No chunks from %s/%s could be embedded; document will retry next run", document.repository, document.path)
                continue
            self.store.replace_document(document, indexed_chunks, vectors, self.settings.embedding_model)
            indexed_documents += 1
            embedded_chunks += len(indexed_chunks)
            logger.info("Indexed %s chunks from %s/%s", len(indexed_chunks), document.repository, document.path)

        deleted_documents = self.store.mark_documents_deleted(change_set.deleted)
        return IngestionSummary(
            repositories=resolved_repositories,
            discovered_documents=len(documents),
            indexed_documents=indexed_documents,
            skipped_documents=skipped_documents,
            embedded_chunks=embedded_chunks,
            deleted_documents=deleted_documents,
            finished_at=datetime.now(UTC),
        )


def build_pipeline(settings: Settings | None = None) -> IngestionPipeline:
    """Create the production pipeline while keeping its collaborators easy to fake in tests."""

    resolved_settings = settings or get_settings()
    embedder = SentenceTransformerEmbedder(
        resolved_settings.embedding_model,
        batch_size=resolved_settings.embedding_batch_size,
    )
    if embedder.dimension != resolved_settings.embedding_dimension:
        raise ValueError(
            "Configured EMBEDDING_DIMENSION does not match the selected sentence-transformers model "
            f"({embedder.dimension} != {resolved_settings.embedding_dimension})"
        )
    return IngestionPipeline(
        settings=resolved_settings,
        source=build_document_source(resolved_settings),
        store=PgVectorStore(resolved_settings.database_url, resolved_settings.embedding_dimension),
        embedder=embedder,
    )


def main() -> None:
    """Run one source snapshot; Airflow invokes the same function on its schedule."""

    parser = argparse.ArgumentParser(description="Incrementally index GitHub documentation into pgvector")
    parser.add_argument("--repos", nargs="*", help="Optional repository allow-list for this run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = build_pipeline().run(args.repos)
    print(json.dumps({
        "repositories": summary.repositories,
        "discovered_documents": summary.discovered_documents,
        "indexed_documents": summary.indexed_documents,
        "skipped_documents": summary.skipped_documents,
        "embedded_chunks": summary.embedded_chunks,
        "deleted_documents": summary.deleted_documents,
        "finished_at": summary.finished_at.isoformat(),
    }))


if __name__ == "__main__":
    main()
