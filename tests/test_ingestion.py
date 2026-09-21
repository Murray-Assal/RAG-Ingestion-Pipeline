from __future__ import annotations

from collections.abc import Sequence

from rag_ingestion.config import Settings
from rag_ingestion.embeddings import EmbeddingBatchResult
from rag_ingestion.ingestion import IngestionPipeline
from rag_ingestion.models import DocumentRecord, SourceDocument


class FakeSource:
    def __init__(self, documents: list[SourceDocument]) -> None:
        self.documents = documents

    def list_repositories(self) -> list[str]:
        return ["demo"]

    def crawl(self, repositories: Sequence[str]) -> list[SourceDocument]:
        assert list(repositories) == ["demo"]
        return self.documents


class FakeStore:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentRecord] = {}
        self.ensure_calls = 0
        self.replacements = 0
        self.seen: list[str] = []
        self.deleted: tuple[list[str], list[str]] | None = None

    def ensure_schema(self) -> None:
        self.ensure_calls += 1

    def list_active_documents(self, repositories: Sequence[str]) -> list[DocumentRecord]:
        assert list(repositories) == ["demo"]
        return list(self.documents.values())

    def mark_seen(self, source_url: str, source_sha: str | None) -> None:
        self.seen.append(source_url)

    def replace_document(self, document, chunks, embeddings, embedding_model: str) -> None:
        assert len(chunks) == len(embeddings)
        self.replacements += 1
        self.documents[document.source_url] = DocumentRecord("doc-id", document.source_url, document.content_hash)

    def mark_documents_deleted(self, documents: Sequence[DocumentRecord]) -> int:
        self.deleted = (["demo"], [document.source_url for document in documents])
        return 0


class FakeEmbedder:
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_resilient(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        vectors = self.embed(texts)
        return EmbeddingBatchResult(tuple(enumerate(vectors)), ())


def test_second_snapshot_skips_unchanged_document_without_embedding_again() -> None:
    document = SourceDocument(
        repository="demo",
        path="README.md",
        source_url="https://github.com/example/demo/blob/main/README.md",
        content="# Demo\n\nStable text.",
        content_hash="stable-hash",
        source_sha="git-sha",
    )
    settings = Settings(
        github_repositories=["demo"],
        embedding_dimension=3,
        chunk_max_tokens=32,
        chunk_overlap_tokens=4,
    )
    store = FakeStore()
    embedder = FakeEmbedder()
    pipeline = IngestionPipeline(settings, FakeSource([document]), store, embedder)

    first_run = pipeline.run()
    second_run = pipeline.run()

    assert first_run.indexed_documents == 1
    assert first_run.skipped_documents == 0
    assert second_run.indexed_documents == 0
    assert second_run.skipped_documents == 1
    assert len(embedder.calls) == 1
    assert store.replacements == 1
    assert store.seen == [document.source_url]
    assert store.deleted == (["demo"], [])
