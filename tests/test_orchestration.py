from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from rag_ingestion.config import Settings
from rag_ingestion.embeddings import EmbeddingBatchResult
from rag_ingestion.models import DocumentRecord, SourceDocument
from rag_ingestion.orchestration import (
    IngestionWorkflow,
    load_orchestration_settings,
    send_failure_notification,
)


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
        self.records: dict[str, DocumentRecord] = {}
        self.replace_calls: list[str] = []
        self.finished_metrics: dict[str, int] | None = None
        self.failed_runs: list[tuple[str, str]] = []
        self.seen: list[str] = []
        self.run_number = 0

    def ensure_schema(self) -> None:
        pass

    def start_ingestion_run(self) -> str:
        self.run_number += 1
        return f"run-{self.run_number}"

    def finish_ingestion_run(self, run_id: str, **metrics: int) -> None:
        self.finished_metrics = metrics

    def fail_ingestion_run(self, run_id: str, error_message: str) -> None:
        self.failed_runs.append((run_id, error_message))

    def list_active_documents(self, repositories: Sequence[str]) -> list[DocumentRecord]:
        return list(self.records.values())

    def mark_seen(self, source_url: str, source_sha: str | None) -> None:
        self.seen.append(source_url)

    def replace_document(self, document, chunks, embeddings, embedding_model: str) -> None:
        assert len(chunks) == len(embeddings)
        self.replace_calls.append(document.path)
        self.records[document.source_url] = DocumentRecord(
            id=f"record-{len(self.records)}",
            source_url=document.source_url,
            content_hash=document.content_hash,
        )

    def mark_documents_deleted(self, documents: Sequence[DocumentRecord]) -> int:
        for document in documents:
            self.records.pop(document.source_url, None)
        return len(documents)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_resilient(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        self.calls.append(list(texts))
        return EmbeddingBatchResult(
            vectors=tuple((index, [0.1, 0.2, 0.3]) for index in range(len(texts))),
            failed_indices=(),
        )


def document(path: str, content: str, content_hash: str) -> SourceDocument:
    return SourceDocument(
        repository="demo",
        path=path,
        source_url=f"https://github.com/example/demo/blob/main/{path}",
        content=content,
        content_hash=content_hash,
    )


def run_workflow(workflow: IngestionWorkflow) -> dict[str, object]:
    run = workflow.start_ingestion_run()
    snapshot = workflow.fetch_docs(run)
    changes = workflow.detect_changes(snapshot)
    chunked = workflow.chunk_changed_docs(changes)
    embedded = workflow.embed_chunks(chunked)
    upserted = workflow.upsert_vector_store(embedded)
    state_update = workflow.update_state_store(upserted)
    return workflow.record_run_metadata(state_update)


def make_workflow(source: FakeSource, store: FakeStore, embedder: FakeEmbedder) -> IngestionWorkflow:
    settings = Settings(
        github_repositories=["demo"],
        embedding_dimension=3,
        chunk_max_tokens=32,
        chunk_overlap_tokens=4,
    )
    return IngestionWorkflow(settings, source, store, lambda: embedder)


def test_second_orchestrated_run_with_unchanged_docs_writes_no_vectors() -> None:
    source = FakeSource([document("README.md", "# Demo\n\nStable.", "hash-one")])
    store = FakeStore()
    embedder = FakeEmbedder()
    workflow = make_workflow(source, store, embedder)

    run_workflow(workflow)
    store.replace_calls.clear()
    embedder.calls.clear()
    summary = run_workflow(workflow)

    assert store.replace_calls == []
    assert embedder.calls == []
    assert summary["chunks_written"] == 0
    assert summary["docs_added"] == 0
    assert summary["docs_updated"] == 0


def test_orchestrated_change_replaces_only_the_changed_document() -> None:
    original_readme = document("README.md", "# Demo\n\nOriginal.", "readme-v1")
    guide = document("docs/guide.md", "# Guide\n\nStable.", "guide-v1")
    source = FakeSource([original_readme, guide])
    store = FakeStore()
    embedder = FakeEmbedder()
    workflow = make_workflow(source, store, embedder)

    run_workflow(workflow)
    store.replace_calls.clear()
    source.documents = [document("README.md", "# Demo\n\nChanged.", "readme-v2"), guide]
    summary = run_workflow(workflow)

    assert store.replace_calls == ["README.md"]
    assert summary["docs_added"] == 0
    assert summary["docs_updated"] == 1
    assert summary["chunks_written"] == 1


def test_failure_notification_falls_back_to_an_error_log_without_a_webhook(caplog) -> None:
    with caplog.at_level(logging.ERROR):
        send_failure_notification("embedding task failed", webhook_url=None)

    assert "embedding task failed" in caplog.text


def test_dag_schedule_and_retry_policy_are_loaded_from_project_configuration() -> None:
    project_root = Path(__file__).resolve().parents[1]

    settings = load_orchestration_settings(project_root / "config" / "settings.yaml")

    assert settings.schedule == "0 3 * * *"
    assert settings.retries == 2
    assert settings.retry_exponential_backoff is True
