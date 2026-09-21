"""Serializable pipeline stages used by Airflow's incremental-ingestion DAG."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from rag_ingestion.change_detector import ChangeDetector
from rag_ingestion.chunking import MarkdownChunker
from rag_ingestion.config import Settings, get_settings
from rag_ingestion.database import PgVectorStore
from rag_ingestion.embeddings import EmbeddingProvider, SentenceTransformerEmbedder
from rag_ingestion.models import Chunk, DocumentRecord, SourceDocument
from rag_ingestion.source_connector import build_document_source

logger = logging.getLogger(__name__)

Payload = dict[str, Any]
EmbedderFactory = Callable[[], EmbeddingProvider]


class DocumentSource(Protocol):
    """The source operations needed by the fetch task."""

    def list_repositories(self) -> list[str]: ...

    def crawl(self, repositories: Sequence[str]) -> list[SourceDocument]: ...


class WorkflowStore(Protocol):
    """The state/vector operations invoked across independent Airflow tasks."""

    def ensure_schema(self) -> None: ...

    def start_ingestion_run(self) -> str: ...

    def finish_ingestion_run(self, run_id: str, **metrics: int) -> None: ...

    def fail_ingestion_run(self, run_id: str, error_message: str) -> None: ...

    def list_active_documents(self, repositories: Sequence[str]) -> list[DocumentRecord]: ...

    def mark_seen(self, source_url: str, source_sha: str | None) -> None: ...

    def replace_document(
        self,
        document: SourceDocument,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None: ...

    def mark_documents_deleted(self, documents: Sequence[DocumentRecord]) -> int: ...


@dataclass(frozen=True, slots=True)
class OrchestrationSettings:
    """DAG scheduling and alert configuration loaded from config/settings.yaml."""

    schedule: str
    retries: int
    retry_delay_minutes: int
    retry_exponential_backoff: bool
    slack_webhook_env: str


def load_orchestration_settings(path: Path | None = None) -> OrchestrationSettings:
    """Read the small configuration subset Airflow needs without loading embedding code."""

    default_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    settings_path = path or default_path
    try:
        parsed = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Could not load orchestration settings from {settings_path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("settings.yaml must contain a mapping")
    orchestration = parsed.get("orchestration", {})
    alerts = parsed.get("failure_alerts", {})
    if not isinstance(orchestration, dict) or not isinstance(alerts, dict):
        raise ValueError("orchestration and failure_alerts configuration must be mappings")
    schedule = orchestration.get("schedule")
    retries = orchestration.get("retries")
    retry_delay_minutes = orchestration.get("retry_delay_minutes")
    if not isinstance(schedule, str) or not schedule.strip():
        raise ValueError("orchestration.schedule must be a non-empty string")
    if not isinstance(retries, int) or retries < 0:
        raise ValueError("orchestration.retries must be a non-negative integer")
    if not isinstance(retry_delay_minutes, int) or retry_delay_minutes < 1:
        raise ValueError("orchestration.retry_delay_minutes must be a positive integer")
    retry_exponential_backoff = orchestration.get("retry_exponential_backoff", True)
    if not isinstance(retry_exponential_backoff, bool):
        raise ValueError("orchestration.retry_exponential_backoff must be a boolean")
    slack_webhook_env = alerts.get("slack_webhook_env", "SLACK_WEBHOOK_URL")
    if not isinstance(slack_webhook_env, str) or not slack_webhook_env:
        raise ValueError("failure_alerts.slack_webhook_env must be a non-empty string")
    return OrchestrationSettings(
        schedule=schedule,
        retries=retries,
        retry_delay_minutes=retry_delay_minutes,
        retry_exponential_backoff=retry_exponential_backoff,
        slack_webhook_env=slack_webhook_env,
    )


def send_failure_notification(
    message: str,
    webhook_url: str | None,
    post: Callable[..., httpx.Response] = httpx.post,
) -> None:
    """Post a compact Slack alert when configured, otherwise emit an actionable error log."""

    if not webhook_url:
        logger.error("Airflow ingestion failure: %s", message)
        return
    try:
        response = post(webhook_url, json={"text": message}, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("Unable to send Slack failure notification: %s", message)


def _document_payload(document: SourceDocument) -> Payload:
    return {
        "repository": document.repository,
        "path": document.path,
        "source_url": document.source_url,
        "content": document.content,
        "content_hash": document.content_hash,
        "source_sha": document.source_sha,
    }


def _document_from_payload(payload: Payload) -> SourceDocument:
    return SourceDocument(
        repository=str(payload["repository"]),
        path=str(payload["path"]),
        source_url=str(payload["source_url"]),
        content=str(payload["content"]),
        content_hash=str(payload["content_hash"]),
        source_sha=str(payload["source_sha"]) if payload.get("source_sha") is not None else None,
    )


def _record_payload(record: DocumentRecord) -> Payload:
    return {"id": record.id, "source_url": record.source_url, "content_hash": record.content_hash}


def _record_from_payload(payload: Payload) -> DocumentRecord:
    return DocumentRecord(
        id=str(payload["id"]),
        source_url=str(payload["source_url"]),
        content_hash=str(payload["content_hash"]),
    )


def _chunk_payload(chunk: Chunk) -> Payload:
    return {
        "id": chunk.id,
        "ordinal": chunk.ordinal,
        "content": chunk.content,
        "header_path": list(chunk.header_path),
        "token_count": chunk.token_count,
    }


def _chunk_from_payload(payload: Payload) -> Chunk:
    header_path = payload["header_path"]
    if not isinstance(header_path, list):
        raise ValueError("Chunk header_path must be a list")
    return Chunk(
        id=str(payload["id"]),
        ordinal=int(payload["ordinal"]),
        content=str(payload["content"]),
        header_path=tuple(str(item) for item in header_path),
        token_count=int(payload["token_count"]),
    )


class IngestionWorkflow:
    """Purely staged orchestration, with JSON-compatible payloads between Airflow tasks."""

    def __init__(
        self,
        settings: Settings,
        source: DocumentSource,
        store: WorkflowStore,
        embedder_factory: EmbedderFactory,
    ) -> None:
        self.settings = settings
        self.source = source
        self.store = store
        self.embedder_factory = embedder_factory
        self.chunker = MarkdownChunker(settings.chunk_max_tokens, settings.chunk_overlap_tokens)

    def start_ingestion_run(self) -> Payload:
        """Create an audit row before any source or vector work begins."""

        self.store.ensure_schema()
        return {"run_id": self.store.start_ingestion_run(), "started_at": datetime.now(UTC).isoformat()}

    def fetch_docs(self, run_context: Payload) -> Payload:
        """Fetch a complete source snapshot before comparing or mutating index state."""

        repositories = tuple(self.settings.github_repositories or self.source.list_repositories())
        if not repositories:
            raise RuntimeError("No repositories found for the configured source")
        documents = self.source.crawl(repositories)
        return {
            "run": run_context,
            "repositories": list(repositories),
            "documents": [_document_payload(document) for document in documents],
        }

    def detect_changes(self, snapshot: Payload) -> Payload:
        """Classify the snapshot against the active state-store records."""

        repositories = [str(repository) for repository in snapshot["repositories"]]
        documents = [_document_from_payload(payload) for payload in snapshot["documents"]]
        change_set = ChangeDetector().detect(documents, self.store.list_active_documents(repositories))
        return {
            "run": snapshot["run"],
            "repositories": repositories,
            "new": [_document_payload(document) for document in change_set.new],
            "updated": [_document_payload(document) for document in change_set.updated],
            "unchanged": [_document_payload(document) for document in change_set.unchanged],
            "deleted": [_record_payload(record) for record in change_set.deleted],
        }

    def chunk_changed_docs(self, changes: Payload) -> Payload:
        """Chunk only new and updated documents, leaving unchanged files out of the work set."""

        changed = [*changes["new"], *changes["updated"]]
        return {
            "changes": changes,
            "documents": [
                {
                    "document": document_payload,
                    "chunks": [
                        _chunk_payload(chunk)
                        for chunk in self.chunker.chunk(_document_from_payload(document_payload))
                    ],
                }
                for document_payload in changed
            ],
        }

    def embed_chunks(self, chunked: Payload) -> Payload:
        """Batch-embed changed chunks and retain successful chunks if one fails."""

        embedder = self.embedder_factory()
        embedded_documents: list[Payload] = []
        for item in chunked["documents"]:
            chunks = [_chunk_from_payload(payload) for payload in item["chunks"]]
            result = embedder.embed_resilient([chunk.content for chunk in chunks])
            successful_pairs = sorted(result.vectors, key=lambda pair: pair[0])
            embedded_documents.append(
                {
                    "document": item["document"],
                    "chunks": [_chunk_payload(chunks[index]) for index, _ in successful_pairs],
                    "embeddings": [vector for _, vector in successful_pairs],
                    "failed_chunks": len(result.failed_indices),
                    "skip_persist": bool(chunks) and not successful_pairs,
                }
            )
        return {"changes": chunked["changes"], "documents": embedded_documents}

    def upsert_vector_store(self, embedded: Payload) -> Payload:
        """Replace changed document vectors transactionally; state finalization stays in the next task."""

        chunks_written = 0
        failed_chunks = 0
        skipped_documents = 0
        persisted_source_urls: list[str] = []
        for item in embedded["documents"]:
            failed_chunks += int(item["failed_chunks"])
            document = _document_from_payload(item["document"])
            if item["skip_persist"]:
                skipped_documents += 1
                logger.error("No chunks from %s/%s could be embedded", document.repository, document.path)
                continue
            chunks = [_chunk_from_payload(payload) for payload in item["chunks"]]
            embeddings = [[float(value) for value in vector] for vector in item["embeddings"]]
            self.store.replace_document(document, chunks, embeddings, self.settings.embedding_model)
            persisted_source_urls.append(document.source_url)
            chunks_written += len(chunks)
        return {
            "changes": embedded["changes"],
            "chunks_written": chunks_written,
            "failed_chunks": failed_chunks,
            "skipped_documents": skipped_documents,
            "persisted_source_urls": persisted_source_urls,
        }

    def update_state_store(self, upserted: Payload) -> Payload:
        """Advance timestamps for unchanged files and deactivate deleted documents after writes succeed."""

        changes = upserted["changes"]
        for document_payload in changes["unchanged"]:
            document = _document_from_payload(document_payload)
            self.store.mark_seen(document.source_url, document.source_sha)
        deleted_count = self.store.mark_documents_deleted(
            [_record_from_payload(payload) for payload in changes["deleted"]]
        )
        persisted_source_urls = {str(url) for url in upserted["persisted_source_urls"]}
        return {
            "run": changes["run"],
            "docs_added": sum(
                document["source_url"] in persisted_source_urls for document in changes["new"]
            ),
            "docs_updated": sum(
                document["source_url"] in persisted_source_urls for document in changes["updated"]
            ),
            "docs_deleted": deleted_count,
            "docs_unchanged": len(changes["unchanged"]),
            "chunks_written": int(upserted["chunks_written"]),
            "failed_chunks": int(upserted["failed_chunks"]),
            "skipped_documents": int(upserted["skipped_documents"]),
        }

    def record_run_metadata(self, state_update: Payload) -> Payload:
        """Finish the database audit row and return an XCom-visible operator summary."""

        run_context = state_update["run"]
        self.store.finish_ingestion_run(
            str(run_context["run_id"]),
            docs_added=int(state_update["docs_added"]),
            docs_updated=int(state_update["docs_updated"]),
            docs_deleted=int(state_update["docs_deleted"]),
            chunks_written=int(state_update["chunks_written"]),
        )
        started_at = datetime.fromisoformat(str(run_context["started_at"]))
        duration_seconds = max(0.0, (datetime.now(UTC) - started_at).total_seconds())
        return {
            "run_id": str(run_context["run_id"]),
            "status": "success",
            "documents_processed": (
                int(state_update["docs_added"])
                + int(state_update["docs_updated"])
                + int(state_update["docs_unchanged"])
            ),
            "docs_added": int(state_update["docs_added"]),
            "docs_updated": int(state_update["docs_updated"]),
            "docs_deleted": int(state_update["docs_deleted"]),
            "chunks_written": int(state_update["chunks_written"]),
            "failed_chunks": int(state_update["failed_chunks"]),
            "skipped_documents": int(state_update["skipped_documents"]),
            "duration_seconds": round(duration_seconds, 3),
        }


def build_workflow(settings: Settings | None = None) -> IngestionWorkflow:
    """Build a production workflow without loading the embedding model before the embed task."""

    resolved_settings = settings or get_settings()

    def make_embedder() -> SentenceTransformerEmbedder:
        embedder = SentenceTransformerEmbedder(
            resolved_settings.embedding_model,
            batch_size=resolved_settings.embedding_batch_size,
        )
        if embedder.dimension != resolved_settings.embedding_dimension:
            raise ValueError(
                "Configured EMBEDDING_DIMENSION does not match the selected "
                "sentence-transformers model"
            )
        return embedder

    return IngestionWorkflow(
        settings=resolved_settings,
        source=build_document_source(resolved_settings),
        store=PgVectorStore(resolved_settings.database_url, resolved_settings.embedding_dimension),
        embedder_factory=make_embedder,
    )
