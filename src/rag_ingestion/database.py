"""Transactional pgvector persistence and similarity search."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json

from rag_ingestion.models import Chunk, DocumentRecord, SearchMatch, SourceDocument


def _vector_literal(values: Sequence[float]) -> str:
    """Serialize a vector for Postgres's vector input syntax without SQL interpolation."""

    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


class PgVectorStore:
    """Storage adapter with document-level idempotency and chunk-level replacement."""

    def __init__(self, database_url: str, embedding_dimension: int) -> None:
        self.database_url = database_url
        self.embedding_dimension = embedding_dimension

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            yield connection

    def ensure_schema(self) -> None:
        """Create the extension, tables, and ANN index if they do not exist."""

        dimension = int(self.embedding_dimension)
        if dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        statements = f"""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                repository TEXT NOT NULL,
                path TEXT NOT NULL,
                source_url TEXT NOT NULL UNIQUE,
                source_sha TEXT,
                content_hash TEXT NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS documents_repository_active_idx
                ON documents (repository, is_active);

            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                header_path JSONB NOT NULL DEFAULT '[]'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding vector({dimension}) NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (document_id, ordinal)
            );

            CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks USING hnsw (embedding vector_cosine_ops);

            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id UUID PRIMARY KEY,
                started_at TIMESTAMPTZ NOT NULL,
                finished_at TIMESTAMPTZ,
                status TEXT NOT NULL CHECK (status IN ('running', 'success', 'failed')),
                docs_added INTEGER NOT NULL DEFAULT 0,
                docs_updated INTEGER NOT NULL DEFAULT 0,
                docs_deleted INTEGER NOT NULL DEFAULT 0,
                chunks_written INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            );

            CREATE INDEX IF NOT EXISTS ingestion_runs_started_at_idx
                ON ingestion_runs (started_at DESC);
        """
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(statements)

    def get_document(self, source_url: str) -> DocumentRecord | None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT id::text, source_url, content_hash FROM documents WHERE source_url = %s",
                (source_url,),
            )
            row = cursor.fetchone()
        return DocumentRecord(**row) if row else None

    def list_active_documents(self, repositories: Sequence[str]) -> list[DocumentRecord]:
        """Read the current state snapshot used by the change detector."""

        if not repositories:
            return []
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id::text, source_url, content_hash
                FROM documents
                WHERE repository = ANY(%s) AND is_active = TRUE
                """,
                (list(repositories),),
            )
            rows = cursor.fetchall()
        return [DocumentRecord(**row) for row in rows]

    def mark_seen(self, source_url: str, source_sha: str | None) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                SET source_sha = %s, is_active = TRUE, last_seen_at = NOW()
                WHERE source_url = %s
                """,
                (source_sha, source_url),
            )

    def mark_documents_deleted(self, documents: Sequence[DocumentRecord]) -> int:
        """Deactivate missing documents and remove their vectors in one transaction."""

        source_urls = [document.source_url for document in documents]
        if not source_urls:
            return 0
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE documents
                SET is_active = FALSE, updated_at = NOW()
                WHERE source_url = ANY(%s) AND is_active = TRUE
                RETURNING id
                """,
                (source_urls,),
            )
            document_ids = [row["id"] for row in cursor.fetchall()]
            if document_ids:
                cursor.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", (document_ids,))
            return len(document_ids)

    def replace_document(
        self,
        document: SourceDocument,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
        embedding_model: str,
    ) -> None:
        """Atomically upsert a changed document and replace all of its chunk vectors."""

        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        if any(len(vector) != self.embedding_dimension for vector in embeddings):
            raise ValueError(
                f"Embedding dimension must be {self.embedding_dimension} for the configured pgvector column"
            )
        document_id = uuid.uuid5(uuid.NAMESPACE_URL, document.source_url)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (
                    id, repository, path, source_url, source_sha, content_hash, is_active,
                    first_indexed_at, last_seen_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, TRUE, NOW(), NOW(), NOW())
                ON CONFLICT (source_url) DO UPDATE SET
                    repository = EXCLUDED.repository,
                    path = EXCLUDED.path,
                    source_sha = EXCLUDED.source_sha,
                    content_hash = EXCLUDED.content_hash,
                    is_active = TRUE,
                    last_seen_at = NOW(),
                    updated_at = NOW()
                RETURNING id
                """,
                (
                    document_id,
                    document.repository,
                    document.path,
                    document.source_url,
                    document.source_sha,
                    document.content_hash,
                ),
            )
            document_row = cursor.fetchone()
            if not document_row:
                raise RuntimeError("Document upsert did not return an id")
            persisted_document_id = document_row["id"]
            cursor.execute("DELETE FROM chunks WHERE document_id = %s", (persisted_document_id,))
            if chunks:
                cursor.executemany(
                    """
                    INSERT INTO chunks (
                        id, document_id, ordinal, content, token_count, header_path, metadata,
                        content_hash, embedding_model, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                    )
                    """,
                    [
                        (
                            uuid.UUID(chunk.id),
                            persisted_document_id,
                            chunk.ordinal,
                            chunk.content,
                            chunk.token_count,
                            Json(list(chunk.header_path)),
                            Json({"repository": document.repository, "path": document.path}),
                            document.content_hash,
                            embedding_model,
                            _vector_literal(vector),
                        )
                        for chunk, vector in zip(chunks, embeddings, strict=True)
                    ],
                )

    def delete_missing_documents(self, repositories: Sequence[str], seen_urls: Sequence[str]) -> int:
        """Delete indexed documents removed from a complete source snapshot."""

        if not repositories:
            return 0
        with self._connection() as connection, connection.cursor() as cursor:
            if seen_urls:
                cursor.execute(
                    """
                    DELETE FROM documents
                    WHERE repository = ANY(%s) AND source_url <> ALL(%s)
                    """,
                    (list(repositories), list(seen_urls)),
                )
            else:
                cursor.execute("DELETE FROM documents WHERE repository = ANY(%s)", (list(repositories),))
            return cursor.rowcount

    def start_ingestion_run(self) -> str:
        """Create an auditable run record before orchestration writes its final counters."""

        run_id = uuid.uuid4()
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO ingestion_runs (id, started_at, status)
                VALUES (%s, NOW(), 'running')
                """,
                (run_id,),
            )
        return str(run_id)

    def finish_ingestion_run(
        self,
        run_id: str,
        *,
        docs_added: int,
        docs_updated: int,
        docs_deleted: int,
        chunks_written: int,
    ) -> None:
        """Persist successful run metrics for Airflow and operational audit queries."""

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = NOW(), status = 'success', docs_added = %s,
                    docs_updated = %s, docs_deleted = %s, chunks_written = %s,
                    error_message = NULL
                WHERE id = %s
                """,
                (docs_added, docs_updated, docs_deleted, chunks_written, uuid.UUID(run_id)),
            )

    def fail_ingestion_run(self, run_id: str, error_message: str) -> None:
        """Persist a concise failure reason while preserving whatever data existed before the run."""

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE ingestion_runs
                SET finished_at = NOW(), status = 'failed', error_message = %s
                WHERE id = %s
                """,
                (error_message[:4_000], uuid.UUID(run_id)),
            )

    def get_ingestion_run(self, run_id: str) -> dict[str, object] | None:
        """Read an ingestion-run record for task-level observability and tests."""

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id::text AS id, started_at, finished_at, status, docs_added,
                    docs_updated, docs_deleted, chunks_written, error_message
                FROM ingestion_runs
                WHERE id = %s
                """,
                (uuid.UUID(run_id),),
            )
            return cursor.fetchone()

    def search(self, query_embedding: Sequence[float], limit: int) -> list[SearchMatch]:
        """Return active chunks nearest to the normalized query embedding by cosine distance."""

        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(f"Query embedding must have {self.embedding_dimension} values")
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    c.content,
                    d.source_url,
                    d.repository,
                    d.path,
                    c.header_path,
                    1 - (c.embedding <=> %s::vector) AS score
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.is_active = TRUE
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (_vector_literal(query_embedding), _vector_literal(query_embedding), limit),
            )
            rows = cursor.fetchall()
        return [
            SearchMatch(
                content=row["content"],
                source_url=row["source_url"],
                repository=row["repository"],
                path=row["path"],
                header_path=tuple(row["header_path"]),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def active_chunk_count(self) -> int:
        """Return the active index size so the API can distinguish empty from irrelevant results."""

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS count
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.is_active = TRUE
                """
            )
            row = cursor.fetchone()
        return int(row["count"]) if row else 0

    def healthcheck(self) -> bool:
        """Return whether Postgres is reachable and can execute a trivial query."""

        try:
            with self._connection() as connection, connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                return cursor.fetchone() is not None
        except psycopg.Error:
            return False
