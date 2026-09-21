"""Deterministic document change classification for incremental ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag_ingestion.models import DocumentRecord, SourceDocument


@dataclass(frozen=True, slots=True)
class ChangeSet:
    """All source changes relative to the active documents persisted in Postgres."""

    new: tuple[SourceDocument, ...]
    updated: tuple[SourceDocument, ...]
    unchanged: tuple[SourceDocument, ...]
    deleted: tuple[DocumentRecord, ...]

    @property
    def changed(self) -> tuple[SourceDocument, ...]:
        """Documents which require chunking and embedding in this run."""

        return (*self.new, *self.updated)


class ChangeDetector:
    """Compare a complete source snapshot to persisted content hashes."""

    def detect(
        self,
        fetched_documents: Sequence[SourceDocument],
        existing_documents: Iterable[DocumentRecord],
    ) -> ChangeSet:
        existing_by_url = {document.source_url: document for document in existing_documents}
        fetched_urls: set[str] = set()
        new: list[SourceDocument] = []
        updated: list[SourceDocument] = []
        unchanged: list[SourceDocument] = []

        for document in fetched_documents:
            if document.source_url in fetched_urls:
                raise ValueError(
                    f"Source snapshot contains duplicate document URL: {document.source_url}"
                )
            fetched_urls.add(document.source_url)
            existing = existing_by_url.get(document.source_url)
            if existing is None:
                new.append(document)
            elif existing.content_hash == document.content_hash:
                unchanged.append(document)
            else:
                updated.append(document)

        deleted = tuple(
            document
            for source_url, document in existing_by_url.items()
            if source_url not in fetched_urls
        )
        return ChangeSet(tuple(new), tuple(updated), tuple(unchanged), deleted)
