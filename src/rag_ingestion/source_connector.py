"""Source-selection and local Markdown document ingestion."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import quote

from rag_ingestion.config import Settings
from rag_ingestion.models import SourceDocument

_MARKDOWN_EXTENSIONS = (".md", ".mdx", ".rst")


class LocalDocumentSourceError(RuntimeError):
    """A local documentation snapshot could not be read safely."""


def is_document_path(path: str) -> bool:
    """Return whether a repository-relative path is a supported documentation file."""

    return path.lower().endswith(_MARKDOWN_EXTENSIONS)


class LocalDocumentSource:
    """Read Markdown files from a configured directory using stable local source URLs."""

    def __init__(self, root: Path, max_document_bytes: int) -> None:
        self.root = root.expanduser().resolve()
        self.max_document_bytes = max_document_bytes
        self.repository = self.root.name or "local-docs"

    def list_repositories(self) -> list[str]:
        return [self.repository]

    def crawl(self, repositories: Sequence[str]) -> list[SourceDocument]:
        """Return one complete, sorted snapshot or fail before yielding partial documents."""

        if repositories and self.repository not in repositories:
            return []
        if not self.root.is_dir():
            raise LocalDocumentSourceError(f"Local documentation directory does not exist: {self.root}")

        documents: list[SourceDocument] = []
        for file_path in sorted(self.root.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                resolved_path = file_path.resolve(strict=True)
                relative_path = resolved_path.relative_to(self.root).as_posix()
            except (OSError, ValueError):
                # Ignore symlinks that escape the configured source root.
                continue
            if not is_document_path(relative_path):
                continue
            try:
                raw = resolved_path.read_bytes()
            except OSError as exc:
                raise LocalDocumentSourceError(f"Could not read {relative_path}: {exc}") from exc
            if len(raw) > self.max_document_bytes:
                raise LocalDocumentSourceError(
                    f"Documentation file {relative_path} exceeds max_document_bytes"
                )
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LocalDocumentSourceError(f"Documentation file {relative_path} is not UTF-8") from exc
            content_hash = hashlib.sha256(raw).hexdigest()
            documents.append(
                SourceDocument(
                    repository=self.repository,
                    path=relative_path,
                    source_url=(
                        f"local://{quote(self.repository, safe='')}/{quote(relative_path, safe='/')}"
                    ),
                    content=content,
                    content_hash=content_hash,
                    source_sha=content_hash,
                )
            )
        return documents


def build_document_source(settings: Settings):
    """Select local ingestion when configured; GitHub remains the production default."""

    if settings.local_docs_path is not None:
        return LocalDocumentSource(settings.local_docs_path, settings.max_document_bytes)
    # Import lazily to keep this module independent of the HTTP client.
    from rag_ingestion.github_source import GitHubDocumentSource

    return GitHubDocumentSource(settings)
