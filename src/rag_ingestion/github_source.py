"""GitHub REST source for READMEs and documentation files."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Sequence
from urllib.parse import quote

import httpx

from rag_ingestion.config import Settings
from rag_ingestion.models import SourceDocument
from rag_ingestion.source_connector import is_document_path


class GitHubSourceError(RuntimeError):
    """GitHub could not provide a complete, safe source snapshot."""


class GitHubDocumentSource:
    """Fetch repository READMEs and files under documentation-oriented paths."""

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "rag-ingestion-pipeline",
        }
        if settings.github_token:
            headers["Authorization"] = f"Bearer {settings.github_token}"
        self.client = client or httpx.Client(
            base_url=settings.github_api_url.rstrip("/"),
            headers=headers,
            timeout=settings.request_timeout_seconds,
            follow_redirects=True,
        )

    def list_repositories(self) -> list[str]:
        """List non-fork repositories when no explicit allow-list is configured."""

        repositories: list[str] = []
        page = 1
        while True:
            payload = self._get_json(
                f"/users/{self.settings.github_owner}/repos",
                params={"type": "owner", "sort": "updated", "per_page": 100, "page": page},
            )
            if not isinstance(payload, list):
                raise GitHubSourceError("GitHub returned an unexpected repository list")
            repositories.extend(
                item["name"]
                for item in payload
                if isinstance(item, dict) and not item.get("fork", False) and item.get("name")
            )
            if len(payload) < 100:
                return repositories
            page += 1

    def crawl(self, repositories: Sequence[str]) -> list[SourceDocument]:
        """Retrieve all supported documentation files from the supplied repositories."""

        documents: list[SourceDocument] = []
        for repository in repositories:
            branch = self._default_branch(repository)
            tree = self._get_json(
                f"/repos/{self.settings.github_owner}/{quote(repository, safe='')}/git/trees/{quote(branch, safe='')}",
                params={"recursive": "1"},
            )
            if not isinstance(tree, dict) or tree.get("truncated"):
                raise GitHubSourceError(
                    f"Repository tree for {repository!r} is unavailable or truncated; refusing partial index"
                )
            entries = tree.get("tree", [])
            if not isinstance(entries, list):
                raise GitHubSourceError(f"Repository tree for {repository!r} has an invalid shape")
            paths = sorted(
                item["path"]
                for item in entries
                if isinstance(item, dict)
                and item.get("type") == "blob"
                and isinstance(item.get("path"), str)
                and is_document_path(item["path"])
            )
            documents.extend(self._fetch_document(repository, branch, path) for path in paths)
        return documents

    def _default_branch(self, repository: str) -> str:
        repo = self._get_json(f"/repos/{self.settings.github_owner}/{quote(repository, safe='')}")
        branch = repo.get("default_branch") if isinstance(repo, dict) else None
        if not isinstance(branch, str) or not branch:
            raise GitHubSourceError(f"Could not find a default branch for {repository!r}")
        return branch

    def _fetch_document(self, repository: str, branch: str, path: str) -> SourceDocument:
        encoded_path = quote(path, safe="/")
        payload = self._get_json(
            f"/repos/{self.settings.github_owner}/{quote(repository, safe='')}/contents/{encoded_path}",
            params={"ref": branch},
        )
        if not isinstance(payload, dict) or payload.get("type") != "file":
            raise GitHubSourceError(f"Expected a file response for {repository}/{path}")
        content = payload.get("content")
        encoding = payload.get("encoding")
        if not isinstance(content, str) or encoding != "base64":
            raise GitHubSourceError(f"GitHub did not return base64 content for {repository}/{path}")
        raw = base64.b64decode(content, validate=False)
        if len(raw) > self.settings.max_document_bytes:
            raise GitHubSourceError(f"Documentation file {repository}/{path} exceeds max_document_bytes")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GitHubSourceError(f"Documentation file {repository}/{path} is not UTF-8") from exc
        source_url = (
            f"https://github.com/{self.settings.github_owner}/{repository}/blob/{branch}/"
            f"{quote(path, safe='/')}"
        )
        return SourceDocument(
            repository=repository,
            path=path,
            source_url=source_url,
            content=text,
            content_hash=hashlib.sha256(raw).hexdigest(),
            source_sha=payload.get("sha") if isinstance(payload.get("sha"), str) else None,
        )

    def _get_json(self, path: str, params: dict[str, object] | None = None) -> object:
        try:
            response = self.client.get(path, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GitHubSourceError(f"GitHub request failed for {path}: {exc}") from exc
