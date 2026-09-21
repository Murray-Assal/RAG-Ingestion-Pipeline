"""FastAPI entry point for local development and the retrieval API contract."""

from rag_ingestion.api import app, create_app

__all__ = ["app", "create_app"]
