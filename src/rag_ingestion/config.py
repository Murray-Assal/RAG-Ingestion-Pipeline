"""Environment-backed configuration used by the CLI, API, and Airflow DAG."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe defaults for a public GitHub portfolio project."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    github_owner: str = "Murray-Assal"
    github_token: str | None = None
    github_repositories: Annotated[list[str], NoDecode] = Field(default_factory=list)
    github_api_url: str = "https://api.github.com"
    local_docs_path: Path | None = None
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    chunk_max_tokens: int = Field(default=350, ge=32)
    chunk_overlap_tokens: int = Field(default=40, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    similarity_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    answer_generation_enabled: bool = False
    answer_generation_model: str = "llama3.2"
    answer_generation_url: str = "http://host.docker.internal:11434"
    answer_generation_timeout_seconds: float = Field(default=30.0, gt=0)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_document_bytes: int = Field(default=1_000_000, ge=1_024)

    @field_validator("github_repositories", mode="before")
    @classmethod
    def parse_repositories(cls, value: object) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            if value.lstrip().startswith("["):
                parsed = json.loads(value)
                if not isinstance(parsed, list):
                    raise ValueError("GITHUB_REPOSITORIES JSON must be an array")
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in value.split(",") if item.strip()]
        raise ValueError("GITHUB_REPOSITORIES must be a comma-separated string or list")


@lru_cache
def get_settings() -> Settings:
    """Return one immutable settings instance per process."""

    return Settings()
