"""Embedding adapter that keeps the project free of paid model APIs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbeddingBatchResult:
    """Successful vectors and failed input positions from a resilient batch request."""

    vectors: tuple[tuple[int, list[float]], ...]
    failed_indices: tuple[int, ...]


class EmbeddingProvider(Protocol):
    """Minimal contract used by both the ingestion pipeline and API."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_resilient(self, texts: Sequence[str]) -> EmbeddingBatchResult: ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_resilient(self, texts: Sequence[str]) -> EmbeddingBatchResult:
        """Embed in batches, isolating a failed batch down to its individual bad chunks."""

        vectors: list[tuple[int, list[float]]] = []
        failed_indices: list[int] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                batch_vectors = self.embed(batch)
                if len(batch_vectors) != len(batch):
                    raise RuntimeError("Embedding provider returned a different number of vectors than texts")
                vectors.extend((start + index, vector) for index, vector in enumerate(batch_vectors))
            except Exception:
                logger.exception(
                    "Embedding batch %s-%s failed; retrying each chunk so the run can continue",
                    start,
                    start + len(batch) - 1,
                )
                for index, text in enumerate(batch, start=start):
                    try:
                        vector = self.embed([text])
                        if len(vector) != 1:
                            raise RuntimeError("Embedding provider returned an invalid single-chunk response")
                        vectors.append((index, vector[0]))
                    except Exception:
                        logger.exception("Skipping chunk %s after embedding failure", index)
                        failed_indices.append(index)
        return EmbeddingBatchResult(tuple(vectors), tuple(failed_indices))
