from collections.abc import Sequence

from rag_ingestion.embeddings import SentenceTransformerEmbedder


class FlakyEmbedder(SentenceTransformerEmbedder):
    def __init__(self) -> None:
        super().__init__("unused", batch_size=4)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) > 1:
            raise RuntimeError("simulated batch failure")
        if texts[0] == "broken":
            raise RuntimeError("simulated chunk failure")
        return [[0.1, 0.2, 0.3]]


def test_embedding_failure_isolated_to_bad_chunk_after_batch_retry() -> None:
    result = FlakyEmbedder().embed_resilient(["healthy", "broken", "also healthy"])

    assert [index for index, _ in result.vectors] == [0, 2]
    assert result.failed_indices == (1,)
