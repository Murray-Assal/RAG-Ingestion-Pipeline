from __future__ import annotations

from collections.abc import Sequence

import psycopg
from fastapi.testclient import TestClient

from rag_ingestion.api import RetrievalService, create_app
from rag_ingestion.config import Settings
from rag_ingestion.models import SearchMatch


class FakeEmbedder:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.questions.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    def __init__(self, matches: list[SearchMatch], active_chunk_count: int | Exception = 1) -> None:
        self.matches = matches
        self.active_chunk_count_value = active_chunk_count
        self.requested_limit: int | None = None

    def active_chunk_count(self) -> int:
        if isinstance(self.active_chunk_count_value, Exception):
            raise self.active_chunk_count_value
        return self.active_chunk_count_value

    def search(self, query_embedding: Sequence[float], limit: int) -> list[SearchMatch]:
        self.requested_limit = limit
        return self.matches[:limit]

    def healthcheck(self) -> bool:
        return True


def match(score: float) -> SearchMatch:
    return SearchMatch(
        content="Run `rag-ingest` to refresh the index.",
        source_url="https://github.com/example/demo/blob/main/README.md",
        repository="demo",
        path="README.md",
        header_path=("Usage",),
        score=score,
    )


def client_for(store: FakeStore, *, threshold: float = 0.35) -> TestClient:
    settings = Settings(embedding_dimension=3, similarity_threshold=threshold, top_k=5)
    service = RetrievalService(FakeStoreAdapter(store), FakeEmbedder(), default_top_k=5, similarity_threshold=threshold)
    return TestClient(create_app(service=service, settings=settings))


class FakeStoreAdapter:
    """Keeps the fake's focused test surface compatible with the production store type."""

    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def active_chunk_count(self) -> int:
        return self.store.active_chunk_count()

    def search(self, query_embedding: Sequence[float], limit: int) -> list[SearchMatch]:
        return self.store.search(query_embedding, limit)

    def healthcheck(self) -> bool:
        return self.store.healthcheck()


class FakeAnswerGenerator:
    def __init__(self, answer: str | None = "Use the ingestion command [README.md].") -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[SearchMatch]]] = []

    def generate(self, question: str, matches: Sequence[SearchMatch]) -> str:
        self.calls.append((question, list(matches)))
        if self.answer is None:
            raise RuntimeError("local model unavailable")
        return self.answer


def enabled_settings() -> Settings:
    return Settings(
        embedding_dimension=3,
        similarity_threshold=0.35,
        top_k=5,
        answer_generation_enabled=True,
    )


def test_query_returns_top_k_grounded_chunks_with_source_metadata() -> None:
    store = FakeStore([match(0.95), match(0.91)])

    with client_for(store) as client:
        response = client.post("/query", json={"question": "How do I refresh docs?", "top_k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "How do I refresh docs?"
    assert body["results"][0]["path"] == "README.md"
    assert body["results"][0]["header_path"] == ["Usage"]
    assert store.requested_limit == 1


def test_query_filters_matches_below_configured_similarity_threshold() -> None:
    store = FakeStore([match(0.34)])

    with client_for(store, threshold=0.35) as client:
        response = client.post("/query", json={"question": "An unrelated question"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_query_returns_clear_error_for_empty_index() -> None:
    with client_for(FakeStore([], active_chunk_count=0)) as client:
        response = client.post("/query", json={"question": "How do I refresh docs?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "empty_index"


def test_query_returns_clear_error_when_vector_store_is_unreachable() -> None:
    unavailable = psycopg.OperationalError("connection refused")

    with client_for(FakeStore([], active_chunk_count=unavailable)) as client:
        response = client.post("/query", json={"question": "How do I refresh docs?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "vector_store_unavailable"


def test_query_optionally_generates_a_grounded_answer_when_enabled() -> None:
    store = FakeStore([match(0.95)])
    generator = FakeAnswerGenerator()
    service = RetrievalService(
        FakeStoreAdapter(store),
        FakeEmbedder(),
        default_top_k=5,
        similarity_threshold=0.35,
    )

    with TestClient(create_app(service=service, settings=enabled_settings(), answer_generator=generator)) as client:
        response = client.post("/query", json={"question": "How do I refresh docs?"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Use the ingestion command [README.md]."
    assert response.json()["answer_status"] == "generated"
    assert generator.calls[0][1][0].path == "README.md"


def test_query_still_returns_context_when_enabled_answer_provider_is_unavailable() -> None:
    store = FakeStore([match(0.95)])
    service = RetrievalService(
        FakeStoreAdapter(store),
        FakeEmbedder(),
        default_top_k=5,
        similarity_threshold=0.35,
    )

    with TestClient(
        create_app(service=service, settings=enabled_settings(), answer_generator=FakeAnswerGenerator(None))
    ) as client:
        response = client.post("/query", json={"question": "How do I refresh docs?"})

    assert response.status_code == 200
    assert response.json()["results"][0]["path"] == "README.md"
    assert response.json()["answer"] is None
    assert response.json()["answer_status"] == "unavailable"


def test_query_has_zero_answer_generator_dependency_when_flag_is_off(monkeypatch) -> None:
    store = FakeStore([match(0.95)])
    service = RetrievalService(
        FakeStoreAdapter(store),
        FakeEmbedder(),
        default_top_k=5,
        similarity_threshold=0.35,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("answer generator must not be built while the flag is off")

    monkeypatch.setattr("rag_ingestion.answer_generation.build_answer_generator", fail_if_called)
    disabled_settings = Settings(
        embedding_dimension=3,
        similarity_threshold=0.35,
        top_k=5,
        answer_generation_enabled=False,
    )
    with TestClient(create_app(service=service, settings=disabled_settings)) as client:
        response = client.post("/query", json={"question": "How do I refresh docs?"})

    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["answer_status"] == "disabled"
