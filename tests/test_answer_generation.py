from rag_ingestion.answer_generation import OllamaAnswerGenerator
from rag_ingestion.models import SearchMatch


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, str]:
        return {"response": "Run the ingestion command."}


def match() -> SearchMatch:
    return SearchMatch(
        content="Use `rag-ingest` to refresh docs.",
        source_url="https://github.com/example/demo/blob/main/docs/setup.md",
        repository="demo",
        path="docs/setup.md",
        header_path=("Setup",),
        score=0.95,
    )


def test_ollama_generator_passes_retrieved_context_and_adds_missing_path_citation() -> None:
    request: dict[str, object] = {}

    def post(url: str, **kwargs) -> FakeResponse:
        request["url"] = url
        request.update(kwargs)
        return FakeResponse()

    generator = OllamaAnswerGenerator("llama3.2", "http://ollama:11434/", 5, post=post)
    answer = generator.generate("How do I refresh docs?", [match()])

    assert request["url"] == "http://ollama:11434/api/generate"
    assert "docs/setup.md" in request["json"]["prompt"]
    assert "[docs/setup.md]" in answer
