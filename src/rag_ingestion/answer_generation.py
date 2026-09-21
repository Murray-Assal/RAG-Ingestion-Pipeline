"""Optional local-LLM answer synthesis, isolated from ingestion and retrieval."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import httpx

from rag_ingestion.models import SearchMatch


class AnswerGenerationError(RuntimeError):
    """The optional answer provider could not create a grounded response."""


class OllamaAnswerGenerator:
    """Synthesize an answer with a locally hosted Ollama model and explicit citations."""

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout_seconds: float,
        post: Callable[..., httpx.Response] = httpx.post,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.post = post

    def generate(self, question: str, matches: Sequence[SearchMatch]) -> str:
        """Return a concise answer grounded solely in the retrieved, cited documentation."""

        if not matches:
            raise AnswerGenerationError("No retrieved chunks are available to ground an answer")
        prompt = self._build_prompt(question, matches)
        try:
            response = self.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AnswerGenerationError("The configured local answer model is unavailable") from exc
        answer = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(answer, str) or not answer.strip():
            raise AnswerGenerationError("The configured local answer model returned no answer")
        return self._ensure_citations(answer.strip(), matches)

    @staticmethod
    def _build_prompt(question: str, matches: Sequence[SearchMatch]) -> str:
        context = "\n\n".join(
            (
                f"Source path: {match.path}\n"
                f"Source URL: {match.source_url}\n"
                f"Section: {' > '.join(match.header_path) or 'root'}\n"
                f"Content:\n{match.content}"
            )
            for match in matches
        )
        return (
            "Answer the question using only the documentation context below. "
            "If the context does not support an answer, say so. Cite source file paths in square "
            "brackets, for example [docs/install.md]. Do not invent facts or citations.\n\n"
            f"Question: {question}\n\nDocumentation context:\n{context}"
        )

    @staticmethod
    def _ensure_citations(answer: str, matches: Sequence[SearchMatch]) -> str:
        """Guarantee source-path provenance even when the local model omits its citations."""

        source_paths = list(dict.fromkeys(match.path for match in matches))
        if any(f"[{path}]" in answer for path in source_paths):
            return answer
        citations = ", ".join(f"[{path}]" for path in source_paths)
        return f"{answer}\n\nSources: {citations}"


def build_answer_generator(
    model: str,
    base_url: str,
    timeout_seconds: float,
) -> OllamaAnswerGenerator:
    """Create the optional provider only when answer generation has been enabled."""

    return OllamaAnswerGenerator(model, base_url, timeout_seconds)
