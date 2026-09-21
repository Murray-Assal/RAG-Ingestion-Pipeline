from rag_ingestion.chunking import MarkdownChunker, estimate_tokens
from rag_ingestion.models import SourceDocument


def make_document(content: str) -> SourceDocument:
    return SourceDocument(
        repository="demo",
        path="README.md",
        source_url="https://github.com/example/demo/blob/main/README.md",
        content=content,
        content_hash="a" * 64,
    )


def test_fenced_code_block_is_never_split_and_keeps_its_section() -> None:
    code = "```python\n" + "\n".join(f"print({number})" for number in range(80)) + "\n```"
    document = make_document(f"# Install\n\nUse this command.\n\n{code}\n\n## Verify\n\nRun the check.")

    chunks = MarkdownChunker(max_tokens=50, overlap_tokens=5).chunk(document)

    code_chunks = [chunk for chunk in chunks if "```python" in chunk.content]
    assert len(code_chunks) == 1
    assert code in code_chunks[0].content
    assert code_chunks[0].header_path == ("Install",)
    assert any(chunk.header_path == ("Install", "Verify") for chunk in chunks)


def test_long_prose_splits_at_bounded_word_or_sentence_boundaries() -> None:
    prose = " ".join(f"word{number}" for number in range(120))
    chunks = MarkdownChunker(max_tokens=40, overlap_tokens=5).chunk(make_document(f"# Notes\n\n{prose}"))

    assert len(chunks) > 1
    assert all(chunk.header_path == ("Notes",) for chunk in chunks)
    assert all(estimate_tokens(chunk.content) <= 40 for chunk in chunks)


def test_chunk_metadata_preserves_nested_header_path_and_position() -> None:
    document = make_document("# Install\n\nOverview.\n\n## Prerequisites\n\nPython is required.")

    chunks = MarkdownChunker(max_tokens=32, overlap_tokens=4).chunk(document)

    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert [chunk.header_path for chunk in chunks] == [("Install",), ("Install", "Prerequisites")]
