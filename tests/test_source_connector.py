import hashlib

from rag_ingestion.source_connector import LocalDocumentSource


def test_local_source_reads_markdown_recursively_and_ignores_non_document_files(tmp_path) -> None:
    docs = tmp_path / "docs"
    nested = docs / "guides"
    nested.mkdir(parents=True)
    content = "# Local guide\n"
    (docs / "README.md").write_text(content, encoding="utf-8")
    (nested / "install.mdx").write_text("# Install\n", encoding="utf-8")
    (docs / "notes.txt").write_text("not documentation", encoding="utf-8")

    source = LocalDocumentSource(docs, max_document_bytes=1_024)
    documents = source.crawl(source.list_repositories())

    documents_by_path = {document.path: document for document in documents}
    assert set(documents_by_path) == {"README.md", "guides/install.mdx"}
    readme = documents_by_path["README.md"]
    assert readme.content_hash == hashlib.sha256(readme.content.encode()).hexdigest()
    assert readme.source_url.startswith("local://docs/")
