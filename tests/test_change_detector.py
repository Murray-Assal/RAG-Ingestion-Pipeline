from rag_ingestion.change_detector import ChangeDetector
from rag_ingestion.models import DocumentRecord, SourceDocument


def document(name: str, content_hash: str) -> SourceDocument:
    return SourceDocument(
        repository="demo",
        path=f"docs/{name}.md",
        source_url=f"https://github.com/example/demo/blob/main/docs/{name}.md",
        content=f"# {name}",
        content_hash=content_hash,
    )


def record(name: str, content_hash: str) -> DocumentRecord:
    source_url = f"https://github.com/example/demo/blob/main/docs/{name}.md"
    return DocumentRecord(id=f"{name}-id", source_url=source_url, content_hash=content_hash)


def test_change_detector_classifies_new_updated_unchanged_and_deleted_documents() -> None:
    changes = ChangeDetector().detect(
        [
            document("unchanged", "same"),
            document("updated", "new-hash"),
            document("new", "new-hash"),
        ],
        [
            record("unchanged", "same"),
            record("updated", "old-hash"),
            record("deleted", "old-hash"),
        ],
    )

    assert [item.path for item in changes.new] == ["docs/new.md"]
    assert [item.path for item in changes.updated] == ["docs/updated.md"]
    assert [item.path for item in changes.unchanged] == ["docs/unchanged.md"]
    assert [item.source_url for item in changes.deleted] == [record("deleted", "old-hash").source_url]
