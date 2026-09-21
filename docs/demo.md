# Ingest-to-query walkthrough

This short, replayable terminal capture is the portfolio demo for the project. It uses the production Docker commands; the document and chunk counts shown are representative because they depend on the configured repository snapshot.

1. Start the pgvector database and FastAPI service.
2. Run `rag-ingest`, which snapshots the source and only embeds changed documents.
3. Submit a question to `POST /query` and receive a score, header path, source URL, and content.

To replay the terminal capture locally with [asciinema](https://asciinema.org/):

```bash
asciinema play docs/demo.cast
```

The same flow is shown in the [SVG preview](assets/ingest-to-query-demo.svg).
