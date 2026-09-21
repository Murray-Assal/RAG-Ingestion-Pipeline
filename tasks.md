# RAG Ingestion Pipeline — Tasks

Each task references the requirement(s) and design section it implements. Work top to bottom — later phases depend on earlier ones. Check a box only once the task's tests pass.

## Phase 1: Scaffolding

- [x] 1.1 Create repo structure per design §5 (`src/`, `dags/`, `api/`, `config/`, `tests/`)
- [x] 1.2 Write `docker-compose.yml`: postgres (pgvector-enabled image), airflow-webserver, airflow-scheduler, api service — design §8
- [x] 1.3 Write `config/settings.yaml` with source repo, chunk size, embedding model, similarity threshold, schedule — design §7
- [x] 1.4 Write Postgres migration creating `documents`, `chunks` (with `vector(384)` column), `ingestion_runs` tables — design §3

## Phase 2: Core Pipeline Components

- [x] 2.1 Implement `source_connector.py`: fetch markdown files + sha from GitHub API, with local-folder fallback — Req 1.1
- [x] 2.2 Implement `state_store.py`: read/write `documents` and `ingestion_runs` tables — design §2.6
- [x] 2.3 Implement `change_detector.py`: hash comparison producing `{new, updated, unchanged, deleted}` — Req 1.2, 1.3
- [x] 2.4 Unit test: change detector correctly classifies all four states given mock state store data
- [x] 2.5 Implement `chunker.py`: split on markdown headers/paragraphs, preserve fenced code blocks, split oversized sections — Req 2.1, 2.2, 2.4
- [x] 2.6 Unit test: a document with a code block spanning what would be a chunk boundary stays intact
- [x] 2.7 Unit test: chunk metadata includes correct header path and position — Req 2.3
- [x] 2.8 Implement `embedding_service.py`: batched embedding via local sentence-transformers model — Req 3.1, 3.2
- [x] 2.9 Add per-chunk failure handling so one bad chunk doesn't halt the batch — Req 3.3
- [x] 2.10 Implement `vector_store.py`: transactional delete-then-insert per document, similarity search method — Req 4.1–4.3, design §6

## Phase 3: Orchestration

- [x] 3.1 Write `dags/ingest_docs_dag.py` wiring: fetch → detect → chunk → embed → upsert → update_state → record_run — design §4
- [x] 3.2 Configure task-level retry policy (e.g. 2 retries, exponential backoff) — Req 5.2
- [x] 3.3 Add `on_failure_callback` posting to Slack webhook (fallback to log if unset) — Req 5.3
- [x] 3.4 Add final task recording run stats to `ingestion_runs` — Req 5.4, 4.4
- [x] 3.5 Set DAG schedule from `config/settings.yaml` — Req 5.1
- [x] 3.6 Integration test: run DAG twice back-to-back with no source changes; assert zero writes on second run — Non-functional Req 1
- [x] 3.7 Integration test: modify one source doc; assert only that doc's chunks are replaced, others untouched

## Phase 4: Retrieval API

- [x] 4.1 Implement `api/main.py` `/query` endpoint: embed query, similarity search, return top-k with metadata — Req 6.1
- [x] 4.2 Add similarity threshold filtering (return empty list below threshold, not noise) — Req 6.2
- [x] 4.3 Add error handling for empty/unreachable vector store (clear error, not empty 200) — Req 6.3
- [x] 4.4 Manual test: query against a freshly ingested corpus, confirm relevant chunks return with correct source paths

## Phase 5: Optional Stretch — Answer Generation

- [x] 5.1 Implement answer-generation module behind a config flag, decoupled from core retrieval — Req 7.1, 7.2
- [x] 5.2 Confirm `/query` still works with the flag off (core pipeline has zero dependency on this module)

## Phase 6: Portfolio Polish

- [x] 6.1 Write README: architecture diagram (reuse design §1 mermaid), setup steps, sample query + response screenshot
- [x] 6.2 Add GitHub Actions workflow running unit + integration tests on push
- [x] 6.3 Record a short demo (ingest run → query → response) for the portfolio writeup
