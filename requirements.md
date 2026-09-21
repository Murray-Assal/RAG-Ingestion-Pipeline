# RAG Ingestion Pipeline — Requirements

## Overview
A data engineering pipeline that ingests documentation (GitHub repo docs / markdown), chunks and embeds it, stores it in a vector database, and keeps the index fresh via scheduled incremental re-indexing. A thin retrieval API sits on top to prove the pipeline works end-to-end. The focus is the **pipeline and orchestration**, not the LLM.

## 1. Document Ingestion

**User Story:** As a pipeline operator, I want the system to pull documentation from a configured source, so that new and updated docs enter the pipeline automatically.

**Acceptance Criteria:**
1. WHEN the ingestion job runs THEN the system SHALL fetch all markdown files from a configured GitHub repo (or local docs folder as a fallback source).
2. WHEN a fetched file's content differs from the last known version THEN the system SHALL mark it as changed.
3. WHEN a file existed previously but is no longer present in the source THEN the system SHALL mark it as deleted.
4. IF the source is unreachable (network/auth error) THEN the system SHALL fail the run without corrupting previously ingested data, and SHALL log the failure reason.

## 2. Chunking

**User Story:** As a retrieval system, I want documents split into semantically coherent chunks, so that retrieved context is useful rather than arbitrary character slices.

**Acceptance Criteria:**
1. WHEN a document is chunked THEN the system SHALL split on markdown headers and paragraph boundaries rather than fixed character counts.
2. WHEN a chunk would break a fenced code block THEN the system SHALL keep the code block intact within a single chunk.
3. WHEN a chunk is produced THEN the system SHALL attach metadata: source file path, header path (e.g. "Install > Prerequisites"), and chunk position.
4. IF a single section exceeds a configured max token size THEN the system SHALL split it further while preserving code block integrity.

## 3. Embedding Generation

**User Story:** As a pipeline operator, I want chunks converted into vector embeddings, so that semantic search is possible.

**Acceptance Criteria:**
1. WHEN a new or changed chunk is processed THEN the system SHALL generate an embedding using a configured local model (e.g. sentence-transformers).
2. WHEN embeddings are generated THEN the system SHALL batch requests rather than embedding one chunk at a time.
3. IF embedding generation fails for a chunk THEN the system SHALL log the failure and continue processing remaining chunks (no single failure halts the run).

## 4. Vector Storage & Incremental Indexing

**User Story:** As a pipeline operator, I want only changed content re-embedded and re-indexed, so that re-runs are fast and cheap rather than reprocessing everything.

**Acceptance Criteria:**
1. WHEN a document is unchanged since the last run THEN the system SHALL skip re-embedding its chunks.
2. WHEN a document is marked changed THEN the system SHALL delete its old chunks from the vector store and insert the newly generated ones.
3. WHEN a document is marked deleted THEN the system SHALL remove its chunks from the vector store.
4. WHEN an ingestion run completes THEN the system SHALL record run metadata (timestamp, docs added/updated/deleted, chunk counts) for auditability.

## 5. Orchestration & Scheduling

**User Story:** As a pipeline operator, I want the ingestion pipeline to run on a schedule and be observable, so that the index stays fresh without manual intervention.

**Acceptance Criteria:**
1. WHEN the scheduled interval elapses THEN the orchestrator (Airflow) SHALL trigger the ingestion DAG automatically.
2. WHEN a DAG task fails THEN the system SHALL retry according to a configured retry policy before marking the run failed.
3. WHEN a DAG run fails after retries THEN the system SHALL send a notification (e.g. Slack/log alert).
4. WHEN a DAG run succeeds THEN the system SHALL expose run stats (docs processed, duration, chunk count) in a way that's visible without reading logs.

## 6. Retrieval API

**User Story:** As an end user, I want to ask a question and receive relevant source chunks, so that I can verify the pipeline retrieves accurate context.

**Acceptance Criteria:**
1. WHEN a query is submitted to the API THEN the system SHALL embed the query and return the top-k most similar chunks with their source metadata.
2. WHEN no chunks meet a minimum similarity threshold THEN the system SHALL return an empty result rather than low-relevance noise.
3. IF the vector store is empty or unreachable THEN the API SHALL return a clear error rather than an empty 200 response.

## 7. (Optional Stretch) Answer Generation

**User Story:** As an end user, I want a natural-language answer, not just raw chunks, so the demo is compelling in an interview.

**Acceptance Criteria:**
1. WHEN retrieval returns chunks THEN the system MAY pass them to an LLM to synthesize an answer citing source file paths.
2. IF this component is unavailable THEN the core pipeline (ingestion → retrieval) SHALL still function independently — this is explicitly decoupled from the DE portion.

## Non-Functional Requirements

1. Idempotency: re-running the ingestion DAG with no source changes SHALL produce no changes to the vector store.
2. All components SHALL run locally via Docker Compose with no paid API dependencies required for the core pipeline.
3. Configuration (source repo, chunk size, schedule) SHALL be externalized, not hardcoded.
