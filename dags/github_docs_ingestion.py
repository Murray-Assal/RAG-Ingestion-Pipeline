"""Scheduled Airflow entry point for incremental GitHub documentation indexing."""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def ingest_github_docs() -> None:
    """Import late so Airflow's DAG parser does not load the embedding model."""

    from rag_ingestion.ingestion import build_pipeline

    summary = build_pipeline().run()
    print(
        "Ingestion completed: "
        f"indexed={summary.indexed_documents}, skipped={summary.skipped_documents}, "
        f"chunks={summary.embedded_chunks}, deleted={summary.deleted_documents}"
    )


with DAG(
    dag_id="github_docs_incremental_index",
    description="Re-crawl GitHub docs and only re-embed changed files",
    start_date=datetime(2025, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "data-engineering", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["rag", "ingestion", "github", "pgvector"],
) as dag:
    PythonOperator(task_id="incrementally_index_docs", python_callable=ingest_github_docs)
