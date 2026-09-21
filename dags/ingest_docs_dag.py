"""Airflow DAG for scheduled, observable, incremental documentation indexing."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

from rag_ingestion.orchestration import (
    build_workflow,
    load_orchestration_settings,
    send_failure_notification,
)

logger = logging.getLogger(__name__)
orchestration_settings = load_orchestration_settings()


def start_ingestion_run() -> dict[str, Any]:
    return build_workflow().start_ingestion_run()


def fetch_docs(run_context: dict[str, Any]) -> dict[str, Any]:
    return build_workflow().fetch_docs(run_context)


def detect_changes(snapshot: dict[str, Any]) -> dict[str, Any]:
    return build_workflow().detect_changes(snapshot)


def chunk_changed_docs(changes: dict[str, Any]) -> dict[str, Any]:
    return build_workflow().chunk_changed_docs(changes)


def embed_chunks(chunked: dict[str, Any]) -> dict[str, Any]:
    return build_workflow().embed_chunks(chunked)


def upsert_vector_store(embedded: dict[str, Any]) -> dict[str, Any]:
    return build_workflow().upsert_vector_store(embedded)


def update_state_store(upserted: dict[str, Any]) -> dict[str, Any]:
    return build_workflow().update_state_store(upserted)


def record_run_metadata(state_update: dict[str, Any]) -> dict[str, Any]:
    return build_workflow().record_run_metadata(state_update)


def on_task_failure(context: dict[str, Any]) -> None:
    """Record the failed audit row and alert Slack or logs after Airflow exhausts retries."""

    task_instance = context.get("ti")
    run_context = None
    if task_instance is not None:
        run_context = task_instance.xcom_pull(task_ids="start_ingestion_run")
    exception = context.get("exception")
    task_id = context.get("task_instance").task_id if context.get("task_instance") else "unknown"
    message = f"Ingestion DAG task {task_id!r} failed after retries: {exception!s}"
    if isinstance(run_context, dict) and run_context.get("run_id"):
        try:
            build_workflow().store.fail_ingestion_run(str(run_context["run_id"]), message)
        except Exception:
            logger.exception("Could not mark ingestion run %s as failed", run_context["run_id"])
    send_failure_notification(message, os.getenv(orchestration_settings.slack_webhook_env))


default_args = {
    "owner": "data-engineering",
    "retries": orchestration_settings.retries,
    "retry_delay": timedelta(minutes=orchestration_settings.retry_delay_minutes),
    "retry_exponential_backoff": orchestration_settings.retry_exponential_backoff,
    "on_failure_callback": on_task_failure,
}

with DAG(
    dag_id="ingest_docs_incremental",
    description="Fetch, incrementally re-index, and audit GitHub or local documentation",
    start_date=datetime(2025, 1, 1),
    schedule=orchestration_settings.schedule,
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["rag", "ingestion", "github", "pgvector"],
) as dag:
    start = PythonOperator(task_id="start_ingestion_run", python_callable=start_ingestion_run)
    fetch = PythonOperator(
        task_id="fetch_docs",
        python_callable=fetch_docs,
        op_kwargs={"run_context": start.output},
    )
    detect = PythonOperator(
        task_id="detect_changes",
        python_callable=detect_changes,
        op_kwargs={"snapshot": fetch.output},
    )
    chunk = PythonOperator(
        task_id="chunk_changed_docs",
        python_callable=chunk_changed_docs,
        op_kwargs={"changes": detect.output},
    )
    embed = PythonOperator(
        task_id="embed_chunks",
        python_callable=embed_chunks,
        op_kwargs={"chunked": chunk.output},
    )
    upsert = PythonOperator(
        task_id="upsert_vector_store",
        python_callable=upsert_vector_store,
        op_kwargs={"embedded": embed.output},
    )
    update_state = PythonOperator(
        task_id="update_state_store",
        python_callable=update_state_store,
        op_kwargs={"upserted": upsert.output},
    )
    record = PythonOperator(
        task_id="record_run_metadata",
        python_callable=record_run_metadata,
        op_kwargs={"state_update": update_state.output},
    )

    start >> fetch >> detect >> chunk >> embed >> upsert >> update_state >> record
