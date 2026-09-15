from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryProcessingRequest:
    request_id: str
    task_id: str
    query: str


@dataclass(frozen=True)
class ProcessedQuery:
    request_id: str
    task_id: str
    original_query: str
    retrieval_queries: tuple[str, ...]


@dataclass(frozen=True)
class QueryProcessingObservation:
    request_id: str
    task_id: str
    query_processor_contract_version: str
    query_processor_version: str
    retrieval_query_count: int
    processing_latency_ms: float
