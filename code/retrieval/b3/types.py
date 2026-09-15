from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class HybridSearchStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"


@dataclass(frozen=True)
class HybridSearchPlan:
    request_id: str
    task_id: str
    query: str
    dense_candidate_limit: int
    bm25_candidate_limit: int


@dataclass(frozen=True)
class HybridSearchConfig:
    collection_name: str
    uri: str
    hybrid_search_timeout_ms: int
    dense_candidate_limit: int
    bm25_candidate_limit: int
    fusion_candidate_limit: int
    rrf_k: int


@dataclass(frozen=True)
class HybridSearchObservation:
    request_id: str
    task_id: str
    retrieval_contract_version: str
    dense_encode_latency_ms: float | None
    hybrid_search_status: str
    hybrid_search_latency_ms: float | None
    hybrid_output_count: int | None
    error_type: str | None
