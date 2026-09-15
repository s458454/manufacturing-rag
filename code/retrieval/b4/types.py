from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str
    rrf_score: float
    fused_rank: int


@dataclass(frozen=True)
class FusionResult:
    request_id: str
    task_id: str
    query: str
    candidates: tuple[FusedCandidate, ...]


@dataclass(frozen=True)
class FusionObservation:
    request_id: str
    task_id: str
    fusion_contract_version: str
    rrf_k: int
    dense_candidate_limit: int
    bm25_candidate_limit: int
    fusion_candidate_limit: int
    hybrid_search_latency_ms: float | None
    fused_candidate_count: int | None
    fusion_status: str
    error_type: str | None
