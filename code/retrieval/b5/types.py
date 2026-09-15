from __future__ import annotations

from dataclasses import dataclass

from retrieval.b4.types import FusedCandidate


@dataclass(frozen=True)
class RerankedCandidate:
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str
    rrf_score: float
    fused_rank: int
    rerank_score: float
    rerank_rank: int


@dataclass(frozen=True)
class RerankingResult:
    request_id: str
    task_id: str
    query: str
    candidates: tuple[RerankedCandidate, ...]


@dataclass(frozen=True)
class RerankerConfig:
    model_name_or_path: str
    model_revision: str
    batch_size: int


@dataclass(frozen=True)
class RerankerObservation:
    request_id: str
    task_id: str
    reranker_contract_version: str
    reranker_model: str
    reranker_input_count: int
    reranker_output_count: int | None
    max_pair_tokens: int | None
    rerank_latency_ms: float
    rerank_status: str
    error_type: str | None


def fused_to_reranked(
    candidate: FusedCandidate,
    *,
    rerank_score: float,
    rerank_rank: int,
) -> RerankedCandidate:
    return RerankedCandidate(
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        section_id=candidate.section_id,
        chunk_index=candidate.chunk_index,
        page_start=candidate.page_start,
        page_end=candidate.page_end,
        content=candidate.content,
        rrf_score=candidate.rrf_score,
        fused_rank=candidate.fused_rank,
        rerank_score=rerank_score,
        rerank_rank=rerank_rank,
    )
