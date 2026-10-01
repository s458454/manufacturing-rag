"""B6 evidence records. Evidence ids and rerank scores are not fields."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveredEvidence:
    document_id: str
    recovery_mode: str
    section_id: str | None
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    content: str
    evidence_priority: int
    matched_chunk_ids: tuple[str, ...]
    context_chunk_ids: tuple[str, ...]
    supporting_rerank_ranks: tuple[int, ...]
    token_count: int


@dataclass(frozen=True)
class RecoveryResult:
    evidence: tuple[RecoveredEvidence, ...]
    evidence_budget: int
    evidence_token_count: int
