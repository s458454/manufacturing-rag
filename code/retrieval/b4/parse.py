"""Parse Milvus RRF order into B4 FusionResult. No application-layer re-sort."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from retrieval.b3.bindings import (
    FIELD_CHUNK_ID,
    FIELD_CHUNK_INDEX,
    FIELD_CONTENT,
    FIELD_DOCUMENT_ID,
    FIELD_PAGE_END,
    FIELD_PAGE_START,
    FIELD_SECTION_ID,
)
from retrieval.b3.errors import HybridSearchError
from retrieval.b4.constants import FUSION_CANDIDATE_LIMIT
from retrieval.b4.types import FusedCandidate, FusionResult


def parse_fusion_hits(
    *,
    request_id: str,
    task_id: str,
    query: str,
    hits: Sequence[Mapping[str, Any]],
    fusion_candidate_limit: int = FUSION_CANDIDATE_LIMIT,
) -> FusionResult:
    if len(hits) > fusion_candidate_limit:
        raise HybridSearchError(
            "hybrid_search returned more candidates than fusion_candidate_limit: "
            f"{len(hits)} > {fusion_candidate_limit}"
        )
    candidates: list[FusedCandidate] = []
    seen: set[str] = set()
    for index, hit in enumerate(hits, start=1):
        candidate = _parse_one_hit(hit, fused_rank=index)
        if candidate.chunk_id in seen:
            raise HybridSearchError(
                f"duplicate chunk_id in fused hits: {candidate.chunk_id}"
            )
        seen.add(candidate.chunk_id)
        candidates.append(candidate)
    return FusionResult(
        request_id=request_id,
        task_id=task_id,
        query=query,
        candidates=tuple(candidates),
    )


def _parse_one_hit(hit: Mapping[str, Any], *, fused_rank: int) -> FusedCandidate:
    entity = hit.get("entity", hit)
    if not isinstance(entity, Mapping):
        raise HybridSearchError("fused hit entity is not a mapping")
    chunk_id = _require_str(
        entity.get(FIELD_CHUNK_ID, hit.get("id")),
        FIELD_CHUNK_ID,
    )
    score_raw = hit.get("distance", hit.get("score"))
    try:
        rrf_score = float(score_raw)
    except (TypeError, ValueError) as exc:
        raise HybridSearchError("fused hit is missing a numeric rrf_score") from exc
    return FusedCandidate(
        chunk_id=chunk_id,
        document_id=_require_str(entity.get(FIELD_DOCUMENT_ID), FIELD_DOCUMENT_ID),
        section_id=_require_str(entity.get(FIELD_SECTION_ID), FIELD_SECTION_ID),
        chunk_index=_require_int(entity.get(FIELD_CHUNK_INDEX), FIELD_CHUNK_INDEX),
        page_start=_require_int(entity.get(FIELD_PAGE_START), FIELD_PAGE_START),
        page_end=_require_int(entity.get(FIELD_PAGE_END), FIELD_PAGE_END),
        content=_require_str(entity.get(FIELD_CONTENT), FIELD_CONTENT),
        rrf_score=rrf_score,
        fused_rank=fused_rank,
    )


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise HybridSearchError(f"fused hit missing {field_name}")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise HybridSearchError(f"fused hit missing integer {field_name}")
    try:
        as_int = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise HybridSearchError(f"fused hit missing integer {field_name}") from exc
    try:
        as_float = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise HybridSearchError(f"fused hit missing integer {field_name}") from exc
    if as_float != float(as_int):
        raise HybridSearchError(f"fused hit missing integer {field_name}")
    return as_int
