from __future__ import annotations

import json
import logging
from collections.abc import Callable

from retrieval.b4.types import FusionObservation

_LOGGER = logging.getLogger("retrieval.b4")

Observer = Callable[[FusionObservation], None]


def emit_fusion_observation(
    observation: FusionObservation,
    observer: Observer | None,
) -> None:
    payload = {
        "request_id": observation.request_id,
        "task_id": observation.task_id,
        "fusion_contract_version": observation.fusion_contract_version,
        "rrf_k": observation.rrf_k,
        "dense_candidate_limit": observation.dense_candidate_limit,
        "bm25_candidate_limit": observation.bm25_candidate_limit,
        "fusion_candidate_limit": observation.fusion_candidate_limit,
        "hybrid_search_latency_ms": observation.hybrid_search_latency_ms,
        "fused_candidate_count": observation.fused_candidate_count,
        "fusion_status": observation.fusion_status,
        "error_type": observation.error_type,
    }
    if observer is not None:
        observer(observation)
        return
    _LOGGER.info("b4.fusion %s", json.dumps(payload, ensure_ascii=False))
