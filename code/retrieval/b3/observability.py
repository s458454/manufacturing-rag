from __future__ import annotations

import json
import logging
from collections.abc import Callable

from retrieval.b3.types import HybridSearchObservation

_LOGGER = logging.getLogger("retrieval.b3")

Observer = Callable[[HybridSearchObservation], None]


def emit_hybrid_search_observation(
    observation: HybridSearchObservation,
    observer: Observer | None,
) -> None:
    payload = {
        "request_id": observation.request_id,
        "task_id": observation.task_id,
        "retrieval_contract_version": observation.retrieval_contract_version,
        "dense_encode_latency_ms": observation.dense_encode_latency_ms,
        "hybrid_search_status": observation.hybrid_search_status,
        "hybrid_search_latency_ms": observation.hybrid_search_latency_ms,
        "hybrid_output_count": observation.hybrid_output_count,
        "error_type": observation.error_type,
    }
    if observer is not None:
        observer(observation)
        return
    _LOGGER.info("b3.hybrid_search %s", json.dumps(payload, ensure_ascii=False))
