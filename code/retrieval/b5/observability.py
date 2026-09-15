from __future__ import annotations

import json
import logging
from collections.abc import Callable

from retrieval.b5.types import RerankerObservation

_LOGGER = logging.getLogger("retrieval.b5")

Observer = Callable[[RerankerObservation], None]


def emit_reranker_observation(
    observation: RerankerObservation,
    observer: Observer | None,
) -> None:
    payload = {
        "request_id": observation.request_id,
        "task_id": observation.task_id,
        "reranker_contract_version": observation.reranker_contract_version,
        "reranker_model": observation.reranker_model,
        "reranker_input_count": observation.reranker_input_count,
        "reranker_output_count": observation.reranker_output_count,
        "max_pair_tokens": observation.max_pair_tokens,
        "rerank_latency_ms": observation.rerank_latency_ms,
        "rerank_status": observation.rerank_status,
        "error_type": observation.error_type,
    }
    if observer is not None:
        observer(observation)
        return
    _LOGGER.info("b5.rerank %s", json.dumps(payload, ensure_ascii=False))
