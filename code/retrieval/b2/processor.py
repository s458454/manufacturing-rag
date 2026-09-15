"""Identity Query Processor: input query equals output retrieval query."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from retrieval.b2.errors import QueryProcessingContractError
from retrieval.b2.types import (
    ProcessedQuery,
    QueryProcessingObservation,
    QueryProcessingRequest,
)

QUERY_PROCESSOR_CONTRACT_VERSION = "b2-contract-v1"
QUERY_PROCESSOR_VERSION = "identity-v1"

_LOGGER = logging.getLogger("retrieval.b2")

Observer = Callable[[QueryProcessingObservation], None]


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise QueryProcessingContractError(
            f"B2 contract violated: {field_name} must be a non-empty str"
        )
    return value


def _emit_observation(
    observation: QueryProcessingObservation,
    observer: Observer | None,
) -> None:
    if observer is not None:
        observer(observation)
        return
    _LOGGER.info(
        "b2.query_processing %s",
        json.dumps(
            {
                "request_id": observation.request_id,
                "task_id": observation.task_id,
                "query_processor_contract_version": (
                    observation.query_processor_contract_version
                ),
                "query_processor_version": observation.query_processor_version,
                "retrieval_query_count": observation.retrieval_query_count,
                "processing_latency_ms": observation.processing_latency_ms,
            },
            ensure_ascii=False,
        ),
    )


def process_query(
    request: QueryProcessingRequest,
    *,
    observer: Observer | None = None,
) -> ProcessedQuery:
    """Return an identity ProcessedQuery. Does not read history or mutate text."""

    started = time.perf_counter()
    request_id = _require_non_empty_str(request.request_id, "request_id")
    task_id = _require_non_empty_str(request.task_id, "task_id")
    query = _require_non_empty_str(request.query, "query")

    processed = ProcessedQuery(
        request_id=request_id,
        task_id=task_id,
        original_query=query,
        retrieval_queries=(query,),
    )
    if not (
        len(processed.retrieval_queries) == 1
        and processed.retrieval_queries[0] == processed.original_query
        and processed.original_query == request.query
        and processed.request_id == request.request_id
        and processed.task_id == request.task_id
    ):
        raise QueryProcessingContractError("B2 identity invariants failed")

    _emit_observation(
        QueryProcessingObservation(
            request_id=processed.request_id,
            task_id=processed.task_id,
            query_processor_contract_version=QUERY_PROCESSOR_CONTRACT_VERSION,
            query_processor_version=QUERY_PROCESSOR_VERSION,
            retrieval_query_count=1,
            processing_latency_ms=(time.perf_counter() - started) * 1000.0,
        ),
        observer,
    )
    return processed
