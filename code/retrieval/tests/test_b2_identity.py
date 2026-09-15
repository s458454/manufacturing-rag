"""B2 identity processor: no mutation, no history, contract fail-closed."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b2.errors import QueryProcessingContractError  # noqa: E402
from retrieval.b2.processor import (  # noqa: E402
    QUERY_PROCESSOR_CONTRACT_VERSION,
    QUERY_PROCESSOR_VERSION,
    process_query,
)
from retrieval.b2.types import QueryProcessingRequest  # noqa: E402


def test_identity_preserves_every_character() -> None:
    request = QueryProcessingRequest(
        request_id="req-123",
        task_id="req-123:0",
        query="NASA STD 5006A 对铝合金焊接有什么要求？",
    )
    processed = process_query(request)
    assert processed.request_id == "req-123"
    assert processed.task_id == "req-123:0"
    assert processed.original_query == request.query
    assert processed.retrieval_queries == (request.query,)
    assert processed.retrieval_queries[0] == processed.original_query


def test_identity_does_not_strip_or_normalize_identifiers() -> None:
    query = "  NASA STD 5006A\n"
    processed = process_query(
        QueryProcessingRequest(request_id="req-1", task_id="req-1:2", query=query)
    )
    assert processed.original_query == query
    assert processed.retrieval_queries == (query,)
    assert "NASA-STD-5006A" not in processed.original_query


def test_task_id_from_original_index_is_not_renumbered() -> None:
    processed = process_query(
        QueryProcessingRequest(
            request_id="req-123",
            task_id="req-123:2",
            query="查 5006A 焊接要求",
        )
    )
    assert processed.task_id == "req-123:2"


def test_empty_or_non_str_fields_are_contract_errors() -> None:
    with pytest.raises(QueryProcessingContractError, match="request_id"):
        process_query(QueryProcessingRequest(request_id="", task_id="t", query="q"))
    with pytest.raises(QueryProcessingContractError, match="task_id"):
        process_query(QueryProcessingRequest(request_id="r", task_id="", query="q"))
    with pytest.raises(QueryProcessingContractError, match="query"):
        process_query(QueryProcessingRequest(request_id="r", task_id="t", query=""))
    with pytest.raises(QueryProcessingContractError, match="query"):
        process_query(
            QueryProcessingRequest(request_id="r", task_id="t", query=None)  # type: ignore[arg-type]
        )


def test_observation_has_no_query_body(caplog: pytest.LogCaptureFixture) -> None:
    query = "secret-query-body-should-not-be-logged"
    observations: list[object] = []
    with caplog.at_level(logging.INFO, logger="retrieval.b2"):
        process_query(
            QueryProcessingRequest(request_id="req-9", task_id="req-9:0", query=query),
            observer=observations.append,
        )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.request_id == "req-9"
    assert observation.task_id == "req-9:0"
    assert observation.query_processor_contract_version == QUERY_PROCESSOR_CONTRACT_VERSION
    assert observation.query_processor_version == QUERY_PROCESSOR_VERSION
    assert observation.retrieval_query_count == 1
    assert observation.processing_latency_ms >= 0
    assert not hasattr(observation, "query")
    assert query not in caplog.text
    process_query(
        QueryProcessingRequest(request_id="req-10", task_id="req-10:0", query=query)
    )
    assert query not in caplog.text
