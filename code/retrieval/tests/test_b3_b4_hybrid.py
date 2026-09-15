"""B3+B4 one hybrid_search: no status filter, full Leaf fields, Milvus order."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b3.bindings import LEAF_OUTPUT_FIELDS  # noqa: E402
from retrieval.b3.errors import DenseQueryEncodingError, HybridSearchError  # noqa: E402
from retrieval.b3.retrieve import run_hybrid_search  # noqa: E402
from retrieval.b3.serialize import serialize_query  # noqa: E402
from retrieval.tests.helpers import (  # noqa: E402
    fake_dense_vector,
    hybrid_config,
    leaf_hit,
    processed_query,
)


class RecordingStore:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self.hits = hits
        self.calls: list[dict[str, Any]] = []

    def hybrid_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return list(self.hits)


def _encode(_query: str) -> tuple[float, ...]:
    return fake_dense_vector()


def test_hybrid_search_kwargs_match_frozen_contract() -> None:
    store = RecordingStore(
        [leaf_hit("c1", score=0.04, content="welding filler"), leaf_hit("c2", score=0.02)]
    )
    query = "NASA-STD-5006A 焊接"
    result = run_hybrid_search(
        processed_query(query),
        encode_query=_encode,
        store=store,
        config=hybrid_config(5000),
    )
    assert len(store.calls) == 1
    call = store.calls[0]
    assert call["timeout_seconds"] == 5.0
    assert call["dense_limit"] == 20
    assert call["bm25_limit"] == 20
    assert call["fusion_limit"] == 20
    assert call["rrf_k"] == 60
    assert call["query_text"] == query
    assert call["query_text"] != serialize_query(query)
    assert call["output_fields"] == LEAF_OUTPUT_FIELDS
    assert "status" not in call
    assert "filter" not in call
    assert "expr" not in call
    assert call["output_fields"] == (
        "chunk_id",
        "document_id",
        "section_id",
        "chunk_index",
        "page_start",
        "page_end",
        "content",
    )
    assert [candidate.chunk_id for candidate in result.candidates] == ["c1", "c2"]
    assert result.candidates[0].fused_rank == 1
    assert result.candidates[0].rrf_score == 0.04
    assert result.candidates[0].content == "welding filler"
    assert result.query == query


def test_empty_hits_are_legal_not_failed() -> None:
    store = RecordingStore([])
    observations: list[object] = []
    result = run_hybrid_search(
        processed_query("q"),
        encode_query=_encode,
        store=store,
        config=hybrid_config(),
        hybrid_observer=observations.append,
    )
    assert result.candidates == ()
    assert observations[0].hybrid_search_status == "empty"
    assert observations[0].error_type is None


def test_application_does_not_retry_failed_search() -> None:
    class FailingStore:
        def __init__(self) -> None:
            self.calls = 0

        def hybrid_search(self, **kwargs: Any) -> list[dict[str, Any]]:
            self.calls += 1
            raise RuntimeError("milvus down")

    store = FailingStore()
    with pytest.raises(HybridSearchError, match="milvus down"):
        run_hybrid_search(
            processed_query("q"),
            encode_query=_encode,
            store=store,
            config=hybrid_config(),
        )
    assert store.calls == 1


def test_encoding_failure_does_not_call_hybrid_search() -> None:
    store = RecordingStore([leaf_hit("c1", score=0.1)])

    def boom(_query: str) -> tuple[float, ...]:
        raise DenseQueryEncodingError("encoder unavailable")

    with pytest.raises(DenseQueryEncodingError, match="encoder unavailable"):
        run_hybrid_search(
            processed_query("q"),
            encode_query=boom,
            store=store,
            config=hybrid_config(),
        )
    assert store.calls == []


def test_fused_rank_follows_milvus_order_not_score_sort() -> None:
    store = RecordingStore(
        [
            leaf_hit("low", score=0.01, content="first"),
            leaf_hit("high", score=0.99, content="second"),
        ]
    )
    result = run_hybrid_search(
        processed_query("q"),
        encode_query=_encode,
        store=store,
        config=hybrid_config(),
    )
    assert [candidate.chunk_id for candidate in result.candidates] == ["low", "high"]
    assert result.candidates[0].fused_rank == 1
    assert result.candidates[1].fused_rank == 2


def test_observability_omits_query_body(caplog: pytest.LogCaptureFixture) -> None:
    query = "secret-hybrid-query-body"
    store = RecordingStore([])
    with caplog.at_level(logging.INFO, logger="retrieval.b3"):
        run_hybrid_search(
            processed_query(query),
            encode_query=_encode,
            store=store,
            config=hybrid_config(),
        )
    assert query not in caplog.text
