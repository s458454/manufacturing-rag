"""B3+B4 production path: one Milvus hybrid_search, then B4 parse in returned order."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from retrieval.b2.types import ProcessedQuery
from retrieval.b3.bindings import FORBIDDEN_SEARCH_FILTER_KEYS, LEAF_OUTPUT_FIELDS
from retrieval.b3.config import timeout_ms_to_seconds
from retrieval.b3.constants import (
    BM25_CANDIDATE_LIMIT,
    DENSE_CANDIDATE_LIMIT,
    RETRIEVAL_CONTRACT_VERSION,
)
from retrieval.b3.errors import (
    DenseQueryEncodingError,
    HybridRetrievalError,
    HybridSearchConfigurationError,
    HybridSearchError,
)
from retrieval.b3.observability import emit_hybrid_search_observation
from retrieval.b3.plan import build_hybrid_search_plan
from retrieval.b3.store import HybridSearchStore
from retrieval.b3.types import HybridSearchConfig, HybridSearchObservation, HybridSearchStatus
from retrieval.b4.constants import FUSION_CANDIDATE_LIMIT, FUSION_CONTRACT_VERSION, RRF_K
from retrieval.b4.observability import emit_fusion_observation
from retrieval.b4.parse import parse_fusion_hits
from retrieval.b4.types import FusionObservation, FusionResult

EncodeQuery = Callable[[str], Sequence[float]]


def _validate_frozen_config(config: HybridSearchConfig) -> None:
    if config.dense_candidate_limit != DENSE_CANDIDATE_LIMIT:
        raise HybridSearchConfigurationError("dense_candidate_limit must be 20")
    if config.bm25_candidate_limit != BM25_CANDIDATE_LIMIT:
        raise HybridSearchConfigurationError("bm25_candidate_limit must be 20")
    if config.fusion_candidate_limit != FUSION_CANDIDATE_LIMIT:
        raise HybridSearchConfigurationError("fusion_candidate_limit must be 20")
    if config.rrf_k != RRF_K:
        raise HybridSearchConfigurationError("rrf_k must be 60")


def run_hybrid_search(
    processed: ProcessedQuery,
    *,
    encode_query: EncodeQuery,
    store: HybridSearchStore,
    config: HybridSearchConfig,
    hybrid_observer: Callable[[HybridSearchObservation], None] | None = None,
    fusion_observer: Callable[[FusionObservation], None] | None = None,
) -> FusionResult:
    """Execute B3 encode + one hybrid_search + B4 parse. Application retry is OFF."""

    _validate_frozen_config(config)
    plan = build_hybrid_search_plan(processed)
    encode_started = time.perf_counter()
    encode_latency_ms: float | None = None
    search_latency_ms: float | None = None
    try:
        try:
            dense_vector = tuple(float(value) for value in encode_query(plan.query))
        except DenseQueryEncodingError:
            encode_latency_ms = (time.perf_counter() - encode_started) * 1000.0
            raise
        except Exception as exc:
            encode_latency_ms = (time.perf_counter() - encode_started) * 1000.0
            raise DenseQueryEncodingError(
                f"dense query encoding failed: {exc}"
            ) from exc
        encode_latency_ms = (time.perf_counter() - encode_started) * 1000.0
        if len(dense_vector) != 2560:
            raise DenseQueryEncodingError(
                "query vector dimension mismatch: "
                f"expected 2560, got {len(dense_vector)}"
            )
        timeout_seconds = timeout_ms_to_seconds(config.hybrid_search_timeout_ms)
        search_kwargs = {
            "collection_name": config.collection_name,
            "dense_vector": dense_vector,
            "query_text": plan.query,
            "dense_limit": config.dense_candidate_limit,
            "bm25_limit": config.bm25_candidate_limit,
            "fusion_limit": config.fusion_candidate_limit,
            "rrf_k": config.rrf_k,
            "output_fields": LEAF_OUTPUT_FIELDS,
            "timeout_seconds": timeout_seconds,
        }
        illegal = FORBIDDEN_SEARCH_FILTER_KEYS.intersection(search_kwargs)
        if illegal:
            raise HybridSearchError(
                "B3 must not pass collection status filters: "
                + ", ".join(sorted(illegal))
            )
        search_started = time.perf_counter()
        hits = store.hybrid_search(**search_kwargs)
        search_latency_ms = (time.perf_counter() - search_started) * 1000.0
        result = parse_fusion_hits(
            request_id=plan.request_id,
            task_id=plan.task_id,
            query=plan.query,
            hits=hits,
            fusion_candidate_limit=config.fusion_candidate_limit,
        )
    except HybridRetrievalError as exc:
        _emit_failed(
            plan.request_id,
            plan.task_id,
            encode_latency_ms=encode_latency_ms,
            search_latency_ms=search_latency_ms,
            error_type=exc.error_type,
            hybrid_observer=hybrid_observer,
            fusion_observer=fusion_observer,
            config=config,
        )
        raise
    except Exception as exc:
        wrapped = HybridSearchError(f"hybrid_search failed: {exc}")
        _emit_failed(
            plan.request_id,
            plan.task_id,
            encode_latency_ms=encode_latency_ms,
            search_latency_ms=search_latency_ms,
            error_type=wrapped.error_type,
            hybrid_observer=hybrid_observer,
            fusion_observer=fusion_observer,
            config=config,
        )
        raise wrapped from exc

    status = (
        HybridSearchStatus.EMPTY.value
        if len(result.candidates) == 0
        else HybridSearchStatus.SUCCESS.value
    )
    emit_hybrid_search_observation(
        HybridSearchObservation(
            request_id=plan.request_id,
            task_id=plan.task_id,
            retrieval_contract_version=RETRIEVAL_CONTRACT_VERSION,
            dense_encode_latency_ms=encode_latency_ms,
            hybrid_search_status=status,
            hybrid_search_latency_ms=search_latency_ms,
            hybrid_output_count=len(result.candidates),
            error_type=None,
        ),
        hybrid_observer,
    )
    emit_fusion_observation(
        FusionObservation(
            request_id=plan.request_id,
            task_id=plan.task_id,
            fusion_contract_version=FUSION_CONTRACT_VERSION,
            rrf_k=config.rrf_k,
            dense_candidate_limit=config.dense_candidate_limit,
            bm25_candidate_limit=config.bm25_candidate_limit,
            fusion_candidate_limit=config.fusion_candidate_limit,
            hybrid_search_latency_ms=search_latency_ms,
            fused_candidate_count=len(result.candidates),
            fusion_status=status,
            error_type=None,
        ),
        fusion_observer,
    )
    return result


def _emit_failed(
    request_id: str,
    task_id: str,
    *,
    encode_latency_ms: float | None,
    search_latency_ms: float | None,
    error_type: str,
    hybrid_observer: Callable[[HybridSearchObservation], None] | None,
    fusion_observer: Callable[[FusionObservation], None] | None,
    config: HybridSearchConfig,
) -> None:
    emit_hybrid_search_observation(
        HybridSearchObservation(
            request_id=request_id,
            task_id=task_id,
            retrieval_contract_version=RETRIEVAL_CONTRACT_VERSION,
            dense_encode_latency_ms=encode_latency_ms,
            hybrid_search_status=HybridSearchStatus.FAILED.value,
            hybrid_search_latency_ms=search_latency_ms,
            hybrid_output_count=None,
            error_type=error_type,
        ),
        hybrid_observer,
    )
    emit_fusion_observation(
        FusionObservation(
            request_id=request_id,
            task_id=task_id,
            fusion_contract_version=FUSION_CONTRACT_VERSION,
            rrf_k=config.rrf_k,
            dense_candidate_limit=config.dense_candidate_limit,
            bm25_candidate_limit=config.bm25_candidate_limit,
            fusion_candidate_limit=config.fusion_candidate_limit,
            hybrid_search_latency_ms=search_latency_ms,
            fused_candidate_count=None,
            fusion_status=HybridSearchStatus.FAILED.value,
            error_type=error_type,
        ),
        fusion_observer,
    )
