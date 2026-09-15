"""Milvus hybrid_search store. No application-layer status filter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from retrieval.b3.bindings import (
    DENSE_METRIC_TYPE,
    FORBIDDEN_SEARCH_FILTER_KEYS,
    LEAF_OUTPUT_FIELDS,
    FIELD_DENSE_VECTOR,
    FIELD_SPARSE_VECTOR,
    SPARSE_METRIC_TYPE,
)
from retrieval.b3.errors import HybridSearchError


class HybridSearchStore(Protocol):
    def hybrid_search(
        self,
        *,
        collection_name: str,
        dense_vector: Sequence[float],
        query_text: str,
        dense_limit: int,
        bm25_limit: int,
        fusion_limit: int,
        rrf_k: int,
        output_fields: Sequence[str],
        timeout_seconds: float,
    ) -> Sequence[Mapping[str, Any]]:
        """Return fused hits in Milvus order. Must not apply a status filter."""


def _reject_filter_kwargs(kwargs: Mapping[str, Any]) -> None:
    illegal = FORBIDDEN_SEARCH_FILTER_KEYS.intersection(kwargs)
    if illegal:
        raise HybridSearchError(
            "B3 must not pass collection status filters: "
            + ", ".join(sorted(illegal))
        )


class MilvusHybridSearchStore:
    """Production store: one MilvusClient.hybrid_search with RRFRanker."""

    def __init__(self, uri: str) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise HybridSearchError(
                "pymilvus is required for production hybrid_search"
            ) from exc
        self.uri = uri
        self.client = MilvusClient(uri=uri)

    def hybrid_search(
        self,
        *,
        collection_name: str,
        dense_vector: Sequence[float],
        query_text: str,
        dense_limit: int,
        bm25_limit: int,
        fusion_limit: int,
        rrf_k: int,
        output_fields: Sequence[str],
        timeout_seconds: float,
    ) -> Sequence[Mapping[str, Any]]:
        try:
            from pymilvus import AnnSearchRequest, RRFRanker
        except ImportError as exc:
            raise HybridSearchError(
                "pymilvus is required for production hybrid_search"
            ) from exc

        dense_req = AnnSearchRequest(
            data=[list(dense_vector)],
            anns_field=FIELD_DENSE_VECTOR,
            param={"metric_type": DENSE_METRIC_TYPE},
            limit=dense_limit,
        )
        bm25_req = AnnSearchRequest(
            data=[query_text],
            anns_field=FIELD_SPARSE_VECTOR,
            param={"metric_type": SPARSE_METRIC_TYPE},
            limit=bm25_limit,
        )
        call_kwargs = {
            "collection_name": collection_name,
            "reqs": [dense_req, bm25_req],
            "ranker": RRFRanker(rrf_k),
            "limit": fusion_limit,
            "output_fields": list(output_fields),
            "timeout": timeout_seconds,
        }
        _reject_filter_kwargs(call_kwargs)
        if tuple(output_fields) != LEAF_OUTPUT_FIELDS:
            raise HybridSearchError(
                "hybrid_search output_fields must be the frozen Leaf set"
            )
        try:
            raw = self.client.hybrid_search(**call_kwargs)
        except HybridSearchError:
            raise
        except Exception as exc:
            raise HybridSearchError(f"Milvus hybrid_search failed: {exc}") from exc
        return _normalize_hybrid_hits(raw)


def _normalize_hybrid_hits(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        raise HybridSearchError("hybrid_search returned no result structure")
    rows = raw
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
        if len(raw) == 1 and isinstance(raw[0], Sequence) and not isinstance(
            raw[0], (str, bytes, Mapping)
        ):
            rows = raw[0]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise HybridSearchError("hybrid_search returned an illegal hit list")
    normalized: list[dict[str, Any]] = []
    for hit in rows:
        normalized.append(_hit_to_mapping(hit))
    return normalized


def _hit_to_mapping(hit: Any) -> dict[str, Any]:
    if isinstance(hit, Mapping):
        entity = hit.get("entity", hit)
        if not isinstance(entity, Mapping):
            entity = {}
        return {
            "id": hit.get("id"),
            "distance": hit.get("distance", hit.get("score")),
            "entity": dict(entity),
        }
    entity = getattr(hit, "entity", None)
    if entity is None:
        entity = {}
    elif not isinstance(entity, Mapping):
        try:
            entity = dict(entity)
        except Exception as exc:
            raise HybridSearchError("hybrid_search hit entity is not a mapping") from exc
    return {
        "id": getattr(hit, "id", None),
        "distance": getattr(hit, "distance", getattr(hit, "score", None)),
        "entity": dict(entity),
    }
