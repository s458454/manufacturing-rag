"""Load B3/B4 hybrid search runtime config. Timeout is required, not an optimum."""

from __future__ import annotations

import os
from collections.abc import Mapping

from retrieval.b3.constants import BM25_CANDIDATE_LIMIT, DENSE_CANDIDATE_LIMIT
from retrieval.b3.errors import HybridSearchConfigurationError
from retrieval.b3.types import HybridSearchConfig
from retrieval.b4.constants import FUSION_CANDIDATE_LIMIT, RRF_K


def timeout_ms_to_seconds(timeout_ms: int) -> float:
    """Convert the frozen millisecond contract field to PyMilvus seconds."""

    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
        raise HybridSearchConfigurationError(
            "hybrid_search_timeout_ms must be a positive integer"
        )
    return timeout_ms / 1000.0


def load_hybrid_search_config(
    environ: Mapping[str, str] | None = None,
) -> HybridSearchConfig:
    env = os.environ if environ is None else environ
    uri = str(env.get("B3_MILVUS_URI", "")).strip()
    collection_name = str(env.get("B3_COLLECTION_NAME", "")).strip()
    timeout_raw = str(env.get("B3_HYBRID_SEARCH_TIMEOUT_MS", "")).strip()
    missing = [
        name
        for name, value in (
            ("B3_MILVUS_URI", uri),
            ("B3_COLLECTION_NAME", collection_name),
            ("B3_HYBRID_SEARCH_TIMEOUT_MS", timeout_raw),
        )
        if not value
    ]
    if missing:
        raise HybridSearchConfigurationError(
            "B3 config missing required values: " + ", ".join(missing)
        )
    try:
        timeout_ms = int(timeout_raw)
    except ValueError as exc:
        raise HybridSearchConfigurationError(
            "B3_HYBRID_SEARCH_TIMEOUT_MS must be an integer millisecond value"
        ) from exc
    if timeout_ms <= 0:
        raise HybridSearchConfigurationError(
            "B3_HYBRID_SEARCH_TIMEOUT_MS must be positive"
        )
    return HybridSearchConfig(
        collection_name=collection_name,
        uri=uri,
        hybrid_search_timeout_ms=timeout_ms,
        dense_candidate_limit=DENSE_CANDIDATE_LIMIT,
        bm25_candidate_limit=BM25_CANDIDATE_LIMIT,
        fusion_candidate_limit=FUSION_CANDIDATE_LIMIT,
        rrf_k=RRF_K,
    )
