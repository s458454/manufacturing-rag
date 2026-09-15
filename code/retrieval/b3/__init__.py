"""B3 Hybrid Retrieval: Dense query adapter + hybrid_search request construction."""

from retrieval.b3.config import load_hybrid_search_config, timeout_ms_to_seconds
from retrieval.b3.constants import (
    BM25_CANDIDATE_LIMIT,
    DENSE_CANDIDATE_LIMIT,
    DENSE_QUERY_SERIALIZER_VERSION,
    RETRIEVAL_CONTRACT_VERSION,
)
from retrieval.b3.encoder import DenseQueryEncoder
from retrieval.b3.errors import (
    DenseQueryEncodingError,
    HybridRetrievalError,
    HybridSearchConfigurationError,
    HybridSearchContractError,
    HybridSearchError,
)
from retrieval.b3.plan import build_hybrid_search_plan
from retrieval.b3.retrieve import run_hybrid_search
from retrieval.b3.serialize import serialize_query
from retrieval.b3.store import MilvusHybridSearchStore
from retrieval.b3.types import (
    HybridSearchConfig,
    HybridSearchObservation,
    HybridSearchPlan,
    HybridSearchStatus,
)

__all__ = [
    "BM25_CANDIDATE_LIMIT",
    "DENSE_CANDIDATE_LIMIT",
    "DENSE_QUERY_SERIALIZER_VERSION",
    "RETRIEVAL_CONTRACT_VERSION",
    "DenseQueryEncoder",
    "DenseQueryEncodingError",
    "HybridRetrievalError",
    "HybridSearchConfig",
    "HybridSearchConfigurationError",
    "HybridSearchContractError",
    "HybridSearchError",
    "HybridSearchObservation",
    "HybridSearchPlan",
    "HybridSearchStatus",
    "MilvusHybridSearchStore",
    "build_hybrid_search_plan",
    "load_hybrid_search_config",
    "run_hybrid_search",
    "serialize_query",
    "timeout_ms_to_seconds",
]
