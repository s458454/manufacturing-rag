"""B5 Cross-Encoder Reranking: Qwen3-Reranker-0.6B on RRF Leaf candidates."""

from retrieval.b5.config import load_reranker_config
from retrieval.b5.constants import (
    RERANK_INPUT_LIMIT,
    RERANK_OUTPUT_LIMIT,
    RERANKER_CONTRACT_VERSION,
    RERANKER_INSTRUCTION,
    RERANKER_LOGICAL_ID,
    RERANKER_MODEL,
    RERANKER_REVISION,
)
from retrieval.b5.errors import (
    RerankerConfigurationError,
    RerankerError,
    RerankerInferenceError,
    RerankerInputTooLongError,
    RerankerModelError,
)
from retrieval.b5.reranker import Qwen3LeafReranker, load_production_reranker, rerank
from retrieval.b5.template import format_rerank_body, format_rerank_pair
from retrieval.b5.types import (
    RerankedCandidate,
    RerankerConfig,
    RerankerObservation,
    RerankingResult,
)

__all__ = [
    "Qwen3LeafReranker",
    "RERANKER_CONTRACT_VERSION",
    "RERANKER_INSTRUCTION",
    "RERANKER_LOGICAL_ID",
    "RERANKER_MODEL",
    "RERANKER_REVISION",
    "RERANK_INPUT_LIMIT",
    "RERANK_OUTPUT_LIMIT",
    "RerankedCandidate",
    "RerankerConfig",
    "RerankerConfigurationError",
    "RerankerError",
    "RerankerInferenceError",
    "RerankerInputTooLongError",
    "RerankerModelError",
    "RerankerObservation",
    "RerankingResult",
    "format_rerank_body",
    "format_rerank_pair",
    "load_production_reranker",
    "load_reranker_config",
    "rerank",
]
