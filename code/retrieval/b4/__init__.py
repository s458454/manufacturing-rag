"""B4 RRF Fusion: parse Milvus RRFRanker order. No second RRF."""

from retrieval.b4.constants import FUSION_CANDIDATE_LIMIT, FUSION_CONTRACT_VERSION, RRF_K
from retrieval.b4.errors import HybridSearchError
from retrieval.b4.parse import parse_fusion_hits
from retrieval.b4.types import FusedCandidate, FusionObservation, FusionResult

__all__ = [
    "FUSION_CANDIDATE_LIMIT",
    "FUSION_CONTRACT_VERSION",
    "FusedCandidate",
    "FusionObservation",
    "FusionResult",
    "HybridSearchError",
    "RRF_K",
    "parse_fusion_hits",
]
