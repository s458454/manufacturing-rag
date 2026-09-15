"""B2 Identity Query Processor (identity-v1)."""

from retrieval.b2.errors import QueryProcessingContractError
from retrieval.b2.processor import (
    QUERY_PROCESSOR_CONTRACT_VERSION,
    QUERY_PROCESSOR_VERSION,
    process_query,
)
from retrieval.b2.types import (
    ProcessedQuery,
    QueryProcessingObservation,
    QueryProcessingRequest,
)

__all__ = [
    "QUERY_PROCESSOR_CONTRACT_VERSION",
    "QUERY_PROCESSOR_VERSION",
    "ProcessedQuery",
    "QueryProcessingContractError",
    "QueryProcessingObservation",
    "QueryProcessingRequest",
    "process_query",
]
