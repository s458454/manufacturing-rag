"""B3 fail-closed errors. No user-facing copy is generated here."""


class HybridRetrievalError(Exception):
    """Base B3/B4 hybrid retrieval error."""

    error_type: str = "hybrid_retrieval_error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.error_type)


class HybridSearchContractError(HybridRetrievalError):
    error_type = "hybrid_search_contract_error"


class HybridSearchConfigurationError(HybridRetrievalError):
    error_type = "hybrid_search_configuration_error"


class DenseQueryEncodingError(HybridRetrievalError):
    error_type = "dense_query_encoding_error"


class HybridSearchError(HybridRetrievalError):
    error_type = "hybrid_search_error"
