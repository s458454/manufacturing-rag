"""B2 contract errors. These are orchestration/programming errors, not user Query errors."""


class QueryProcessingContractError(Exception):
    """Caller violated the B2 input contract."""
