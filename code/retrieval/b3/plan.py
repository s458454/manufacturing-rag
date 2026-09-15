"""Build the B3 HybridSearchPlan from a single B2 ProcessedQuery."""

from retrieval.b2.types import ProcessedQuery
from retrieval.b3.constants import BM25_CANDIDATE_LIMIT, DENSE_CANDIDATE_LIMIT
from retrieval.b3.errors import HybridSearchContractError
from retrieval.b3.types import HybridSearchPlan


def _require_non_empty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise HybridSearchContractError(
            f"B3 contract violated: {field_name} must be a non-empty str"
        )
    return value


def build_hybrid_search_plan(processed: ProcessedQuery) -> HybridSearchPlan:
    request_id = _require_non_empty_str(processed.request_id, "request_id")
    task_id = _require_non_empty_str(processed.task_id, "task_id")
    original_query = _require_non_empty_str(processed.original_query, "original_query")
    queries = processed.retrieval_queries
    if not isinstance(queries, tuple) or len(queries) != 1:
        raise HybridSearchContractError(
            "B3 v1 accepts exactly one B2 retrieval query"
        )
    query = _require_non_empty_str(queries[0], "retrieval_queries[0]")
    if query != original_query:
        raise HybridSearchContractError(
            "B3 v1 requires retrieval_queries[0] == original_query"
        )
    return HybridSearchPlan(
        request_id=request_id,
        task_id=task_id,
        query=query,
        dense_candidate_limit=DENSE_CANDIDATE_LIMIT,
        bm25_candidate_limit=BM25_CANDIDATE_LIMIT,
    )
