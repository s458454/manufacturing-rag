"""B3 plan: one identity B2 query, independent Dense/BM25 limits."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b2.types import ProcessedQuery  # noqa: E402
from retrieval.b3.constants import BM25_CANDIDATE_LIMIT, DENSE_CANDIDATE_LIMIT  # noqa: E402
from retrieval.b3.errors import HybridSearchContractError  # noqa: E402
from retrieval.b3.plan import build_hybrid_search_plan  # noqa: E402
from retrieval.tests.helpers import processed_query  # noqa: E402


def test_plan_keeps_independent_candidate_limits() -> None:
    plan = build_hybrid_search_plan(processed_query("q"))
    assert plan.dense_candidate_limit == DENSE_CANDIDATE_LIMIT == 20
    assert plan.bm25_candidate_limit == BM25_CANDIDATE_LIMIT == 20
    assert plan.dense_candidate_limit == plan.bm25_candidate_limit
    assert plan.query == "q"


def test_plan_rejects_multi_query_and_rewritten_query() -> None:
    with pytest.raises(HybridSearchContractError, match="exactly one"):
        build_hybrid_search_plan(
            ProcessedQuery(
                request_id="r",
                task_id="r:0",
                original_query="q",
                retrieval_queries=("q", "q2"),
            )
        )
    with pytest.raises(HybridSearchContractError, match="original_query"):
        build_hybrid_search_plan(
            ProcessedQuery(
                request_id="r",
                task_id="r:0",
                original_query="q",
                retrieval_queries=("rewritten",),
            )
        )
