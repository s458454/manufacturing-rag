"""B4 parse: Milvus order is fused_rank; required Leaf fields only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b3.errors import HybridSearchError  # noqa: E402
from retrieval.b4.parse import parse_fusion_hits  # noqa: E402
from retrieval.tests.helpers import leaf_hit  # noqa: E402


def test_parse_assigns_fused_rank_from_returned_order() -> None:
    result = parse_fusion_hits(
        request_id="r",
        task_id="r:0",
        query="q",
        hits=[leaf_hit("a", score=0.2), leaf_hit("b", score=0.1)],
    )
    assert result.candidates[0].fused_rank == 1
    assert result.candidates[1].fused_rank == 2
    assert result.candidates[0].chunk_id == "a"


def test_parse_rejects_duplicate_chunk_id_and_missing_content() -> None:
    with pytest.raises(HybridSearchError, match="duplicate chunk_id"):
        parse_fusion_hits(
            request_id="r",
            task_id="r:0",
            query="q",
            hits=[leaf_hit("a", score=0.2), leaf_hit("a", score=0.1)],
        )
    broken = leaf_hit("a", score=0.2)
    del broken["entity"]["content"]
    with pytest.raises(HybridSearchError, match="content"):
        parse_fusion_hits(
            request_id="r",
            task_id="r:0",
            query="q",
            hits=[broken],
        )


def test_parse_rejects_more_than_fusion_limit() -> None:
    hits = [leaf_hit(f"c{index}", score=0.1) for index in range(21)]
    with pytest.raises(HybridSearchError, match="fusion_candidate_limit"):
        parse_fusion_hits(
            request_id="r",
            task_id="r:0",
            query="q",
            hits=hits,
        )
