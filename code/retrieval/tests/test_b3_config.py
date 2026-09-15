"""B3 timeout is milliseconds; PyMilvus hybrid_search timeout is seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b3.config import load_hybrid_search_config, timeout_ms_to_seconds  # noqa: E402
from retrieval.b3.errors import HybridSearchConfigurationError  # noqa: E402
from retrieval.b4.constants import RRF_K  # noqa: E402


def test_timeout_ms_converts_to_seconds_explicitly() -> None:
    assert timeout_ms_to_seconds(5000) == 5.0
    assert timeout_ms_to_seconds(1500) == 1.5


def test_timeout_must_be_explicit_in_env() -> None:
    env = {
        "B3_MILVUS_URI": "http://localhost:19530",
        "B3_COLLECTION_NAME": "a6_leaf_v1",
    }
    with pytest.raises(HybridSearchConfigurationError, match="B3_HYBRID_SEARCH_TIMEOUT_MS"):
        load_hybrid_search_config(env)
    env["B3_HYBRID_SEARCH_TIMEOUT_MS"] = "5000"
    config = load_hybrid_search_config(env)
    assert config.hybrid_search_timeout_ms == 5000
    assert config.rrf_k == RRF_K == 60
    assert timeout_ms_to_seconds(config.hybrid_search_timeout_ms) == 5.0
