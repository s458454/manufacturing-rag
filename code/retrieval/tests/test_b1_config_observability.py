"""B1 config loading and observability field completeness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.config import load_router_config  # noqa: E402
from retrieval.b1.errors import RouterConfigurationError  # noqa: E402
from retrieval.b1.types import RouterObservation  # noqa: E402


REQUIRED_OBSERVATION_FIELDS = {
    "request_id",
    "conversation_id",
    "router_contract_version",
    "router_prompt_version",
    "router_schema_version",
    "model_alias",
    "history_turn_count",
    "history_tokens",
    "current_query_weighted_length",
    "router_mode",
    "task_count",
    "router_latency_ms",
    "error_type",
}


def test_observation_fields_match_contract() -> None:
    assert set(RouterObservation.__dataclass_fields__) == REQUIRED_OBSERVATION_FIELDS
    forbidden = {"query", "history", "assistant_response", "raw_user_query"}
    assert forbidden.isdisjoint(REQUIRED_OBSERVATION_FIELDS)


def test_timeout_must_be_explicit() -> None:
    env = {
        "B1_VLLM_BASE_URL": "http://127.0.0.1:8000/v1",
        "B1_MODEL_ALIAS": "Qwen/Qwen2.5-7B-Instruct",
        "B1_TOKENIZER_NAME_OR_PATH": "/models/Qwen2.5-7B-Instruct",
    }
    with pytest.raises(RouterConfigurationError, match="B1_REQUEST_TIMEOUT_MS"):
        load_router_config(env)
    env["B1_REQUEST_TIMEOUT_MS"] = "15000"
    config = load_router_config(env)
    assert config.request_timeout_ms == 15000
    env["B1_STRUCTURED_DECODING_BACKEND"] = "auto"
    with pytest.raises(RouterConfigurationError, match="must be xgrammar"):
        load_router_config(env)
    env["B1_STRUCTURED_DECODING_BACKEND"] = "xgrammar:no-fallback"
    with pytest.raises(RouterConfigurationError, match="must be xgrammar"):
        load_router_config(env)
    env["B1_STRUCTURED_DECODING_BACKEND"] = "xgrammar"
    config = load_router_config(env)
    assert config.structured_decoding_backend == "xgrammar"
