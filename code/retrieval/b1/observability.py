from __future__ import annotations

import json
import logging
from collections.abc import Callable

from retrieval.b1.constants import (
    ROUTER_CONTRACT_VERSION,
    ROUTER_PROMPT_VERSION,
    ROUTER_SCHEMA_VERSION,
)
from retrieval.b1.types import RouterObservation

_LOGGER = logging.getLogger("retrieval.b1")

Observer = Callable[[RouterObservation], None]


def emit_router_observation(
    observation: RouterObservation,
    observer: Observer | None,
) -> None:
    payload = {
        "request_id": observation.request_id,
        "conversation_id": observation.conversation_id,
        "router_contract_version": observation.router_contract_version,
        "router_prompt_version": observation.router_prompt_version,
        "router_schema_version": observation.router_schema_version,
        "model_alias": observation.model_alias,
        "history_turn_count": observation.history_turn_count,
        "history_tokens": observation.history_tokens,
        "current_query_weighted_length": observation.current_query_weighted_length,
        "router_mode": observation.router_mode,
        "task_count": observation.task_count,
        "router_latency_ms": observation.router_latency_ms,
        "error_type": observation.error_type,
    }
    if observer is not None:
        observer(observation)
        return
    _LOGGER.info("b1.router %s", json.dumps(payload, ensure_ascii=False))


def observation_versions() -> dict[str, str]:
    return {
        "router_contract_version": ROUTER_CONTRACT_VERSION,
        "router_prompt_version": ROUTER_PROMPT_VERSION,
        "router_schema_version": ROUTER_SCHEMA_VERSION,
    }
