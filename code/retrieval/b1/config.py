"""Load B1 runtime config. Timeout is required and is not an algorithm optimum."""

from __future__ import annotations

import os
from collections.abc import Mapping

from retrieval.b1.constants import STRUCTURED_DECODING_BACKEND
from retrieval.b1.errors import RouterConfigurationError
from retrieval.b1.types import RouterConfig


def load_router_config(
    environ: Mapping[str, str] | None = None,
) -> RouterConfig:
    env = os.environ if environ is None else environ
    base_url = str(env.get("B1_VLLM_BASE_URL", "")).strip()
    model_alias = str(env.get("B1_MODEL_ALIAS", "")).strip()
    tokenizer_name_or_path = str(env.get("B1_TOKENIZER_NAME_OR_PATH", "")).strip()
    timeout_raw = str(env.get("B1_REQUEST_TIMEOUT_MS", "")).strip()
    backend = str(
        env.get("B1_STRUCTURED_DECODING_BACKEND", STRUCTURED_DECODING_BACKEND)
    ).strip()
    api_key = str(env.get("B1_VLLM_API_KEY", "EMPTY")).strip() or "EMPTY"

    missing = [
        name
        for name, value in (
            ("B1_VLLM_BASE_URL", base_url),
            ("B1_MODEL_ALIAS", model_alias),
            ("B1_TOKENIZER_NAME_OR_PATH", tokenizer_name_or_path),
            ("B1_REQUEST_TIMEOUT_MS", timeout_raw),
        )
        if not value
    ]
    if missing:
        raise RouterConfigurationError(
            "B1 config missing required values: " + ", ".join(missing)
        )
    try:
        timeout_ms = int(timeout_raw)
    except ValueError as exc:
        raise RouterConfigurationError(
            "B1_REQUEST_TIMEOUT_MS must be an integer millisecond value"
        ) from exc
    if timeout_ms <= 0:
        raise RouterConfigurationError("B1_REQUEST_TIMEOUT_MS must be positive")
    if backend != STRUCTURED_DECODING_BACKEND:
        raise RouterConfigurationError(
            "B1 structured decoding backend must be xgrammar"
        )
    return RouterConfig(
        vllm_base_url=base_url,
        model_alias=model_alias,
        tokenizer_name_or_path=tokenizer_name_or_path,
        request_timeout_ms=timeout_ms,
        structured_decoding_backend=backend,
        api_key=api_key,
    )
