"""Load B5 runtime config. Model path is required; batch size may be overridden."""

from __future__ import annotations

import os
from collections.abc import Mapping

from retrieval.b5.constants import (
    RERANKER_DEFAULT_BATCH_SIZE,
    RERANKER_REVISION,
)
from retrieval.b5.errors import RerankerConfigurationError
from retrieval.b5.types import RerankerConfig


def load_reranker_config(
    environ: Mapping[str, str] | None = None,
) -> RerankerConfig:
    env = os.environ if environ is None else environ
    model_name_or_path = str(env.get("B5_MODEL_NAME_OR_PATH", "")).strip()
    if not model_name_or_path:
        raise RerankerConfigurationError("B5_MODEL_NAME_OR_PATH is required")
    revision = str(env.get("B5_MODEL_REVISION", RERANKER_REVISION)).strip()
    if not revision:
        raise RerankerConfigurationError("B5_MODEL_REVISION must not be empty")
    batch_raw = str(env.get("B5_BATCH_SIZE", "")).strip()
    if batch_raw == "":
        batch_size = RERANKER_DEFAULT_BATCH_SIZE
    else:
        try:
            batch_size = int(batch_raw)
        except ValueError as exc:
            raise RerankerConfigurationError(
                "B5_BATCH_SIZE must be a positive integer"
            ) from exc
    if batch_size <= 0:
        raise RerankerConfigurationError("B5_BATCH_SIZE must be a positive integer")
    return RerankerConfig(
        model_name_or_path=model_name_or_path,
        model_revision=revision,
        batch_size=batch_size,
    )
