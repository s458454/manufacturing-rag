"""vLLM OpenAI-compatible Router client. One call, engine-level xgrammar JSON Schema, no retry."""

from __future__ import annotations

from retrieval.b1.constants import (
    ROUTER_MAX_NEW_TOKENS,
    ROUTER_TEMPERATURE,
    ROUTER_TOP_P,
    STRUCTURED_DECODING_BACKEND,
)
from retrieval.b1.errors import RouterConfigurationError, RouterUnavailableError
from retrieval.b1.schema import ROUTER_GUIDED_JSON_SCHEMA
from retrieval.b1.types import RouterConfig

try:
    from openai import OpenAI as OpenAIClient
except ImportError:  # pragma: no cover
    OpenAIClient = None  # type: ignore[misc,assignment]


class VLLMRouterClient:
    def __init__(self, config: RouterConfig) -> None:
        if config.structured_decoding_backend != STRUCTURED_DECODING_BACKEND:
            raise RouterConfigurationError(
                "B1 structured decoding backend must be xgrammar"
            )
        if config.request_timeout_ms <= 0:
            raise RouterConfigurationError("B1_REQUEST_TIMEOUT_MS must be positive")
        if OpenAIClient is None:
            raise RouterConfigurationError(
                "openai is required for the B1 vLLM client"
            )
        self._config = config
        self._client = OpenAIClient(
            base_url=config.vllm_base_url,
            api_key=config.api_key,
            timeout=config.request_timeout_ms / 1000.0,
            max_retries=0,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._config.model_alias,
                messages=messages,
                temperature=ROUTER_TEMPERATURE,
                top_p=ROUTER_TOP_P,
                max_tokens=ROUTER_MAX_NEW_TOKENS,
                stream=False,
                extra_body={
                    "guided_json": ROUTER_GUIDED_JSON_SCHEMA,
                    "guided_decoding_backend": STRUCTURED_DECODING_BACKEND,
                },
            )
        except Exception as exc:
            raise RouterUnavailableError(
                "B1 vLLM completion failed"
            ) from exc
        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise RouterUnavailableError(
                "B1 vLLM returned an illegal response structure"
            ) from exc
        if not isinstance(content, str) or content.strip() == "":
            raise RouterUnavailableError("B1 vLLM returned empty content")
        return content
