"""vLLM client must pin engine-level xgrammar and must not retry."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.client import VLLMRouterClient  # noqa: E402
from retrieval.b1.constants import STRUCTURED_DECODING_BACKEND  # noqa: E402
from retrieval.b1.errors import RouterConfigurationError, RouterUnavailableError  # noqa: E402
from retrieval.b1.schema import ROUTER_GUIDED_JSON_SCHEMA, ROUTER_JSON_SCHEMA  # noqa: E402
from retrieval.tests.helpers import router_config  # noqa: E402


def test_client_sends_xgrammar_guided_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            message = SimpleNamespace(
                content='{"mode":"clarify","tasks":[],"candidate_query":null}'
            )
            choice = SimpleNamespace(message=message)
            return SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import retrieval.b1.client as client_mod

    monkeypatch.setattr(client_mod, "OpenAIClient", FakeOpenAI)

    client = VLLMRouterClient(router_config())
    content = client.complete([{"role": "user", "content": "hi"}])
    assert "clarify" in content
    assert captured["init"]["max_retries"] == 0
    assert captured["temperature"] == 0
    assert captured["top_p"] == 1.0
    assert captured["stream"] is False
    extra = captured["extra_body"]
    assert extra["guided_decoding_backend"] == "xgrammar"
    assert extra["guided_decoding_backend"] == STRUCTURED_DECODING_BACKEND
    assert extra["guided_json"] == ROUTER_GUIDED_JSON_SCHEMA
    assert extra["guided_json"] != ROUTER_JSON_SCHEMA
    assert "minItems" not in _schema_keys(extra["guided_json"])
    assert "maxItems" not in _schema_keys(extra["guided_json"])
    assert "minLength" not in _schema_keys(extra["guided_json"])


def _schema_keys(node: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        keys.update(node)
        for value in node.values():
            keys.update(_schema_keys(value))
    elif isinstance(node, list):
        for item in node:
            keys.update(_schema_keys(item))
    return keys


def test_guided_schema_is_xgrammar_projection_only() -> None:
    guided_keys = _schema_keys(ROUTER_GUIDED_JSON_SCHEMA)
    frozen_keys = _schema_keys(ROUTER_JSON_SCHEMA)
    assert "oneOf" not in guided_keys
    assert "const" not in guided_keys
    assert "minItems" not in guided_keys
    assert "maxItems" not in guided_keys
    assert "minLength" not in guided_keys
    assert ROUTER_JSON_SCHEMA["properties"]["tasks"]["minItems"] == 0
    assert ROUTER_JSON_SCHEMA["properties"]["tasks"]["maxItems"] == 8
    assert ROUTER_JSON_SCHEMA["properties"]["tasks"]["items"]["properties"]["query"][
        "minLength"
    ] == 1
    assert "minItems" in frozen_keys
    assert "maxItems" in frozen_keys
    assert "minLength" in frozen_keys
    assert ROUTER_GUIDED_JSON_SCHEMA["properties"]["mode"]["enum"] == [
        "direct",
        "confirm_rewrite",
        "clarify",
        "task_limit",
    ]
    assert ROUTER_GUIDED_JSON_SCHEMA["properties"]["tasks"]["type"] == "array"
    assert ROUTER_GUIDED_JSON_SCHEMA["properties"]["candidate_query"] == {
        "type": ["string", "null"]
    }


def test_client_rejects_no_fallback_request_backend() -> None:
    config = router_config()
    bad = config.__class__(
        vllm_base_url=config.vllm_base_url,
        model_alias=config.model_alias,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        request_timeout_ms=config.request_timeout_ms,
        structured_decoding_backend="xgrammar:no-fallback",
    )
    with pytest.raises(RouterConfigurationError, match="must be xgrammar"):
        VLLMRouterClient(bad)


def test_client_rejects_non_xgrammar_backend() -> None:
    config = router_config()
    bad = config.__class__(
        vllm_base_url=config.vllm_base_url,
        model_alias=config.model_alias,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        request_timeout_ms=config.request_timeout_ms,
        structured_decoding_backend="auto",
    )
    with pytest.raises(RouterConfigurationError, match="must be xgrammar"):
        VLLMRouterClient(bad)


def test_client_wraps_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs: object) -> object:
            raise ConnectionError("down")

    import retrieval.b1.client as client_mod

    monkeypatch.setattr(client_mod, "OpenAIClient", FakeOpenAI)
    client = VLLMRouterClient(router_config())
    with pytest.raises(RouterUnavailableError):
        client.complete([{"role": "user", "content": "hi"}])
