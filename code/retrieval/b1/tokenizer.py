"""Qwen2.5-7B-Instruct tokenizer loading and chat-template token accounting."""

from __future__ import annotations

import json
from typing import Any

from retrieval.b1.errors import RouterConfigurationError
from retrieval.b1.schema import ROUTER_JSON_SCHEMA
from retrieval.b1.types import ConversationTurn

_MIN_TRANSFORMERS = (4, 37, 0)


def _parse_version_tuple(value: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for chunk in value.split("."):
        digits = "".join(char for char in chunk if char.isdigit())
        parts.append(int(digits) if digits else 0)
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def load_router_tokenizer(tokenizer_id: str) -> Any:
    identifier = tokenizer_id.strip()
    if not identifier:
        raise RouterConfigurationError("B1 tokenizer identifier is empty")
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        PackageNotFoundError = Exception  # type: ignore[misc,assignment]
        version = None  # type: ignore[assignment]
    detected = "unknown"
    if version is not None:
        try:
            detected = version("transformers")
        except PackageNotFoundError:
            detected = "unknown"
    if _parse_version_tuple(detected) < _MIN_TRANSFORMERS:
        raise RouterConfigurationError(
            "transformers>=4.37.0 is required for the Qwen2.5 tokenizer; "
            f"found {detected}"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RouterConfigurationError(
            "transformers is required to load the B1 router tokenizer"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(identifier, trust_remote_code=False)
    except Exception as exc:
        raise RouterConfigurationError(
            f"Failed to load B1 tokenizer {identifier!r}"
        ) from exc


def schema_token_count(tokenizer: Any) -> int:
    payload = json.dumps(ROUTER_JSON_SCHEMA, ensure_ascii=False, separators=(",", ":"))
    return count_text_tokens(tokenizer, payload)


def count_text_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> int:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as exc:
        raise RouterConfigurationError(
            "B1 tokenizer failed to apply the Qwen2.5 chat template"
        ) from exc
    return len(encoded)


def history_messages(turns: tuple[ConversationTurn, ...]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.canonical_user_query})
        messages.append({"role": "assistant", "content": turn.assistant_response})
    return messages


def count_history_tokens(
    tokenizer: Any,
    turns: tuple[ConversationTurn, ...],
) -> int:
    if not turns:
        return 0
    return count_chat_tokens(
        tokenizer,
        history_messages(turns),
        add_generation_prompt=False,
    )
