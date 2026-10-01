"""兔子 API (OpenAI-compatible chat) adapter. Fills JevAnswers. Not Jev."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from retrieval.b1.types import ConversationTurn
from retrieval.b1_jev_experimental.constants import (
    ROUTE_KEYS,
    TUZI_API_KEY_ENV,
    TUZI_BASE_URL_ENV,
    TUZI_DEFAULT_BASE_URL,
    TUZI_DEFAULT_MODEL,
    TUZI_MODEL_ENV,
)
from retrieval.b1_jev_experimental.errors import JevProtocolError, JevUnavailableError
from retrieval.b1_jev_experimental.gliclass_adapter import serialize_classifier_text
from retrieval.b1_jev_experimental.policy import parse_jev_answers
from retrieval.b1_jev_experimental.questions import JEV_QUESTIONS
from retrieval.b1_jev_experimental.types import JevAnswers, JevConnection

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

TUZI_ANSWER_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["primary_route", "primary_route_confidence"],
    "properties": {
        "primary_route": {"type": "string", "enum": list(ROUTE_KEYS)},
        "primary_route_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

_SYSTEM = (
    "You classify the current request into exactly one route. Reply with JSON only. "
    "Do not rewrite the query, split tasks, or emit mode. "
    "knowledge = document RAG; system = assistant capability; unsupported = reject. "
    "Do not use knowledge or unsupported as a dump bucket."
)


def chat_completions_url(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if not root:
        raise JevUnavailableError("TUZI_BASE_URL is empty")
    if root.endswith("/chat/completions"):
        return root
    if root.endswith("/v1"):
        return f"{root}/chat/completions"
    return f"{root}/v1/chat/completions"


def resolve_tuzi_connection(
    environ: Mapping[str, str] | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
) -> JevConnection:
    env = os.environ if environ is None else environ
    key = (api_key if api_key is not None else env.get(TUZI_API_KEY_ENV, "")).strip()
    if not key:
        raise JevUnavailableError("set TUZI_API_KEY for 兔子 API smoke")
    chosen_model = (
        model if model is not None else str(env.get(TUZI_MODEL_ENV, "")).strip()
    ) or TUZI_DEFAULT_MODEL
    if endpoint is not None:
        url = endpoint.strip()
    else:
        url = chat_completions_url(
            str(env.get(TUZI_BASE_URL_ENV, "")).strip() or TUZI_DEFAULT_BASE_URL
        )
    return JevConnection(
        provider="tuzi-openai",
        endpoint=url,
        model=chosen_model,
        api_key=key,
    )


def build_tuzi_messages(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
) -> list[dict[str, str]]:
    lines = [serialize_classifier_text(current_query, history), "", "Questions:"]
    for key, spec in JEV_QUESTIONS.items():
        lines.append(f"- {key} ({spec['type']}): {spec['instructions']}")
        criteria = spec.get("criteria")
        if isinstance(criteria, dict):
            for choice, meaning in criteria.items():
                lines.append(f"    {choice}: {meaning}")
    lines.append("")
    lines.append("Return JSON with keys primary_route and primary_route_confidence.")
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _request_body(model: str, messages: Sequence[Mapping[str, str]], *, schema: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": 0,
        "max_tokens": 256,
    }
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "b1_jev_answers_v0",
                "strict": True,
                "schema": TUZI_ANSWER_SCHEMA,
            },
        }
    else:
        body["response_format"] = {"type": "json_object"}
    return body


def _message_text(message: object) -> str:
    if not isinstance(message, dict):
        raise JevProtocolError("tu-zi message must be an object")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    raise JevProtocolError("tu-zi message content missing")


def answers_from_tuzi_content(content: str) -> JevAnswers:
    text = _FENCE.sub("", content.strip()).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise JevProtocolError("tu-zi returned non-JSON") from exc
    return parse_jev_answers(payload)


def answers_from_tuzi_payload(payload: object) -> JevAnswers:
    if not isinstance(payload, dict):
        raise JevProtocolError("tu-zi response must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise JevProtocolError("tu-zi choices missing")
    first = choices[0]
    if not isinstance(first, dict):
        raise JevProtocolError("tu-zi choice must be an object")
    return answers_from_tuzi_content(_message_text(first.get("message")))


def _http_post(
    connection: JevConnection,
    body: Mapping[str, Any],
    *,
    timeout_s: float,
) -> tuple[int, str]:
    request = urllib.request.Request(
        connection.endpoint,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {connection.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise JevUnavailableError(f"tu-zi HTTP {exc.code}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise JevUnavailableError("tu-zi request failed") from exc


def decide_with_tuzi(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
    *,
    api_key: str | None = None,
    model: str | None = None,
    endpoint: str | None = None,
    timeout_s: float = 60.0,
) -> JevAnswers:
    connection = resolve_tuzi_connection(
        api_key=api_key,
        model=model,
        endpoint=endpoint,
    )
    messages = build_tuzi_messages(current_query, history)
    try:
        _, raw = _http_post(
            connection,
            _request_body(connection.model, messages, schema=True),
            timeout_s=timeout_s,
        )
    except JevUnavailableError as exc:
        if "tu-zi HTTP 400:" not in str(exc):
            raise
        _, raw = _http_post(
            connection,
            _request_body(connection.model, messages, schema=False),
            timeout_s=timeout_s,
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JevProtocolError("tu-zi returned non-JSON") from exc
    return answers_from_tuzi_payload(payload)
