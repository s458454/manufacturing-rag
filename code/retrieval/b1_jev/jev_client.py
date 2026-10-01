"""Jev HTTP client for POST /v1/systemone. One call, no retry.

Confidence is parsed so an illegal probability is a protocol error.
The returned label is the only value the candidate router uses.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from retrieval.b1.types import ConversationTurn
from retrieval.b1_jev.questions import (
    JEV_MODEL,
    KNOWLEDGE_MODE_KEYS,
    ROUTE_KEYS,
    JevProtocolError,
    build_jev_mode_request,
    build_jev_request,
)

AI_GATEWAY_API_KEY_ENV = "AI_GATEWAY_API_KEY"
JEV_DECISION_API = "https://api.typesafe.ai/v1/systemone"
JEV_GATEWAY_API = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
JEV_GATEWAY_MODEL = "typesafe-ai/jev"
TUZI_API_KEY_ENV = "TUZI_API_KEY"
TUZI_BASE_URL_ENV = "TUZI_BASE_URL"
TUZI_DEFAULT_BASE_URL = "https://api.tu-zi.com"
TUZI_JEV_MODEL = "jev-1.13"
TUZI_JEV_API = "https://api.tu-zi.com/v1/systemone"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"

_OK_CODES = {None, 0, 200, "0", "ok", "success"}


class JevUnavailableError(Exception):
    """Timeout, HTTP failure, or network failure. Not a business route."""

    error_type = "jev_unavailable"


@dataclass(frozen=True)
class JevConnection:
    provider: str
    endpoint: str
    model: str
    api_key: str


def systemone_url(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if not root:
        raise JevUnavailableError("Jev base URL is empty")
    if root.endswith("/systemone"):
        return root
    if root.endswith("/v1"):
        return f"{root}/systemone"
    return f"{root}/v1/systemone"


def _require_unit(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise JevProtocolError(f"{name} must be a probability")
    number = float(value)
    if number != number or number < 0.0 or number > 1.0:
        raise JevProtocolError(f"{name} must be in [0, 1]")
    return number


def _choice_confidence(answer: dict[str, Any]) -> float:
    if "confidence" in answer:
        return _require_unit("confidence", answer["confidence"])
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        try:
            return _require_unit("confidence", max(probabilities.values()))
        except (TypeError, ValueError) as exc:
            raise JevProtocolError("choice confidence is illegal") from exc
    raise JevProtocolError("choice answer missing confidence")


def _extract_answers_map(payload: dict[str, Any]) -> dict[str, Any]:
    code = payload.get("code")
    if code not in _OK_CODES:
        detail = payload.get("message") or payload.get("msg") or str(code)
        raise JevUnavailableError(f"Jev upstream error: {detail}")
    answers = payload.get("answers")
    if isinstance(answers, dict):
        return answers
    data = payload.get("data")
    if isinstance(data, dict):
        nested = data.get("answers")
        if isinstance(nested, dict):
            return nested
        if isinstance(data.get("primary_route"), dict) or isinstance(
            data.get("knowledge_mode"), dict
        ):
            return data
    if isinstance(payload.get("primary_route"), dict) or isinstance(
        payload.get("knowledge_mode"), dict
    ):
        return payload
    raise JevProtocolError("Jev answers must be an object")


def route_label_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev response must be an object")
    block = _extract_answers_map(payload)
    route = block.get("primary_route")
    if not isinstance(route, dict):
        raise JevProtocolError("choice answers must be objects")
    route_key = route.get("choice")
    if route_key not in ROUTE_KEYS:
        raise JevProtocolError("Jev returned an illegal choice")
    _choice_confidence(route)
    return route_key


def mode_label_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev response must be an object")
    block = _extract_answers_map(payload)
    mode = block.get("knowledge_mode")
    if not isinstance(mode, dict):
        raise JevProtocolError("choice answers must be objects")
    mode_key = mode.get("choice")
    if mode_key not in KNOWLEDGE_MODE_KEYS:
        raise JevProtocolError("Jev returned an illegal choice")
    _choice_confidence(mode)
    return mode_key


def resolve_jev_connection(
    environ: Mapping[str, str] | None = None,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> JevConnection:
    env = os.environ if environ is None else environ
    explicit = api_key.strip() if api_key is not None else ""
    tuzi_key = str(env.get(TUZI_API_KEY_ENV, "")).strip()
    gateway_key = (
        explicit if api_key is not None else str(env.get(AI_GATEWAY_API_KEY_ENV, "")).strip()
    )
    typesafe_key = str(env.get(TYPESAFE_API_KEY_ENV, "")).strip()
    if endpoint is not None:
        url = endpoint.strip()
        if "ai-gateway.vercel.sh" in url:
            key = gateway_key or typesafe_key or tuzi_key
            if not key:
                raise JevUnavailableError(
                    "AI_GATEWAY_API_KEY is required for Vercel AI Gateway"
                )
            return JevConnection(
                provider="vercel-ai-gateway",
                endpoint=url,
                model=JEV_GATEWAY_MODEL,
                api_key=key,
            )
        if "tu-zi.com" in url:
            key = explicit or tuzi_key or gateway_key or typesafe_key
            if not key:
                raise JevUnavailableError("TUZI_API_KEY is required for 兔子 Jev")
            if url.rstrip("/").endswith("tu-zi.com") or url.rstrip("/").endswith("/v1"):
                url = systemone_url(url)
            return JevConnection(
                provider="tuzi-jev",
                endpoint=url,
                model=TUZI_JEV_MODEL,
                api_key=key,
            )
        key = typesafe_key or gateway_key or tuzi_key or explicit
        if not key:
            raise JevUnavailableError("an API key is required for live Jev calls")
        return JevConnection(
            provider="typesafe",
            endpoint=url,
            model=JEV_MODEL,
            api_key=key,
        )
    if tuzi_key:
        base = str(env.get(TUZI_BASE_URL_ENV, "")).strip() or TUZI_DEFAULT_BASE_URL
        return JevConnection(
            provider="tuzi-jev",
            endpoint=systemone_url(base) if base != TUZI_DEFAULT_BASE_URL else TUZI_JEV_API,
            model=TUZI_JEV_MODEL,
            api_key=tuzi_key,
        )
    if gateway_key:
        return JevConnection(
            provider="vercel-ai-gateway",
            endpoint=JEV_GATEWAY_API,
            model=JEV_GATEWAY_MODEL,
            api_key=gateway_key,
        )
    if typesafe_key:
        return JevConnection(
            provider="typesafe",
            endpoint=JEV_DECISION_API,
            model=JEV_MODEL,
            api_key=typesafe_key,
        )
    raise JevUnavailableError(
        "set TUZI_API_KEY (兔子 jev-1.13), AI_GATEWAY_API_KEY, or TYPESAFE_API_KEY"
    )


def _post_systemone(
    connection: JevConnection,
    body: Mapping[str, object],
    *,
    timeout_s: float,
) -> dict[str, Any]:
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
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise JevUnavailableError(f"Jev HTTP {exc.code}: {detail[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise JevUnavailableError("Jev request failed") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JevProtocolError("Jev returned non-JSON") from exc
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev response must be an object")
    return payload


class SystemOneJudge:
    """Two separate /v1/systemone calls. The router decides whether to ask the second."""

    def __init__(self, connection: JevConnection, *, timeout_s: float = 60.0) -> None:
        self._connection = connection
        self._timeout_s = timeout_s

    def primary_route(
        self,
        current_query: str,
        history: Sequence[ConversationTurn],
    ) -> str:
        payload = _post_systemone(
            self._connection,
            build_jev_request(
                current_query,
                history,
                model=self._connection.model,
            ),
            timeout_s=self._timeout_s,
        )
        return route_label_from_payload(payload)

    def knowledge_mode(
        self,
        current_query: str,
        history: Sequence[ConversationTurn],
    ) -> str:
        payload = _post_systemone(
            self._connection,
            build_jev_mode_request(
                current_query,
                history,
                model=self._connection.model,
            ),
            timeout_s=self._timeout_s,
        )
        return mode_label_from_payload(payload)
