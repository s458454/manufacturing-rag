"""Jev HTTP client. 兔子 /v1/systemone preferred; QueryRouter never imports this."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from retrieval.b1.types import ConversationTurn
from retrieval.b1_jev_experimental.constants import (
    AI_GATEWAY_API_KEY_ENV,
    JEV_DECISION_API,
    JEV_GATEWAY_API,
    JEV_GATEWAY_MODEL,
    JEV_MODEL,
    KNOWLEDGE_MODE_KEYS,
    ROUTE_KEYS,
    TUZI_API_KEY_ENV,
    TUZI_BASE_URL_ENV,
    TUZI_DEFAULT_BASE_URL,
    TUZI_JEV_API,
    TUZI_JEV_MODEL,
    TYPESAFE_API_KEY_ENV,
)
from retrieval.b1_jev_experimental.errors import JevProtocolError, JevUnavailableError
from retrieval.b1_jev_experimental.policy import parse_jev_answers, parse_jev_mode_answer
from retrieval.b1_jev_experimental.questions import build_jev_mode_request, build_jev_request
from retrieval.b1_jev_experimental.types import JevAnswers, JevConnection, JevModeAnswer

_OK_CODES = {None, 0, 200, "0", "ok", "success"}


def systemone_url(base_url: str) -> str:
    root = base_url.strip().rstrip("/")
    if not root:
        raise JevUnavailableError("Jev base URL is empty")
    if root.endswith("/systemone"):
        return root
    if root.endswith("/v1"):
        return f"{root}/systemone"
    return f"{root}/v1/systemone"


def _choice_confidence(answer: dict[str, Any]) -> float:
    if "confidence" in answer:
        return float(answer["confidence"])
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict) and probabilities:
        return float(max(probabilities.values()))
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


def answers_from_api_payload(payload: object) -> JevAnswers:
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev response must be an object")
    block = _extract_answers_map(payload)
    route = block.get("primary_route")
    if not isinstance(route, dict):
        raise JevProtocolError("choice answers must be objects")
    route_key = route.get("choice")
    if route_key not in ROUTE_KEYS:
        raise JevProtocolError("Jev returned an illegal choice")
    return parse_jev_answers(
        {
            "primary_route": route_key,
            "primary_route_confidence": _choice_confidence(route),
        }
    )


def answers_from_mode_payload(payload: object) -> JevModeAnswer:
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev response must be an object")
    block = _extract_answers_map(payload)
    mode = block.get("knowledge_mode")
    if not isinstance(mode, dict):
        raise JevProtocolError("choice answers must be objects")
    mode_key = mode.get("choice")
    if mode_key not in KNOWLEDGE_MODE_KEYS:
        raise JevProtocolError("Jev returned an illegal choice")
    return parse_jev_mode_answer(
        {
            "knowledge_mode": mode_key,
            "knowledge_mode_confidence": _choice_confidence(mode),
        }
    )


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
        raise JevUnavailableError(
            f"Jev HTTP {exc.code}: {detail[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise JevUnavailableError("Jev request failed") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JevProtocolError("Jev returned non-JSON") from exc
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev response must be an object")
    return payload


def decide_with_jev(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    timeout_s: float = 60.0,
) -> JevAnswers:
    connection = resolve_jev_connection(api_key=api_key, endpoint=endpoint)
    return answers_from_api_payload(
        _post_systemone(
            connection,
            build_jev_request(current_query, history, model=connection.model),
            timeout_s=timeout_s,
        )
    )


def decide_knowledge_mode_with_jev(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    timeout_s: float = 60.0,
) -> JevModeAnswer:
    connection = resolve_jev_connection(api_key=api_key, endpoint=endpoint)
    return answers_from_mode_payload(
        _post_systemone(
            connection,
            build_jev_mode_request(current_query, history, model=connection.model),
            timeout_s=timeout_s,
        )
    )
