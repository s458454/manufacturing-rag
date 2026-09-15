"""Deterministic business validation after JSON Schema parsing."""

from __future__ import annotations

from typing import Any

from retrieval.b1.constants import MAX_ROUTER_TASKS
from retrieval.b1.errors import (
    EmptyQueryError,
    InvalidQueryError,
    QueryTooLongError,
    RouterProtocolError,
)
from retrieval.b1.schema import ROUTER_JSON_SCHEMA
from retrieval.b1.types import RoutedTask, RouterResult
from retrieval.b1.validation import validate_query_text

_MODES = frozenset(ROUTER_JSON_SCHEMA["properties"]["mode"]["enum"])
_ROUTES = frozenset(
    ROUTER_JSON_SCHEMA["properties"]["tasks"]["items"]["properties"]["route"]["enum"]
)


def _require_object(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RouterProtocolError("Router output is not a JSON object")
    extra = set(payload) - {"mode", "tasks", "candidate_query"}
    if extra:
        raise RouterProtocolError("Router output contains extra fields")
    for field in ("mode", "tasks", "candidate_query"):
        if field not in payload:
            raise RouterProtocolError(f"Router output missing {field}")
    return payload


def _parse_mode(value: object) -> str:
    if not isinstance(value, str) or value not in _MODES:
        raise RouterProtocolError("Router mode is invalid")
    return value


def _parse_tasks(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise RouterProtocolError("Router tasks is not an array")
    if len(value) > MAX_ROUTER_TASKS:
        raise RouterProtocolError("Router tasks exceed maxItems=8")
    parsed: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise RouterProtocolError("Router task is not an object")
        extra = set(item) - {"route", "query"}
        if extra:
            raise RouterProtocolError("Router task contains extra fields")
        if "route" not in item or "query" not in item:
            raise RouterProtocolError("Router task missing route or query")
        route = item["route"]
        query = item["query"]
        if not isinstance(route, str) or route not in _ROUTES:
            raise RouterProtocolError("Router task route is invalid")
        if not isinstance(query, str) or query == "":
            raise RouterProtocolError("Router task query is invalid")
        parsed.append({"route": route, "query": query})
    return parsed


def _revalidate_generated_query(query: str) -> str:
    try:
        return validate_query_text(query)
    except (EmptyQueryError, InvalidQueryError, QueryTooLongError) as exc:
        raise RouterProtocolError("Router-generated query failed B1.0 validation") from exc


def validate_router_payload(payload: object, current_query: str) -> RouterResult:
    data = _require_object(payload)
    mode = _parse_mode(data["mode"])
    tasks = _parse_tasks(data["tasks"])
    candidate_query = data["candidate_query"]
    if candidate_query is not None and not isinstance(candidate_query, str):
        raise RouterProtocolError("candidate_query must be string or null")

    if mode == "direct":
        if not 1 <= len(tasks) <= MAX_ROUTER_TASKS or candidate_query is not None:
            raise RouterProtocolError("direct mode field combination is invalid")
        if len(tasks) == 1 and tasks[0]["query"] != current_query:
            raise RouterProtocolError("direct single task must equal current query")
    elif mode == "confirm_rewrite":
        if tasks or candidate_query is None or str(candidate_query).strip() == "":
            raise RouterProtocolError("confirm_rewrite field combination is invalid")
    else:
        if tasks or candidate_query is not None:
            raise RouterProtocolError(f"{mode} field combination is invalid")

    normalized_tasks: list[RoutedTask] = []
    seen: set[tuple[str, str]] = set()
    for item in tasks:
        query = _revalidate_generated_query(item["query"])
        key = (item["route"], query)
        if key in seen:
            raise RouterProtocolError("duplicate (route, query) tasks")
        seen.add(key)
        normalized_tasks.append(RoutedTask(route=item["route"], query=query))

    normalized_candidate: str | None = None
    if candidate_query is not None:
        normalized_candidate = _revalidate_generated_query(candidate_query)

    if mode == "direct" and len(normalized_tasks) == 1:
        if normalized_tasks[0].query != current_query:
            raise RouterProtocolError("direct single task must equal current query")

    return RouterResult(
        mode=mode,  # type: ignore[arg-type]
        tasks=tuple(normalized_tasks),
        candidate_query=normalized_candidate,
    )
