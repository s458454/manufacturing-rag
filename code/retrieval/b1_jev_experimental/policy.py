"""Parse Jev choices. Does not compose production mode or apply thresholds."""

from __future__ import annotations

from retrieval.b1_jev_experimental.constants import KNOWLEDGE_MODE_KEYS, ROUTE_KEYS
from retrieval.b1_jev_experimental.errors import JevProtocolError
from retrieval.b1_jev_experimental.types import JevAnswers, JevModeAnswer


def _require_unit(name: str, value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise JevProtocolError(f"{name} must be a probability")
    number = float(value)
    if number != number or number < 0.0 or number > 1.0:
        raise JevProtocolError(f"{name} must be in [0, 1]")
    return number


def parse_jev_answers(payload: object) -> JevAnswers:
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev answers must be an object")
    route = payload.get("primary_route")
    if route not in ROUTE_KEYS:
        raise JevProtocolError("primary_route is not a valid choice")
    confidence = payload.get("primary_route_confidence", 1.0)
    return JevAnswers(
        primary_route=route,  # type: ignore[arg-type]
        primary_route_confidence=_require_unit("primary_route_confidence", confidence),
    )


def parse_jev_mode_answer(payload: object) -> JevModeAnswer:
    if not isinstance(payload, dict):
        raise JevProtocolError("Jev mode answers must be an object")
    mode = payload.get("knowledge_mode")
    if mode not in KNOWLEDGE_MODE_KEYS:
        raise JevProtocolError("knowledge_mode is not a valid choice")
    confidence = payload.get("knowledge_mode_confidence", 1.0)
    return JevModeAnswer(
        knowledge_mode=mode,  # type: ignore[arg-type]
        knowledge_mode_confidence=_require_unit("knowledge_mode_confidence", confidence),
    )
