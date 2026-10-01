from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from retrieval.b1.types import BusinessRoute

KnowledgeMode = Literal[
    "direct",
    "multi",
    "task_limit",
    "confirm_rewrite",
    "clarify",
]


@dataclass(frozen=True)
class JevAnswers:
    primary_route: BusinessRoute
    primary_route_confidence: float


@dataclass(frozen=True)
class JevModeAnswer:
    knowledge_mode: KnowledgeMode
    knowledge_mode_confidence: float


@dataclass(frozen=True)
class JevConnection:
    provider: str
    endpoint: str
    model: str
    api_key: str
