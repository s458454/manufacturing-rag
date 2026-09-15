from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

RouterMode = Literal["direct", "confirm_rewrite", "clarify", "task_limit"]
BusinessRoute = Literal["knowledge", "system", "unsupported"]


@dataclass(frozen=True)
class QueryRequest:
    request_id: str
    conversation_id: str
    query: str


@dataclass(frozen=True)
class ConversationTurn:
    raw_user_query: str
    canonical_user_query: str
    assistant_response: str


@dataclass(frozen=True)
class RoutedTask:
    route: BusinessRoute
    query: str


@dataclass(frozen=True)
class RouterResult:
    mode: RouterMode
    tasks: tuple[RoutedTask, ...]
    candidate_query: str | None


@dataclass(frozen=True)
class RouterConfig:
    vllm_base_url: str
    model_alias: str
    tokenizer_name_or_path: str
    request_timeout_ms: int
    structured_decoding_backend: str = "xgrammar"
    api_key: str = "EMPTY"


@dataclass(frozen=True)
class RouterObservation:
    request_id: str
    conversation_id: str
    router_contract_version: str
    router_prompt_version: str
    router_schema_version: str
    model_alias: str
    history_turn_count: int
    history_tokens: int
    current_query_weighted_length: int
    router_mode: str | None
    task_count: int | None
    router_latency_ms: float
    error_type: str | None


class SystemHandler(Protocol):
    """Capability-answer hook. v1 ships the interface only; no FAQ body."""

    def handle_system_task(
        self,
        *,
        request_id: str,
        task_id: str,
        query: str,
    ) -> str:
        """Return the system-capability answer for one routed system task."""


class ConversationStore(Protocol):
    def get_log(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        """Return the full conversation log, including turns dropped from memory."""

    def get_active_memory(
        self, conversation_id: str
    ) -> tuple[ConversationTurn, ...]:
        """Return Router Active Memory only."""

    def set_active_memory(
        self,
        conversation_id: str,
        turns: tuple[ConversationTurn, ...],
    ) -> None:
        """Replace Active Memory after FIFO eviction. Does not rewrite the log."""

    def append_log(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Append one completed turn to the full conversation log."""


class RouterClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        """Run one constrained Router completion. No retries."""
