"""Conversation log vs Router Active Memory. Persistence medium is in-process only."""

from __future__ import annotations

from typing import Any

from retrieval.b1.constants import MAX_ROUTER_MEMORY_TOKENS
from retrieval.b1.tokenizer import count_history_tokens
from retrieval.b1.types import ConversationTurn


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._logs: dict[str, list[ConversationTurn]] = {}
        self._active: dict[str, list[ConversationTurn]] = {}

    def get_log(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        return tuple(self._logs.get(conversation_id, ()))

    def get_active_memory(
        self, conversation_id: str
    ) -> tuple[ConversationTurn, ...]:
        return tuple(self._active.get(conversation_id, ()))

    def set_active_memory(
        self,
        conversation_id: str,
        turns: tuple[ConversationTurn, ...],
    ) -> None:
        self._active[conversation_id] = list(turns)

    def append_log(self, conversation_id: str, turn: ConversationTurn) -> None:
        self._logs.setdefault(conversation_id, []).append(turn)


def trim_active_memory(
    turns: tuple[ConversationTurn, ...],
    tokenizer: Any,
    *,
    limit: int = MAX_ROUTER_MEMORY_TOKENS,
) -> tuple[ConversationTurn, ...]:
    remaining = list(turns)
    while remaining and count_history_tokens(tokenizer, tuple(remaining)) > limit:
        remaining.pop(0)
    return tuple(remaining)


def commit_turn(
    store: InMemoryConversationStore,
    conversation_id: str,
    turn: ConversationTurn,
    tokenizer: Any,
) -> None:
    """Record a completed business turn. Caller must not invoke this for clarify,
    task_limit, Router errors, or a pending rewrite confirmation.
    """

    store.append_log(conversation_id, turn)
    active = store.get_active_memory(conversation_id)
    single = (turn,)
    if count_history_tokens(tokenizer, single) > MAX_ROUTER_MEMORY_TOKENS:
        return
    store.set_active_memory(
        conversation_id,
        trim_active_memory(active + single, tokenizer),
    )
