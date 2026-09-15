"""Active Memory FIFO and scheme A commit_turn."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.constants import MAX_ROUTER_MEMORY_TOKENS  # noqa: E402
from retrieval.b1.memory import InMemoryConversationStore, commit_turn  # noqa: E402
from retrieval.b1.tokenizer import count_history_tokens  # noqa: E402
from retrieval.b1.types import ConversationTurn  # noqa: E402
from retrieval.tests.helpers import CountingTokenizer  # noqa: E402


def _turn(raw: str, canonical: str, assistant: str) -> ConversationTurn:
    return ConversationTurn(
        raw_user_query=raw,
        canonical_user_query=canonical,
        assistant_response=assistant,
    )


def test_commit_turn_appends_log_and_active_memory() -> None:
    store = InMemoryConversationStore()
    tokenizer = CountingTokenizer()
    turn = _turn("那钢呢？", "钢的焊接要求", "根据标准……")
    commit_turn(store, "conv-1", turn, tokenizer)
    assert store.get_log("conv-1") == (turn,)
    assert store.get_active_memory("conv-1") == (turn,)


def test_single_oversize_turn_stays_in_log_not_active_memory() -> None:
    store = InMemoryConversationStore()
    tokenizer = CountingTokenizer()
    huge = _turn("q", "q", "a" * (MAX_ROUTER_MEMORY_TOKENS + 50))
    commit_turn(store, "conv-1", huge, tokenizer)
    assert store.get_log("conv-1") == (huge,)
    assert store.get_active_memory("conv-1") == ()


def test_fifo_drops_oldest_complete_turn() -> None:
    store = InMemoryConversationStore()
    tokenizer = CountingTokenizer()
    first = _turn("t1", "t1", "a" * 9000)
    second = _turn("t2", "t2", "b" * 9000)
    commit_turn(store, "conv-1", first, tokenizer)
    commit_turn(store, "conv-1", second, tokenizer)
    active = store.get_active_memory("conv-1")
    assert active == (second,)
    assert store.get_log("conv-1") == (first, second)
    assert count_history_tokens(tokenizer, active) <= MAX_ROUTER_MEMORY_TOKENS
