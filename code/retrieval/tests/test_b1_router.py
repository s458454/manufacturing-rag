"""B1 QueryRouter with a fake client: one shot, confirmation, no commit_turn."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.constants import (  # noqa: E402
    ROUTER_CONTEXT_LIMIT,
    ROUTER_CONTRACT_VERSION,
    ROUTER_MAX_NEW_TOKENS,
    USER_REPLY_CONFIRM_REWRITE_TEMPLATE,
    USER_REPLY_ROUTER_UNAVAILABLE,
)
from retrieval.b1.dispatcher import router_mode_user_reply  # noqa: E402
from retrieval.b1.errors import (  # noqa: E402
    EmptyQueryError,
    RewriteRejected,
    RouterConfigurationError,
    RouterProtocolError,
    RouterUnavailableError,
)
from retrieval.b1.memory import InMemoryConversationStore, commit_turn  # noqa: E402
from retrieval.b1.router import QueryRouter  # noqa: E402
from retrieval.b1.types import ConversationTurn, QueryRequest  # noqa: E402
from retrieval.tests.helpers import (  # noqa: E402
    CountingTokenizer,
    FakeRouterClient,
    router_config,
)

CURRENT = "NASA-STD-5006A 对铝合金焊接有什么要求？"


class RecordingStore(InMemoryConversationStore):
    def __init__(self) -> None:
        super().__init__()
        self.commit_called = False

    def append_log(self, conversation_id: str, turn: ConversationTurn) -> None:
        self.commit_called = True
        super().append_log(conversation_id, turn)


class OverflowTokenizer(CountingTokenizer):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> list[int] | str:
        encoded = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        assert isinstance(encoded, str)
        extra = 0
        if any(message["role"] == "system" for message in messages) and len(messages) > 2:
            extra = ROUTER_CONTEXT_LIMIT
        n_tokens = len(encoded) + extra
        if tokenize:
            return [0] * n_tokens
        return encoded


def _router(payload: Any, *, store: RecordingStore | None = None, tokenizer: Any = None) -> QueryRouter:
    return QueryRouter(
        config=router_config(),
        client=FakeRouterClient(payload),
        tokenizer=tokenizer or CountingTokenizer(),
        store=store or RecordingStore(),
        observer=None,
    )


def test_direct_knowledge_does_not_commit_turn() -> None:
    store = RecordingStore()
    observations: list[Any] = []
    router = QueryRouter(
        config=router_config(),
        client=FakeRouterClient(
            {
                "mode": "direct",
                "tasks": [{"route": "knowledge", "query": CURRENT}],
                "candidate_query": None,
            }
        ),
        tokenizer=CountingTokenizer(),
        store=store,
        observer=observations.append,
    )
    result = router.route(
        QueryRequest(request_id="req-1", conversation_id="conv-1", query=CURRENT)
    )
    assert result.mode == "direct"
    assert store.commit_called is False
    assert store.get_active_memory("conv-1") == ()
    observation = observations[0]
    assert observation.router_contract_version == ROUTER_CONTRACT_VERSION
    assert observation.router_mode == "direct"
    assert observation.task_count == 1
    assert observation.error_type is None
    assert CURRENT not in json.dumps(observation.__dict__, ensure_ascii=False)


def test_empty_query_does_not_call_model() -> None:
    client = FakeRouterClient({"mode": "clarify", "tasks": [], "candidate_query": None})
    router = QueryRouter(
        config=router_config(),
        client=client,
        tokenizer=CountingTokenizer(),
        store=InMemoryConversationStore(),
    )
    with pytest.raises(EmptyQueryError):
        router.route(QueryRequest(request_id="req-1", conversation_id="c", query="  "))
    assert client.calls == 0


def test_model_is_called_once() -> None:
    client = FakeRouterClient(
        {
            "mode": "direct",
            "tasks": [{"route": "knowledge", "query": CURRENT}],
            "candidate_query": None,
        }
    )
    router = QueryRouter(
        config=router_config(),
        client=client,
        tokenizer=CountingTokenizer(),
        store=InMemoryConversationStore(),
    )
    router.route(QueryRequest(request_id="req-1", conversation_id="c", query=CURRENT))
    assert client.calls == 1


def test_invalid_json_is_protocol_error() -> None:
    router = _router("not-json")
    with pytest.raises(RouterProtocolError):
        router.route(QueryRequest(request_id="req-1", conversation_id="c", query=CURRENT))


def test_unavailable_model_has_frozen_reply() -> None:
    def boom(_messages: list[dict[str, str]]) -> str:
        raise RouterUnavailableError("down")

    router = _router(boom)
    with pytest.raises(RouterUnavailableError) as exc:
        router.route(QueryRequest(request_id="req-1", conversation_id="c", query=CURRENT))
    assert exc.value.user_reply == USER_REPLY_ROUTER_UNAVAILABLE


def test_confirm_rewrite_reply_template() -> None:
    candidate = "NASA-STD-5006A 对钢材焊接有什么要求？"
    router = _router(
        {
            "mode": "confirm_rewrite",
            "tasks": [],
            "candidate_query": candidate,
        }
    )
    result = router.route(
        QueryRequest(request_id="req-2", conversation_id="c", query="那钢呢？")
    )
    assert result.mode == "confirm_rewrite"
    assert router_mode_user_reply(result) == USER_REPLY_CONFIRM_REWRITE_TEMPLATE.format(
        candidate_query=candidate
    )


def test_pending_yes_reenters_router_with_candidate() -> None:
    candidate = "NASA-STD-5006A 对钢材焊接有什么要求？"
    client = FakeRouterClient(
        lambda messages: json.dumps(
            {
                "mode": "direct",
                "tasks": [
                    {
                        "route": "knowledge",
                        "query": messages[-1]["content"],
                    }
                ],
                "candidate_query": None,
            },
            ensure_ascii=False,
        )
    )
    router = QueryRouter(
        config=router_config(),
        client=client,
        tokenizer=CountingTokenizer(),
        store=InMemoryConversationStore(),
    )
    result = router.route(
        QueryRequest(request_id="req-3", conversation_id="c", query="是"),
        pending_rewrite=candidate,
    )
    assert result.mode == "direct"
    assert result.tasks[0].query == candidate
    assert client.messages is not None
    assert client.messages[-1]["content"] == candidate


def test_pending_reject_does_not_call_model() -> None:
    client = FakeRouterClient({"mode": "clarify", "tasks": [], "candidate_query": None})
    router = QueryRouter(
        config=router_config(),
        client=client,
        tokenizer=CountingTokenizer(),
        store=InMemoryConversationStore(),
    )
    with pytest.raises(RewriteRejected):
        router.route(
            QueryRequest(request_id="req-4", conversation_id="c", query="不"),
            pending_rewrite=CURRENT,
        )
    assert client.calls == 0


def test_dynamic_context_check_fifo_then_configuration_error() -> None:
    store = InMemoryConversationStore()
    tokenizer = OverflowTokenizer()
    commit_turn(
        store,
        "conv-1",
        ConversationTurn("old", "old", "answer"),
        CountingTokenizer(),
    )
    client = FakeRouterClient(
        {
            "mode": "direct",
            "tasks": [{"route": "knowledge", "query": CURRENT}],
            "candidate_query": None,
        }
    )
    router = QueryRouter(
        config=router_config(),
        client=client,
        tokenizer=tokenizer,
        store=store,
    )
    result = router.route(
        QueryRequest(request_id="req-5", conversation_id="conv-1", query=CURRENT)
    )
    assert result.mode == "direct"
    assert store.get_active_memory("conv-1") == ()
    assert client.calls == 1


def test_empty_memory_still_over_limit_is_configuration_error() -> None:
    class AlwaysOverTokenizer(CountingTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            tokenize: bool = True,
            add_generation_prompt: bool = True,
            **kwargs: Any,
        ) -> list[int] | str:
            if tokenize:
                return [0] * (ROUTER_CONTEXT_LIMIT - ROUTER_MAX_NEW_TOKENS + 1)
            return "x"

        def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]:
            return [0]

    with pytest.raises(RouterConfigurationError):
        QueryRouter(
            config=router_config(),
            client=FakeRouterClient({}),
            tokenizer=AlwaysOverTokenizer(),
            store=InMemoryConversationStore(),
        )
