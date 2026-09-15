"""B1 Query Router: deterministic core plus a single vLLM constrained decode."""

from __future__ import annotations

import json
import time
from typing import Any

from retrieval.b1.business_validation import validate_router_payload
from retrieval.b1.confirmation import handle_rewrite_confirmation
from retrieval.b1.constants import (
    MAX_QUERY_LENGTH,
    ROUTER_CONTEXT_LIMIT,
    ROUTER_CONTRACT_VERSION,
    ROUTER_MAX_NEW_TOKENS,
    ROUTER_NON_HISTORY_RESERVE,
    ROUTER_PROMPT_VERSION,
    ROUTER_SCHEMA_VERSION,
)
from retrieval.b1.errors import (
    InvalidQueryError,
    RewriteRejected,
    RouterConfigurationError,
    RouterError,
    RouterProtocolError,
    RouterUnavailableError,
)
from retrieval.b1.observability import emit_router_observation
from retrieval.b1.prompt import ROUTER_SYSTEM_PROMPT
from retrieval.b1.tokenizer import (
    count_chat_tokens,
    count_history_tokens,
    history_messages,
    schema_token_count,
)
from retrieval.b1.types import (
    ConversationStore,
    QueryRequest,
    RouterClient,
    RouterConfig,
    RouterObservation,
    RouterResult,
)
from retrieval.b1.validation import query_length, validate_query_request, validate_query_text


class QueryRouter:
    def __init__(
        self,
        *,
        config: RouterConfig,
        client: RouterClient,
        tokenizer: Any,
        store: ConversationStore,
        observer: Any | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._tokenizer = tokenizer
        self._store = store
        self._observer = observer
        self._schema_tokens = schema_token_count(tokenizer)
        self._verify_startup_budget()

    def _verify_startup_budget(self) -> None:
        max_query = "x" * MAX_QUERY_LENGTH
        messages = self._messages((), max_query)
        non_history = self._actual_input_tokens(messages)
        if non_history > ROUTER_NON_HISTORY_RESERVE:
            raise RouterConfigurationError(
                "System Prompt + Schema + max Current Query + chat template "
                f"exceed ROUTER_NON_HISTORY_RESERVE={ROUTER_NON_HISTORY_RESERVE}"
            )

    def _messages(
        self,
        turns: tuple[Any, ...],
        current_query: str,
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        messages.extend(history_messages(turns))
        messages.append({"role": "user", "content": current_query})
        return messages

    def _actual_input_tokens(self, messages: list[dict[str, str]]) -> int:
        return (
            count_chat_tokens(
                self._tokenizer,
                messages,
                add_generation_prompt=True,
            )
            + self._schema_tokens
        )

    def _fit_messages(
        self,
        conversation_id: str,
        turns: tuple[Any, ...],
        current_query: str,
    ) -> tuple[list[dict[str, str]], tuple[Any, ...], int]:
        remaining = list(turns)
        while True:
            messages = self._messages(tuple(remaining), current_query)
            actual = self._actual_input_tokens(messages)
            if actual + ROUTER_MAX_NEW_TOKENS <= ROUTER_CONTEXT_LIMIT:
                fitted = tuple(remaining)
                if fitted != turns:
                    self._store.set_active_memory(conversation_id, fitted)
                return messages, fitted, actual
            if not remaining:
                raise RouterConfigurationError(
                    "actual_input_tokens + ROUTER_MAX_NEW_TOKENS exceed "
                    "ROUTER_CONTEXT_LIMIT with empty Active Memory"
                )
            remaining.pop(0)

    def _observe(
        self,
        *,
        request_id: str,
        conversation_id: str,
        history_turn_count: int,
        history_tokens: int,
        current_query_weighted_length: int,
        router_mode: str | None,
        task_count: int | None,
        started: float,
        error_type: str | None,
    ) -> None:
        emit_router_observation(
            RouterObservation(
                request_id=request_id,
                conversation_id=conversation_id,
                router_contract_version=ROUTER_CONTRACT_VERSION,
                router_prompt_version=ROUTER_PROMPT_VERSION,
                router_schema_version=ROUTER_SCHEMA_VERSION,
                model_alias=self._config.model_alias,
                history_turn_count=history_turn_count,
                history_tokens=history_tokens,
                current_query_weighted_length=current_query_weighted_length,
                router_mode=router_mode,
                task_count=task_count,
                router_latency_ms=(time.perf_counter() - started) * 1000.0,
                error_type=error_type,
            ),
            self._observer,
        )

    def route(
        self,
        request: QueryRequest,
        *,
        pending_rewrite: str | None = None,
    ) -> RouterResult:
        started = time.perf_counter()
        request_id = request.request_id if isinstance(request.request_id, str) else ""
        conversation_id = (
            request.conversation_id if isinstance(request.conversation_id, str) else ""
        )
        history = self._store.get_active_memory(conversation_id) if conversation_id else ()
        history_tokens = count_history_tokens(self._tokenizer, history)
        weighted_length = 0
        try:
            current_query = self._resolve_current_query(request, pending_rewrite)
            weighted_length = query_length(current_query)
            history = self._store.get_active_memory(request.conversation_id)
            history_tokens = count_history_tokens(self._tokenizer, history)
            messages, fitted, _actual = self._fit_messages(
                request.conversation_id,
                history,
                current_query,
            )
            history_tokens = count_history_tokens(self._tokenizer, fitted)
            try:
                raw = self._client.complete(messages)
            except RouterError:
                raise
            except Exception as exc:
                raise RouterUnavailableError("B1 router inference failed") from exc
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RouterProtocolError("Router output is not valid JSON") from exc
            result = validate_router_payload(payload, current_query)
            self._observe(
                request_id=request.request_id,
                conversation_id=request.conversation_id,
                history_turn_count=len(fitted),
                history_tokens=history_tokens,
                current_query_weighted_length=weighted_length,
                router_mode=result.mode,
                task_count=len(result.tasks),
                started=started,
                error_type=None,
            )
            return result
        except RewriteRejected:
            self._observe(
                request_id=request_id,
                conversation_id=conversation_id,
                history_turn_count=len(history),
                history_tokens=history_tokens,
                current_query_weighted_length=query_length(str(request.query))
                if isinstance(request.query, str)
                else 0,
                router_mode=None,
                task_count=None,
                started=started,
                error_type=None,
            )
            raise
        except RouterUnavailableError as exc:
            self._observe(
                request_id=request.request_id
                if isinstance(request.request_id, str)
                else request_id,
                conversation_id=request.conversation_id
                if isinstance(request.conversation_id, str)
                else conversation_id,
                history_turn_count=len(history),
                history_tokens=history_tokens,
                current_query_weighted_length=weighted_length,
                router_mode=None,
                task_count=None,
                started=started,
                error_type=exc.error_type,
            )
            raise
        except RouterError as exc:
            self._observe(
                request_id=request.request_id
                if isinstance(request.request_id, str)
                else request_id,
                conversation_id=request.conversation_id
                if isinstance(request.conversation_id, str)
                else conversation_id,
                history_turn_count=len(history),
                history_tokens=history_tokens,
                current_query_weighted_length=weighted_length,
                router_mode=None,
                task_count=None,
                started=started,
                error_type=exc.error_type,
            )
            raise

    def _resolve_current_query(
        self,
        request: QueryRequest,
        pending_rewrite: str | None,
    ) -> str:
        if pending_rewrite is not None:
            if not isinstance(request.query, str):
                raise InvalidQueryError("query must be a str")
            confirmed = handle_rewrite_confirmation(request.query, pending_rewrite)
            if confirmed is not None:
                return validate_query_text(confirmed)
        return validate_query_request(request)
