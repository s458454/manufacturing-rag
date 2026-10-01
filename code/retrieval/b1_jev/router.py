"""Production B1 router. Jev judges; Qwen only splits or rewrites.

Does not call the retired QueryRouter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from retrieval.b1.confirmation import handle_rewrite_confirmation
from retrieval.b1.constants import (
    ROUTER_MAX_NEW_TOKENS,
    ROUTER_TEMPERATURE,
    ROUTER_TOP_P,
    STRUCTURED_DECODING_BACKEND,
)
from retrieval.b1.errors import (
    InvalidQueryError,
    RouterConfigurationError,
    RouterError,
    RouterProtocolError,
    RouterUnavailableError,
)
from retrieval.b1.memory import trim_active_memory
from retrieval.b1.types import (
    BusinessRoute,
    ConversationStore,
    ConversationTurn,
    QueryRequest,
    RoutedTask,
    RouterConfig,
    RouterResult,
)
from retrieval.b1.validation import validate_query_request, validate_query_text
from retrieval.b1_jev.jev_client import JevUnavailableError
from retrieval.b1_jev.multi_splitter import TextComplete, split_current_query
from retrieval.b1_jev.questions import (
    KNOWLEDGE_MODE_KEYS,
    ROUTE_KEYS,
    JevProtocolError,
)
from retrieval.b1_jev.rewrite_generator import rewrite_candidate


class JevJudge(Protocol):
    def primary_route(
        self,
        current_query: str,
        history: Sequence[ConversationTurn],
    ) -> str:
        """Return one of knowledge, system, unsupported."""

    def knowledge_mode(
        self,
        current_query: str,
        history: Sequence[ConversationTurn],
    ) -> str:
        """Return one knowledge_mode label. Called only after knowledge."""


class JevQueryRouter:
    def __init__(
        self,
        *,
        store: ConversationStore,
        tokenizer: Any,
        judge: JevJudge,
        complete_text: TextComplete | None = None,
    ) -> None:
        self._store = store
        self._tokenizer = tokenizer
        self._judge = judge
        self._complete_text = complete_text

    def route(
        self,
        request: QueryRequest,
        *,
        pending_rewrite: str | None = None,
    ) -> RouterResult:
        current_query = self._resolve_current_query(request, pending_rewrite)
        history = self._active_memory(request.conversation_id)
        route = self._primary_route(current_query, history)
        if route != "knowledge":
            return _single_task(route, current_query)
        mode = self._knowledge_mode(current_query, history)
        if mode == "direct":
            return _single_task("knowledge", current_query)
        if mode in ("task_limit", "clarify"):
            return RouterResult(mode=mode, tasks=(), candidate_query=None)
        if mode == "multi":
            queries = split_current_query(current_query, self._complete_text_or_unavailable())
            return RouterResult(
                mode="direct",
                tasks=tuple(
                    RoutedTask(route="knowledge", query=query) for query in queries
                ),
                candidate_query=None,
            )
        if mode == "confirm_rewrite":
            candidate = rewrite_candidate(
                current_query,
                history,
                self._complete_text_or_unavailable(),
            )
            return RouterResult(
                mode="confirm_rewrite",
                tasks=(),
                candidate_query=candidate,
            )
        raise RouterProtocolError("Jev returned an illegal choice")

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

    def _active_memory(self, conversation_id: str) -> tuple[ConversationTurn, ...]:
        history = self._store.get_active_memory(conversation_id)
        fitted = trim_active_memory(history, self._tokenizer)
        if fitted != history:
            self._store.set_active_memory(conversation_id, fitted)
        return fitted

    def _primary_route(
        self,
        current_query: str,
        history: Sequence[ConversationTurn],
    ) -> str:
        try:
            route = self._judge.primary_route(current_query, history)
        except JevUnavailableError as exc:
            raise RouterUnavailableError("Jev primary_route is unavailable") from exc
        except JevProtocolError as exc:
            raise RouterProtocolError("Jev primary_route protocol error") from exc
        except RouterError:
            raise
        except Exception as exc:
            raise RouterUnavailableError("Jev primary_route is unavailable") from exc
        if route not in ROUTE_KEYS:
            raise RouterProtocolError("Jev returned an illegal choice")
        return route

    def _knowledge_mode(
        self,
        current_query: str,
        history: Sequence[ConversationTurn],
    ) -> str:
        try:
            mode = self._judge.knowledge_mode(current_query, history)
        except JevUnavailableError as exc:
            raise RouterUnavailableError("Jev knowledge_mode is unavailable") from exc
        except JevProtocolError as exc:
            raise RouterProtocolError("Jev knowledge_mode protocol error") from exc
        except RouterError:
            raise
        except Exception as exc:
            raise RouterUnavailableError("Jev knowledge_mode is unavailable") from exc
        if mode not in KNOWLEDGE_MODE_KEYS:
            raise RouterProtocolError("Jev returned an illegal choice")
        return mode

    def _complete_text_or_unavailable(self) -> TextComplete:
        if self._complete_text is not None:
            return self._complete_text

        def _missing(
            messages: list[dict[str, str]],
            schema: dict[str, Any],
        ) -> str:
            raise RouterUnavailableError("Qwen text generator is not configured")

        return _missing


def _single_task(route: str, query: str) -> RouterResult:
    checked: BusinessRoute = route  # type: ignore[assignment]
    return RouterResult(
        mode="direct",
        tasks=(RoutedTask(route=checked, query=query),),
        candidate_query=None,
    )


def qwen_text_complete(config: RouterConfig) -> TextComplete:
    """vLLM chat completion for splitter and rewriter schemas only."""

    if config.structured_decoding_backend != STRUCTURED_DECODING_BACKEND:
        raise RouterConfigurationError("B1 structured decoding backend must be xgrammar")
    if config.request_timeout_ms <= 0:
        raise RouterConfigurationError("B1_REQUEST_TIMEOUT_MS must be positive")
    holder: dict[str, Any] = {"client": None}

    def complete(messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        if holder["client"] is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RouterUnavailableError(
                    "openai is required for Qwen text generation"
                ) from exc
            holder["client"] = OpenAI(
                base_url=config.vllm_base_url,
                api_key=config.api_key,
                timeout=config.request_timeout_ms / 1000.0,
                max_retries=0,
            )
        try:
            response = holder["client"].chat.completions.create(
                model=config.model_alias,
                messages=messages,
                temperature=ROUTER_TEMPERATURE,
                top_p=ROUTER_TOP_P,
                max_tokens=ROUTER_MAX_NEW_TOKENS,
                stream=False,
                extra_body={
                    "guided_json": schema,
                    "guided_decoding_backend": STRUCTURED_DECODING_BACKEND,
                },
            )
            content = response.choices[0].message.content
        except RouterError:
            raise
        except Exception as exc:
            raise RouterUnavailableError("Qwen text generation failed") from exc
        if not isinstance(content, str) or content.strip() == "":
            raise RouterUnavailableError("Qwen text generation returned empty content")
        return content

    return complete
