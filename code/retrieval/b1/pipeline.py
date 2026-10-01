"""Production path: Jev judges, then the existing dispatcher sends knowledge tasks to B2."""

from __future__ import annotations

from typing import Any

from retrieval.b1.config import load_router_config
from retrieval.b1.dispatcher import dispatch_knowledge_tasks
from retrieval.b1.errors import RouterUnavailableError
from retrieval.b1.types import ConversationStore, QueryRequest, RouterConfig, RouterResult
from retrieval.b1_jev.jev_client import (
    JevUnavailableError,
    SystemOneJudge,
    resolve_jev_connection,
)
from retrieval.b1_jev.router import JevQueryRouter, qwen_text_complete
from retrieval.b2.types import ProcessedQuery


def build_jev_router(
    *,
    store: ConversationStore,
    tokenizer: Any,
    config: RouterConfig | None = None,
) -> JevQueryRouter:
    """Wire the live Jev judge and the Qwen splitter/rewriter. No old QueryRouter."""

    resolved = load_router_config() if config is None else config
    try:
        connection = resolve_jev_connection()
    except JevUnavailableError as exc:
        raise RouterUnavailableError("Jev is not configured") from exc
    judge = SystemOneJudge(
        connection,
        timeout_s=resolved.request_timeout_ms / 1000.0,
    )
    return JevQueryRouter(
        store=store,
        tokenizer=tokenizer,
        judge=judge,
        complete_text=qwen_text_complete(resolved),
    )


def route_for_b2(
    router: JevQueryRouter,
    request: QueryRequest,
    *,
    pending_rewrite: str | None = None,
) -> tuple[RouterResult, tuple[ProcessedQuery, ...]]:
    """Route one request and process every knowledge task through B2, in order."""

    result = router.route(request, pending_rewrite=pending_rewrite)
    processed = dispatch_knowledge_tasks(request.request_id, result)
    return result, processed
