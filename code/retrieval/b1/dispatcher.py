"""Deterministic Dispatcher. Serial execution is an implementation choice, not a B1 invariant."""

from __future__ import annotations

from collections.abc import Iterator

from retrieval.b1.constants import (
    USER_REPLY_CLARIFY,
    USER_REPLY_CONFIRM_REWRITE_TEMPLATE,
    USER_REPLY_TASK_LIMIT,
    USER_REPLY_UNSUPPORTED,
    USER_REPLY_UNSUPPORTED_SUBTASK,
)
from retrieval.b1.types import RoutedTask, RouterResult, SystemHandler
from retrieval.b2.processor import process_query
from retrieval.b2.types import ProcessedQuery, QueryProcessingRequest


def task_id_for(request_id: str, task_index: int) -> str:
    return f"{request_id}:{task_index}"


def iter_direct_tasks(
    request_id: str,
    result: RouterResult,
) -> Iterator[tuple[str, RoutedTask]]:
    if result.mode != "direct":
        return
    for index, task in enumerate(result.tasks):
        yield task_id_for(request_id, index), task


def knowledge_processing_requests(
    request_id: str,
    result: RouterResult,
) -> tuple[QueryProcessingRequest, ...]:
    requests: list[QueryProcessingRequest] = []
    for task_id, task in iter_direct_tasks(request_id, result):
        if task.route == "knowledge":
            requests.append(
                QueryProcessingRequest(
                    request_id=request_id,
                    task_id=task_id,
                    query=task.query,
                )
            )
    return tuple(requests)


def dispatch_knowledge_tasks(
    request_id: str,
    result: RouterResult,
) -> tuple[ProcessedQuery, ...]:
    """Process knowledge tasks one by one in original task-index order."""

    processed: list[ProcessedQuery] = []
    for request in knowledge_processing_requests(request_id, result):
        processed.append(process_query(request))
    return tuple(processed)


def unsupported_user_reply(*, subtask: bool) -> str:
    if subtask:
        return USER_REPLY_UNSUPPORTED_SUBTASK
    return USER_REPLY_UNSUPPORTED


def router_mode_user_reply(result: RouterResult) -> str | None:
    if result.mode == "confirm_rewrite":
        return USER_REPLY_CONFIRM_REWRITE_TEMPLATE.format(
            candidate_query=result.candidate_query
        )
    if result.mode == "clarify":
        return USER_REPLY_CLARIFY
    if result.mode == "task_limit":
        return USER_REPLY_TASK_LIMIT
    return None


def system_task_ids(
    request_id: str,
    result: RouterResult,
) -> tuple[tuple[str, RoutedTask], ...]:
    """Expose system tasks for an injected SystemHandler. B1 does not write FAQ text."""

    return tuple(
        (task_id, task)
        for task_id, task in iter_direct_tasks(request_id, result)
        if task.route == "system"
    )


def dispatch_system_tasks(
    request_id: str,
    result: RouterResult,
    handler: SystemHandler,
) -> tuple[str, ...]:
    replies: list[str] = []
    for task_id, task in system_task_ids(request_id, result):
        replies.append(
            handler.handle_system_task(
                request_id=request_id,
                task_id=task_id,
                query=task.query,
            )
        )
    return tuple(replies)
