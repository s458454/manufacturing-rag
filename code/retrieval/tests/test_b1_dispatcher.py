"""Dispatcher: original task_index, serial knowledge → B2, SystemHandler is interface-only."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.dispatcher import (  # noqa: E402
    dispatch_knowledge_tasks,
    knowledge_processing_requests,
    system_task_ids,
    unsupported_user_reply,
)
from retrieval.b1.types import RoutedTask, RouterResult  # noqa: E402


def test_knowledge_task_ids_keep_original_indexes() -> None:
    result = RouterResult(
        mode="direct",
        tasks=(
            RoutedTask(route="knowledge", query="查 5006A"),
            RoutedTask(route="system", query="你支持哪些功能"),
            RoutedTask(route="knowledge", query="再查检验要求"),
        ),
        candidate_query=None,
    )
    requests = knowledge_processing_requests("req-123", result)
    assert [item.task_id for item in requests] == ["req-123:0", "req-123:2"]
    processed = dispatch_knowledge_tasks("req-123", result)
    assert [item.task_id for item in processed] == ["req-123:0", "req-123:2"]
    assert processed[0].retrieval_queries == ("查 5006A",)
    assert processed[1].original_query == "再查检验要求"


def test_system_tasks_are_exposed_without_faq_text() -> None:
    result = RouterResult(
        mode="direct",
        tasks=(
            RoutedTask(route="knowledge", query="查 5006A"),
            RoutedTask(route="system", query="你支持 ERP 吗？"),
        ),
        candidate_query=None,
    )
    systems = system_task_ids("req-1", result)
    assert len(systems) == 1
    assert systems[0][0] == "req-1:1"
    assert systems[0][1].query == "你支持 ERP 吗？"


def test_unsupported_replies_are_frozen() -> None:
    assert unsupported_user_reply(subtask=False) == (
        "该请求超出了当前助手职责范围，请调整需求后重试。"
    )
    assert unsupported_user_reply(subtask=True) == (
        "该子请求超出了当前助手职责范围，未执行。"
    )
