"""Qwen text splitter. It does not choose route or knowledge_mode."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from retrieval.b1.errors import (
    EmptyQueryError,
    InvalidQueryError,
    QueryTooLongError,
    RouterError,
    RouterUnavailableError,
)
from retrieval.b1.validation import validate_query_text

USER_REPLY_SPLITTER_FAILED = (
    "当前请求包含多个独立问题，但自动拆分失败，请将问题分别提交后重试。"
)

SPLITTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tasks"],
    "properties": {
        "tasks": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}

# xgrammar rejects minItems, maxItems, and minLength. Those checks stay in
# parse_split_payload.
SPLITTER_GUIDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tasks"],
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                },
            },
        },
    },
}

_SYSTEM = (
    "你是文本拆分器。只输出 JSON 对象，且只能包含 tasks。"
    "不要判断 route，不要判断处理形态，不要回答用户问题。"
)

TextComplete = Callable[[list[dict[str, str]], dict[str, Any]], str]


class SplitterFailed(RouterError):
    error_type = "splitter_error"
    user_reply = USER_REPLY_SPLITTER_FAILED

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.user_reply)


_RULES = (
    "把当前 Query 拆成自包含的 task.query。只使用当前 Query 里已经写明的信息。\n"
    "1. 输出 2～8 个任务。\n"
    "2. 每个 query 只表达一个独立任务。\n"
    "3. 每个 query 脱离原始 Query 后仍应能够独立理解。\n"
    "4. 保留用户原来的标准号、对象、条件、范围和限制。\n"
    "5. 允许补全当前 Query 中已经明确存在、但在局部表达中被省略的信息。\n"
    "6. 禁止使用对话历史中的信息。\n"
    "7. 禁止增加当前 Query 中不存在的新标准、材料、设备、条件或事实。\n"
    "8. 禁止遗漏任务。\n"
    "9. 禁止重复任务。\n"
    "10. 保持用户原始任务顺序。\n"
    "11. 使用最少数量的语义完整独立任务。「A，以及 B」是两个任务，不能合成一条。\n"
    "12. 不回答用户问题。\n"
    '只输出 {"tasks":[{"query":"..."}]}。\n'
    "例子里的标准号、对象不能抄进当前 Query 的输出。"
)

_EXAMPLE_YIJI = (
    "GB/T 5094.1 设备位号怎么编制，以及 IEC 60204-1 急停按钮颜色怎么规定。"
)
_EXAMPLE_YIJI_JSON = (
    '{"tasks":[{"query":"GB/T 5094.1 设备位号怎么编制"},'
    '{"query":"IEC 60204-1 急停按钮颜色怎么规定"}]}'
)
_EXAMPLE_FENBIE = (
    "分别说明 ISO 2768-1 线性尺寸公差、ASME Y14.5 基准如何选择、GB/T 1184 未注形位公差。"
)
_EXAMPLE_FENBIE_JSON = (
    '{"tasks":[{"query":"说明 ISO 2768-1 线性尺寸公差"},'
    '{"query":"说明 ASME Y14.5 基准如何选择"},'
    '{"query":"说明 GB/T 1184 未注形位公差"}]}'
)


def splitter_messages(current_query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _RULES},
        {"role": "user", "content": f"当前 Query：\n{_EXAMPLE_YIJI}"},
        {"role": "assistant", "content": _EXAMPLE_YIJI_JSON},
        {"role": "user", "content": f"当前 Query：\n{_EXAMPLE_FENBIE}"},
        {"role": "assistant", "content": _EXAMPLE_FENBIE_JSON},
        {"role": "user", "content": f"当前 Query：\n{current_query}"},
    ]


def parse_split_payload(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, str):
        raise SplitterFailed("splitter output is not text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SplitterFailed("splitter output is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SplitterFailed("splitter payload must be an object")
    if set(payload) != {"tasks"}:
        raise SplitterFailed("splitter payload must contain only tasks")
    tasks = payload["tasks"]
    if not isinstance(tasks, list) or not 2 <= len(tasks) <= 8:
        raise SplitterFailed("splitter task count is illegal")
    queries: list[str] = []
    seen: set[str] = set()
    for item in tasks:
        if not isinstance(item, dict) or set(item) != {"query"}:
            raise SplitterFailed("splitter task must contain only query")
        query = item["query"]
        if not isinstance(query, str):
            raise SplitterFailed("splitter query must be a string")
        try:
            checked = validate_query_text(query)
        except (EmptyQueryError, InvalidQueryError, QueryTooLongError) as exc:
            raise SplitterFailed("splitter query failed validation") from exc
        if checked in seen:
            raise SplitterFailed("splitter queries are duplicated")
        seen.add(checked)
        queries.append(checked)
    return tuple(queries)


def split_current_query(current_query: str, complete: TextComplete) -> tuple[str, ...]:
    try:
        raw = complete(splitter_messages(current_query), SPLITTER_GUIDED_SCHEMA)
    except SplitterFailed:
        raise
    except (RouterUnavailableError, RouterError) as exc:
        raise SplitterFailed("splitter call failed") from exc
    except Exception as exc:
        raise SplitterFailed("splitter call failed") from exc
    return parse_split_payload(raw)
