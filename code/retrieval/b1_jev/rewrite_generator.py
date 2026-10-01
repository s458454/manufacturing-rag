"""Qwen context rewrite generator. It does not choose route or knowledge_mode."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from retrieval.b1.errors import (
    EmptyQueryError,
    InvalidQueryError,
    QueryTooLongError,
    RouterError,
    RouterUnavailableError,
)
from retrieval.b1.types import ConversationTurn
from retrieval.b1.validation import validate_query_text
from retrieval.b1_jev.multi_splitter import TextComplete

USER_REPLY_REWRITE_FAILED = "当前请求依赖上下文，请完整说明需要查询的问题后重试。"

REWRITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_query"],
    "properties": {
        "candidate_query": {"type": "string", "minLength": 1},
    },
}

# xgrammar rejects minLength. Empty text is rejected in parse_rewrite_payload.
REWRITER_GUIDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_query"],
    "properties": {
        "candidate_query": {"type": "string"},
    },
}

_SYSTEM = (
    "你是查询改写器。只输出 JSON 对象，且只能包含 candidate_query。"
    "不要判断 route，不要判断处理形态，不要回答用户问题。"
)


class RewriteFailed(RouterError):
    error_type = "rewrite_error"
    user_reply = USER_REPLY_REWRITE_FAILED

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.user_reply)


def rewriter_messages(
    current_query: str,
    history: Sequence[ConversationTurn],
) -> list[dict[str, str]]:
    lines: list[str] = []
    for index, turn in enumerate(history, start=1):
        lines.append(f"[{index}] canonical_user_query: {turn.canonical_user_query}")
        lines.append(f"[{index}] assistant_response: {turn.assistant_response}")
    memory = "\n".join(lines) if lines else "(empty)"
    rules = (
        "根据当前问题和对话历史，生成一条完整、自包含的查询。\n"
        "规则：\n"
        "1. 以历史中最近相关的完整问题为模板。\n"
        "2. 当前问题改变了对象、材料、属性或条件时，替换模板中的对应部分。\n"
        "3. 被替换的旧对象或旧条件不得保留。\n"
        "4. 不得把历史问题和当前问题直接拼接。\n"
        "5. 只使用输入中已有的信息，不回答问题。\n"
        "6. 只输出 candidate_query。\n"
        f"Active Memory：\n{memory}\n\n"
        f"当前 Query：\n{current_query}"
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": rules},
    ]


def parse_rewrite_payload(raw: object) -> str:
    if not isinstance(raw, str):
        raise RewriteFailed("rewriter output is not text")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RewriteFailed("rewriter output is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RewriteFailed("rewriter payload must be an object")
    if set(payload) != {"candidate_query"}:
        raise RewriteFailed("rewriter payload must contain only candidate_query")
    candidate = payload["candidate_query"]
    if not isinstance(candidate, str):
        raise RewriteFailed("candidate_query must be a string")
    try:
        return validate_query_text(candidate)
    except (EmptyQueryError, InvalidQueryError, QueryTooLongError) as exc:
        raise RewriteFailed("candidate_query failed validation") from exc


def rewrite_candidate(
    current_query: str,
    history: Sequence[ConversationTurn],
    complete: TextComplete,
) -> str:
    try:
        raw = complete(rewriter_messages(current_query, history), REWRITER_GUIDED_SCHEMA)
    except RewriteFailed:
        raise
    except (RouterUnavailableError, RouterError) as exc:
        raise RewriteFailed("rewriter call failed") from exc
    except Exception as exc:
        raise RewriteFailed("rewriter call failed") from exc
    return parse_rewrite_payload(raw)
