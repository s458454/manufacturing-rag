"""Jev questions: route first; knowledge_mode only on a second request. Not QueryRouter."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from retrieval.b1.prompt import ROUTER_SYSTEM_PROMPT
from retrieval.b1.types import ConversationTurn
from retrieval.b1_jev_experimental.constants import (
    JEV_MAX_BODY_BYTES,
    JEV_MODE_QUESTION_KEYS,
    JEV_MODEL,
    JEV_QUESTION_KEYS,
    KNOWLEDGE_MODE_KEYS,
    ROUTE_KEYS,
)
from retrieval.b1_jev_experimental.errors import JevProtocolError

_ROUTE_GUARD = (
    "只判断当前请求的业务 route，三选一："
    "knowledge=走文档 RAG；system=走助手说明；unsupported=直接拒绝（当前系统做不了）。"
    "不要输出说明书里的 JSON，不要生成 candidate_query，不要切分任务，不要用 mode。"
)

_MODE_GUARD = (
    "本题只判断 knowledge 请求的处理形态，五选一。"
    "direct=一条已完整、可直接执行的知识问题；"
    "multi=2～8 条彼此独立的知识问题（现网仍是 mode=direct，这里单独标出）；"
    "task_limit=超过 8 条独立知识问题；"
    "confirm_rewrite=当前句缺关键对象，但历史能唯一补全；"
    "clarify=无法唯一补全，或多种解释都合理。"
    "不要输出说明书里的 JSON，不要生成 candidate_query，不要写出切分后的 task 文本。"
)


def _instructions(guard: str, task: str) -> str:
    return (
        "现网路由器说明书（其中的 route / mode 定义有效）：\n"
        f"{ROUTER_SYSTEM_PROMPT.strip()}\n\n"
        f"{guard}\n"
        f"{task}"
    )


JEV_QUESTIONS: dict[str, dict[str, object]] = {
    "primary_route": {
        "type": "choice",
        "instructions": _instructions(
            _ROUTE_GUARD,
            "本题（choice）：当前请求属于哪个 route。"
            "问「标准对 ERP 如何规定」是 knowledge；"
            "问「你能不能查库存 / 能不能改 PLC」是 system；"
            "「导出 ERP 库存 / 写作编程 / 上网 / 看图」是 unsupported。"
            "knowledge 和 unsupported 都不是兜底。",
        ),
        "criteria": {
            "knowledge": "走文档知识检索（RAG）",
            "system": "走助手能力说明",
            "unsupported": "直接拒绝：文档 RAG 之外的执行",
        },
    },
}

JEV_MODE_QUESTIONS: dict[str, dict[str, object]] = {
    "knowledge_mode": {
        "type": "choice",
        "instructions": _instructions(
            _MODE_GUARD,
            "本题（choice）：当前 knowledge 请求属于哪种处理形态。"
            "「NASA-STD-5006A 焊缝验收标准有哪些」是 direct；"
            "一次分别问 2～8 个独立文档问题是 multi；"
            "一次明确列出 9 个及以上独立文档问题是 task_limit；"
            "历史是铝合金要求、当前是「钛呢？」是 confirm_rewrite；"
            "无历史的「按那个处理一下」是 clarify。"
            "比较同一对象的多个属性、一次检索可回答的总结，视为 direct，不是 multi。",
        ),
        "criteria": {
            "direct": "一条已完整的知识问题，可直接检索",
            "multi": "2～8 条彼此独立的知识问题（现网 mode 仍是 direct）",
            "task_limit": "超过 8 条独立知识问题",
            "confirm_rewrite": "缺对象，历史能唯一补全，须先确认",
            "clarify": "补不全或多种解释都合理，须再问",
        },
    },
}


def build_jev_state(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
) -> dict[str, object]:
    return {
        "current_query": current_query,
        "history": [
            {
                "raw_user_query": turn.raw_user_query,
                "canonical_user_query": turn.canonical_user_query,
                "assistant_response": turn.assistant_response,
            }
            for turn in history
        ],
    }


def _validate_choice_question(
    name: str,
    spec: object,
    expected_keys: tuple[str, ...],
) -> None:
    if not isinstance(spec, dict):
        raise JevProtocolError(f"{name} must be an object")
    if spec.get("type") != "choice":
        raise JevProtocolError(f"{name} type must be choice")
    if not isinstance(spec.get("instructions"), str) or spec["instructions"] == "":
        raise JevProtocolError(f"{name} instructions must be a non-empty string")
    criteria = spec.get("criteria")
    if not isinstance(criteria, dict) or tuple(criteria) != expected_keys:
        raise JevProtocolError(f"{name} criteria keys are illegal")


def validate_jev_wire_request(request: object) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise JevProtocolError("Jev request must be an object")
    extra = set(request) - {"model", "state", "questions"}
    if extra:
        raise JevProtocolError(f"Jev request has illegal fields: {sorted(extra)}")
    model = request.get("model")
    state = request.get("state")
    questions = request.get("questions")
    if not isinstance(model, str) or model == "":
        raise JevProtocolError("model must be a non-empty string")
    if not isinstance(state, (str, dict)):
        raise JevProtocolError("state must be a string or object")
    if isinstance(state, dict):
        if "current_query" not in state or "history" not in state:
            raise JevProtocolError("state object must contain current_query and history")
    if not isinstance(questions, dict) or not questions:
        raise JevProtocolError("questions must be a non-empty object")
    keys = tuple(questions)
    if keys == JEV_QUESTION_KEYS:
        _validate_choice_question("primary_route", questions["primary_route"], ROUTE_KEYS)
    elif keys == JEV_MODE_QUESTION_KEYS:
        _validate_choice_question(
            "knowledge_mode",
            questions["knowledge_mode"],
            KNOWLEDGE_MODE_KEYS,
        )
    else:
        raise JevProtocolError(
            "questions must contain only primary_route or only knowledge_mode"
        )
    encoded = json.dumps(request, ensure_ascii=False).encode("utf-8")
    if len(encoded) > JEV_MAX_BODY_BYTES:
        raise JevProtocolError(
            f"Jev request is {len(encoded)} bytes; tu-zi limit is {JEV_MAX_BODY_BYTES}"
        )
    return request


def _build_request(
    current_query: str,
    history: Sequence[ConversationTurn],
    *,
    model: str,
    questions: dict[str, dict[str, object]],
) -> dict[str, object]:
    return validate_jev_wire_request(
        {
            "model": model,
            "state": build_jev_state(current_query, history),
            "questions": questions,
        }
    )


def build_jev_request(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
    *,
    model: str = JEV_MODEL,
) -> dict[str, object]:
    return _build_request(current_query, history, model=model, questions=JEV_QUESTIONS)


def build_jev_mode_request(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
    *,
    model: str = JEV_MODEL,
) -> dict[str, object]:
    return _build_request(
        current_query,
        history,
        model=model,
        questions=JEV_MODE_QUESTIONS,
    )
