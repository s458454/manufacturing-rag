"""B1 business validation after Schema parse."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.business_validation import validate_router_payload  # noqa: E402
from retrieval.b1.errors import RouterProtocolError  # noqa: E402

CURRENT = "NASA-STD-5006A 对铝合金焊接有什么要求？"


def test_direct_single_task_must_equal_current_query() -> None:
    result = validate_router_payload(
        {
            "mode": "direct",
            "tasks": [{"route": "knowledge", "query": CURRENT}],
            "candidate_query": None,
        },
        CURRENT,
    )
    assert result.mode == "direct"
    assert result.tasks[0].query == CURRENT


def test_direct_single_task_rewrite_is_protocol_error() -> None:
    with pytest.raises(RouterProtocolError):
        validate_router_payload(
            {
                "mode": "direct",
                "tasks": [{"route": "knowledge", "query": CURRENT + "（润色）"}],
                "candidate_query": None,
            },
            CURRENT,
        )


def test_duplicate_tasks_are_protocol_error() -> None:
    with pytest.raises(RouterProtocolError, match="duplicate"):
        validate_router_payload(
            {
                "mode": "direct",
                "tasks": [
                    {"route": "knowledge", "query": "A"},
                    {"route": "knowledge", "query": "A"},
                ],
                "candidate_query": None,
            },
            CURRENT,
        )


def test_generated_overlong_query_is_protocol_error_not_user_too_long() -> None:
    with pytest.raises(RouterProtocolError):
        validate_router_payload(
            {
                "mode": "confirm_rewrite",
                "tasks": [],
                "candidate_query": "焊" * 501,
            },
            CURRENT,
        )


def test_clarify_and_task_limit_shapes() -> None:
    clarify = validate_router_payload(
        {"mode": "clarify", "tasks": [], "candidate_query": None},
        CURRENT,
    )
    assert clarify.mode == "clarify"
    limit = validate_router_payload(
        {"mode": "task_limit", "tasks": [], "candidate_query": None},
        CURRENT,
    )
    assert limit.mode == "task_limit"


def test_confirm_rewrite_requires_candidate() -> None:
    with pytest.raises(RouterProtocolError):
        validate_router_payload(
            {"mode": "confirm_rewrite", "tasks": [], "candidate_query": None},
            CURRENT,
        )
    result = validate_router_payload(
        {
            "mode": "confirm_rewrite",
            "tasks": [],
            "candidate_query": "NASA-STD-5006A 对钢材焊接有什么要求？",
        },
        CURRENT,
    )
    assert result.candidate_query == "NASA-STD-5006A 对钢材焊接有什么要求？"
