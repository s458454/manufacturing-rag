"""Route + knowledge-mode diagnostic for exposed B1 JSONL. Not QueryRouter."""

from __future__ import annotations

from collections.abc import Mapping

from retrieval.b1.errors import RouterError
from retrieval.b1.types import ConversationTurn
from retrieval.b1.validation import validate_query_text
from retrieval.b1_jev_experimental.types import JevAnswers, JevModeAnswer

_SKIP_BUCKETS = frozenset({"clarify", "mixed"})
_KNOWLEDGE_ROUTE_BUCKETS = frozenset({"confirm_rewrite", "task_limit_gt8"})
_DIRECT_BUCKETS = frozenset({"knowledge_single", "long_tail", "zh_en"})


def history_from_case(case: Mapping[str, object]) -> tuple[ConversationTurn, ...]:
    turns: list[ConversationTurn] = []
    for item in case.get("history") or []:
        if not isinstance(item, Mapping):
            raise ValueError("history turn must be an object")
        turns.append(
            ConversationTurn(
                raw_user_query=str(item["raw_user_query"]),
                canonical_user_query=str(item["canonical_user_query"]),
                assistant_response=str(item["assistant_response"]),
            )
        )
    return tuple(turns)


def gold_route(case: Mapping[str, object]) -> str | None:
    expected = case["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError("expected must be an object")
    if expected.get("error_type"):
        return None
    bucket = str(case["bucket"])
    if bucket in _SKIP_BUCKETS:
        return None
    routes: list[str] = []
    for item in expected.get("routes") or []:
        if isinstance(item, str) and item not in routes:
            routes.append(item)
    if len(routes) == 1:
        return routes[0]
    if bucket in _KNOWLEDGE_ROUTE_BUCKETS:
        return "knowledge"
    raise ValueError(f"no single gold route for {case.get('id')} bucket={bucket}")


def gold_knowledge_mode(case: Mapping[str, object]) -> str | None:
    if gold_route(case) != "knowledge":
        return None
    bucket = str(case["bucket"])
    if bucket in _DIRECT_BUCKETS:
        return "direct"
    if bucket == "multi_2_to_8":
        return "multi"
    if bucket == "task_limit_gt8":
        return "task_limit"
    if bucket == "confirm_rewrite":
        return "confirm_rewrite"
    raise ValueError(f"no knowledge_mode gold for {case.get('id')} bucket={bucket}")


def route_lane(case: Mapping[str, object]) -> str:
    expected = case["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError("expected must be an object")
    if expected.get("error_type"):
        return "control"
    if str(case["bucket"]) in _SKIP_BUCKETS:
        return "skip"
    return "route"


def mode_lane(case: Mapping[str, object]) -> str:
    if gold_knowledge_mode(case) is None:
        return "skip"
    return "mode"


def score_control(case: Mapping[str, object]) -> dict[str, object]:
    expected = case["expected"]
    if not isinstance(expected, Mapping):
        raise ValueError("expected must be an object")
    wanted = expected.get("error_type")
    try:
        validate_query_text(case.get("query"))
        got = None
        passed = False
        reason = f"expected {wanted} got none"
    except RouterError as exc:
        got = exc.error_type
        passed = got == wanted
        reason = "error_type" if passed else f"expected {wanted} got {got}"
    return {
        "lane": "control",
        "mark": "PASS" if passed else "FAIL",
        "pass": passed,
        "gold_route": None,
        "got_route": None,
        "reason": reason,
        "got_error": got,
        "mode_lane": "skip",
        "mode_mark": "SKIP",
        "mode_pass": None,
        "gold_mode": None,
        "got_mode": None,
        "mode_reason": "control",
    }


def score_route(case: Mapping[str, object], answers: JevAnswers) -> dict[str, object]:
    lane = route_lane(case)
    got = answers.primary_route
    if lane == "skip":
        return {
            "lane": "skip",
            "mark": "SKIP",
            "pass": None,
            "gold_route": None,
            "got_route": got,
            "reason": f"skip_bucket {case['bucket']}",
            "got_error": None,
            "confidence": answers.primary_route_confidence,
        }
    wanted = gold_route(case)
    passed = got == wanted
    return {
        "lane": "route",
        "mark": "PASS" if passed else "FAIL",
        "pass": passed,
        "gold_route": wanted,
        "got_route": got,
        "reason": "ok" if passed else f"route {got}",
        "got_error": None,
        "confidence": answers.primary_route_confidence,
    }


def score_mode(
    case: Mapping[str, object],
    mode: JevModeAnswer | None,
    *,
    asked: bool,
) -> dict[str, object]:
    wanted = gold_knowledge_mode(case)
    got = mode.knowledge_mode if mode is not None else None
    if wanted is None:
        return {
            "mode_lane": "skip",
            "mode_mark": "SKIP",
            "mode_pass": None,
            "gold_mode": None,
            "got_mode": got,
            "mode_reason": (
                f"skip_mode {case['bucket']}" if asked else "not_knowledge_gold"
            ),
        }
    if not asked or mode is None:
        return {
            "mode_lane": "mode",
            "mode_mark": "SKIP",
            "mode_pass": None,
            "gold_mode": wanted,
            "got_mode": None,
            "mode_reason": "route_not_knowledge",
        }
    passed = got == wanted
    return {
        "mode_lane": "mode",
        "mode_mark": "PASS" if passed else "FAIL",
        "mode_pass": passed,
        "gold_mode": wanted,
        "got_mode": got,
        "mode_reason": "ok" if passed else f"mode {got}",
        "mode_confidence": mode.knowledge_mode_confidence,
    }
