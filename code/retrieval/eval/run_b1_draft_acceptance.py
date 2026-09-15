"""Run the draft B1 acceptance set against a live vLLM Router.

This is not a frozen B1 close-out. It scores mode / route sequence / single-task
identity / confirm_rewrite candidate text. It does not freeze thresholds.

Requires the same env as verify_b1_router_server.py:
  B1_VLLM_BASE_URL
  B1_MODEL_ALIAS
  B1_TOKENIZER_NAME_OR_PATH
  B1_REQUEST_TIMEOUT_MS
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.client import VLLMRouterClient
from retrieval.b1.config import load_router_config
from retrieval.b1.errors import RouterError
from retrieval.b1.memory import InMemoryConversationStore, commit_turn
from retrieval.b1.router import QueryRouter
from retrieval.b1.tokenizer import load_router_tokenizer
from retrieval.b1.types import ConversationTurn, QueryRequest

SET_PATH = SCRIPT_DIR / "b1_draft_acceptance_set.jsonl"


def _load_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line in SET_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _score(case: dict[str, object], *, result: object, error: RouterError | None) -> dict[str, object]:
    expected = case["expected"]
    expected_error = expected.get("error_type")
    got_error = None if error is None else error.error_type
    if expected_error:
        passed = got_error == expected_error
        return {
            "passed": passed,
            "got_mode": None,
            "got_routes": [],
            "got_error": got_error,
            "reason": "error_type" if passed else f"expected {expected_error} got {got_error}",
        }
    if error is not None:
        return {
            "passed": False,
            "got_mode": None,
            "got_routes": [],
            "got_error": got_error,
            "reason": f"unexpected {got_error}",
        }
    got_mode = result.mode
    got_routes = [task.route for task in result.tasks]
    if got_mode != expected["mode"]:
        return {
            "passed": False,
            "got_mode": got_mode,
            "got_routes": got_routes,
            "got_error": None,
            "reason": f"mode {got_mode}",
        }
    if got_routes != list(expected["routes"]):
        return {
            "passed": False,
            "got_mode": got_mode,
            "got_routes": got_routes,
            "got_error": None,
            "reason": f"routes {got_routes}",
        }
    if expected.get("single_query_equals_current") and result.tasks:
        current = str(case["query"])
        if result.tasks[0].query != current:
            return {
                "passed": False,
                "got_mode": got_mode,
                "got_routes": got_routes,
                "got_error": None,
                "reason": "single task query rewritten",
            }
    expected_candidate = expected.get("candidate_query")
    if result.candidate_query != expected_candidate:
        return {
            "passed": False,
            "got_mode": got_mode,
            "got_routes": got_routes,
            "got_error": None,
            "reason": "candidate_query mismatch",
        }
    return {
        "passed": True,
        "got_mode": got_mode,
        "got_routes": got_routes,
        "got_error": None,
        "reason": "ok",
    }


def main() -> None:
    required = (
        "B1_VLLM_BASE_URL",
        "B1_MODEL_ALIAS",
        "B1_TOKENIZER_NAME_OR_PATH",
        "B1_REQUEST_TIMEOUT_MS",
    )
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        raise SystemExit("missing " + ",".join(missing))
    config = load_router_config()
    tokenizer = load_router_tokenizer(config.tokenizer_name_or_path)
    client = VLLMRouterClient(config)
    cases = _load_cases()
    by_bucket: dict[str, list[bool]] = {}
    print(f"set={SET_PATH}")
    print(f"case_count={len(cases)}")
    print("status=DRAFT_NOT_FROZEN")
    print("role=DEV_DIAGNOSTIC_NOT_HOLDOUT")
    for case in cases:
        store = InMemoryConversationStore()
        conversation_id = str(case["id"])
        router = QueryRouter(
            config=config,
            client=client,
            tokenizer=tokenizer,
            store=store,
        )
        for turn in case.get("history") or []:
            commit_turn(
                store,
                conversation_id,
                ConversationTurn(
                    raw_user_query=str(turn["raw_user_query"]),
                    canonical_user_query=str(turn["canonical_user_query"]),
                    assistant_response=str(turn["assistant_response"]),
                ),
                tokenizer,
            )
        result = None
        error: RouterError | None = None
        try:
            result = router.route(
                QueryRequest(
                    request_id=str(case["id"]),
                    conversation_id=conversation_id,
                    query=str(case["query"]),
                )
            )
        except RouterError as exc:
            error = exc
        scored = _score(case, result=result, error=error)
        by_bucket.setdefault(str(case["bucket"]), []).append(bool(scored["passed"]))
        mark = "PASS" if scored["passed"] else "FAIL"
        print(
            f"{mark} {case['id']} bucket={case['bucket']} "
            f"expected={case['expected'].get('mode') or case['expected'].get('error_type')} "
            f"got_mode={scored['got_mode']} got_error={scored['got_error']} "
            f"reason={scored['reason']}"
        )
    print("---")
    total = 0
    passed = 0
    for bucket, flags in by_bucket.items():
        total += len(flags)
        passed += sum(flags)
        print(f"bucket_{bucket}={sum(flags)}/{len(flags)}")
    print(f"mode_route_score={passed}/{total}")
    print("b1_closed=no")


if __name__ == "__main__":
    main()
