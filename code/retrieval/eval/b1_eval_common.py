"""Shared live-vLLM scoring for B1 case JSONL sets."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

_SCRIPT_DIR = Path(__file__).resolve().parent
_CODE_ROOT = _SCRIPT_DIR.parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from retrieval.b1.client import VLLMRouterClient
from retrieval.b1.config import load_router_config
from retrieval.b1.constants import (
    ROUTER_CONTRACT_VERSION,
    ROUTER_PROMPT_VERSION,
    ROUTER_SCHEMA_VERSION,
)
from retrieval.b1.errors import (
    RouterError,
    RouterProtocolError,
    RouterUnavailableError,
)
from retrieval.b1.memory import InMemoryConversationStore, commit_turn
from retrieval.b1.router import QueryRouter
from retrieval.b1.tokenizer import load_router_tokenizer
from retrieval.b1.types import ConversationTurn, QueryRequest, RouterObservation

CONTROL_BUCKET = "control"
SEMANTIC_OVERALL_GATE = 0.95
BUCKET_GATE = 0.80
EXPECTED_SEMANTIC = 60
EXPECTED_CONTROL = 4
EXPECTED_TOTAL = 64


class TappingClient:
    def __init__(self, inner: VLLMRouterClient) -> None:
        self._inner = inner
        self.last_content: str | None = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.last_content = self._inner.complete(messages)
        return self.last_content


def load_cases(set_path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line in set_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _one_line(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.split())


def parse_mode(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        mode = payload.get("mode")
        if isinstance(mode, str):
            return mode
    return None


def payload_preview(raw: str | None, current_query: str) -> str:
    if not raw:
        return "raw=<empty>"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return f"raw_unparsed={_one_line(raw)}"
    if not isinstance(payload, dict):
        return f"raw_type={type(payload).__name__}"
    mode = payload.get("mode")
    tasks = payload.get("tasks")
    candidate = payload.get("candidate_query")
    task_count: object
    routes: list[str] = []
    queries: list[str] = []
    if isinstance(tasks, list):
        task_count = len(tasks)
        for item in tasks:
            if not isinstance(item, dict):
                routes.append("?")
                continue
            route = item.get("route")
            routes.append(route if isinstance(route, str) else "?")
            query = item.get("query")
            if isinstance(query, str):
                queries.append(query)
    else:
        task_count = "n/a"
    query_eq = "n/a"
    if task_count == 1 and queries:
        query_eq = "yes" if queries[0] == current_query else "no"
    return (
        f"mode={mode} task_count={task_count} routes={routes} "
        f"query_eq={query_eq} candidate={candidate!r}"
    )


def print_failure_dump(
    case: dict[str, object],
    *,
    error: RouterError | None,
    raw: str | None,
) -> None:
    print(f"  current={_one_line(str(case['query']))}")
    if error is not None:
        print(f"  error_detail={_one_line(str(error))}")
        cause = error.__cause__
        if cause is not None:
            print(f"  cause={type(cause).__name__}: {_one_line(str(cause))}")
    print(f"  preview={payload_preview(raw, str(case['query']))}")
    if raw:
        print(f"  raw={_one_line(raw)}")
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = None
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if isinstance(tasks, list):
        for index, item in enumerate(tasks):
            if isinstance(item, dict) and isinstance(item.get("query"), str):
                print(
                    f"  task[{index}].route={item.get('route')} "
                    f"query={_one_line(item['query'])}"
                )
    if isinstance(payload, dict) and payload.get("candidate_query"):
        print(f"  candidate_query={_one_line(str(payload['candidate_query']))}")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def semantic_pass(case: dict[str, object], result: object) -> tuple[bool, str]:
    expected = case["expected"]
    got_mode = result.mode
    got_routes = [task.route for task in result.tasks]
    if got_mode != expected["mode"]:
        return False, f"mode {got_mode}"
    if got_routes != list(expected["routes"]):
        return False, f"routes {got_routes}"
    if expected.get("single_query_equals_current") and result.tasks:
        current = str(case["query"])
        if result.tasks[0].query != current:
            return False, "single task query rewritten"
    if result.candidate_query != expected.get("candidate_query"):
        return False, "candidate_query mismatch"
    return True, "ok"


def score_case(
    case: dict[str, object],
    *,
    result: object,
    error: RouterError | None,
    raw: str | None,
    latency_ms: float | None,
) -> dict[str, object]:
    expected = case["expected"]
    expected_error = expected.get("error_type")
    got_error = None if error is None else error.error_type
    predicted_mode = None if result is None else result.mode
    if predicted_mode is None:
        predicted_mode = parse_mode(raw)
    gold_mode = expected.get("mode")
    false_task_limit = (
        gold_mode not in (None, "task_limit") and predicted_mode == "task_limit"
    )
    unavailable = isinstance(error, RouterUnavailableError)
    protocol_error = isinstance(error, RouterProtocolError)
    protocol_valid = error is None and result is not None
    if expected_error:
        control_pass = got_error == expected_error
        return {
            "channel": "control",
            "control_pass": control_pass,
            "protocol_valid": None,
            "semantic_pass": None,
            "unavailable": False,
            "false_task_limit": False,
            "got_mode": predicted_mode,
            "got_error": got_error,
            "latency_ms": latency_ms,
            "reason": "error_type" if control_pass else f"expected {expected_error} got {got_error}",
        }
    if unavailable:
        return {
            "channel": "infra",
            "control_pass": None,
            "protocol_valid": None,
            "semantic_pass": None,
            "unavailable": True,
            "false_task_limit": false_task_limit,
            "got_mode": predicted_mode,
            "got_error": got_error,
            "latency_ms": latency_ms,
            "reason": "router_unavailable",
        }
    if not protocol_valid:
        reason = f"unexpected {got_error}" if got_error else "missing result"
        if protocol_error:
            reason = "router_protocol_error"
        return {
            "channel": "semantic",
            "control_pass": None,
            "protocol_valid": False,
            "semantic_pass": False,
            "unavailable": False,
            "false_task_limit": false_task_limit,
            "got_mode": predicted_mode,
            "got_error": got_error,
            "latency_ms": latency_ms,
            "reason": reason,
        }
    passed, reason = semantic_pass(case, result)
    return {
        "channel": "semantic",
        "control_pass": None,
        "protocol_valid": True,
        "semantic_pass": passed,
        "unavailable": False,
        "false_task_limit": false_task_limit,
        "got_mode": predicted_mode,
        "got_error": None,
        "latency_ms": latency_ms,
        "reason": reason,
    }


def gate(passed: int, total: int, threshold: float) -> bool:
    if total <= 0:
        return False
    return (passed / total) >= threshold


def run_eval_set(
    set_path: Path,
    *,
    status: str,
    role: str,
    gates_met_label: str,
    extra_headers: Iterable[str] = (),
) -> None:
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
    inner = VLLMRouterClient(config)
    client = TappingClient(inner)
    cases = load_cases(set_path)
    if len(cases) != EXPECTED_TOTAL:
        raise SystemExit(f"set must contain {EXPECTED_TOTAL} cases")
    print(f"set={set_path}")
    print(f"case_count={len(cases)}")
    print(f"status={status}")
    print(f"router_contract_version={ROUTER_CONTRACT_VERSION}")
    print(f"router_prompt_version={ROUTER_PROMPT_VERSION}")
    print(f"router_schema_version={ROUTER_SCHEMA_VERSION}")
    print(f"role={role}")
    for line in extra_headers:
        print(line)
    rows: list[dict[str, object]] = []
    for case in cases:
        observations: list[RouterObservation] = []
        store = InMemoryConversationStore()
        conversation_id = str(case["id"])
        router = QueryRouter(
            config=config,
            client=client,
            tokenizer=tokenizer,
            store=store,
            observer=observations.append,
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
        client.last_content = None
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
        latency_ms = observations[-1].router_latency_ms if observations else None
        scored = score_case(
            case,
            result=result,
            error=error,
            raw=client.last_content,
            latency_ms=latency_ms,
        )
        scored["id"] = case["id"]
        scored["bucket"] = case["bucket"]
        scored["expected_mode"] = case["expected"].get("mode") or case["expected"].get(
            "error_type"
        )
        rows.append(scored)
        if scored["channel"] == "control":
            mark = "PASS" if scored["control_pass"] else "FAIL"
        elif scored["unavailable"]:
            mark = "INFRA"
        elif scored["semantic_pass"]:
            mark = "PASS"
        else:
            mark = "FAIL"
        print(
            f"{mark} {case['id']} bucket={case['bucket']} "
            f"expected={scored['expected_mode']} got_mode={scored['got_mode']} "
            f"got_error={scored['got_error']} reason={scored['reason']}"
        )
        if mark in {"FAIL", "INFRA"}:
            print_failure_dump(case, error=error, raw=client.last_content)
    control_rows = [row for row in rows if row["channel"] == "control"]
    semantic_rows = [row for row in rows if row["bucket"] != CONTROL_BUCKET]
    healthy = [row for row in semantic_rows if not row["unavailable"]]
    protocol_pass = sum(1 for row in healthy if row["protocol_valid"])
    semantic_exact = sum(1 for row in healthy if row["semantic_pass"])
    unavailable_n = sum(1 for row in semantic_rows if row["unavailable"])
    false_task_limit_n = sum(1 for row in semantic_rows if row["false_task_limit"])
    gt8_rows = [row for row in healthy if row["bucket"] == "task_limit_gt8"]
    gt8_pass = sum(
        1
        for row in gt8_rows
        if row["protocol_valid"] and row["got_mode"] == "task_limit" and row["semantic_pass"]
    )
    control_pass = sum(1 for row in control_rows if row["control_pass"])
    latencies = [
        float(row["latency_ms"])
        for row in semantic_rows
        if isinstance(row["latency_ms"], (int, float))
    ]
    print("---")
    print(f"control={control_pass}/{len(control_rows)}")
    print(f"protocol_healthy={protocol_pass}/{len(healthy)}")
    print(f"semantic_exact={semantic_exact}/{len(healthy)}")
    print(
        f"all_cases_incl_control={control_pass + semantic_exact}/{len(control_rows) + len(healthy)}"
    )
    print(f"unavailable={unavailable_n}")
    print(f"false_task_limit={false_task_limit_n}")
    print(f"task_limit_gt8={gt8_pass}/{len(gt8_rows)}")
    bucket_ok = True
    by_bucket: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_bucket.setdefault(str(row["bucket"]), []).append(row)
    for bucket, bucket_rows in by_bucket.items():
        if bucket == CONTROL_BUCKET:
            passed = sum(1 for row in bucket_rows if row["control_pass"])
            total = len(bucket_rows)
            print(f"bucket_{bucket}={passed}/{total}")
            continue
        eligible = [row for row in bucket_rows if not row["unavailable"]]
        passed = sum(1 for row in eligible if row["semantic_pass"])
        total = len(eligible)
        print(f"bucket_{bucket}={passed}/{total}")
        if not gate(passed, total, BUCKET_GATE):
            bucket_ok = False
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    print(f"latency_p50_ms={None if p50 is None else round(p50, 1)}")
    print(f"latency_p95_ms={None if p95 is None else round(p95, 1)}")
    print("latency_frozen=no")
    gate_control = control_pass == EXPECTED_CONTROL and len(control_rows) == EXPECTED_CONTROL
    gate_protocol = len(healthy) == EXPECTED_SEMANTIC and protocol_pass == len(healthy)
    gate_semantic = gate(semantic_exact, len(healthy), SEMANTIC_OVERALL_GATE)
    gate_false_task_limit = false_task_limit_n == 0
    gate_gt8 = len(gt8_rows) == 4 and gt8_pass == 4
    gate_unavailable = unavailable_n == 0
    gates_met = (
        gate_control
        and gate_protocol
        and gate_semantic
        and bucket_ok
        and gate_false_task_limit
        and gate_gt8
        and gate_unavailable
    )
    print(f"gate_control={'yes' if gate_control else 'no'}")
    print(f"gate_protocol={'yes' if gate_protocol else 'no'}")
    print(f"gate_semantic_overall={'yes' if gate_semantic else 'no'}")
    print(f"gate_bucket_min_80={'yes' if bucket_ok else 'no'}")
    print(f"gate_false_task_limit={'yes' if gate_false_task_limit else 'no'}")
    print(f"gate_gt8_task_limit={'yes' if gate_gt8 else 'no'}")
    print(f"gate_unavailable={'yes' if gate_unavailable else 'no'}")
    print(f"{gates_met_label}={'yes' if gates_met else 'no'}")
    print("b1_closed=no")
