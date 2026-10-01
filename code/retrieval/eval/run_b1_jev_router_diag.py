"""Exposed full-result diagnostic for the B1-Jev candidate router.

Not QueryRouter. Not a CLOSED set. clarify and mixed stay skip.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.config import load_router_config
from retrieval.b1.dispatcher import knowledge_processing_requests
from retrieval.b1.errors import (
    RouterConfigurationError,
    RouterError,
    RouterProtocolError,
    RouterUnavailableError,
)
from retrieval.b1.memory import InMemoryConversationStore
from retrieval.b1.types import ConversationTurn, QueryRequest
from retrieval.b1.validation import validate_query_text
from retrieval.b1_jev.jev_client import (
    JevUnavailableError,
    SystemOneJudge,
    resolve_jev_connection,
)
from retrieval.b1_jev.multi_splitter import SplitterFailed
from retrieval.b1_jev.rewrite_generator import RewriteFailed
from retrieval.b1_jev.router import JevQueryRouter, qwen_text_complete

SCRIPT_DIR = Path(__file__).resolve().parent
SETS = {
    "holdout": SCRIPT_DIR / "b1_holdout_v1.jsonl",
    "blind": SCRIPT_DIR / "b1_blind_acceptance_v1.jsonl",
}
MULTI_GOLD_PATH = SCRIPT_DIR / "b1_jev_multi_task_gold_v1.jsonl"
SKIP_BUCKETS = frozenset({"clarify", "mixed"})
AMBIGUOUS_IDS = frozenset({"ba-051"})
_ANCHOR = re.compile(
    r"NASA-STD-\d+(?:\.\d+)?[A-Z]?"
    r"|NASA-HDBK-\d+(?:\.\d+)?(?:-\d+)?"
    r"|NISTIR\s*\d+"
    r"|YH5WR-\d+/\d+"
    r"|水贝|兆瓦"
)
DIRECT_BUCKETS = frozenset({"knowledge_single", "long_tail", "zh_en", "system", "unsupported"})
KNOWLEDGE_MODE_BY_BUCKET = {
    "knowledge_single": "direct",
    "long_tail": "direct",
    "zh_en": "direct",
    "multi_2_to_8": "multi",
    "task_limit_gt8": "task_limit",
    "confirm_rewrite": "confirm_rewrite",
}


class _CharTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> list[int] | str:
        text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        if tokenize:
            return [0] * len(text)
        return text


class RecordingJudge:
    def __init__(self, inner: SystemOneJudge) -> None:
        self._inner = inner
        self.primary_route: str | None = None
        self.knowledge_mode: str | None = None
        self.primary_calls = 0
        self.mode_calls = 0

    def reset(self) -> None:
        self.primary_route = None
        self.knowledge_mode = None
        self.primary_calls = 0
        self.mode_calls = 0

    def primary_route_call(
        self,
        current_query: str,
        history: tuple[ConversationTurn, ...],
    ) -> str:
        self.primary_calls += 1
        label = self._inner.primary_route(current_query, history)
        self.primary_route = label
        return label

    def knowledge_mode_call(
        self,
        current_query: str,
        history: tuple[ConversationTurn, ...],
    ) -> str:
        self.mode_calls += 1
        label = self._inner.knowledge_mode(current_query, history)
        self.knowledge_mode = label
        return label


class _BoundJudge:
    def __init__(self, recording: RecordingJudge) -> None:
        self._recording = recording

    def primary_route(
        self,
        current_query: str,
        history: tuple[ConversationTurn, ...],
    ) -> str:
        return self._recording.primary_route_call(current_query, history)

    def knowledge_mode(
        self,
        current_query: str,
        history: tuple[ConversationTurn, ...],
    ) -> str:
        return self._recording.knowledge_mode_call(current_query, history)


def diagnostic_status() -> dict[str, str]:
    return {"b1_closed": "yes", "main_path": "connected"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} row is not an object")
            rows.append(payload)
    return rows


def load_multi_gold(path: Path = MULTI_GOLD_PATH) -> dict[str, list[str]]:
    gold: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        if row.get("closed") is not False:
            raise ValueError(f"multi gold {row.get('id')} must stay closed=false")
        tasks = row.get("tasks")
        if not isinstance(tasks, list) or not all(isinstance(item, str) for item in tasks):
            raise ValueError(f"multi gold {row.get('id')} tasks must be strings")
        gold[str(row["id"])] = list(tasks)
    return gold


def classify_case(case: dict[str, Any]) -> str:
    if str(case.get("id")) in AMBIGUOUS_IDS:
        return "ambiguous"
    expected = case.get("expected")
    if isinstance(expected, dict) and expected.get("error_type"):
        return "control"
    if str(case.get("bucket")) in SKIP_BUCKETS:
        return "skip"
    return "result"


def task_anchors(text: str) -> list[str]:
    return [item.replace(" ", "") for item in _ANCHOR.findall(text)]


class CountingComplete:
    """Count Splitter vs Rewriter calls. Does not change generation."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.splitter_calls = 0
        self.rewriter_calls = 0

    def reset(self) -> None:
        self.splitter_calls = 0
        self.rewriter_calls = 0

    def __call__(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        required = schema.get("required") if isinstance(schema, dict) else None
        keys = set(required) if isinstance(required, list) else set()
        if "tasks" in keys:
            self.splitter_calls += 1
        elif "candidate_query" in keys:
            self.rewriter_calls += 1
        return self._inner(messages, schema)


def gold_primary_route(case: dict[str, Any]) -> str | None:
    """Single-route gold already present on the exposed row. Never invents one for clarify."""

    if classify_case(case) != "result":
        return None
    expected = case.get("expected")
    if not isinstance(expected, dict):
        return None
    routes: list[str] = []
    for item in expected.get("routes") or []:
        if isinstance(item, str) and item not in routes:
            routes.append(item)
    if len(routes) == 1:
        return routes[0]
    bucket = str(case.get("bucket"))
    if bucket in {"confirm_rewrite", "task_limit_gt8", "multi_2_to_8"}:
        return "knowledge"
    return None


def gold_knowledge_mode(case: dict[str, Any]) -> str | None:
    if gold_primary_route(case) != "knowledge":
        return None
    return KNOWLEDGE_MODE_BY_BUCKET.get(str(case.get("bucket")))


def history_from_case(case: dict[str, Any]) -> tuple[ConversationTurn, ...]:
    turns: list[ConversationTurn] = []
    for item in case.get("history") or []:
        if not isinstance(item, dict):
            raise ValueError("history turn must be an object")
        turns.append(
            ConversationTurn(
                raw_user_query=str(item["raw_user_query"]),
                canonical_user_query=str(item["canonical_user_query"]),
                assistant_response=str(item["assistant_response"]),
            )
        )
    return tuple(turns)


def _base_row(case: dict[str, Any], lane: str) -> dict[str, Any]:
    return {
        "id": case.get("id"),
        "bucket": case.get("bucket"),
        "lane": lane,
        "pass": None,
        "reason": lane,
        "primary_route": None,
        "knowledge_mode": None,
        "result_mode": None,
        "task_count": None,
        "routes": None,
        "queries": None,
        "candidate_query": None,
        "unavailable": False,
        "protocol_error": False,
        "splitter_error": False,
        "rewrite_error": False,
        "splitter_called": False,
        "rewriter_called": False,
        "dispatcher_knowledge_request_count": 0,
        "split_exact": None,
        "error_type": None,
    }


def skip_row(case: dict[str, Any]) -> dict[str, Any]:
    row = _base_row(case, "skip")
    row["reason"] = f"skip_bucket {case.get('bucket')}"
    row["gold_primary_route"] = None
    return row


def score_control(case: dict[str, Any], error_type: str | None) -> dict[str, Any]:
    expected = case["expected"]
    wanted = expected.get("error_type") if isinstance(expected, dict) else None
    passed = error_type == wanted
    row = _base_row(case, "control")
    row["pass"] = passed
    row["reason"] = "ok" if passed else f"expected {wanted} got {error_type}"
    row["got_error"] = error_type
    return row


def _fail(row: dict[str, Any], reason: str) -> dict[str, Any]:
    row["pass"] = False
    row["reason"] = reason
    return row


def score_result(
    case: dict[str, Any],
    observed: dict[str, Any],
    multi_gold: dict[str, list[str]],
) -> dict[str, Any]:
    lane = classify_case(case)
    if lane == "skip":
        return skip_row(case)
    if lane == "ambiguous":
        row = _base_row(case, "ambiguous")
        _copy_observed(row, observed)
        row["pass"] = None
        row["reason"] = "ambiguous"
        return row
    row = _base_row(case, "result")
    _copy_observed(row, observed)
    error_type = observed.get("error_type")
    if error_type == "router_unavailable":
        row["unavailable"] = True
        row["reason"] = "unavailable"
        return row
    if error_type == "router_protocol_error":
        row["protocol_error"] = True
        return _fail(row, "protocol_error")
    if error_type == "splitter_error":
        row["splitter_error"] = True
        return _fail(row, "splitter_error")
    if error_type == "rewrite_error":
        row["rewrite_error"] = True
        return _fail(row, "rewrite_error")
    if error_type:
        return _fail(row, f"error {error_type}")

    wanted_route = gold_primary_route(case)
    if row["primary_route"] != wanted_route:
        return _fail(row, f"primary_route {row['primary_route']}")
    wanted_mode = gold_knowledge_mode(case)
    if wanted_mode is not None and row["knowledge_mode"] != wanted_mode:
        return _fail(row, f"knowledge_mode {row['knowledge_mode']}")
    if wanted_mode is None and row["knowledge_mode"] is not None:
        return _fail(row, f"unexpected knowledge_mode {row['knowledge_mode']}")

    expected = case["expected"]
    if not isinstance(expected, dict):
        return _fail(row, "expected missing")
    bucket = str(case.get("bucket"))
    routes = list(observed.get("routes") or [])
    queries = list(observed.get("queries") or [])
    if bucket in DIRECT_BUCKETS or expected.get("single_query_equals_current") is True:
        current = validate_query_text(case.get("query"))
        if (
            observed.get("result_mode") == "direct"
            and routes == [wanted_route]
            and queries == [current]
            and observed.get("candidate_query") is None
        ):
            row["pass"] = True
            row["reason"] = "ok"
            return row
        return _fail(row, "direct result mismatch")
    if bucket == "multi_2_to_8":
        return _score_multi(row, case, expected, routes, queries, multi_gold)
    if bucket == "task_limit_gt8":
        if (
            observed.get("result_mode") == "task_limit"
            and observed.get("task_count") == 0
            and observed.get("candidate_query") is None
        ):
            row["pass"] = True
            row["reason"] = "ok"
            return row
        return _fail(row, "task_limit result mismatch")
    if bucket == "confirm_rewrite":
        if (
            observed.get("result_mode") == "confirm_rewrite"
            and observed.get("task_count") == 0
            and observed.get("candidate_query") == expected.get("candidate_query")
        ):
            row["pass"] = True
            row["reason"] = "ok"
            return row
        return _fail(row, "confirm_rewrite result mismatch")
    return _fail(row, f"unscored bucket {bucket}")


def _copy_observed(row: dict[str, Any], observed: dict[str, Any]) -> None:
    row["primary_route"] = observed.get("primary_route")
    row["knowledge_mode"] = observed.get("knowledge_mode")
    row["result_mode"] = observed.get("result_mode")
    row["task_count"] = observed.get("task_count")
    row["routes"] = observed.get("routes")
    row["queries"] = observed.get("queries")
    row["candidate_query"] = observed.get("candidate_query")
    row["splitter_called"] = bool(observed.get("splitter_called"))
    row["rewriter_called"] = bool(observed.get("rewriter_called"))
    row["dispatcher_knowledge_request_count"] = int(
        observed.get("dispatcher_knowledge_request_count") or 0
    )
    row["error_type"] = observed.get("error_type")


def _score_multi(
    row: dict[str, Any],
    case: dict[str, Any],
    expected: dict[str, Any],
    routes: list[str],
    queries: list[str],
    multi_gold: dict[str, list[str]],
) -> dict[str, Any]:
    gold_queries = multi_gold.get(str(case.get("id")))
    if gold_queries is None:
        return _fail(row, "experimental_multi_gold_missing")
    exact = queries == gold_queries
    row["split_exact"] = exact
    anchors_ok = len(queries) == len(gold_queries) and all(
        task_anchors(left) == task_anchors(right)
        for left, right in zip(gold_queries, queries)
    )
    if (
        row["result_mode"] == "direct"
        and routes == ["knowledge"] * len(gold_queries)
        and row["candidate_query"] is None
        and len(gold_queries) == len(list(expected.get("routes") or []))
        and anchors_ok
    ):
        row["pass"] = True
        row["reason"] = "ok" if exact else "split_semantic_pass"
        return row
    return _fail(row, "multi result mismatch")


def _call_flags(counter: CountingComplete | None) -> dict[str, object]:
    if counter is None:
        return {"splitter_called": False, "rewriter_called": False}
    return {
        "splitter_called": counter.splitter_calls > 0,
        "rewriter_called": counter.rewriter_calls > 0,
    }


def _dispatcher_count(case_id: str, result: Any | None) -> int:
    if result is None:
        return 0
    return len(knowledge_processing_requests(case_id, result))


def _observation_from_error(
    judge: RecordingJudge,
    error_type: str,
    *,
    counter: CountingComplete | None = None,
) -> dict[str, Any]:
    observed = {
        "primary_route": judge.primary_route,
        "knowledge_mode": judge.knowledge_mode,
        "result_mode": None,
        "task_count": None,
        "routes": None,
        "queries": None,
        "candidate_query": None,
        "error_type": error_type,
        "dispatcher_knowledge_request_count": 0,
    }
    observed.update(_call_flags(counter))
    return observed


def _observation_from_result(
    judge: RecordingJudge,
    result: Any,
    *,
    case_id: str,
    counter: CountingComplete | None = None,
) -> dict[str, Any]:
    observed = {
        "primary_route": judge.primary_route,
        "knowledge_mode": judge.knowledge_mode,
        "result_mode": result.mode,
        "task_count": len(result.tasks),
        "routes": [task.route for task in result.tasks],
        "queries": [task.query for task in result.tasks],
        "candidate_query": result.candidate_query,
        "error_type": None,
        "dispatcher_knowledge_request_count": _dispatcher_count(case_id, result),
    }
    observed.update(_call_flags(counter))
    return observed


def run_case(
    case: dict[str, Any],
    *,
    router: JevQueryRouter,
    judge: RecordingJudge,
    multi_gold: dict[str, list[str]],
    counter: CountingComplete | None = None,
) -> dict[str, Any]:
    lane = classify_case(case)
    if lane == "skip":
        return skip_row(case)
    judge.reset()
    if counter is not None:
        counter.reset()
    case_id = str(case.get("id") or "diag")
    request = QueryRequest(
        request_id=case_id,
        conversation_id=case_id,
        query=case.get("query"),  # type: ignore[arg-type]
    )
    if lane == "control":
        try:
            router.route(request)
            error_type = None
        except RouterError as exc:
            error_type = exc.error_type
        row = score_control(case, error_type)
        if judge.primary_calls or judge.mode_calls:
            row["pass"] = False
            row["reason"] = "control_called_jev"
        return row
    try:
        result = router.route(request)
    except SplitterFailed:
        observed = _observation_from_error(judge, "splitter_error", counter=counter)
    except RewriteFailed:
        observed = _observation_from_error(judge, "rewrite_error", counter=counter)
    except RouterUnavailableError:
        observed = _observation_from_error(judge, "router_unavailable", counter=counter)
    except RouterProtocolError:
        observed = _observation_from_error(judge, "router_protocol_error", counter=counter)
    except RouterError as exc:
        observed = _observation_from_error(judge, exc.error_type, counter=counter)
    else:
        observed = _observation_from_result(
            judge,
            result,
            case_id=case_id,
            counter=counter,
        )
    return score_result(case, observed, multi_gold)


def _mark(row: dict[str, Any]) -> str:
    if row["lane"] == "skip":
        return "SKIP"
    if row["lane"] == "ambiguous":
        return "AMBIGUOUS"
    if row["lane"] == "control":
        return "PASS" if row["pass"] else "FAIL"
    if row["unavailable"]:
        return "INFRA"
    if row["protocol_error"]:
        return "PROTOCOL"
    if row["pass"]:
        return "PASS"
    return "FAIL"


def summarize(set_name: str, rows: list[dict[str, Any]]) -> None:
    status = diagnostic_status()
    control = [row for row in rows if row["lane"] == "control"]
    skipped = [row for row in rows if row["lane"] == "skip"]
    results = [row for row in rows if row["lane"] == "result"]
    ambiguous = [row for row in rows if row["lane"] == "ambiguous"]
    unavailable = [row for row in results if row["unavailable"]]
    protocol = [row for row in results if row["protocol_error"]]
    splitter = [row for row in results if row["splitter_error"]]
    rewrite = [row for row in results if row["rewrite_error"]]
    healthy = [row for row in results if not row["unavailable"]]
    passed = sum(1 for row in healthy if row["pass"])
    control_pass = sum(1 for row in control if row["pass"])
    print("---")
    print(f"set_name={set_name}")
    print("task=candidate_router_result")
    print(f"control={control_pass}/{len(control)}")
    print(f"result_pass={passed}/{len(healthy)}")
    print(f"skip={len(skipped)}")
    print(f"ambiguous={len(ambiguous)}")
    print(f"unavailable={len(unavailable)}")
    print(f"protocol={len(protocol)}")
    print(f"splitter_error={len(splitter)}")
    print(f"rewrite_error={len(rewrite)}")
    print(f"b1_closed={status['b1_closed']}")
    print(f"main_path={status['main_path']}")


def _router_for_case(
    case: dict[str, Any],
    *,
    judge: RecordingJudge,
    complete_text: Any,
    tokenizer: _CharTokenizer,
) -> JevQueryRouter:
    store = InMemoryConversationStore()
    conversation_id = str(case.get("id") or "diag")
    store.set_active_memory(conversation_id, history_from_case(case))
    return JevQueryRouter(
        store=store,
        tokenizer=tokenizer,
        judge=_BoundJudge(judge),
        complete_text=complete_text,
    )


def _missing_qwen(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
) -> str:
    raise RouterUnavailableError("Qwen text generator is not configured")


def build_complete_text() -> Any:
    try:
        config = load_router_config()
    except RouterConfigurationError:
        return _missing_qwen
    return qwen_text_complete(config)


def run_set(
    set_name: str,
    *,
    judge: RecordingJudge,
    complete_text: Any,
    multi_gold: dict[str, list[str]],
    sleep_s: float,
) -> list[dict[str, Any]]:
    path = SETS[set_name]
    cases = load_jsonl(path)
    tokenizer = _CharTokenizer()
    counter = CountingComplete(complete_text)
    print(f"set={path}")
    print(f"case_count={len(cases)}")
    print("status=EXPOSED_CANDIDATE_ROUTER_DIAGNOSTIC")
    print("closed_candidate_set=no")
    rows: list[dict[str, Any]] = []
    for case in cases:
        router = _router_for_case(
            case,
            judge=judge,
            complete_text=counter,
            tokenizer=tokenizer,
        )
        row = run_case(
            case,
            router=router,
            judge=judge,
            multi_gold=multi_gold,
            counter=counter,
        )
        rows.append(row)
        print(
            f"{_mark(row)} {row['id']} bucket={row['bucket']} lane={row['lane']} "
            f"primary_route={row['primary_route']} knowledge_mode={row['knowledge_mode']} "
            f"mode={row['result_mode']} task_count={row['task_count']} "
            f"routes={row['routes']} queries={row['queries']} "
            f"candidate_query={row['candidate_query']} "
            f"splitter_called={row['splitter_called']} rewriter_called={row['rewriter_called']} "
            f"dispatcher_knowledge_request_count={row['dispatcher_knowledge_request_count']} "
            f"split_exact={row['split_exact']} "
            f"unavailable={row['unavailable']} protocol_error={row['protocol_error']} "
            f"splitter_error={row['splitter_error']} rewrite_error={row['rewrite_error']} "
            f"reason={row['reason']}"
        )
        if sleep_s > 0 and row["lane"] == "result":
            time.sleep(sleep_s)
    summarize(set_name, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("holdout", "blind", "both"), default="both")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    try:
        connection = resolve_jev_connection()
    except JevUnavailableError as exc:
        print(f"unavailable={exc}")
        print("b1_closed=yes")
        return 1
    print(f"jev_provider={connection.provider}")
    print(f"jev_endpoint={connection.endpoint}")
    print(f"jev_model={connection.model}")
    judge = RecordingJudge(SystemOneJudge(connection, timeout_s=args.timeout))
    complete_text = build_complete_text()
    multi_gold = load_multi_gold()
    names = ("holdout", "blind") if args.set == "both" else (args.set,)
    for name in names:
        run_set(
            name,
            judge=judge,
            complete_text=complete_text,
            multi_gold=multi_gold,
            sleep_s=args.sleep,
        )
        print("===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
