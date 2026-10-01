"""Offline tests for the B1-Jev candidate router. No live Jev or vLLM calls."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.constants import (  # noqa: E402
    MAX_ROUTER_MEMORY_TOKENS,
    USER_REPLY_CLARIFY,
    USER_REPLY_CONFIRM_REWRITE_TEMPLATE,
    USER_REPLY_ROUTER_PROTOCOL,
    USER_REPLY_ROUTER_UNAVAILABLE,
    USER_REPLY_TASK_LIMIT,
)
from retrieval.b1.dispatcher import (  # noqa: E402
    knowledge_processing_requests,
    router_mode_user_reply,
    system_task_ids,
)
from retrieval.b1.errors import (  # noqa: E402
    EmptyQueryError,
    InvalidQueryError,
    RewriteRejected,
    RouterProtocolError,
    RouterUnavailableError,
)
from retrieval.b1.memory import InMemoryConversationStore  # noqa: E402
from retrieval.b1.pipeline import route_for_b2  # noqa: E402
from retrieval.b1.types import ConversationTurn, QueryRequest  # noqa: E402
from retrieval.b1_jev.jev_client import (  # noqa: E402
    TUZI_JEV_API,
    JevUnavailableError,
    resolve_jev_connection,
    route_label_from_payload,
)
from retrieval.b1_jev.multi_splitter import (  # noqa: E402
    SPLITTER_GUIDED_SCHEMA,
    SPLITTER_SCHEMA,
    USER_REPLY_SPLITTER_FAILED,
    SplitterFailed,
    parse_split_payload,
    split_current_query,
    splitter_messages,
)
from retrieval.b1_jev.questions import (  # noqa: E402
    JEV_MODE_QUESTIONS,
    JEV_QUESTIONS,
    JevProtocolError,
    build_jev_mode_request,
    build_jev_request,
)
from retrieval.b1_jev.rewrite_generator import (  # noqa: E402
    REWRITER_GUIDED_SCHEMA,
    REWRITER_SCHEMA,
    USER_REPLY_REWRITE_FAILED,
    RewriteFailed,
    parse_rewrite_payload,
    rewrite_candidate,
    rewriter_messages,
)
from retrieval.b1_jev.router import JevQueryRouter, qwen_text_complete  # noqa: E402
from retrieval.b1_jev_experimental.questions import (  # noqa: E402
    JEV_MODE_QUESTIONS as EXPERIMENTAL_MODE_QUESTIONS,
)
from retrieval.b1_jev_experimental.questions import (  # noqa: E402
    JEV_QUESTIONS as EXPERIMENTAL_QUESTIONS,
)
from retrieval.tests.helpers import CountingTokenizer  # noqa: E402

CURRENT = "NASA-STD-5006A 焊缝质量验收标准有哪些？"
PACKAGE_ROOT = CODE_ROOT / "retrieval" / "b1_jev"


def _load_diag():
    path = CODE_ROOT / "retrieval" / "eval" / "run_b1_jev_router_diag.py"
    spec = importlib.util.spec_from_file_location("run_b1_jev_router_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load candidate diagnostic")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptedJudge:
    def __init__(
        self,
        route: str = "knowledge",
        mode: str = "direct",
        route_error: BaseException | None = None,
        mode_error: BaseException | None = None,
    ) -> None:
        self.route = route
        self.mode = mode
        self.route_error = route_error
        self.mode_error = mode_error
        self.route_queries: list[str] = []
        self.mode_queries: list[str] = []
        self.histories: list[tuple[ConversationTurn, ...]] = []

    def primary_route(
        self,
        current_query: str,
        history: tuple[ConversationTurn, ...],
    ) -> str:
        self.route_queries.append(current_query)
        self.histories.append(tuple(history))
        if self.route_error is not None:
            raise self.route_error
        return self.route

    def knowledge_mode(
        self,
        current_query: str,
        history: tuple[ConversationTurn, ...],
    ) -> str:
        self.mode_queries.append(current_query)
        if self.mode_error is not None:
            raise self.mode_error
        return self.mode


class CaptureComplete:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls = 0
        self.messages: list[dict[str, str]] | None = None
        self.schema: dict[str, Any] | None = None

    def __call__(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
        self.calls += 1
        self.messages = messages
        self.schema = schema
        if isinstance(self.payload, BaseException):
            raise self.payload
        return json.dumps(self.payload, ensure_ascii=False)


def _request(query: str, request_id: str = "req-1") -> QueryRequest:
    return QueryRequest(
        request_id=request_id,
        conversation_id=request_id,
        query=query,
    )


def _router(
    judge: Any,
    complete: CaptureComplete | None = None,
    *,
    store: InMemoryConversationStore | None = None,
    tokenizer: Any = None,
) -> JevQueryRouter:
    return JevQueryRouter(
        store=store or InMemoryConversationStore(),
        tokenizer=tokenizer or CountingTokenizer(),
        judge=judge,
        complete_text=complete,
    )


def test_questions_match_experimental_v0_wording() -> None:
    assert JEV_QUESTIONS == EXPERIMENTAL_QUESTIONS
    assert JEV_MODE_QUESTIONS == EXPERIMENTAL_MODE_QUESTIONS
    route_request = build_jev_request("q", model="jev-1.13")
    mode_request = build_jev_mode_request("q", model="jev-1.13")
    assert tuple(route_request["questions"]) == ("primary_route",)
    assert tuple(mode_request["questions"]) == ("knowledge_mode",)


def test_tuzi_connection_is_preferred_and_confidence_is_not_a_policy() -> None:
    connection = resolve_jev_connection(
        {
            "TUZI_API_KEY": "tuzi-secret",
            "AI_GATEWAY_API_KEY": "gateway-secret",
            "TYPESAFE_API_KEY": "typesafe-secret",
        }
    )
    assert connection.provider == "tuzi-jev"
    assert connection.endpoint == TUZI_JEV_API
    assert connection.model == "jev-1.13"
    with pytest.raises(JevUnavailableError):
        resolve_jev_connection({})
    assert (
        route_label_from_payload(
            {"answers": {"primary_route": {"choice": "knowledge", "confidence": 0.01}}}
        )
        == "knowledge"
    )
    with pytest.raises(JevProtocolError):
        route_label_from_payload(
            {"answers": {"primary_route": {"choice": "mixed", "confidence": 0.9}}}
        )
    with pytest.raises(JevProtocolError):
        route_label_from_payload({"answers": {"primary_route": {"choice": "knowledge"}}})


def test_candidate_package_does_not_import_the_old_router() -> None:
    forbidden_names = {"QueryRouter", "VLLMRouterClient"}
    for path in PACKAGE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "ROUTER_GUIDED_JSON_SCHEMA" not in text
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in {"retrieval.b1.router", "retrieval.b1.client"}
                assert not module.startswith("retrieval.b1.router")
                assert not module.startswith("retrieval.b1.client")
                for alias in node.names:
                    assert alias.name not in forbidden_names
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_names
    for name in ("multi_splitter.py", "rewrite_generator.py"):
        text = (PACKAGE_ROOT / name).read_text(encoding="utf-8")
        assert "ROUTER_SYSTEM_PROMPT" not in text
        assert "ROUTER_GUIDED_JSON_SCHEMA" not in text
    source = inspect.getsource(qwen_text_complete)
    assert "max_retries=0" in source
    assert "guided_json" in source
    assert "ROUTER_GUIDED_JSON_SCHEMA" not in source
    b1_init = (CODE_ROOT / "retrieval" / "b1" / "__init__.py").read_text(encoding="utf-8")
    root_init = (CODE_ROOT / "retrieval" / "__init__.py").read_text(encoding="utf-8")
    pipeline = (CODE_ROOT / "retrieval" / "b1" / "pipeline.py").read_text(encoding="utf-8")
    assert "build_jev_router" in b1_init
    assert "route_for_b2" in b1_init
    assert "JevQueryRouter" in b1_init
    assert "VLLMRouterClient" not in b1_init
    assert "from retrieval.b1.router import" not in b1_init
    assert "build_jev_router" in root_init
    assert "route_for_b2" in root_init
    assert '"QueryRouter"' not in root_init
    assert '"VLLMRouterClient"' not in root_init
    assert "dispatch_knowledge_tasks" in pipeline
    assert "JevQueryRouter" in pipeline


def test_route_for_b2_sends_knowledge_tasks_and_skips_system() -> None:
    result, processed = route_for_b2(
        _router(ScriptedJudge(route="knowledge", mode="direct")),
        _request(CURRENT),
    )
    assert result.mode == "direct"
    assert len(processed) == 1
    assert processed[0].request_id == "req-1"
    assert processed[0].original_query == CURRENT
    assert processed[0].retrieval_queries == (CURRENT,)

    _, system_processed = route_for_b2(
        _router(ScriptedJudge(route="system")),
        _request("你现在接入了哪些类型的技术文档？"),
    )
    assert system_processed == ()

    split = CaptureComplete(
        {
            "tasks": [
                {"query": "GB/T 1184 未注形位公差怎么规定"},
                {"query": "ISO 2768-1 线性尺寸公差怎么规定"},
            ]
        }
    )
    multi, multi_processed = route_for_b2(
        _router(ScriptedJudge(route="knowledge", mode="multi"), split),
        _request("分别说明 GB/T 1184 未注形位公差，以及 ISO 2768-1 线性尺寸公差。"),
    )
    assert multi.mode == "direct"
    assert len(multi_processed) == 2
    assert multi_processed[0].retrieval_queries[0] == multi.tasks[0].query
    assert multi_processed[1].retrieval_queries[0] == multi.tasks[1].query


def test_direct_system_unsupported_and_fixed_modes_do_not_call_qwen() -> None:
    cases = (
        ("knowledge", "direct", "direct", "knowledge", CURRENT),
        ("system", "direct", "direct", "system", "你现在接入了哪些类型的技术文档？"),
        ("unsupported", "direct", "direct", "unsupported", "把 ERP 里的实时库存导出来。"),
        ("knowledge", "task_limit", "task_limit", None, CURRENT),
        ("knowledge", "clarify", "clarify", None, CURRENT),
    )
    for route, mode, result_mode, task_route, query in cases:
        complete = CaptureComplete({"tasks": []})
        judge = ScriptedJudge(route=route, mode=mode)
        result = _router(judge, complete).route(_request(query))
        assert complete.calls == 0
        assert result.mode == result_mode
        assert result.candidate_query is None
        if route == "knowledge":
            assert judge.mode_queries == [query.strip()]
        else:
            assert judge.mode_queries == []
        if result_mode == "direct":
            assert len(result.tasks) == 1
            assert result.tasks[0].route == task_route
            assert result.tasks[0].query == query
        else:
            assert result.tasks == ()


def test_clarify_assembly_is_the_existing_empty_result() -> None:
    result = _router(ScriptedJudge(route="knowledge", mode="clarify")).route(_request(CURRENT))
    assert result.mode == "clarify"
    assert result.tasks == ()
    assert result.candidate_query is None
    assert router_mode_user_reply(result) == USER_REPLY_CLARIFY
    assert knowledge_processing_requests("req-1", result) == ()


def test_task_limit_and_system_use_existing_dispatcher_exits() -> None:
    limit = _router(ScriptedJudge(route="knowledge", mode="task_limit")).route(_request(CURRENT))
    assert limit.mode == "task_limit"
    assert limit.tasks == ()
    assert router_mode_user_reply(limit) == USER_REPLY_TASK_LIMIT

    system = _router(ScriptedJudge(route="system")).route(_request("这个助手能做什么？"))
    assert system_task_ids("req-1", system)[0][1].query == "这个助手能做什么？"
    assert knowledge_processing_requests("req-1", system) == ()
    assert router_mode_user_reply(system) is None


def test_multi_builds_direct_knowledge_tasks_without_history() -> None:
    store = InMemoryConversationStore()
    store.set_active_memory(
        "req-1",
        (ConversationTurn("RAW_ONLY_MARKER", "UNIQUE_CANONICAL", "UNIQUE_ANSWER"),),
    )
    complete = CaptureComplete(
        {"tasks": [{"query": "第一个独立问题"}, {"query": "第二个独立问题"}]}
    )
    result = _router(
        ScriptedJudge(route="knowledge", mode="multi"),
        complete,
        store=store,
    ).route(_request("第一个独立问题，以及第二个独立问题。"))
    assert complete.calls == 1
    assert complete.schema == SPLITTER_GUIDED_SCHEMA
    assert "mode" not in complete.schema["properties"]
    assert "candidate_query" not in complete.schema["properties"]
    guided = json.dumps(complete.schema)
    assert "minItems" not in guided
    assert "maxItems" not in guided
    assert "minLength" not in guided
    assert SPLITTER_SCHEMA["properties"]["tasks"]["minItems"] == 2
    assert SPLITTER_SCHEMA["properties"]["tasks"]["maxItems"] == 8
    rendered = "\n".join(message["content"] for message in complete.messages or [])
    assert "UNIQUE_CANONICAL" not in rendered
    assert "RAW_ONLY_MARKER" not in rendered
    assert result.mode == "direct"
    assert result.candidate_query is None
    assert [task.route for task in result.tasks] == ["knowledge", "knowledge"]
    assert [task.query for task in result.tasks] == ["第一个独立问题", "第二个独立问题"]
    requests = knowledge_processing_requests("req-1", result)
    assert [item.query for item in requests] == ["第一个独立问题", "第二个独立问题"]


def test_splitter_rejects_illegal_payloads_without_repair() -> None:
    with pytest.raises(SplitterFailed) as one:
        parse_split_payload(json.dumps({"tasks": [{"query": "只有一条"}]}))
    assert one.value.user_reply == USER_REPLY_SPLITTER_FAILED
    with pytest.raises(SplitterFailed):
        parse_split_payload(
            json.dumps({"tasks": [{"query": f"任务{index}"} for index in range(9)]})
        )
    with pytest.raises(SplitterFailed):
        parse_split_payload(
            json.dumps({"tasks": [{"query": "相同问题"}, {"query": " 相同问题 "}]})
        )
    with pytest.raises(SplitterFailed):
        parse_split_payload(
            json.dumps({"tasks": [{"query": "甲"}, {"query": "乙", "route": "knowledge"}]})
        )
    with pytest.raises(SplitterFailed):
        parse_split_payload("not-json")
    with pytest.raises(SplitterFailed) as down:
        split_current_query(CURRENT, CaptureComplete(RuntimeError("down")))
    assert down.value.user_reply == USER_REPLY_SPLITTER_FAILED
    assert not isinstance(down.value, RouterUnavailableError)


def test_confirm_rewrite_uses_memory_and_existing_prompt() -> None:
    history = (ConversationTurn("RAW_ONLY_MARKER", "UNIQUE_CANONICAL", "UNIQUE_ANSWER"),)
    complete = CaptureComplete({"candidate_query": "NASA-STD-5006A 对钢材焊接有什么要求？"})
    store = InMemoryConversationStore()
    store.set_active_memory("req-1", history)
    result = _router(
        ScriptedJudge(route="knowledge", mode="confirm_rewrite"),
        complete,
        store=store,
    ).route(_request("那钢呢？"))
    rendered = "\n".join(message["content"] for message in complete.messages or [])
    assert "UNIQUE_CANONICAL" in rendered
    assert "UNIQUE_ANSWER" in rendered
    assert "RAW_ONLY_MARKER" not in rendered
    assert complete.schema == REWRITER_GUIDED_SCHEMA
    assert set(complete.schema["properties"]) == {"candidate_query"}
    assert "minLength" not in json.dumps(complete.schema)
    assert REWRITER_SCHEMA["properties"]["candidate_query"]["minLength"] == 1
    assert result.mode == "confirm_rewrite"
    assert result.tasks == ()
    assert result.candidate_query == "NASA-STD-5006A 对钢材焊接有什么要求？"
    assert router_mode_user_reply(result) == USER_REPLY_CONFIRM_REWRITE_TEMPLATE.format(
        candidate_query=result.candidate_query
    )
    assert "canonical_user_query" in rewriter_messages("那钢呢？", history)[1]["content"]


def test_rewrite_failure_has_its_own_reply() -> None:
    with pytest.raises(RewriteFailed) as empty:
        parse_rewrite_payload(json.dumps({"candidate_query": "  "}))
    assert empty.value.user_reply == USER_REPLY_REWRITE_FAILED
    with pytest.raises(RewriteFailed):
        parse_rewrite_payload(
            json.dumps({"candidate_query": "完整问题", "route": "knowledge"})
        )
    with pytest.raises(RewriteFailed) as down:
        rewrite_candidate(CURRENT, (), CaptureComplete(RuntimeError("down")))
    assert down.value.user_reply == USER_REPLY_REWRITE_FAILED
    with pytest.raises(RewriteFailed) as unconfigured:
        _router(ScriptedJudge(route="knowledge", mode="confirm_rewrite")).route(
            _request("那钢呢？")
        )
    assert unconfigured.value.user_reply == USER_REPLY_REWRITE_FAILED


def test_jev_failures_use_router_replies_and_do_not_call_qwen() -> None:
    complete = CaptureComplete({"tasks": []})
    with pytest.raises(RouterUnavailableError) as unavailable:
        _router(
            ScriptedJudge(route_error=JevUnavailableError("timeout")),
            complete,
        ).route(_request(CURRENT))
    assert unavailable.value.user_reply == USER_REPLY_ROUTER_UNAVAILABLE
    with pytest.raises(RouterProtocolError) as protocol:
        _router(
            ScriptedJudge(route_error=JevProtocolError("bad choice")),
            complete,
        ).route(_request(CURRENT))
    assert protocol.value.user_reply == USER_REPLY_ROUTER_PROTOCOL
    with pytest.raises(RouterProtocolError):
        _router(ScriptedJudge(route="knowledge", mode="mixed"), complete).route(
            _request(CURRENT)
        )
    assert complete.calls == 0


def test_confirmation_reenters_the_candidate_router() -> None:
    candidate = "NASA-STD-5006A 对钢材焊接有什么要求？"
    judge = ScriptedJudge(route="knowledge", mode="direct")
    confirmed = _router(judge).route(_request("是"), pending_rewrite=candidate)
    assert judge.route_queries == [candidate]
    assert confirmed.tasks[0].query == candidate

    rejected = ScriptedJudge()
    with pytest.raises(RewriteRejected):
        _router(rejected).route(_request("不"), pending_rewrite=candidate)
    assert rejected.route_queries == []

    fresh = ScriptedJudge(route="system")
    _router(fresh).route(_request("这个助手能做什么？"), pending_rewrite=candidate)
    assert fresh.route_queries == ["这个助手能做什么？"]


def test_b10_rejects_before_jev() -> None:
    judge = ScriptedJudge()
    with pytest.raises(EmptyQueryError):
        _router(judge).route(_request("   "))
    with pytest.raises(InvalidQueryError):
        _router(judge).route(_request("\u0001NASA-STD-5006A"))
    assert judge.route_queries == []


def test_memory_is_trimmed_with_the_existing_fifo_budget() -> None:
    class AlwaysOver(CountingTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            tokenize: bool = True,
            add_generation_prompt: bool = True,
            **kwargs: Any,
        ) -> list[int] | str:
            if tokenize:
                return [0] * (MAX_ROUTER_MEMORY_TOKENS + 1)
            return "x"

    store = InMemoryConversationStore()
    store.set_active_memory(
        "req-1",
        (ConversationTurn("old", "OLD_CANON", "old answer"),),
    )
    judge = ScriptedJudge(route="system")
    _router(judge, store=store, tokenizer=AlwaysOver()).route(_request("你能做什么？"))
    assert judge.histories == [()]
    assert store.get_active_memory("req-1") == ()


def test_oversized_jev_body_is_protocol_without_a_second_memory_policy() -> None:
    class ZeroTokenizer(CountingTokenizer):
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            tokenize: bool = True,
            add_generation_prompt: bool = True,
            **kwargs: Any,
        ) -> list[int] | str:
            if tokenize:
                return []
            return ""

    store = InMemoryConversationStore()
    store.set_active_memory("req-1", (ConversationTurn("q", "q", "x" * 40000),))

    class WireJudge:
        def primary_route(
            self,
            current_query: str,
            history: tuple[ConversationTurn, ...],
        ) -> str:
            build_jev_request(current_query, history, model="jev-1.13")
            return "knowledge"

        def knowledge_mode(
            self,
            current_query: str,
            history: tuple[ConversationTurn, ...],
        ) -> str:
            raise AssertionError("knowledge_mode")

    with pytest.raises(RouterProtocolError) as exc:
        _router(WireJudge(), store=store, tokenizer=ZeroTokenizer()).route(_request(CURRENT))
    assert exc.value.user_reply == USER_REPLY_ROUTER_PROTOCOL


def test_splitter_messages_keep_examples_off_the_live_query() -> None:
    messages = splitter_messages(CURRENT)
    rendered = "\n".join(message["content"] for message in messages)
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == f"当前 Query：\n{CURRENT}"
    assert "GB/T 5094.1" in rendered
    assert "分别说明 ISO 2768-1" in rendered
    assert "canonical_user_query" not in rendered


def test_exposed_clarify_and_mixed_stay_unscored() -> None:
    diag = _load_diag()
    assert diag.diagnostic_status() == {"b1_closed": "yes", "main_path": "connected"}
    gold = diag.load_multi_gold()
    for set_name in ("holdout", "blind"):
        cases = diag.load_jsonl(diag.SETS[set_name])
        for case in cases:
            bucket = case["bucket"]
            if bucket in {"clarify", "mixed"}:
                assert diag.classify_case(case) == "skip"
                assert diag.gold_primary_route(case) is None
                assert diag.gold_knowledge_mode(case) is None
                row = diag.score_result(
                    case,
                    {
                        "primary_route": "knowledge",
                        "knowledge_mode": "clarify",
                        "result_mode": "clarify",
                        "task_count": 0,
                        "routes": [],
                        "queries": [],
                        "candidate_query": None,
                        "error_type": None,
                    },
                    gold,
                )
                assert row["lane"] == "skip"
                assert row["pass"] is None
                assert row["gold_primary_route"] is None
                called = {"route": False}

                class BoomRouter:
                    def route(self, request: QueryRequest) -> None:
                        called["route"] = True
                        raise AssertionError("skip case entered the router")

                skipped = diag.run_case(
                    case,
                    router=BoomRouter(),
                    judge=object(),
                    multi_gold=gold,
                )
                assert called["route"] is False
                assert skipped["lane"] == "skip"
            elif bucket == "multi_2_to_8":
                assert case["id"] in gold
                assert len(gold[case["id"]]) == len(case["expected"]["routes"])
                assert 2 <= len(gold[case["id"]]) <= 8
            else:
                assert case["id"] not in gold


def test_diagnostic_scores_full_results_without_inventing_clarify_route() -> None:
    diag = _load_diag()
    holdout = diag.load_jsonl(diag.SETS["holdout"])
    gold = diag.load_multi_gold()
    direct = next(case for case in holdout if case["id"] == "h-001")
    direct_row = diag.score_result(
        direct,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "direct",
            "result_mode": "direct",
            "task_count": 1,
            "routes": ["knowledge"],
            "queries": [direct["query"]],
            "candidate_query": None,
            "error_type": None,
        },
        gold,
    )
    assert direct_row["pass"] is True
    multi = next(case for case in holdout if case["id"] == "h-039")
    multi_row = diag.score_result(
        multi,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "multi",
            "result_mode": "direct",
            "task_count": len(gold["h-039"]),
            "routes": ["knowledge"] * len(gold["h-039"]),
            "queries": gold["h-039"],
            "candidate_query": None,
            "error_type": None,
        },
        gold,
    )
    assert multi_row["pass"] is True
    missed = diag.score_result(
        multi,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "multi",
            "result_mode": "direct",
            "task_count": 2,
            "routes": ["knowledge", "knowledge"],
            "queries": ["不是实验 gold", "也不是"],
            "candidate_query": None,
            "error_type": None,
        },
        gold,
    )
    assert missed["pass"] is False
    limit = next(case for case in holdout if case["bucket"] == "task_limit_gt8")
    limit_row = diag.score_result(
        limit,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "task_limit",
            "result_mode": "task_limit",
            "task_count": 0,
            "routes": [],
            "queries": [],
            "candidate_query": None,
            "error_type": None,
        },
        gold,
    )
    assert limit_row["pass"] is True
    assert diag.gold_primary_route(limit) == "knowledge"
    rewrite = next(case for case in holdout if case["id"] == "h-051")
    rewrite_row = diag.score_result(
        rewrite,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "confirm_rewrite",
            "result_mode": "confirm_rewrite",
            "task_count": 0,
            "routes": [],
            "queries": [],
            "candidate_query": rewrite["expected"]["candidate_query"],
            "error_type": None,
        },
        gold,
    )
    assert rewrite_row["pass"] is True
    split_fail = diag.score_result(
        multi,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "multi",
            "result_mode": None,
            "task_count": None,
            "routes": None,
            "queries": None,
            "candidate_query": None,
            "error_type": "splitter_error",
        },
        gold,
    )
    assert split_fail["pass"] is False
    assert split_fail["splitter_error"] is True
    assert split_fail["protocol_error"] is False


def test_diagnostic_control_does_not_call_jev() -> None:
    diag = _load_diag()
    holdout = diag.load_jsonl(diag.SETS["holdout"])
    control = next(case for case in holdout if case["id"] == "h-061")

    class Inner:
        def primary_route(self, current_query: str, history: tuple[Any, ...]) -> str:
            raise AssertionError("control called jev")

        def knowledge_mode(self, current_query: str, history: tuple[Any, ...]) -> str:
            raise AssertionError("control called jev")

    judge = diag.RecordingJudge(Inner())
    router = JevQueryRouter(
        store=InMemoryConversationStore(),
        tokenizer=CountingTokenizer(),
        judge=diag._BoundJudge(judge),
        complete_text=None,
    )
    row = diag.run_case(control, router=router, judge=judge, multi_gold={})
    assert row["lane"] == "control"
    assert row["pass"] is True
    assert judge.primary_calls == 0
    assert judge.mode_calls == 0


def test_multi_calls_splitter_once() -> None:
    complete = CaptureComplete(
        {"tasks": [{"query": "第一个独立问题"}, {"query": "第二个独立问题"}]}
    )
    result = _router(
        ScriptedJudge(route="knowledge", mode="multi"),
        complete,
    ).route(_request("第一个独立问题，以及第二个独立问题。"))
    assert complete.calls == 1
    assert complete.schema is not None
    assert complete.schema["required"] == ["tasks"]
    assert "candidate_query" not in complete.schema["properties"]
    assert result.mode == "direct"
    assert len(result.tasks) == 2


def test_rewrite_calls_rewriter_once() -> None:
    complete = CaptureComplete({"candidate_query": "NASA-STD-5006A 对钢材焊接有什么要求？"})
    result = _router(
        ScriptedJudge(route="knowledge", mode="confirm_rewrite"),
        complete,
    ).route(_request("那钢呢？"))
    assert complete.calls == 1
    assert complete.schema is not None
    assert complete.schema["required"] == ["candidate_query"]
    assert "tasks" not in complete.schema["properties"]
    assert result.mode == "confirm_rewrite"
    assert result.tasks == ()


def test_illegal_split_and_qwen_down_raise_splitter_error() -> None:
    illegal = CaptureComplete({"tasks": [{"query": "甲\u0001"}, {"query": "乙"}]})
    with pytest.raises(SplitterFailed) as bad_query:
        _router(ScriptedJudge(route="knowledge", mode="multi"), illegal).route(
            _request(CURRENT)
        )
    assert bad_query.value.error_type == "splitter_error"
    down = CaptureComplete(RouterUnavailableError("down"))
    with pytest.raises(SplitterFailed) as unavailable:
        _router(ScriptedJudge(route="knowledge", mode="multi"), down).route(_request(CURRENT))
    assert unavailable.value.error_type == "splitter_error"


def test_illegal_candidate_raises_rewrite_error() -> None:
    complete = CaptureComplete({"candidate_query": "钢\u0001材"})
    with pytest.raises(RewriteFailed) as exc:
        _router(
            ScriptedJudge(route="knowledge", mode="confirm_rewrite"),
            complete,
        ).route(_request("那钢呢？"))
    assert exc.value.error_type == "rewrite_error"


def test_ba051_is_ambiguous_and_ba044_prefix_still_passes() -> None:
    diag = _load_diag()
    blind = diag.load_jsonl(diag.SETS["blind"])
    gold = diag.load_multi_gold()
    ambiguous = next(case for case in blind if case["id"] == "ba-051")
    assert diag.classify_case(ambiguous) == "ambiguous"
    ambiguous_row = diag.score_result(
        ambiguous,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "clarify",
            "result_mode": "clarify",
            "task_count": 0,
            "routes": [],
            "queries": [],
            "candidate_query": None,
            "error_type": None,
        },
        gold,
    )
    assert ambiguous_row["lane"] == "ambiguous"
    assert ambiguous_row["pass"] is None
    assert ambiguous_row["reason"] == "ambiguous"
    prefixed = next(case for case in blind if case["id"] == "ba-044")
    queries = [f"说明 {item}" for item in gold["ba-044"]]
    prefixed_row = diag.score_result(
        prefixed,
        {
            "primary_route": "knowledge",
            "knowledge_mode": "multi",
            "result_mode": "direct",
            "task_count": len(queries),
            "routes": ["knowledge"] * len(queries),
            "queries": queries,
            "candidate_query": None,
            "error_type": None,
            "splitter_called": True,
            "rewriter_called": False,
            "dispatcher_knowledge_request_count": len(queries),
        },
        gold,
    )
    assert prefixed_row["pass"] is True
    assert prefixed_row["split_exact"] is False
    assert prefixed_row["reason"] == "split_semantic_pass"
    from retrieval.b1.types import RoutedTask, RouterResult

    routed = RouterResult(
        mode="direct",
        tasks=(
            RoutedTask(route="knowledge", query="甲"),
            RoutedTask(route="knowledge", query="乙"),
            RoutedTask(route="knowledge", query="丙"),
        ),
        candidate_query=None,
    )
    assert diag._dispatcher_count("req-1", routed) == 3
