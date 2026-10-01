"""B1-Jev experimental route classifier. Does not call TypeSafe or vLLM."""

from __future__ import annotations

import inspect

from retrieval.b1.prompt import ROUTER_SYSTEM_PROMPT
from retrieval.b1.types import ConversationTurn
from retrieval.b1_jev_experimental.client import (
    answers_from_api_payload,
    answers_from_mode_payload,
    resolve_jev_connection,
)
from retrieval.b1_jev_experimental.constants import (
    EXPERIMENTAL_CONTRACT_ID,
    JEV_GATEWAY_API,
    JEV_GATEWAY_MODEL,
    JEV_MODE_QUESTION_KEYS,
    JEV_QUESTION_KEYS,
    KNOWLEDGE_MODE_KEYS,
    ROUTE_KEYS,
    TUZI_DEFAULT_MODEL,
    TUZI_JEV_API,
    TUZI_JEV_MODEL,
)
from retrieval.b1_jev_experimental.diag import (
    gold_knowledge_mode,
    gold_route,
    mode_lane,
    route_lane,
    score_control,
    score_mode,
    score_route,
)
from retrieval.b1_jev_experimental.errors import JevProtocolError, JevUnavailableError
from retrieval.b1_jev_experimental.gliclass_adapter import (
    answers_from_gliclass_mode_scores,
    answers_from_gliclass_scores,
    resolve_gliclass_model_path,
    serialize_classifier_text,
)
from retrieval.b1_jev_experimental.policy import parse_jev_answers, parse_jev_mode_answer
from retrieval.b1_jev_experimental.questions import (
    JEV_MODE_QUESTIONS,
    JEV_QUESTIONS,
    build_jev_mode_request,
    build_jev_request,
    validate_jev_wire_request,
)
from retrieval.b1_jev_experimental.tuzi_adapter import (
    answers_from_tuzi_content,
    answers_from_tuzi_payload,
    build_tuzi_messages,
    chat_completions_url,
    resolve_tuzi_connection,
)
from retrieval.b1_jev_experimental.types import JevAnswers


def _answers(route: str = "knowledge", confidence: float = 0.9) -> JevAnswers:
    return parse_jev_answers(
        {"primary_route": route, "primary_route_confidence": confidence}
    )


HISTORY = (
    ConversationTurn(
        raw_user_query="NASA-STD-5006A 对铝合金焊接有什么要求？",
        canonical_user_query="NASA-STD-5006A 对铝合金焊接有什么要求？",
        assistant_response="...",
    ),
)


def test_questions_are_a_single_route_choice() -> None:
    assert tuple(JEV_QUESTIONS) == JEV_QUESTION_KEYS == ("primary_route",)
    spec = JEV_QUESTIONS["primary_route"]
    assert spec["type"] == "choice"
    assert tuple(spec["criteria"]) == ROUTE_KEYS
    assert "self_contained" not in JEV_QUESTIONS
    assert "task_count" not in JEV_QUESTIONS
    assert "knowledge_mode" not in JEV_QUESTIONS


def test_mode_questions_are_five_way() -> None:
    assert tuple(JEV_MODE_QUESTIONS) == JEV_MODE_QUESTION_KEYS == ("knowledge_mode",)
    spec = JEV_MODE_QUESTIONS["knowledge_mode"]
    assert spec["type"] == "choice"
    assert tuple(spec["criteria"]) == KNOWLEDGE_MODE_KEYS


def test_questions_embed_the_qwen_router_prompt() -> None:
    policy = ROUTER_SYSTEM_PROMPT.strip()
    route = str(JEV_QUESTIONS["primary_route"]["instructions"])
    mode = str(JEV_MODE_QUESTIONS["knowledge_mode"]["instructions"])
    for text in (route, mode):
        assert policy in text
        assert "钛呢？" in text
        assert "按那个处理一下" in text
        assert "不要输出说明书里的 JSON" in text
        assert "不要生成 candidate_query" in text
    assert "不要用 mode" in route
    assert "五选一" in mode


def test_request_state_contains_query_and_history_only() -> None:
    request = build_jev_request("那钢呢？", HISTORY)
    assert list(request) == ["model", "state", "questions"]
    assert request["state"]["current_query"] == "那钢呢？"
    assert request["state"]["history"][0]["raw_user_query"].startswith("NASA-STD-5006A")
    assert request["questions"] == JEV_QUESTIONS
    validate_jev_wire_request(request)


def test_jev_request_stays_under_gateway_body_limit() -> None:
    long_query = "请分别回答：" + "；".join(f"问题{index} NASA-STD-5006A" for index in range(1, 12))
    request = build_jev_request(long_query, HISTORY, model="jev-1.13")
    assert request["model"] == "jev-1.13"
    mode_request = build_jev_mode_request("那钢呢？", HISTORY, model="jev-1.13")
    assert tuple(mode_request["questions"]) == ("knowledge_mode",)
    validate_jev_wire_request(mode_request)


def test_wire_request_rejects_chat_completions_shape() -> None:
    try:
        validate_jev_wire_request(
            {
                "model": "jev-1.13",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
    except JevProtocolError:
        return
    raise AssertionError("expected JevProtocolError")


def test_parse_rejects_extra_mode_fields() -> None:
    answers = parse_jev_answers(
        {
            "primary_route": "system",
            "primary_route_confidence": 0.7,
            "self_contained": 0.1,
            "task_count": "2",
        }
    )
    assert answers.primary_route == "system"
    assert not hasattr(answers, "self_contained")
    assert not hasattr(answers, "task_count")


def test_illegal_route_is_protocol_error() -> None:
    try:
        parse_jev_answers({"primary_route": "clarify"})
    except JevProtocolError:
        return
    raise AssertionError("expected JevProtocolError")


def test_illegal_knowledge_mode_is_protocol_error() -> None:
    try:
        parse_jev_mode_answer({"knowledge_mode": "knowledge"})
    except JevProtocolError:
        return
    raise AssertionError("expected JevProtocolError")


def test_api_payload_parser() -> None:
    sample = {
        "primary_route": {
            "type": "choice",
            "choice": "system",
            "confidence": 0.91,
        }
    }
    answers = answers_from_api_payload({"answers": sample})
    assert answers.primary_route == "system"
    assert answers.primary_route_confidence == 0.91
    wrapped = answers_from_api_payload(
        {"code": 0, "message": "ok", "data": {"answers": sample}}
    )
    assert wrapped.primary_route == "system"


def test_mode_payload_parser() -> None:
    sample = {
        "knowledge_mode": {
            "type": "choice",
            "choice": "confirm_rewrite",
            "confidence": 0.84,
        }
    }
    answers = answers_from_mode_payload({"answers": sample})
    assert answers.knowledge_mode == "confirm_rewrite"
    assert answers.knowledge_mode_confidence == 0.84


def test_community_error_code_is_unavailable() -> None:
    try:
        answers_from_api_payload({"code": 1, "message": "quota exceeded"})
    except JevUnavailableError as exc:
        assert "quota exceeded" in str(exc)
    else:
        raise AssertionError("expected JevUnavailableError")


def test_gliclass_text_includes_query_and_history() -> None:
    text = serialize_classifier_text("那钢呢？", HISTORY)
    assert "那钢呢？" in text
    assert "NASA-STD-5006A" in text
    assert "empty" not in text


def test_gliclass_scores_map_to_answers() -> None:
    answers = answers_from_gliclass_scores(
        primary_route=[
            {"label": "knowledge", "score": 0.6},
            {"label": "system", "score": 0.3},
            {"label": "unsupported", "score": 0.1},
        ],
    )
    assert answers.primary_route == "knowledge"
    assert 0.59 < answers.primary_route_confidence < 0.61
    mode = answers_from_gliclass_mode_scores(
        knowledge_mode=[
            {"label": "direct", "score": 0.1},
            {"label": "multi", "score": 0.7},
            {"label": "task_limit", "score": 0.05},
            {"label": "confirm_rewrite", "score": 0.1},
            {"label": "clarify", "score": 0.05},
        ],
    )
    assert mode.knowledge_mode == "multi"


def test_tuzi_jev_is_preferred_when_tuzi_key_is_set() -> None:
    connection = resolve_jev_connection(
        {
            "TUZI_API_KEY": "tuzi-secret",
            "AI_GATEWAY_API_KEY": "gateway-secret",
            "TYPESAFE_API_KEY": "typesafe-secret",
        }
    )
    assert connection.provider == "tuzi-jev"
    assert connection.endpoint == TUZI_JEV_API
    assert connection.model == TUZI_JEV_MODEL
    request = build_jev_request("q", model=connection.model)
    assert request["model"] == "jev-1.13"
    assert tuple(request["questions"]) == ("primary_route",)


def test_vercel_gateway_is_preferred_when_ai_gateway_key_is_set() -> None:
    connection = resolve_jev_connection(
        {
            "AI_GATEWAY_API_KEY": "gateway-secret",
            "TYPESAFE_API_KEY": "typesafe-secret",
        }
    )
    assert connection.provider == "vercel-ai-gateway"
    assert connection.endpoint == JEV_GATEWAY_API
    assert connection.model == JEV_GATEWAY_MODEL
    request = build_jev_request("q", model=connection.model)
    assert request["model"] == "typesafe-ai/jev"


def test_typesafe_direct_when_only_typesafe_key_is_set() -> None:
    connection = resolve_jev_connection({"TYPESAFE_API_KEY": "typesafe-secret"})
    assert connection.provider == "typesafe"
    assert connection.model == "jev-latest"


def test_gliclass_default_path_has_weights_on_this_machine() -> None:
    path = resolve_gliclass_model_path()
    assert (path / "model.safetensors").is_file()


def test_tuzi_chat_url_does_not_double_v1() -> None:
    assert (
        chat_completions_url("https://api.tu-zi.com")
        == "https://api.tu-zi.com/v1/chat/completions"
    )
    assert (
        chat_completions_url("https://api.tu-zi.com/v1")
        == "https://api.tu-zi.com/v1/chat/completions"
    )


def test_tuzi_missing_key_is_unavailable() -> None:
    try:
        resolve_tuzi_connection({})
    except JevUnavailableError as exc:
        assert "TUZI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected JevUnavailableError")


def test_tuzi_connection_uses_key_and_optional_model() -> None:
    connection = resolve_tuzi_connection({"TUZI_API_KEY": "tuzi-secret"})
    assert connection.provider == "tuzi-openai"
    assert connection.endpoint == "https://api.tu-zi.com/v1/chat/completions"
    assert connection.model == TUZI_DEFAULT_MODEL
    named = resolve_tuzi_connection(
        {"TUZI_API_KEY": "tuzi-secret", "TUZI_MODEL": "gpt-4o"}
    )
    assert named.model == "gpt-4o"


def test_tuzi_messages_reuse_jev_questions() -> None:
    messages = build_tuzi_messages("那钢呢？", HISTORY)
    assert messages[0]["role"] == "system"
    body = messages[1]["content"]
    assert "那钢呢？" in body
    assert "NASA-STD-5006A" in body
    assert "primary_route" in body
    assert "self_contained" not in body
    assert "task_count" not in body


def test_tuzi_payload_parser_strips_fences() -> None:
    answers = answers_from_tuzi_content(
        """```json
{"primary_route": "system", "primary_route_confidence": 0.77}
```"""
    )
    assert answers.primary_route == "system"
    assert answers.primary_route_confidence == 0.77


def test_tuzi_choices_payload() -> None:
    answers = answers_from_tuzi_payload(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"primary_route": "knowledge", '
                            '"primary_route_confidence": 0.6}'
                        ),
                    }
                }
            ]
        }
    )
    assert answers.primary_route == "knowledge"


def test_gold_route_uses_unique_expected_routes() -> None:
    case = {
        "id": "t-001",
        "bucket": "zh_en",
        "query": "How does NASA-STD-4003A define class S bonding?",
        "history": [],
        "expected": {
            "mode": "direct",
            "routes": ["knowledge"],
        },
    }
    assert route_lane(case) == "route"
    assert gold_route(case) == "knowledge"
    scored = score_route(case, _answers("knowledge"))
    assert scored["mark"] == "PASS"


def test_multi_knowledge_is_still_knowledge() -> None:
    case = {
        "id": "t-002",
        "bucket": "multi_2_to_8",
        "query": "a 以及 b",
        "history": [],
        "expected": {
            "mode": "direct",
            "routes": ["knowledge", "knowledge"],
        },
    }
    assert gold_route(case) == "knowledge"
    scored = score_route(case, _answers("knowledge"))
    assert scored["mark"] == "PASS"


def test_mixed_and_clarify_are_skipped() -> None:
    mixed = {
        "id": "t-003",
        "bucket": "mixed",
        "query": "查标准并登录 MES",
        "history": [],
        "expected": {"mode": "direct", "routes": ["knowledge", "unsupported"]},
    }
    clarify = {
        "id": "t-004",
        "bucket": "clarify",
        "query": "那个呢？",
        "history": [],
        "expected": {"mode": "clarify", "routes": []},
    }
    assert route_lane(mixed) == "skip"
    assert route_lane(clarify) == "skip"
    assert gold_route(mixed) is None
    assert gold_route(clarify) is None
    assert score_route(mixed, _answers("knowledge"))["mark"] == "SKIP"
    assert score_route(clarify, _answers("system"))["mark"] == "SKIP"


def test_confirm_rewrite_and_task_limit_gold_as_knowledge() -> None:
    confirm = {
        "id": "t-005",
        "bucket": "confirm_rewrite",
        "query": "那钢呢？",
        "history": [],
        "expected": {"mode": "confirm_rewrite", "routes": []},
    }
    limit = {
        "id": "t-006",
        "bucket": "task_limit_gt8",
        "query": "请分别回答 9 个问题",
        "history": [],
        "expected": {"mode": "task_limit", "routes": []},
    }
    assert gold_route(confirm) == "knowledge"
    assert gold_route(limit) == "knowledge"
    assert score_route(confirm, _answers("knowledge"))["mark"] == "PASS"
    assert score_route(limit, _answers("unsupported"))["mark"] == "FAIL"


def test_gold_knowledge_mode_mapping() -> None:
    singles = {
        "id": "t-010",
        "bucket": "knowledge_single",
        "query": "q",
        "history": [],
        "expected": {"mode": "direct", "routes": ["knowledge"]},
    }
    multi = {
        "id": "t-011",
        "bucket": "multi_2_to_8",
        "query": "a 以及 b",
        "history": [],
        "expected": {"mode": "direct", "routes": ["knowledge", "knowledge"]},
    }
    confirm = {
        "id": "t-012",
        "bucket": "confirm_rewrite",
        "query": "那钢呢？",
        "history": [],
        "expected": {"mode": "confirm_rewrite", "routes": []},
    }
    limit = {
        "id": "t-013",
        "bucket": "task_limit_gt8",
        "query": "9 问",
        "history": [],
        "expected": {"mode": "task_limit", "routes": []},
    }
    system = {
        "id": "t-014",
        "bucket": "system",
        "query": "你能做什么",
        "history": [],
        "expected": {"mode": "direct", "routes": ["system"]},
    }
    assert gold_knowledge_mode(singles) == "direct"
    assert gold_knowledge_mode(multi) == "multi"
    assert gold_knowledge_mode(confirm) == "confirm_rewrite"
    assert gold_knowledge_mode(limit) == "task_limit"
    assert gold_knowledge_mode(system) is None
    assert mode_lane(multi) == "mode"
    assert mode_lane(system) == "skip"
    assert (
        score_mode(
            multi,
            parse_jev_mode_answer({"knowledge_mode": "multi"}),
            asked=True,
        )["mode_mark"]
        == "PASS"
    )
    assert (
        score_mode(
            confirm,
            parse_jev_mode_answer({"knowledge_mode": "direct"}),
            asked=True,
        )["mode_mark"]
        == "FAIL"
    )
    skipped = score_mode(confirm, None, asked=False)
    assert skipped["mode_mark"] == "SKIP"
    assert skipped["mode_reason"] == "route_not_knowledge"


def test_control_uses_b10_validation() -> None:
    case = {
        "id": "t-007",
        "bucket": "control",
        "query": "   ",
        "history": [],
        "expected": {"error_type": "empty_query"},
    }
    assert route_lane(case) == "control"
    scored = score_control(case)
    assert scored["mark"] == "PASS"


def test_experimental_package_does_not_export_mode_composition() -> None:
    import retrieval.b1_jev_experimental as pkg

    assert not hasattr(pkg, "compose_policy")
    assert not hasattr(pkg, "assemble_phase1")
    assert not hasattr(pkg, "PolicyThresholds")
    assert pkg.EXPERIMENTAL_CONTRACT_ID == EXPERIMENTAL_CONTRACT_ID
    assert pkg.JEV_MODE_QUESTION_KEYS == ("knowledge_mode",)


def test_production_router_does_not_import_experimental_package() -> None:
    from retrieval import b1 as b1_pkg
    from retrieval.b1 import pipeline as pipeline_mod
    from retrieval.b1_jev import router as router_mod

    assert "b1_jev_experimental" not in inspect.getsource(router_mod)
    assert "b1_jev_experimental" not in inspect.getsource(pipeline_mod)
    assert "b1_jev_experimental" not in inspect.getsource(b1_pkg)
