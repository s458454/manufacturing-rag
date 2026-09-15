"""Blind acceptance set integrity. Does not call vLLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.constants import (  # noqa: E402
    ROUTER_CONTRACT_VERSION,
    ROUTER_PROMPT_VERSION,
    ROUTER_SCHEMA_VERSION,
)
from retrieval.b1.prompt import ROUTER_SYSTEM_PROMPT  # noqa: E402
from retrieval.b1.schema import ROUTER_GUIDED_JSON_SCHEMA, ROUTER_JSON_SCHEMA  # noqa: E402
from retrieval.b1.validation import query_length  # noqa: E402

EVAL_DIR = CODE_ROOT / "retrieval" / "eval"
HOLDOUT_PATH = EVAL_DIR / "b1_holdout_v1.jsonl"
DRAFT_PATH = EVAL_DIR / "b1_draft_acceptance_set.jsonl"
BLIND_PATH = EVAL_DIR / "b1_blind_acceptance_v1.jsonl"

EXPECTED_BUCKET_COUNTS = {
    "control": 4,
    "knowledge_single": 12,
    "long_tail": 5,
    "zh_en": 5,
    "system": 5,
    "unsupported": 5,
    "mixed": 6,
    "multi_2_to_8": 8,
    "task_limit_gt8": 4,
    "confirm_rewrite": 5,
    "clarify": 5,
}

PROMPT_EXEMPLARS = (
    "NASA-STD-8739.12A 校准周期怎么规定？",
    "NASA-STD-6016C 对铝合金材料有什么要求？",
    "钛呢？",
    "NASA-STD-6016C 对钛材料有什么要求？",
    "按那个处理一下。",
)

V2_KNOWN_FAILS = (
    "How does NASA-STD-4003A define class S bonding?",
    "你现在接入了哪些类型的技术文档？",
    "把 ERP 里的实时库存导出来。",
    "帮我用 Python 写一个压接电阻计算脚本。",
    "搜索网上最新的 NASA 焊接新闻。",
    "胶粘剂贮存呢？",
    "持续运行电压呢？",
    "另一种也要。",
    "两份都按那个来。",
)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _all_texts(case: dict[str, object]) -> list[str]:
    texts = [str(case["query"])]
    for turn in case.get("history") or []:
        texts.append(str(turn["raw_user_query"]))
        texts.append(str(turn["canonical_user_query"]))
    candidate = case["expected"].get("candidate_query")
    if isinstance(candidate, str):
        texts.append(candidate)
    return texts


def _collect_texts(cases: list[dict[str, object]]) -> set[str]:
    texts: set[str] = set()
    for case in cases:
        texts.update(_all_texts(case))
    return texts


def test_blind_shape_and_buckets() -> None:
    cases = _load_jsonl(BLIND_PATH)
    assert len(cases) == 64
    assert [case["id"] for case in cases] == [f"ba-{index:03d}" for index in range(1, 65)]
    counts: dict[str, int] = {}
    for case in cases:
        counts[str(case["bucket"])] = counts.get(str(case["bucket"]), 0) + 1
    assert counts == EXPECTED_BUCKET_COUNTS


def test_blind_does_not_leak_prior_sets_or_exemplars() -> None:
    blind = _load_jsonl(BLIND_PATH)
    holdout = _load_jsonl(HOLDOUT_PATH)
    draft = _load_jsonl(DRAFT_PATH)
    prior = _collect_texts(holdout) | _collect_texts(draft)
    prior.update(PROMPT_EXEMPLARS)
    prior.update(V2_KNOWN_FAILS)
    prior.update(
        (
            "随便处理一下。",
            "那个呢？",
            "那个标准呢？",
        )
    )
    overlap = _collect_texts(blind) & prior
    assert not overlap, overlap


def test_blind_gold_invariants() -> None:
    cases = {str(case["id"]): case for case in _load_jsonl(BLIND_PATH)}
    eight = cases["ba-046"]
    assert eight["expected"]["mode"] == "direct"
    assert eight["expected"]["routes"] == ["knowledge"] * 8
    assert "8 个互不依赖" in str(eight["query"])
    for case_id in ("ba-047", "ba-048", "ba-049", "ba-050"):
        assert cases[case_id]["expected"]["mode"] == "task_limit"
        assert cases[case_id]["expected"]["routes"] == []
        assert cases[case_id]["expected"]["candidate_query"] is None
    assert "9 个互不依赖" in str(cases["ba-047"]["query"])
    assert "11 个互不依赖" in str(cases["ba-049"]["query"])
    assert "nine unrelated" in str(cases["ba-050"]["query"])
    too_long = str(cases["ba-063"]["query"])
    assert query_length(too_long) > 2000
    assert cases["ba-061"]["expected"]["error_type"] == "empty_query"
    assert cases["ba-064"]["expected"]["error_type"] == "invalid_query"


def test_prompt_v3_minimal_and_schema_untouched() -> None:
    assert ROUTER_CONTRACT_VERSION == "b1-contract-v1"
    assert ROUTER_SCHEMA_VERSION == "b1-schema-v1"
    assert ROUTER_PROMPT_VERSION == "b1-router-v3"
    assert "逐字复制" in ROUTER_SYSTEM_PROMPT
    assert "第 9 个 task" in ROUTER_SYSTEM_PROMPT
    assert "禁止 mode=direct 静默执行" in ROUTER_SYSTEM_PROMPT
    assert "禁止虚构实体" in ROUTER_SYSTEM_PROMPT
    assert "已接入的数据或文档类型" in ROUTER_SYSTEM_PROMPT
    for banned in V2_KNOWN_FAILS:
        assert banned not in ROUTER_SYSTEM_PROMPT
    assert "oneOf" not in json.dumps(ROUTER_GUIDED_JSON_SCHEMA)
    assert ROUTER_JSON_SCHEMA["properties"]["tasks"]["maxItems"] == 8
    assert "minItems" not in json.dumps(ROUTER_GUIDED_JSON_SCHEMA)
