"""Frozen 64-case holdout integrity. Does not call vLLM."""

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
from retrieval.b1.validation import query_length  # noqa: E402

EVAL_DIR = CODE_ROOT / "retrieval" / "eval"
HOLDOUT_PATH = EVAL_DIR / "b1_holdout_v1.jsonl"
DRAFT_PATH = EVAL_DIR / "b1_draft_acceptance_set.jsonl"

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

V2_EXEMPLAR_STRINGS = (
    "NASA-STD-8739.12A 校准周期怎么规定？",
    "NASA-STD-6016C 对铝合金材料有什么要求？",
    "钛呢？",
    "NASA-STD-6016C 对钛材料有什么要求？",
    "按那个处理一下。",
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


def test_holdout_shape_and_buckets() -> None:
    cases = _load_jsonl(HOLDOUT_PATH)
    assert len(cases) == 64
    assert [case["id"] for case in cases] == [f"h-{index:03d}" for index in range(1, 65)]
    counts: dict[str, int] = {}
    for case in cases:
        counts[str(case["bucket"])] = counts.get(str(case["bucket"]), 0) + 1
    assert counts == EXPECTED_BUCKET_COUNTS


def test_holdout_does_not_leak_draft_or_v2_exemplars() -> None:
    holdout = _load_jsonl(HOLDOUT_PATH)
    draft = _load_jsonl(DRAFT_PATH)
    draft_queries = {str(case["query"]) for case in draft}
    for case in draft:
        for turn in case.get("history") or []:
            draft_queries.add(str(turn["raw_user_query"]))
            draft_queries.add(str(turn["canonical_user_query"]))
    holdout_queries = {str(case["query"]) for case in holdout}
    overlap = holdout_queries & draft_queries
    assert not overlap, overlap
    for case in holdout:
        for text in _all_texts(case):
            assert text not in draft_queries
            assert text not in V2_EXEMPLAR_STRINGS
            assert text not in (
                "随便处理一下。",
                "那个呢？",
                "那个标准呢？",
            )


def test_holdout_gold_invariants() -> None:
    cases = {str(case["id"]): case for case in _load_jsonl(HOLDOUT_PATH)}
    eight = cases["h-046"]
    assert eight["expected"]["mode"] == "direct"
    assert eight["expected"]["routes"] == ["knowledge"] * 8
    assert "8 个彼此独立" in str(eight["query"])
    for case_id in ("h-047", "h-048", "h-049", "h-050"):
        assert cases[case_id]["expected"]["mode"] == "task_limit"
        assert cases[case_id]["expected"]["routes"] == []
        assert cases[case_id]["expected"]["candidate_query"] is None
    assert "9 个彼此独立" in str(cases["h-047"]["query"])
    assert "10 个彼此独立" in str(cases["h-049"]["query"])
    assert "nine independent" in str(cases["h-050"]["query"])
    for case in cases.values():
        if case["bucket"] == "control":
            assert "error_type" in case["expected"]
            continue
        assert case["expected"]["mode"] in {
            "direct",
            "confirm_rewrite",
            "clarify",
            "task_limit",
        }
        if case["expected"].get("single_query_equals_current"):
            assert case["expected"]["routes"]
            assert len(case["expected"]["routes"]) == 1
    too_long = str(cases["h-063"]["query"])
    assert query_length(too_long) > 2000
    assert cases["h-061"]["expected"]["error_type"] == "empty_query"
    assert cases["h-062"]["expected"]["error_type"] == "empty_query"
    assert cases["h-063"]["expected"]["error_type"] == "query_too_long"
    assert cases["h-064"]["expected"]["error_type"] == "invalid_query"


def test_prompt_v3_and_guided_projection() -> None:
    assert ROUTER_CONTRACT_VERSION == "b1-contract-v2"
    assert ROUTER_SCHEMA_VERSION == "b1-result-v1"
    assert ROUTER_PROMPT_VERSION == "b1-jev-v1"
    assert "mode 决策（按顺序，只选一个）" in ROUTER_SYSTEM_PROMPT
    assert "【direct】" in ROUTER_SYSTEM_PROMPT
