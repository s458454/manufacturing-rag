from __future__ import annotations

import json
from typing import Any

from retrieval.b1.types import RouterConfig
from retrieval.b2.types import ProcessedQuery
from retrieval.b3.constants import BM25_CANDIDATE_LIMIT, DENSE_CANDIDATE_LIMIT
from retrieval.b3.types import HybridSearchConfig
from retrieval.b4.constants import FUSION_CANDIDATE_LIMIT, RRF_K
from retrieval.b4.types import FusedCandidate, FusionResult


class CountingTokenizer:
    """Deterministic stand-in: one token per character, including chat markers."""

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs: Any) -> list[int]:
        return [0] * len(text)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> list[int] | str:
        text = "\n".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        if tokenize:
            return [0] * len(text)
        return text


class FakeRouterClient:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls = 0
        self.messages: list[dict[str, str]] | None = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        self.messages = messages
        if callable(self.payload):
            value = self.payload(messages)
        else:
            value = self.payload
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)


def router_config() -> RouterConfig:
    return RouterConfig(
        vllm_base_url="http://127.0.0.1:9/v1",
        model_alias="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name_or_path="fake-tokenizer",
        request_timeout_ms=1234,
        structured_decoding_backend="xgrammar",
    )


def processed_query(
    query: str = "NASA-STD-5006A welding requirements",
    *,
    request_id: str = "req-1",
    task_id: str = "req-1:0",
) -> ProcessedQuery:
    return ProcessedQuery(
        request_id=request_id,
        task_id=task_id,
        original_query=query,
        retrieval_queries=(query,),
    )


def fake_dense_vector(fill: float = 0.01) -> tuple[float, ...]:
    return tuple(fill for _ in range(2560))


def hybrid_config(timeout_ms: int = 5000) -> HybridSearchConfig:
    return HybridSearchConfig(
        collection_name="a6_leaf_v1",
        uri="http://localhost:19530",
        hybrid_search_timeout_ms=timeout_ms,
        dense_candidate_limit=DENSE_CANDIDATE_LIMIT,
        bm25_candidate_limit=BM25_CANDIDATE_LIMIT,
        fusion_candidate_limit=FUSION_CANDIDATE_LIMIT,
        rrf_k=RRF_K,
    )


def fused_candidate(
    chunk_id: str,
    *,
    fused_rank: int,
    rrf_score: float = 0.03,
    content: str = "leaf body",
) -> FusedCandidate:
    return FusedCandidate(
        chunk_id=chunk_id,
        document_id="doc-a",
        section_id="sec-1",
        chunk_index=fused_rank - 1,
        page_start=1,
        page_end=1,
        content=content,
        rrf_score=rrf_score,
        fused_rank=fused_rank,
    )


def fusion_result(*chunk_ids: str, query: str = "q") -> FusionResult:
    candidates = tuple(
        fused_candidate(chunk_id, fused_rank=index)
        for index, chunk_id in enumerate(chunk_ids, start=1)
    )
    return FusionResult(
        request_id="req-1",
        task_id="req-1:0",
        query=query,
        candidates=candidates,
    )


def leaf_hit(
    chunk_id: str,
    *,
    score: float,
    content: str = "leaf body",
    section_id: str = "sec-1",
    chunk_index: int = 0,
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "distance": score,
        "entity": {
            "chunk_id": chunk_id,
            "document_id": "doc-a",
            "section_id": section_id,
            "chunk_index": chunk_index,
            "page_start": 1,
            "page_end": 2,
            "content": content,
        },
    }
