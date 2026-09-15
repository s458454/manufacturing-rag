"""B5 rerank: no silent truncation, score sort then fused_rank, empty skips model."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b5.errors import RerankerInputTooLongError  # noqa: E402
from retrieval.b5.reranker import Qwen3LeafReranker  # noqa: E402
from retrieval.tests.helpers import fusion_result  # noqa: E402


YES_ID = 1
NO_ID = 2


class FakeRerankTokenizer:
    def __init__(self, body_tokens: int = 1) -> None:
        self.padding_side = "left"
        self.pad_token_id = 0
        self.encode_calls: list[dict[str, object]] = []
        self.body_tokens = body_tokens

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"yes": YES_ID, "no": NO_ID}[token]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        **kwargs: object,
    ) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
            }
        )
        if truncation is not False:
            raise AssertionError("B5 must not pass truncation=True or longest_first")
        if text.startswith("<Instruct>:"):
            return [3] * self.body_tokens
        return [3]


class FakeCausalLM:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = list(probabilities)
        self.forward_calls = 0
        self.device = "cpu"

    def eval(self) -> FakeCausalLM:
        return self

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        torch = pytest.importorskip("torch")
        self.forward_calls += 1
        input_ids = kwargs["input_ids"]
        assert input_ids.shape[1] >= 1
        batch = int(input_ids.shape[0])
        vocab = 8
        logits = torch.zeros(batch, 2, vocab)
        for index in range(batch):
            probability = self.probabilities.pop(0)
            probability = min(max(probability, 1e-6), 1 - 1e-6)
            yes_logit = math.log(probability / (1.0 - probability))
            logits[index, -1, YES_ID] = yes_logit
            logits[index, -1, NO_ID] = 0.0
        return SimpleNamespace(logits=logits)


def test_empty_fusion_does_not_run_the_model() -> None:
    model = FakeCausalLM([])
    reranker = Qwen3LeafReranker(
        tokenizer=FakeRerankTokenizer(), model=model, batch_size=8
    )
    result = reranker.rerank(fusion_result())
    assert result.candidates == ()
    assert model.forward_calls == 0


def test_too_long_pair_fails_without_truncation() -> None:
    model = FakeCausalLM([0.5])
    tokenizer = FakeRerankTokenizer(body_tokens=5000)
    reranker = Qwen3LeafReranker(tokenizer=tokenizer, model=model, batch_size=8)
    with pytest.raises(RerankerInputTooLongError, match="max_input_tokens"):
        reranker.rerank(fusion_result("a", query="q"))
    assert model.forward_calls == 0
    assert all(call["truncation"] is False for call in tokenizer.encode_calls)


def test_sorts_by_score_then_fused_rank_and_keeps_top_5() -> None:
    pytest.importorskip("torch")
    model = FakeCausalLM([0.2, 0.9, 0.9, 0.1, 0.4, 0.3])
    reranker = Qwen3LeafReranker(
        tokenizer=FakeRerankTokenizer(), model=model, batch_size=8
    )
    fusion = fusion_result("a", "b", "c", "d", "e", "f")
    result = reranker.rerank(fusion)
    assert [item.chunk_id for item in result.candidates] == ["b", "c", "e", "f", "a"]
    assert [item.rerank_rank for item in result.candidates] == [1, 2, 3, 4, 5]
    assert result.candidates[0].fused_rank == 2
    assert result.candidates[1].fused_rank == 3
    assert result.candidates[0].rrf_score == fusion.candidates[1].rrf_score
    assert model.forward_calls == 1


def test_preserves_query_and_leaf_content_fields() -> None:
    pytest.importorskip("torch")
    model = FakeCausalLM([0.7])
    reranker = Qwen3LeafReranker(
        tokenizer=FakeRerankTokenizer(), model=model, batch_size=8
    )
    fusion = fusion_result("only", query="manufacturing query")
    result = reranker.rerank(fusion)
    assert result.query == "manufacturing query"
    assert result.candidates[0].content == "leaf body"
    assert 0.0 < result.candidates[0].rerank_score < 1.0
