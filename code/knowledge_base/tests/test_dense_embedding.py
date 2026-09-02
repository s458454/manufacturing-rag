"""A5 Dense Embedding unit tests. Ordinary pytest uses stub tokenizer/model."""

from __future__ import annotations

import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.dense_embedding import (  # noqa: E402
    A5EmbeddingError,
    DenseEmbeddingConfig,
    Qwen3DenseEncoder,
    embed_leaves,
    l2_normalize,
    last_token_pool,
)
from knowledge_base.embedding_config import (  # noqa: E402
    DEFAULT_MAX_INPUT_TOKENS,
    DETERMINISM_ATOL,
    DETERMINISM_RTOL,
    EMBEDDING_DIMENSION,
)
from knowledge_base.leaf_chunker import LEAF_FIELD_NAMES, Leaf  # noqa: E402


class FakeTokenizer:
    def __init__(self, padding_side: str = "left") -> None:
        self.padding_side = padding_side
        self.calls: list[dict[str, object]] = []
        self.texts: list[str] = []
        self.pad_id = 0
        self.eos_id = 2

    def _content_ids(self, text: str) -> list[int]:
        return [(ord(char) % 100) + 3 for char in text]

    def __call__(
        self,
        texts: str | list[str],
        add_special_tokens: bool = True,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: str | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "texts": texts,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "padding": padding,
                "return_tensors": return_tensors,
                "max_length": kwargs.get("max_length"),
            }
        )
        single = isinstance(texts, str)
        sequences = [texts] if single else list(texts)
        if single:
            self.texts.append(texts)
        else:
            self.texts.extend(sequences)
        all_ids: list[list[int]] = []
        for text in sequences:
            ids = self._content_ids(text)
            if add_special_tokens:
                ids = ids + [self.eos_id]
            all_ids.append(ids)
        if padding:
            width = max(len(ids) for ids in all_ids)
            padded: list[list[int]] = []
            masks: list[list[int]] = []
            for ids in all_ids:
                gap = width - len(ids)
                if self.padding_side == "left":
                    padded.append([self.pad_id] * gap + ids)
                    masks.append([0] * gap + [1] * len(ids))
                else:
                    padded.append(ids + [self.pad_id] * gap)
                    masks.append([1] * len(ids) + [0] * gap)
            all_ids = padded
            all_masks = masks
        else:
            all_masks = [[1] * len(ids) for ids in all_ids]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_masks, dtype=torch.long),
            }
        if single:
            return {"input_ids": all_ids[0], "attention_mask": all_masks[0]}
        return {"input_ids": all_ids, "attention_mask": all_masks}


class FakeModel:
    def __init__(
        self,
        hidden_size: int = EMBEDDING_DIMENSION,
        *,
        nonfinite: str | None = None,
        output_batch: int | None = None,
    ) -> None:
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.device = torch.device("cpu")
        self.calls = 0
        self.nonfinite = nonfinite
        self.output_batch = output_batch

    def eval(self) -> FakeModel:
        return self

    def to(self, device: str | torch.device) -> FakeModel:
        self.device = torch.device(device)
        return self

    def __call__(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        assert input_ids is not None and attention_mask is not None
        self.calls += 1
        batch_size, width = input_ids.shape
        emit = self.output_batch if self.output_batch is not None else batch_size
        hidden = torch.zeros(emit, width, self.config.hidden_size)
        if self.nonfinite == "nan":
            hidden[:, :, 0] = float("nan")
        elif self.nonfinite == "inf":
            hidden[:, :, 0] = float("inf")
        elif emit == batch_size:
            for index in range(batch_size):
                real = input_ids[index][attention_mask[index].bool()]
                last = int(attention_mask[index].nonzero(as_tuple=False).flatten()[-1])
                hidden[index, last, 0] = real.float().sum() + float(len(real))
                hidden[index, last, 1] = 4.0
                hidden[index, last, 2] = 3.0
        return SimpleNamespace(last_hidden_state=hidden)


def _leaf(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "doc",
    chunk_index: int = 0,
    page_start: int = 1,
    page_end: int = 1,
) -> Leaf:
    return Leaf(
        chunk_id=chunk_id,
        document_id=document_id,
        section_id="sec_1",
        chunk_index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        content=content,
    )


def _cpu_config(**kwargs: object) -> DenseEmbeddingConfig:
    payload: dict[str, object] = {"device": "cpu"}
    payload.update(kwargs)
    return DenseEmbeddingConfig(**payload)  # type: ignore[arg-type]


def _encoder(
    tokenizer: FakeTokenizer | None = None,
    model: FakeModel | None = None,
    **config_kwargs: object,
) -> tuple[Qwen3DenseEncoder, FakeTokenizer, FakeModel]:
    tokenizer = tokenizer or FakeTokenizer()
    model = model or FakeModel()
    encoder = Qwen3DenseEncoder(
        _cpu_config(**config_kwargs),
        tokenizer=tokenizer,
        model=model,
    )
    return encoder, tokenizer, model


def _embed(
    leaves: list[Leaf],
    *,
    tokenizer: FakeTokenizer | None = None,
    model: FakeModel | None = None,
    **config_kwargs: object,
):
    encoder, tokenizer, model = _encoder(tokenizer, model, **config_kwargs)
    result = embed_leaves(
        leaves, config=_cpu_config(**config_kwargs), encoder=encoder
    )
    return result, encoder, tokenizer, model


def test_t81_leaf_content_only() -> None:
    leaves = [
        _leaf("chk_a", "body only"),
        _leaf("chk_b", "second body", chunk_index=1),
    ]
    result, _encoder_obj, tokenizer, _model = _embed(leaves)
    assert tokenizer.texts
    assert set(tokenizer.texts) == {"body only", "second body"}
    for text in tokenizer.texts:
        assert "heading" not in text
        assert "NASA-STD" not in text
        assert not text.startswith("Instruct:")
        assert "Query:" not in text
    assert result.chunk_ids == ("chk_a", "chk_b")


def test_t82_input_order_preserved() -> None:
    leaves = [_leaf("A", "aaa"), _leaf("B", "bb"), _leaf("C", "cccc")]
    result, *_ = _embed(leaves)
    assert result.chunk_ids == ("A", "B", "C")
    assert result.vectors.shape[0] == 3
    assert not np.allclose(result.vectors[0], result.vectors[1])
    assert not np.allclose(result.vectors[1], result.vectors[2])


def test_t83_duplicate_chunk_id_fails() -> None:
    leaves = [_leaf("same", "one"), _leaf("same", "two")]
    with pytest.raises(A5EmbeddingError, match="Duplicate chunk_id"):
        _embed(leaves)


def test_t84_empty_input_fails() -> None:
    with pytest.raises(A5EmbeddingError, match="No leaves"):
        embed_leaves([], config=_cpu_config())


def test_t85_empty_leaf_content_fails() -> None:
    with pytest.raises(A5EmbeddingError, match="Leaf.content is empty"):
        _embed([_leaf("chk", "")])
    with pytest.raises(A5EmbeddingError, match="Leaf.content is empty"):
        _embed([_leaf("chk", "   \n")])


def test_t86_exact_max_length_allowed() -> None:
    content = "x" * (DEFAULT_MAX_INPUT_TOKENS - 1)
    result, encoder, tokenizer, model = _embed([_leaf("chk", content)])
    assert encoder.input_length(content) == DEFAULT_MAX_INPUT_TOKENS
    assert result.vectors.shape == (1, EMBEDDING_DIMENSION)
    assert model.calls >= 1
    assert all(call["truncation"] is False for call in tokenizer.calls)


def test_t87_max_plus_one_fails_without_truncation() -> None:
    content = "x" * DEFAULT_MAX_INPUT_TOKENS
    tokenizer = FakeTokenizer()
    model = FakeModel()
    encoder, tokenizer, model = _encoder(tokenizer, model)
    with pytest.raises(A5EmbeddingError, match="actual input tokens"):
        embed_leaves(
            [_leaf("chk", content)],
            config=_cpu_config(),
            encoder=encoder,
        )
    assert encoder.input_length(content) == DEFAULT_MAX_INPUT_TOKENS + 1
    assert model.calls == 0
    batch_calls = [call for call in tokenizer.calls if call["padding"] is True]
    assert batch_calls == []
    assert all(call["truncation"] is False for call in tokenizer.calls)


def test_t88_tokenizer_invoked_with_no_truncation() -> None:
    _result, _encoder_obj, tokenizer, _model = _embed([_leaf("chk", "hello")])
    assert tokenizer.calls
    for call in tokenizer.calls:
        assert call["truncation"] is False
        assert call["max_length"] is None


def test_t89_left_padding() -> None:
    leaves = [_leaf("short", "a"), _leaf("long", "abcdef")]
    result, encoder, tokenizer, _model = _embed(leaves)
    assert tokenizer.padding_side == "left"
    assert encoder.tokenizer.padding_side == "left"
    batch_calls = [call for call in tokenizer.calls if call["padding"] is True]
    assert batch_calls
    encoded = tokenizer(
        ["a", "abcdef"],
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    assert int(encoded["input_ids"][0, 0].item()) == tokenizer.pad_id
    assert int(encoded["attention_mask"][0, 0].item()) == 0
    assert int(encoded["attention_mask"][0, -1].item()) == 1
    assert int(encoded["attention_mask"][1, 0].item()) == 1
    assert result.vectors.shape[0] == 2


def test_t90_last_token_pooling() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            [[5.0, 0.0], [6.0, 0.0], [7.0, 0.0], [8.0, 0.0]],
        ]
    )
    left_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    left_pooled = last_token_pool(hidden, left_mask)
    assert torch.equal(left_pooled, hidden[:, -1])
    right_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    right_pooled = last_token_pool(hidden, right_mask)
    assert torch.equal(right_pooled[0], hidden[0, 1])
    assert torch.equal(right_pooled[1], hidden[1, 2])


def test_t91_fp32_normalize() -> None:
    pooled = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    out = l2_normalize(pooled)
    assert out.dtype == torch.float32
    assert out.shape == (1, 2)
    np.testing.assert_allclose(
        out.numpy(), np.array([[0.6, 0.8]], dtype=np.float32), rtol=1e-5, atol=1e-6
    )


def test_t92_vector_norm() -> None:
    result, *_ = _embed([_leaf("a", "one"), _leaf("b", "two words")])
    norms = np.linalg.norm(result.vectors, axis=1)
    assert np.all(np.abs(norms - 1.0) <= 1e-4)


def test_t93_dimension() -> None:
    result, *_ = _embed([_leaf("a", "dim check")])
    assert result.dimension == 2560
    assert result.vectors.shape == (1, 2560)


def test_t94_output_dtype() -> None:
    result, *_ = _embed([_leaf("a", "dtype check")])
    assert result.vectors.dtype == np.float32


def test_t95_nonfinite_fails() -> None:
    with pytest.raises(A5EmbeddingError, match="non-finite"):
        _embed([_leaf("a", "nan")], model=FakeModel(nonfinite="nan"))
    with pytest.raises(A5EmbeddingError, match="non-finite"):
        _embed([_leaf("a", "inf")], model=FakeModel(nonfinite="inf"))


def test_t96_output_count_mismatch_fails() -> None:
    with pytest.raises(A5EmbeddingError, match="batch count mismatch"):
        _embed(
            [_leaf("a", "one"), _leaf("b", "two")],
            model=FakeModel(output_batch=1),
        )


def test_t97_output_mapping() -> None:
    leaves = [_leaf("chk_a", "alpha"), _leaf("chk_b", "beta-beta")]
    result, *_ = _embed(leaves)
    assert result.chunk_ids[0] == "chk_a"
    assert result.chunk_ids[1] == "chk_b"
    assert not np.allclose(result.vectors[0], result.vectors[1])


def test_t98_leaf_immutable() -> None:
    leaves = [_leaf("chk", "immutable body")]
    before = [replace(leaf) for leaf in leaves]
    before_payload = [leaf.__dict__.copy() for leaf in leaves]
    result, *_ = _embed(leaves)
    assert [field.name for field in fields(Leaf)] == list(LEAF_FIELD_NAMES)
    assert "dense_vector" not in leaves[0].__dict__
    assert leaves[0] == before[0]
    assert leaves[0].__dict__ == before_payload[0]
    assert result.chunk_ids == ("chk",)


def test_t99_no_query_instruction() -> None:
    leaves = [_leaf("chk", "document body")]
    _result, encoder, tokenizer, _model = _embed(leaves)
    assert not hasattr(encoder, "encode_query")
    assert not hasattr(encoder, "get_detailed_instruct")
    for text in tokenizer.texts:
        assert text == "document body"
        assert not text.startswith("Instruct:")
        assert "Query:" not in text


def test_t100_no_runtime_model_fallback() -> None:
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=FakeTokenizer(),
    ):
        with patch(
            "transformers.AutoModel.from_pretrained",
            side_effect=OSError("4B weights missing"),
        ) as mocked:
            with pytest.raises(A5EmbeddingError, match="Failed to load model"):
                Qwen3DenseEncoder(_cpu_config())
            assert mocked.call_args_list
            for call in mocked.call_args_list:
                blob = " ".join(str(item) for item in call.args) + str(call.kwargs)
                assert "0.6B" not in blob
                assert "0.6b" not in blob.lower()


def test_t101_cuda_unavailable_fails() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(A5EmbeddingError, match="CUDA is unavailable"):
            embed_leaves(
                [_leaf("chk", "hello")],
                config=DenseEmbeddingConfig(device="cuda"),
            )


def test_t102_invalid_batch_size() -> None:
    with pytest.raises(A5EmbeddingError, match="batch_size"):
        _embed([_leaf("chk", "hello")], batch_size=0)
    with pytest.raises(A5EmbeddingError, match="batch_size"):
        _embed([_leaf("chk", "hello")], batch_size=-1)


def test_t103_invalid_max_input() -> None:
    with pytest.raises(A5EmbeddingError, match="max_input_tokens"):
        _embed([_leaf("chk", "hello")], max_input_tokens=0)
    with pytest.raises(A5EmbeddingError, match="max_input_tokens"):
        _embed([_leaf("chk", "hello")], max_input_tokens=-8)


def test_t104_dimension_mismatch_fails() -> None:
    with pytest.raises(A5EmbeddingError, match="dimension mismatch"):
        _encoder(model=FakeModel(hidden_size=1024))


def test_t105_determinism_tolerance() -> None:
    leaves = [_leaf("chk", "same text")]
    run1, *_ = _embed(leaves)
    run2, *_ = _embed(leaves)
    assert np.allclose(
        run1.vectors, run2.vectors, rtol=DETERMINISM_RTOL, atol=DETERMINISM_ATOL
    )
    noisy = run1.vectors + 5e-6
    assert np.allclose(
        run1.vectors, noisy, rtol=DETERMINISM_RTOL, atol=DETERMINISM_ATOL
    )
    assert not np.array_equal(run1.vectors, noisy)


def test_t106_a1_to_a4_regression_files_present() -> None:
    tests_dir = Path(__file__).resolve().parent
    assert (tests_dir / "test_markdown_loader.py").is_file()
    assert (tests_dir / "test_document_registry.py").is_file()
    assert (tests_dir / "test_section_profile.py").is_file()
    assert (tests_dir / "test_leaf_chunker.py").is_file()
    assert (tests_dir / "test_section_hierarchy.py").is_file()
    assert (tests_dir / "test_dense_embedding.py").is_file()
