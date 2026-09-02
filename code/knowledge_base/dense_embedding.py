"""A5 Dense Embedding: Leaf.content → 2560-d L2-normalized float32 vectors.

Reuses A1 loading and A3 Leaf chunking. Does not modify Leaf schema, A4
hierarchy, or implement A6 / query-side instruction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from knowledge_base.chunking_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_TOKENS,
)
from knowledge_base.embedding_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MODEL_NAME_OR_PATH,
    DETERMINISM_ATOL,
    DETERMINISM_RTOL,
    EMBEDDING_DIMENSION,
    NORM_ATOL,
)
from knowledge_base.leaf_chunker import (
    A3ChunkingError,
    LEAF_FIELD_NAMES,
    Leaf,
    chunk_documents,
)
from knowledge_base.markdown_loader import (
    MarkdownLoadingError,
    load_markdown_documents,
)
from knowledge_base.token_count import (
    count_tokens,
    load_tokenizer,
    transformers_version,
)


class A5EmbeddingError(Exception):
    """Fail-fast error for A5 dense embedding."""


@dataclass(frozen=True)
class DenseEmbeddingConfig:
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    batch_size: int = DEFAULT_BATCH_SIZE
    device: str = DEFAULT_DEVICE


@dataclass(frozen=True)
class DenseEmbeddingResult:
    chunk_ids: tuple[str, ...]
    vectors: np.ndarray
    model_name: str
    dimension: int


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Official Qwen3-Embedding last-token pooling."""

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def l2_normalize(pooled: Tensor) -> Tensor:
    """Cast to FP32 then L2-normalize along dim=1."""

    return F.normalize(pooled.float(), p=2, dim=1)


def _token_ids(encoded: Any) -> list[int]:
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def actual_input_length(tokenizer: Any, text: str) -> int:
    """Token length of *text* using A5 inference special-token semantics."""

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return len(_token_ids(encoded))


def validate_embedding_config(config: DenseEmbeddingConfig) -> None:
    if config.max_input_tokens <= 0:
        raise A5EmbeddingError(
            f"max_input_tokens must be > 0, got {config.max_input_tokens}"
        )
    if config.batch_size <= 0:
        raise A5EmbeddingError(
            f"batch_size must be > 0, got {config.batch_size}"
        )
    if not config.model_name_or_path.strip():
        raise A5EmbeddingError("model_name_or_path is empty")


def resolve_device(device: str) -> str:
    spec = device.strip()
    if not spec:
        raise A5EmbeddingError("device is empty")
    if spec == "cpu":
        return spec
    if spec == "cuda" or spec.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise A5EmbeddingError(
                f"device={spec!r} requested but CUDA is unavailable"
            )
        return spec
    raise A5EmbeddingError(f"unsupported device {spec!r}")


def _cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _load_tokenizer_and_model(
    model_name_or_path: str, device: str
) -> tuple[Any, Any]:
    identifier = model_name_or_path.strip()
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise A5EmbeddingError(
            "transformers is required to load Qwen3-Embedding-4B"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            identifier, padding_side="left"
        )
    except Exception as exc:
        raise A5EmbeddingError(
            f"Failed to load tokenizer {identifier!r}"
        ) from exc
    tokenizer.padding_side = "left"
    try:
        try:
            model = AutoModel.from_pretrained(identifier, dtype=torch.float16)
        except TypeError:
            model = AutoModel.from_pretrained(
                identifier, torch_dtype=torch.float16
            )
    except Exception as exc:
        if _cuda_oom(exc):
            raise A5EmbeddingError(
                "CUDA OOM while loading the embedding model; "
                "specify a smaller --batch-size and retry. "
                "Automatic batch/model fallback is forbidden"
            ) from exc
        raise A5EmbeddingError(
            f"Failed to load model {identifier!r}"
        ) from exc
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden != EMBEDDING_DIMENSION:
        raise A5EmbeddingError(
            "model embedding dimension mismatch: "
            f"expected {EMBEDDING_DIMENSION}, got {hidden!r}"
        )
    model.eval()
    try:
        model.to(device)
    except Exception as exc:
        if _cuda_oom(exc):
            raise A5EmbeddingError(
                "CUDA OOM while moving the embedding model; "
                "specify a smaller --batch-size and retry. "
                "Automatic batch/model fallback is forbidden"
            ) from exc
        raise A5EmbeddingError(
            f"Failed to move model to device {device!r}"
        ) from exc
    return tokenizer, model


def _model_device(model: Any, fallback: str) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    return torch.device(fallback)


def _as_int_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(item) for item in values]


def _leaf_snapshot(leaf: Leaf) -> tuple[Any, ...]:
    return tuple(getattr(leaf, name) for name in LEAF_FIELD_NAMES)


def validate_leaves(leaves: Sequence[Leaf]) -> tuple[Leaf, ...]:
    if not leaves:
        raise A5EmbeddingError("No leaves provided")
    seen: set[str] = set()
    snapshots = []
    for leaf in leaves:
        if not isinstance(leaf, Leaf):
            raise A5EmbeddingError(
                f"A5 input must be A3 Leaf instances, got {type(leaf)!r}"
            )
        if not leaf.chunk_id:
            raise A5EmbeddingError("Leaf.chunk_id is empty")
        if leaf.chunk_id in seen:
            raise A5EmbeddingError(
                f"Duplicate chunk_id: {leaf.chunk_id}"
            )
        seen.add(leaf.chunk_id)
        if not leaf.content or not leaf.content.strip():
            raise A5EmbeddingError(
                f"Leaf.content is empty for chunk_id={leaf.chunk_id}"
            )
        snapshots.append(_leaf_snapshot(leaf))
    if len(snapshots) != len(set(snapshots)):
        raise A5EmbeddingError("Duplicate Leaf records in A5 input")
    return tuple(leaves)


def _validate_vectors(
    vectors: np.ndarray, expected_count: int
) -> np.ndarray:
    if not isinstance(vectors, np.ndarray):
        raise A5EmbeddingError(
            f"embedding output must be numpy.ndarray, got {type(vectors)!r}"
        )
    if vectors.ndim != 2:
        raise A5EmbeddingError(
            f"embedding matrix must be 2-D, got shape {vectors.shape}"
        )
    if vectors.shape[0] != expected_count:
        raise A5EmbeddingError(
            "model output batch count mismatch: "
            f"expected {expected_count}, got {vectors.shape[0]}"
        )
    if vectors.shape[1] != EMBEDDING_DIMENSION:
        raise A5EmbeddingError(
            "embedding dimension mismatch: "
            f"expected {EMBEDDING_DIMENSION}, got {vectors.shape[1]}"
        )
    if vectors.dtype != np.float32:
        vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        nan_count = int(np.isnan(vectors).any(axis=1).sum())
        inf_count = int(np.isinf(vectors).any(axis=1).sum())
        raise A5EmbeddingError(
            "non-finite embedding values: "
            f"nan_rows={nan_count} inf_rows={inf_count}"
        )
    norms = np.linalg.norm(vectors, axis=1)
    bad = np.abs(norms - 1.0) > NORM_ATOL
    if np.any(bad):
        raise A5EmbeddingError(
            "L2 norm contract violated: "
            f"invalid_norm_count={int(bad.sum())} "
            f"norm_min={float(norms.min())} norm_max={float(norms.max())}"
        )
    return np.ascontiguousarray(vectors, dtype=np.float32)


class Qwen3DenseEncoder:
    """Document-side Qwen3-Embedding-4B encoder. Adds no instruction."""

    def __init__(
        self,
        config: DenseEmbeddingConfig,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        validate_embedding_config(config)
        self.config = replace(config, device=resolve_device(config.device))
        if (tokenizer is None) ^ (model is None):
            raise A5EmbeddingError(
                "tokenizer and model must both be provided or both omitted"
            )
        if tokenizer is None:
            tokenizer, model = _load_tokenizer_and_model(
                self.config.model_name_or_path, self.config.device
            )
        hidden = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden != EMBEDDING_DIMENSION:
            raise A5EmbeddingError(
                "model embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMENSION}, got {hidden!r}"
            )
        padding_side = getattr(tokenizer, "padding_side", None)
        if padding_side != "left":
            raise A5EmbeddingError(
                f"tokenizer padding_side must be 'left', got {padding_side!r}"
            )
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            try:
                to_fn(self.config.device)
            except Exception as exc:
                if _cuda_oom(exc):
                    raise A5EmbeddingError(
                        "CUDA OOM while moving the embedding model; "
                        "specify a smaller --batch-size and retry. "
                        "Automatic batch/model fallback is forbidden"
                    ) from exc
                raise
        self.tokenizer = tokenizer
        self.model = model

    def input_length(self, text: str) -> int:
        return actual_input_length(self.tokenizer, text)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise A5EmbeddingError("No texts provided")
        lengths = [self.input_length(text) for text in texts]
        for index, (text, length) in enumerate(zip(texts, lengths, strict=True)):
            if not text or not text.strip():
                raise A5EmbeddingError(
                    f"empty embedding text at index {index}"
                )
            if length > self.config.max_input_tokens:
                raise A5EmbeddingError(
                    "actual input tokens exceed max_input_tokens: "
                    f"length={length} max={self.config.max_input_tokens} "
                    f"index={index}"
                )
        parts: list[np.ndarray] = []
        batch_size = self.config.batch_size
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            parts.append(
                self._encode_batch(
                    list(texts[start:end]), lengths[start:end]
                )
            )
        vectors = np.concatenate(parts, axis=0)
        return _validate_vectors(vectors, len(texts))

    def _encode_batch(
        self, texts: list[str], expected_lengths: list[int]
    ) -> np.ndarray:
        try:
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
        except Exception as exc:
            raise A5EmbeddingError("tokenizer batch encoding failed") from exc
        if batch.get("input_ids") is None or batch.get("attention_mask") is None:
            raise A5EmbeddingError(
                "tokenizer did not return input_ids and attention_mask"
            )
        observed = _as_int_list(batch["attention_mask"].sum(dim=1))
        if observed != list(expected_lengths):
            raise A5EmbeddingError(
                "implicit truncation detected: "
                f"expected_lengths={list(expected_lengths)} "
                f"observed_lengths={observed}"
            )
        device = _model_device(self.model, self.config.device)
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        try:
            with torch.no_grad():
                outputs = self.model(**batch)
                hidden = outputs.last_hidden_state
                if hidden.shape[0] != len(texts):
                    raise A5EmbeddingError(
                        "model output batch count mismatch: "
                        f"expected {len(texts)}, got {hidden.shape[0]}"
                    )
                if hidden.shape[-1] != EMBEDDING_DIMENSION:
                    raise A5EmbeddingError(
                        "embedding dimension mismatch: "
                        f"expected {EMBEDDING_DIMENSION}, got {hidden.shape[-1]}"
                    )
                pooled = last_token_pool(hidden, batch["attention_mask"])
                vectors = l2_normalize(pooled)
        except A5EmbeddingError:
            raise
        except Exception as exc:
            if _cuda_oom(exc):
                raise A5EmbeddingError(
                    "CUDA OOM during embedding forward; "
                    "specify a smaller --batch-size and retry. "
                    "Automatic batch/model fallback is forbidden"
                ) from exc
            raise A5EmbeddingError("embedding forward failed") from exc
        array = vectors.detach().cpu().numpy().astype(np.float32, copy=False)
        return _validate_vectors(array, len(texts))


def embed_leaves(
    leaves: Sequence[Leaf],
    *,
    config: DenseEmbeddingConfig,
    encoder: Qwen3DenseEncoder | None = None,
) -> DenseEmbeddingResult:
    """Map each Leaf.content to a dense vector. Does not mutate *leaves*."""

    ordered = validate_leaves(leaves)
    before = [_leaf_snapshot(leaf) for leaf in ordered]
    validate_embedding_config(config)
    resolve_device(config.device)
    if encoder is None:
        encoder = Qwen3DenseEncoder(config)
    else:
        if encoder.config.max_input_tokens != config.max_input_tokens:
            raise A5EmbeddingError(
                "encoder max_input_tokens does not match config"
            )
        if encoder.config.batch_size != config.batch_size:
            raise A5EmbeddingError("encoder batch_size does not match config")
    chunk_ids = tuple(leaf.chunk_id for leaf in ordered)
    for leaf in ordered:
        length = encoder.input_length(leaf.content)
        if length > config.max_input_tokens:
            raise A5EmbeddingError(
                "actual input tokens exceed max_input_tokens: "
                f"length={length} max={config.max_input_tokens} "
                f"chunk_id={leaf.chunk_id}"
            )
    vectors = encoder.encode_documents([leaf.content for leaf in ordered])
    after = [_leaf_snapshot(leaf) for leaf in ordered]
    if before != after:
        raise A5EmbeddingError("A5 mutated Leaf records")
    if [field.name for field in fields(Leaf)] != list(LEAF_FIELD_NAMES):
        raise A5EmbeddingError("Leaf schema changed during A5")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise A5EmbeddingError("output chunk_id contains duplicates")
    if len(chunk_ids) != vectors.shape[0]:
        raise A5EmbeddingError(
            "chunk_id / vector count mismatch: "
            f"{len(chunk_ids)} != {vectors.shape[0]}"
        )
    return DenseEmbeddingResult(
        chunk_ids=chunk_ids,
        vectors=vectors,
        model_name=config.model_name_or_path,
        dimension=EMBEDDING_DIMENSION,
    )


def _percentile(values: list[int], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percent / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return round(ordered[low] * (1.0 - weight) + ordered[high] * weight, 6)


def _vector_norms(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=1)


def _first_eight(vector: np.ndarray) -> list[float]:
    return [float(value) for value in vector[:8].tolist()]


def _select_spotcheck(
    leaves: tuple[Leaf, ...],
    a3_tokens: list[int],
    chunk_size: int,
) -> list[int]:
    shorts = [
        index
        for index, tokens in enumerate(a3_tokens)
        if tokens <= 256
    ]
    normal = [
        index
        for index, tokens in enumerate(a3_tokens)
        if tokens <= chunk_size
    ]
    near_max = sorted(normal, key=lambda index: a3_tokens[index], reverse=True)
    tables = [
        index
        for index, leaf in enumerate(leaves)
        if "| " in leaf.content and a3_tokens[index] <= chunk_size
    ]
    oversize = [
        index
        for index, tokens in enumerate(a3_tokens)
        if tokens > chunk_size
    ]
    cross = [
        index
        for index, leaf in enumerate(leaves)
        if leaf.page_end > leaf.page_start
    ]
    picked: list[int] = []
    for group, count in (
        (shorts, 2),
        (near_max, 2),
        (tables, 2),
        (oversize, 2),
        (cross, 2),
    ):
        added = 0
        for index in group:
            if index not in picked:
                picked.append(index)
                added += 1
            if added >= count:
                break
    for index in range(len(leaves)):
        if len(picked) >= 10:
            break
        if index not in picked:
            picked.append(index)
    return picked[:10]


def _device_report(device: str) -> tuple[str, str | None]:
    if device == "cpu" or not torch.cuda.is_available():
        return device, None
    index = 0
    if device.startswith("cuda:") and device[5:].isdigit():
        index = int(device.split(":")[1])
    try:
        return device, torch.cuda.get_device_name(index)
    except Exception:
        return device, None


def build_embedding_report(
    leaves: tuple[Leaf, ...],
    result: DenseEmbeddingResult,
    encoder: Qwen3DenseEncoder,
    *,
    config: DenseEmbeddingConfig,
    a3_tokenizer: Any,
    chunk_size: int,
    overlap_tokens: int,
    determinism_failure_count: int,
    spot_indices: list[int],
) -> dict[str, Any]:
    a5_tokens = [encoder.input_length(leaf.content) for leaf in leaves]
    a3_tokens = [count_tokens(leaf.content, a3_tokenizer) for leaf in leaves]
    norms = _vector_norms(result.vectors)
    nan_vector_count = int(np.isnan(result.vectors).any(axis=1).sum())
    inf_vector_count = int(np.isinf(result.vectors).any(axis=1).sum())
    invalid_norm_count = int((np.abs(norms - 1.0) > NORM_ATOL).sum())
    chunk_ids = list(result.chunk_ids)
    expected_ids = [leaf.chunk_id for leaf in leaves]
    missing_vector_count = len(set(expected_ids) - set(chunk_ids))
    duplicate_chunk_id_count = len(chunk_ids) - len(set(chunk_ids))
    device, gpu_name = _device_report(config.device)
    spotcheck = []
    for index in spot_indices:
        leaf = leaves[index]
        spotcheck.append(
            {
                "chunk_id": leaf.chunk_id,
                "document_id": leaf.document_id,
                "chunk_index": leaf.chunk_index,
                "page_start": leaf.page_start,
                "page_end": leaf.page_end,
                "token_count": a5_tokens[index],
                "a3_token_count": a3_tokens[index],
                "vector_dimension": int(result.vectors.shape[1]),
                "vector_norm": float(norms[index]),
                "first8": _first_eight(result.vectors[index]),
            }
        )
    return {
        "model_name": result.model_name,
        "transformers_version": transformers_version(),
        "torch_version": torch.__version__,
        "device": device,
        "gpu_name": gpu_name,
        "forward_dtype": "float16",
        "max_input_tokens": config.max_input_tokens,
        "batch_size": config.batch_size,
        "chunk_size": chunk_size,
        "overlap_tokens": overlap_tokens,
        "leaf_count": len(leaves),
        "embedding_count": int(result.vectors.shape[0]),
        "embedding_dimension": int(result.dimension),
        "embedding_dtype": str(result.vectors.dtype),
        "input_token_min": min(a5_tokens) if a5_tokens else None,
        "input_token_p50": _percentile(a5_tokens, 50),
        "input_token_p95": _percentile(a5_tokens, 95),
        "input_token_max": max(a5_tokens) if a5_tokens else None,
        "over_limit_leaf_count": sum(
            1 for length in a5_tokens if length > config.max_input_tokens
        ),
        "nan_vector_count": nan_vector_count,
        "inf_vector_count": inf_vector_count,
        "invalid_norm_count": invalid_norm_count,
        "duplicate_chunk_id_count": duplicate_chunk_id_count,
        "missing_vector_count": missing_vector_count,
        "vector_norm_min": float(norms.min()) if len(norms) else None,
        "vector_norm_max": float(norms.max()) if len(norms) else None,
        "vector_norm_mean": float(norms.mean()) if len(norms) else None,
        "determinism_failure_count": determinism_failure_count,
        "spotcheck": spotcheck,
    }


def format_embedding_summary(report: dict[str, Any]) -> str:
    lines = [
        f"model_name={report['model_name']}",
        f"transformers_version={report['transformers_version']}",
        f"torch_version={report['torch_version']}",
        f"device={report['device']}",
        f"gpu_name={report['gpu_name']}",
        f"forward_dtype={report['forward_dtype']}",
        f"max_input_tokens={report['max_input_tokens']}",
        f"batch_size={report['batch_size']}",
        f"leaf_count={report['leaf_count']}",
        f"embedding_count={report['embedding_count']}",
        f"embedding_dimension={report['embedding_dimension']}",
        f"embedding_dtype={report['embedding_dtype']}",
        f"input_token_min={report['input_token_min']}",
        f"input_token_p50={report['input_token_p50']}",
        f"input_token_p95={report['input_token_p95']}",
        f"input_token_max={report['input_token_max']}",
        f"over_limit_leaf_count={report['over_limit_leaf_count']}",
        f"nan_vector_count={report['nan_vector_count']}",
        f"inf_vector_count={report['inf_vector_count']}",
        f"invalid_norm_count={report['invalid_norm_count']}",
        f"duplicate_chunk_id_count={report['duplicate_chunk_id_count']}",
        f"missing_vector_count={report['missing_vector_count']}",
        f"vector_norm_min={report['vector_norm_min']}",
        f"vector_norm_max={report['vector_norm_max']}",
        f"vector_norm_mean={report['vector_norm_mean']}",
        f"determinism_failure_count={report['determinism_failure_count']}",
        "spotcheck:",
    ]
    for row in report["spotcheck"]:
        lines.append(
            "  "
            f"chunk_id={row['chunk_id']} "
            f"document_id={row['document_id']} "
            f"chunk_index={row['chunk_index']} "
            f"page={row['page_start']}-{row['page_end']} "
            f"tokens={row['token_count']} "
            f"dim={row['vector_dimension']} "
            f"norm={row['vector_norm']:.6f} "
            f"first8={row['first8']}"
        )
    return "\n".join(lines)


def _run_determinism(
    leaves: tuple[Leaf, ...],
    indices: list[int],
    *,
    config: DenseEmbeddingConfig,
    encoder: Qwen3DenseEncoder,
) -> int:
    """Re-embed the same subset twice. Do not compare against the full corpus batch.

    A 10-leaf batch has different left-padding geometry than those same rows
    inside the 271-leaf run. FP16 GPU differences across batch shapes are not
    a contract failure. The A5 contract only requires two identical subset runs.
    """

    subset = tuple(leaves[index] for index in indices)
    run1 = embed_leaves(subset, config=config, encoder=encoder)
    run2 = embed_leaves(subset, config=config, encoder=encoder)
    if np.allclose(
        run1.vectors, run2.vectors, rtol=DETERMINISM_RTOL, atol=DETERMINISM_ATOL
    ):
        return 0
    return 1


def save_embeddings_npz(path: Path, result: DenseEmbeddingResult) -> None:
    np.savez(
        path,
        chunk_ids=np.array(result.chunk_ids),
        vectors=result.vectors,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A5 Dense Embedding. Reuses A1 loading and A3 Leaf chunking. "
            "Embeds Leaf.content only. Does not modify Leaf, A4 hierarchy, "
            "or write a production A6 store."
        ),
        epilog=(
            "From the repository root, put code/ on PYTHONPATH, for example:\n"
            '  PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" '
            "python -m knowledge_base.dense_embedding "
            "--canonical-root <ABS_PATH> "
            "--model Qwen/Qwen3-Embedding-4B "
            "--device cuda "
            "--output /tmp/a5-embeddings.npz "
            "--report /tmp/a5-embedding-report.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Formal A0 output root containing <document_id>/document.md",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help=(
            "Hugging Face id or local directory for Qwen3-Embedding-4B "
            f"(default {DEFAULT_MODEL_NAME_OR_PATH})"
        ),
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"cuda / cuda:N, or cpu for explicit tests (default {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Deployment batch size (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_MAX_INPUT_TOKENS,
        help=f"Inference token budget (default {DEFAULT_MAX_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"A3 Leaf soft limit used only to rebuild Leaves (default {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help=(
            "A3 overlap budget used only to rebuild Leaves "
            f"(default {DEFAULT_OVERLAP_TOKENS})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="npz smoke artifact path; not an A6 production store",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="JSON smoke/acceptance report path",
    )
    args = parser.parse_args(argv)

    config = DenseEmbeddingConfig(
        model_name_or_path=args.model,
        max_input_tokens=args.max_input_tokens,
        batch_size=args.batch_size,
        device=args.device,
    )
    try:
        documents = load_markdown_documents(args.canonical_root)
        a3_tokenizer = load_tokenizer(args.model)
        chunking = chunk_documents(
            documents,
            a3_tokenizer,
            chunk_size=args.chunk_size,
            overlap_tokens=args.overlap,
        )
        leaves = chunking.leaves
        encoder = Qwen3DenseEncoder(config)
        result = embed_leaves(leaves, config=config, encoder=encoder)
        a3_tokens = [count_tokens(leaf.content, a3_tokenizer) for leaf in leaves]
        spot_indices = _select_spotcheck(leaves, a3_tokens, args.chunk_size)
        determinism_failure_count = _run_determinism(
            leaves,
            spot_indices,
            config=config,
            encoder=encoder,
        )
        report = build_embedding_report(
            leaves,
            result,
            encoder,
            config=config,
            a3_tokenizer=a3_tokenizer,
            chunk_size=args.chunk_size,
            overlap_tokens=args.overlap,
            determinism_failure_count=determinism_failure_count,
            spot_indices=spot_indices,
        )
        save_embeddings_npz(args.output, result)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (
        MarkdownLoadingError,
        A3ChunkingError,
        A5EmbeddingError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"canonical_root={args.canonical_root.resolve()}")
    print(f"output={args.output.resolve()}")
    print(f"report={args.report.resolve()}")
    print(format_embedding_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
