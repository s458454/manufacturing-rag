"""B5 Qwen3-Reranker-0.6B. Official token template; silent truncation is OFF."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from retrieval.b4.types import FusionResult
from retrieval.b5.constants import (
    NO_TOKEN,
    RERANK_INPUT_LIMIT,
    RERANK_OUTPUT_LIMIT,
    RERANKER_CONTRACT_VERSION,
    RERANKER_MAX_INPUT_TOKENS,
    RERANKER_MODEL,
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
    YES_TOKEN,
)
from retrieval.b5.errors import (
    RerankerError,
    RerankerInferenceError,
    RerankerInputTooLongError,
    RerankerModelError,
)
from retrieval.b5.observability import emit_reranker_observation
from retrieval.b5.template import format_rerank_body
from retrieval.b5.types import (
    RerankerConfig,
    RerankerObservation,
    RerankingResult,
    fused_to_reranked,
)


def _cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _encode_unterminated(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer.encode(
            text, add_special_tokens=False, truncation=False
        )
    except TypeError:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


class Qwen3LeafReranker:
    """Leaf Cross-Encoder. Inject tokenizer/model in tests; use load_production for CUDA."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        batch_size: int,
        model_name: str = RERANKER_MODEL,
        max_input_tokens: int = RERANKER_MAX_INPUT_TOKENS,
    ) -> None:
        if batch_size <= 0:
            raise RerankerModelError("reranker_batch_size must be positive")
        padding_side = getattr(tokenizer, "padding_side", None)
        if padding_side != "left":
            raise RerankerModelError(
                f"tokenizer padding_side must be 'left', got {padding_side!r}"
            )
        self.tokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.token_true_id = tokenizer.convert_tokens_to_ids(YES_TOKEN)
        self.token_false_id = tokenizer.convert_tokens_to_ids(NO_TOKEN)
        if self.token_true_id is None or self.token_false_id is None:
            raise RerankerModelError("tokenizer is missing yes/no token ids")
        eval_fn = getattr(model, "eval", None)
        if callable(eval_fn):
            eval_fn()
        self.prefix_tokens = _encode_unterminated(tokenizer, RERANKER_PREFIX)
        self.suffix_tokens = _encode_unterminated(tokenizer, RERANKER_SUFFIX)

    def tokenize_pair(self, query: str, document: str) -> list[int]:
        body = format_rerank_body(query, document)
        body_tokens = _encode_unterminated(self.tokenizer, body)
        token_ids = self.prefix_tokens + body_tokens + self.suffix_tokens
        if len(token_ids) > self.max_input_tokens:
            raise RerankerInputTooLongError(
                "reranker pair exceeds max_input_tokens: "
                f"length={len(token_ids)} max={self.max_input_tokens}"
            )
        return token_ids

    def rerank(
        self,
        fusion: FusionResult,
        *,
        observer: Callable[[RerankerObservation], None] | None = None,
    ) -> RerankingResult:
        started = time.perf_counter()
        input_count = len(fusion.candidates)
        if input_count > RERANK_INPUT_LIMIT:
            exc = RerankerError(
                "B5 received more than rerank_input_limit candidates: "
                f"{input_count} > {RERANK_INPUT_LIMIT}"
            )
            self._emit_failed(fusion, started, input_count, exc, observer)
            raise exc
        if input_count == 0:
            emit_reranker_observation(
                RerankerObservation(
                    request_id=fusion.request_id,
                    task_id=fusion.task_id,
                    reranker_contract_version=RERANKER_CONTRACT_VERSION,
                    reranker_model=self.model_name,
                    reranker_input_count=0,
                    reranker_output_count=0,
                    max_pair_tokens=0,
                    rerank_latency_ms=(time.perf_counter() - started) * 1000.0,
                    rerank_status="empty",
                    error_type=None,
                ),
                observer,
            )
            return RerankingResult(
                request_id=fusion.request_id,
                task_id=fusion.task_id,
                query=fusion.query,
                candidates=(),
            )
        try:
            pair_ids = [
                self.tokenize_pair(fusion.query, candidate.content)
                for candidate in fusion.candidates
            ]
            max_pair_tokens = max(len(ids) for ids in pair_ids)
            scores = self._score_batches(pair_ids)
            ranked = sorted(
                zip(fusion.candidates, scores, strict=True),
                key=lambda item: (-item[1], item[0].fused_rank),
            )
            top = ranked[:RERANK_OUTPUT_LIMIT]
            candidates = tuple(
                fused_to_reranked(
                    candidate, rerank_score=float(score), rerank_rank=index
                )
                for index, (candidate, score) in enumerate(top, start=1)
            )
        except RerankerError as exc:
            self._emit_failed(fusion, started, input_count, exc, observer)
            raise
        except Exception as exc:
            wrapped = RerankerInferenceError(f"reranker inference failed: {exc}")
            self._emit_failed(fusion, started, input_count, wrapped, observer)
            raise wrapped from exc

        emit_reranker_observation(
            RerankerObservation(
                request_id=fusion.request_id,
                task_id=fusion.task_id,
                reranker_contract_version=RERANKER_CONTRACT_VERSION,
                reranker_model=self.model_name,
                reranker_input_count=input_count,
                reranker_output_count=len(candidates),
                max_pair_tokens=max_pair_tokens,
                rerank_latency_ms=(time.perf_counter() - started) * 1000.0,
                rerank_status="success",
                error_type=None,
            ),
            observer,
        )
        return RerankingResult(
            request_id=fusion.request_id,
            task_id=fusion.task_id,
            query=fusion.query,
            candidates=candidates,
        )

    def _score_batches(self, pair_ids: Sequence[list[int]]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pair_ids), self.batch_size):
            batch = list(pair_ids[start : start + self.batch_size])
            scores.extend(self._forward(batch))
        if len(scores) != len(pair_ids):
            raise RerankerInferenceError(
                "reranker score count mismatch: "
                f"expected {len(pair_ids)}, got {len(scores)}"
            )
        return scores

    def _forward(self, batch_ids: list[list[int]]) -> list[float]:
        import torch
        import torch.nn.functional as F

        padded = self._pad(batch_ids, torch_module=torch)
        try:
            with torch.inference_mode():
                outputs = self.model(**padded)
                logits = outputs.logits[:, -1, :]
                true_vector = logits[:, self.token_true_id]
                false_vector = logits[:, self.token_false_id]
                stacked = torch.stack([false_vector, true_vector], dim=1)
                log_probs = F.log_softmax(stacked.float(), dim=1)
                scores = log_probs[:, 1].exp().tolist()
        except Exception as exc:
            if _cuda_oom(exc):
                raise RerankerInferenceError(
                    "CUDA OOM during rerank forward; automatic batch fallback is forbidden"
                ) from exc
            raise RerankerInferenceError(
                f"reranker forward failed: {exc}"
            ) from exc
        return [float(score) for score in scores]

    def _pad(self, batch_ids: list[list[int]], *, torch_module: Any) -> dict[str, Any]:
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)
        width = max(len(ids) for ids in batch_ids)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for ids in batch_ids:
            gap = width - len(ids)
            input_ids.append([int(pad_id)] * gap + ids)
            attention_mask.append([0] * gap + [1] * len(ids))
        device = _model_device(self.model)
        tensors = {
            "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
            "attention_mask": torch_module.tensor(
                attention_mask, dtype=torch_module.long
            ),
        }
        if device is not None:
            tensors = {key: value.to(device) for key, value in tensors.items()}
        return tensors

    def _emit_failed(
        self,
        fusion: FusionResult,
        started: float,
        input_count: int,
        exc: RerankerError,
        observer: Callable[[RerankerObservation], None] | None,
    ) -> None:
        emit_reranker_observation(
            RerankerObservation(
                request_id=fusion.request_id,
                task_id=fusion.task_id,
                reranker_contract_version=RERANKER_CONTRACT_VERSION,
                reranker_model=self.model_name,
                reranker_input_count=input_count,
                reranker_output_count=None,
                max_pair_tokens=None,
                rerank_latency_ms=(time.perf_counter() - started) * 1000.0,
                rerank_status="failed",
                error_type=exc.error_type,
            ),
            observer,
        )


def _model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            return None
    return None


def load_production_reranker(config: RerankerConfig) -> Qwen3LeafReranker:
    """Load Qwen3-Reranker-0.6B on CUDA. CPU fallback is forbidden."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RerankerModelError("B5 requires CUDA; CPU fallback is forbidden")
    pretrained_kwargs: dict[str, str] = {}
    local_path = Path(config.model_name_or_path)
    if not local_path.exists():
        pretrained_kwargs["revision"] = config.model_revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
            **pretrained_kwargs,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            **pretrained_kwargs,
        )
        model.to("cuda")
        model.eval()
    except RerankerError:
        raise
    except Exception as exc:
        if _cuda_oom(exc):
            raise RerankerModelError(
                "CUDA OOM while loading Qwen3-Reranker-0.6B; "
                "automatic model fallback is forbidden"
            ) from exc
        raise RerankerModelError(
            f"failed to load Qwen3-Reranker-0.6B: {exc}"
        ) from exc
    reranker = Qwen3LeafReranker(
        tokenizer=tokenizer,
        model=model,
        batch_size=config.batch_size,
        model_name=RERANKER_MODEL,
    )
    reranker._forward([reranker.tokenize_pair("warm", "warm")])
    return reranker


def rerank(
    fusion: FusionResult,
    reranker: Qwen3LeafReranker,
    *,
    observer: Callable[[RerankerObservation], None] | None = None,
) -> RerankingResult:
    return reranker.rerank(fusion, observer=observer)
