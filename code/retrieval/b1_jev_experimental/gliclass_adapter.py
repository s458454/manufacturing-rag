"""Local GLiClass adapter. Route-only. Not imported by QueryRouter."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from retrieval.b1.types import ConversationTurn
from retrieval.b1_jev_experimental.constants import (
    GLICLASS_DEFAULT_MODEL_PATH,
    GLICLASS_MODEL_ENV,
    KNOWLEDGE_MODE_KEYS,
    ROUTE_KEYS,
)
from retrieval.b1_jev_experimental.errors import JevProtocolError, JevUnavailableError
from retrieval.b1_jev_experimental.policy import parse_jev_answers, parse_jev_mode_answer
from retrieval.b1_jev_experimental.questions import JEV_MODE_QUESTIONS, JEV_QUESTIONS
from retrieval.b1_jev_experimental.types import JevAnswers, JevModeAnswer


def resolve_gliclass_model_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = str(env.get(GLICLASS_MODEL_ENV, "")).strip() or GLICLASS_DEFAULT_MODEL_PATH
    path = Path(raw)
    if not path.is_dir():
        raise JevUnavailableError(f"GLiClass model directory not found: {path}")
    if not (path / "model.safetensors").is_file() and not (path / "pytorch_model.bin").is_file():
        raise JevUnavailableError(f"GLiClass weights missing under {path}")
    return path


def serialize_classifier_text(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
) -> str:
    if not isinstance(current_query, str) or current_query == "":
        raise ValueError("current_query must be a non-empty str")
    lines = ["Current query:", current_query]
    if history:
        lines.append("Conversation history (oldest first):")
        for index, turn in enumerate(history, start=1):
            lines.append(f"{index}. user_raw: {turn.raw_user_query}")
            lines.append(f"{index}. user_canonical: {turn.canonical_user_query}")
            lines.append(f"{index}. assistant: {turn.assistant_response}")
    else:
        lines.append("Conversation history: (empty)")
    return "\n".join(lines)


def _score_map(rows: object) -> dict[str, float]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise JevProtocolError("GLiClass output must be a list of scored labels")
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise JevProtocolError("GLiClass label row must be an object")
        label = row.get("label")
        score = row.get("score")
        if not isinstance(label, str) or not isinstance(score, (int, float)):
            raise JevProtocolError("GLiClass row missing label or score")
        scores[label] = float(score)
    if not scores:
        raise JevProtocolError("GLiClass returned no label scores")
    return scores


def _argmax(scores: Mapping[str, float], keys: Sequence[str]) -> tuple[str, float]:
    present = {key: scores[key] for key in keys if key in scores}
    if not present:
        raise JevProtocolError("GLiClass scores missing required labels")
    winner = max(present, key=present.get)
    mass = sum(present.values())
    confidence = present[winner] / mass if mass > 0 else present[winner]
    return winner, confidence


def answers_from_gliclass_scores(*, primary_route: object) -> JevAnswers:
    route_key, route_conf = _argmax(_score_map(primary_route), ROUTE_KEYS)
    return parse_jev_answers(
        {
            "primary_route": route_key,
            "primary_route_confidence": route_conf,
        }
    )


def answers_from_gliclass_mode_scores(*, knowledge_mode: object) -> JevModeAnswer:
    mode_key, mode_conf = _argmax(_score_map(knowledge_mode), KNOWLEDGE_MODE_KEYS)
    return parse_jev_mode_answer(
        {
            "knowledge_mode": mode_key,
            "knowledge_mode_confidence": mode_conf,
        }
    )


def _run(pipeline: object, text: str, labels: Sequence[str]) -> object:
    raw = pipeline(text, list(labels), threshold=0.0)
    if isinstance(raw, Sequence) and raw and isinstance(raw[0], Sequence):
        return raw[0]
    return raw


def decide_with_gliclass(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
    *,
    model_path: str | Path | None = None,
    device: str | None = None,
) -> JevAnswers:
    try:
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise JevUnavailableError(
            "gliclass/torch/transformers are required for local GLiClass smoke"
        ) from exc

    path = (
        resolve_gliclass_model_path({GLICLASS_MODEL_ENV: str(model_path)})
        if model_path is not None
        else resolve_gliclass_model_path()
    )
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(str(path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    pipeline = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="single-label",
        device=device,
    )
    spec = JEV_QUESTIONS["primary_route"]
    text = f"Task: {spec['instructions']}\n\n{serialize_classifier_text(current_query, history)}"
    return answers_from_gliclass_scores(
        primary_route=_run(pipeline, text, list(ROUTE_KEYS)),
    )


def decide_knowledge_mode_with_gliclass(
    current_query: str,
    history: Sequence[ConversationTurn] = (),
    *,
    model_path: str | Path | None = None,
    device: str | None = None,
) -> JevModeAnswer:
    try:
        import torch
        from gliclass import GLiClassModel, ZeroShotClassificationPipeline
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise JevUnavailableError(
            "gliclass/torch/transformers are required for local GLiClass smoke"
        ) from exc

    path = (
        resolve_gliclass_model_path({GLICLASS_MODEL_ENV: str(model_path)})
        if model_path is not None
        else resolve_gliclass_model_path()
    )
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GLiClassModel.from_pretrained(str(path), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    pipeline = ZeroShotClassificationPipeline(
        model,
        tokenizer,
        classification_type="single-label",
        device=device,
    )
    spec = JEV_MODE_QUESTIONS["knowledge_mode"]
    text = f"Task: {spec['instructions']}\n\n{serialize_classifier_text(current_query, history)}"
    return answers_from_gliclass_mode_scores(
        knowledge_mode=_run(pipeline, text, list(KNOWLEDGE_MODE_KEYS)),
    )
