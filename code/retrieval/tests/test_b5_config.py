"""B5 config: model path required; batch size defaults to 8."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b5.config import load_reranker_config  # noqa: E402
from retrieval.b5.constants import RERANKER_REVISION  # noqa: E402
from retrieval.b5.errors import RerankerConfigurationError  # noqa: E402

MANIFEST = (
    Path(__file__).resolve().parents[3] / "models" / "retrieval-models.manifest.json"
)


def test_model_path_is_required() -> None:
    with pytest.raises(RerankerConfigurationError, match="B5_MODEL_NAME_OR_PATH"):
        load_reranker_config({})
    config = load_reranker_config({"B5_MODEL_NAME_OR_PATH": "/models/Qwen3-Reranker-0.6B"})
    assert config.batch_size == 8
    assert config.model_revision == RERANKER_REVISION


def test_manifest_pins_reranker_revision() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    reranker = next(
        asset for asset in payload["assets"] if asset["name"] == "Qwen3-Reranker-0.6B"
    )
    assert reranker["revision"] == RERANKER_REVISION
    assert reranker["logical_id"] == "Qwen/Qwen3-Reranker-0.6B"
