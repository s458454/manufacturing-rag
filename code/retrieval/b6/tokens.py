"""C2 Qwen2.5-7B-Instruct token counting for evidence bodies.

Counts raw evidence text. Chat-template accounting belongs to C1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retrieval.b6.constants import (
    GENERATION_TOKENIZER_REVISION,
)
from retrieval.b6.errors import RecoveryConfigurationError


def count_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return len(encoded)


def load_generation_tokenizer(name_or_path: str) -> Any:
    """Load the pinned local Qwen2.5 artifact. Do not fetch Hugging Face latest."""

    identifier = name_or_path.strip()
    if not identifier:
        raise RecoveryConfigurationError("generation tokenizer path is empty")
    path = Path(identifier)
    if not path.exists():
        raise RecoveryConfigurationError(
            "Qwen2.5-7B-Instruct local artifact is missing at "
            f"{identifier!r}. Install revision {GENERATION_TOKENIZER_REVISION}. "
            "Runtime must not fetch Hugging Face latest."
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RecoveryConfigurationError(
            "transformers is required to load the generation tokenizer"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            identifier,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RecoveryConfigurationError(
            f"Failed to load generation tokenizer {identifier!r} "
            f"revision {GENERATION_TOKENIZER_REVISION}"
        ) from exc
