"""Centralized Qwen3 tokenizer loading and token counting for A3.1 profiling.

All Section/Table token lengths must go through ``count_tokens``. Approximate
counts (``len(text)``, ``split()``, ``len(text) / 4``) are forbidden.
"""

from __future__ import annotations

from typing import Any

_MIN_TRANSFORMERS = (4, 51, 0)


class SectionProfileError(Exception):
    """Fail-fast error for A3.1 structure parse / tokenizer profiling."""


def _parse_version_tuple(value: str) -> tuple[int, int, int]:
    parts: list[int] = []
    for chunk in value.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits) if digits else 0)
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def transformers_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        PackageNotFoundError = Exception  # type: ignore[misc,assignment]
        version = None  # type: ignore[assignment]
    if version is not None:
        try:
            return version("transformers")
        except PackageNotFoundError:
            pass
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return "unknown"


def require_transformers_version() -> str:
    detected = transformers_version()
    if _parse_version_tuple(detected) < _MIN_TRANSFORMERS:
        raise SectionProfileError(
            "transformers>=4.51.0 is required for the Qwen3 tokenizer; "
            f"found {detected}"
        )
    return detected


def load_tokenizer(tokenizer_id: str) -> Any:
    """Load a tokenizer from a Hugging Face id or local path.

    The identifier is caller-supplied. This function does not hard-code a
    server filesystem path. Only the tokenizer is loaded, not embedding weights.
    """

    identifier = tokenizer_id.strip()
    if not identifier:
        raise SectionProfileError("Tokenizer identifier is empty")
    require_transformers_version()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SectionProfileError(
            "transformers is required to load the A3 profiling tokenizer"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(identifier)
    except Exception as exc:
        raise SectionProfileError(
            f"Failed to load tokenizer {identifier!r}"
        ) from exc


def count_tokens(text: str, tokenizer: Any) -> int:
    """Return the token length of *text* without special tokens or truncation."""

    try:
        encoded = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )
    except TypeError:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def tokenize_with_offsets(
    text: str, tokenizer: Any
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return token ids and character offsets for tokenizer-window fallback.

    Uses the same tokenizer object as ``count_tokens``. Content is later sliced
    from the original source; decoded text is never used as Leaf.content.
    """

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
    except TypeError as exc:
        raise SectionProfileError(
            "Tokenizer does not support offset mapping required for "
            "oversize-block sliding split"
        ) from exc
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
        offsets = offsets[0]
    pairs: list[tuple[int, int]] = []
    for item in offsets:
        pairs.append((int(item[0]), int(item[1])))
    return [int(token_id) for token_id in ids], pairs
