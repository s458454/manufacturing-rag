"""Deterministic rewrite-confirmation handler. No LLM."""

from __future__ import annotations

from retrieval.b1.constants import (
    CONFIRM_WORDS,
    REJECT_WORDS,
    TRAILING_CONFIRMATION_PUNCTUATION,
)
from retrieval.b1.errors import RewriteRejected


def normalize_confirmation_text(text: str) -> str:
    stripped = text.strip()
    lowered = "".join(char.lower() if char.isascii() else char for char in stripped)
    if lowered.endswith(TRAILING_CONFIRMATION_PUNCTUATION):
        lowered = lowered[:-1]
    return lowered


def handle_rewrite_confirmation(user_text: str, pending_rewrite: str) -> str | None:
    """Return the confirmed candidate, raise on reject, or None to treat as a new query."""

    normalized = normalize_confirmation_text(user_text)
    if normalized in CONFIRM_WORDS:
        return pending_rewrite
    if normalized in REJECT_WORDS:
        raise RewriteRejected()
    return None
