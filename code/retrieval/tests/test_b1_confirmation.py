"""Rewrite confirmation: exact lexicon, no LLM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.confirmation import handle_rewrite_confirmation  # noqa: E402
from retrieval.b1.constants import USER_REPLY_REWRITE_REJECTED  # noqa: E402
from retrieval.b1.errors import RewriteRejected  # noqa: E402

PENDING = "NASA-STD-5006A 对钢材焊接有什么要求？"


def test_confirm_words_return_pending_rewrite() -> None:
    assert handle_rewrite_confirmation("是", PENDING) == PENDING
    assert handle_rewrite_confirmation(" Yes! ", PENDING) == PENDING
    assert handle_rewrite_confirmation("对，就这样。", PENDING) == PENDING


def test_reject_words_raise_frozen_reply() -> None:
    with pytest.raises(RewriteRejected) as exc:
        handle_rewrite_confirmation("不", PENDING)
    assert exc.value.user_reply == USER_REPLY_REWRITE_REJECTED
    with pytest.raises(RewriteRejected):
        handle_rewrite_confirmation("cancel", PENDING)


def test_other_text_is_a_new_query() -> None:
    assert handle_rewrite_confirmation("那铝呢？", PENDING) is None
