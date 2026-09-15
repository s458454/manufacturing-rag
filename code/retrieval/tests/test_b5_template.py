"""B5 official Qwen3-Reranker template. Instruction is frozen English."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b5.constants import (  # noqa: E402
    RERANKER_INSTRUCTION,
    RERANKER_PREFIX,
    RERANKER_REVISION,
    RERANKER_SUFFIX,
)
from retrieval.b5.template import format_rerank_body, format_rerank_pair  # noqa: E402


def test_official_prefix_and_suffix_tokens() -> None:
    assert RERANKER_PREFIX.startswith("<|im_start|>system\n")
    assert 'Note that the answer can only be "yes" or "no".' in RERANKER_PREFIX
    assert RERANKER_PREFIX.endswith("<|im_start|>user\n")
    assert RERANKER_SUFFIX == (
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_body_uses_official_instruct_query_document_tags() -> None:
    body = format_rerank_body("q1", "leaf")
    assert body == (
        f"<Instruct>: {RERANKER_INSTRUCTION}\n"
        "<Query>: q1\n"
        "<Document>: leaf"
    )
    pair = format_rerank_pair("q1", "leaf")
    assert pair.startswith(RERANKER_PREFIX)
    assert pair.endswith(RERANKER_SUFFIX)
    assert "<Instruct>:" in pair
    assert pair.count("<|im_start|>") == 3


def test_reranker_revision_is_pinned() -> None:
    assert RERANKER_REVISION == "e61197ed45024b0ed8a2d74b80b4d909f1255473"
