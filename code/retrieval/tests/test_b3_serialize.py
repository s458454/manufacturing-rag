"""B3 Dense query serialization: instruction is Dense-only."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b3.serialize import serialize_query  # noqa: E402


def test_serialize_query_uses_frozen_instruction_and_newline() -> None:
    query = "NASA-STD-5006A 对铝合金焊接有什么要求？"
    serialized = serialize_query(query)
    assert serialized == (
        "Instruct: Given a manufacturing and engineering query, "
        "retrieve relevant passages from technical documents that contain "
        "the information needed to answer the query.\n"
        f"Query:{query}"
    )
    assert serialized.startswith("Instruct: ")
    assert "\nQuery:" in serialized
    assert serialized.endswith(query)


def test_serialize_query_does_not_strip_or_rewrite() -> None:
    query = "  6061 vs 7075\n"
    serialized = serialize_query(query)
    assert serialized.endswith(query)
    assert "6061-T6" not in serialized
