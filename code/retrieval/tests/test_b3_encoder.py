"""B3 DenseQueryEncoder adapter: serialize then A5 encode_documents."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b3.encoder import DenseQueryEncoder  # noqa: E402
from retrieval.b3.errors import DenseQueryEncodingError  # noqa: E402
from retrieval.b3.serialize import serialize_query  # noqa: E402
from retrieval.tests.helpers import fake_dense_vector  # noqa: E402


class _RecordingEncoder:
    def __init__(self, vector: tuple[float, ...] | None = None) -> None:
        self.texts: list[str] = []
        self.vector = vector or fake_dense_vector()

    def encode_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.texts.extend(texts)
        return [self.vector]


def test_adapter_sends_serialized_query_to_encode_documents() -> None:
    base = _RecordingEncoder()
    encoder = DenseQueryEncoder(base)
    query = "查 NASA-STD-5006A 焊接要求"
    vector = encoder.encode(query)
    assert base.texts == [serialize_query(query)]
    assert query not in base.texts
    assert len(vector) == 2560
    assert all(item == 0.01 for item in vector)


def test_adapter_does_not_call_private_encode_batch() -> None:
    class Guard:
        def encode_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
            return [fake_dense_vector()]

        def _encode_batch(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("must not call _encode_batch")

    DenseQueryEncoder(Guard()).encode("q")


def test_wrong_dimension_is_encoding_error() -> None:
    class Short:
        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0]]

    with pytest.raises(DenseQueryEncodingError, match="2560"):
        DenseQueryEncoder(Short()).encode("q")


def test_adapter_wraps_base_encoder_failures() -> None:
    class Boom:
        def encode_documents(self, texts: list[str]) -> list[tuple[float, ...]]:
            raise RuntimeError("gpu down")

    with pytest.raises(DenseQueryEncodingError, match="gpu down"):
        DenseQueryEncoder(Boom()).encode("q")
