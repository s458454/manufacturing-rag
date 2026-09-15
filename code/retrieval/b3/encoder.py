"""B3 Dense query adapter. Reuses A5 public encode_documents(); no copy of pooling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from knowledge_base.embedding_config import EMBEDDING_DIMENSION

from retrieval.b3.errors import DenseQueryEncodingError
from retrieval.b3.serialize import serialize_query


class DocumentEncoder(Protocol):
    def encode_documents(self, texts: Sequence[str]) -> Any:
        """Encode texts with the A5 numerical contract. Adds no instruction."""


def _as_row_vector(raw: Any) -> tuple[float, ...]:
    if hasattr(raw, "shape"):
        shape = tuple(int(dim) for dim in raw.shape)
        if len(shape) == 2:
            if shape[0] != 1 or shape[1] != EMBEDDING_DIMENSION:
                raise DenseQueryEncodingError(
                    "query encoder must return one 2560-d vector, "
                    f"got shape {shape}"
                )
            raw = raw[0]
        elif len(shape) != 1 or shape[0] != EMBEDDING_DIMENSION:
            raise DenseQueryEncodingError(
                "query encoder must return one 2560-d vector, "
                f"got shape {shape}"
            )
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise DenseQueryEncodingError("query vector is not a 1-d sequence")
    values = [float(item) for item in raw]
    if len(values) != EMBEDDING_DIMENSION:
        raise DenseQueryEncodingError(
            "query vector dimension mismatch: "
            f"expected {EMBEDDING_DIMENSION}, got {len(values)}"
        )
    if not all(item == item and item not in (float("inf"), float("-inf")) for item in values):
        raise DenseQueryEncodingError("query vector contains NaN or Inf")
    return tuple(values)


class DenseQueryEncoder:
    """Serialize a B2 query, then reuse the A5 public encoder core."""

    def __init__(self, base_encoder: DocumentEncoder) -> None:
        self.base_encoder = base_encoder

    def encode(self, query: str) -> tuple[float, ...]:
        text = serialize_query(query)
        try:
            vectors = self.base_encoder.encode_documents([text])
        except DenseQueryEncodingError:
            raise
        except Exception as exc:
            a5_cls: type[Exception] | tuple[()] = ()
            try:
                from knowledge_base.dense_embedding import A5EmbeddingError

                a5_cls = A5EmbeddingError
            except ImportError:
                a5_cls = ()
            if a5_cls and isinstance(exc, a5_cls):
                raise DenseQueryEncodingError(str(exc) or exc.__class__.__name__) from exc
            raise DenseQueryEncodingError(
                f"dense query encoding failed: {exc}"
            ) from exc
        if hasattr(vectors, "shape"):
            return _as_row_vector(vectors)
        if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
            raise DenseQueryEncodingError("query encoder returned a non-sequence")
        if len(vectors) != 1:
            raise DenseQueryEncodingError(
                "query encoder must return exactly one vector, "
                f"got {len(vectors)}"
            )
        return _as_row_vector(vectors[0])
