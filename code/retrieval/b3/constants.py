"""B3 frozen limits and Dense query serialization. Not algorithm optima."""

from __future__ import annotations

RETRIEVAL_CONTRACT_VERSION = "b3-contract-v1"
DENSE_QUERY_SERIALIZER_VERSION = "b3-dense-query-v1"

DENSE_CANDIDATE_LIMIT = 20
BM25_CANDIDATE_LIMIT = 20

DENSE_INSTRUCTION = (
    "Given a manufacturing and engineering query, retrieve relevant passages "
    "from technical documents that contain the information needed to answer "
    "the query."
)

DENSE_QUERY_MAX_INPUT_TOKENS = 8192
DENSE_QUERY_DIMENSION = 2560
