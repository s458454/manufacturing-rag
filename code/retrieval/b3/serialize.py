"""Dense query instruction serialization. Does not rewrite the B2 query."""

from retrieval.b3.constants import DENSE_INSTRUCTION


def serialize_query(query: str) -> str:
    """Return the frozen Qwen3-Embedding query serialization.

    The instruction is Dense-only. Callers must not send this text to BM25.
    """

    if not isinstance(query, str) or query == "":
        raise ValueError("query must be a non-empty str")
    return (
        "Instruct: Given a manufacturing and engineering query, "
        "retrieve relevant passages from technical documents that contain "
        "the information needed to answer the query.\n"
        f"Query:{query}"
    )


def dense_instruction() -> str:
    return DENSE_INSTRUCTION
