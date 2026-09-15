"""Official Qwen3-Reranker pair serialization. Truncation is forbidden."""

from retrieval.b5.constants import (
    RERANKER_INSTRUCTION,
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
)


def format_rerank_body(query: str, document: str) -> str:
    return (
        f"<Instruct>: {RERANKER_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def format_rerank_pair(query: str, document: str) -> str:
    return RERANKER_PREFIX + format_rerank_body(query, document) + RERANKER_SUFFIX
