"""Production retrieval package: B1–B6.

B6 context recovery runs after B5. Callers must not recover Parent context
before B5 reranking.
"""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "QueryProcessingContractError": "b2.errors",
    "QueryProcessingObservation": "b2.types",
    "QueryProcessingRequest": "b2.types",
    "ProcessedQuery": "b2.types",
    "QUERY_PROCESSOR_CONTRACT_VERSION": "b2.processor",
    "QUERY_PROCESSOR_VERSION": "b2.processor",
    "process_query": "b2.processor",
    "ConversationTurn": "b1.types",
    "QueryRequest": "b1.types",
    "RoutedTask": "b1.types",
    "RouterConfig": "b1.types",
    "RouterObservation": "b1.types",
    "RouterResult": "b1.types",
    "SystemHandler": "b1.types",
    "EmptyQueryError": "b1.errors",
    "InvalidQueryError": "b1.errors",
    "QueryTooLongError": "b1.errors",
    "RewriteRejected": "b1.errors",
    "RouterConfigurationError": "b1.errors",
    "RouterProtocolError": "b1.errors",
    "RouterUnavailableError": "b1.errors",
    "RouterError": "b1.errors",
    "InMemoryConversationStore": "b1.memory",
    "commit_turn": "b1.memory",
    "JevQueryRouter": "b1_jev.router",
    "build_jev_router": "b1.pipeline",
    "route_for_b2": "b1.pipeline",
    "dispatch_knowledge_tasks": "b1.dispatcher",
    "serialize_query": "b3.serialize",
    "DenseQueryEncoder": "b3.encoder",
    "run_hybrid_search": "b3.retrieve",
    "FusionResult": "b4.types",
    "FusedCandidate": "b4.types",
    "rerank": "b5.reranker",
    "Qwen3LeafReranker": "b5.reranker",
    "RerankingResult": "b5.types",
    "recover": "b6.recover",
    "RecoveredEvidence": "b6.types",
    "RecoveryResult": "b6.types",
    "load_recovery_context": "b6.context",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(f"{__name__}.{module_name}", fromlist=[name])
    return getattr(module, name)
