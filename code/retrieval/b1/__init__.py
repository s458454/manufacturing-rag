"""B1 shell plus the Jev production router."""

from retrieval.b1.config import load_router_config
from retrieval.b1.constants import (
    MAX_ROUTER_MEMORY_TOKENS,
    ROUTER_CONTEXT_LIMIT,
    ROUTER_CONTRACT_VERSION,
    ROUTER_MAX_NEW_TOKENS,
    ROUTER_NON_HISTORY_RESERVE,
    ROUTER_OUTPUT_AND_SAFETY_RESERVE,
    ROUTER_PROMPT_VERSION,
    ROUTER_SCHEMA_VERSION,
)
from retrieval.b1.dispatcher import (
    dispatch_knowledge_tasks,
    knowledge_processing_requests,
    router_mode_user_reply,
    system_task_ids,
    unsupported_user_reply,
)
from retrieval.b1.errors import (
    EmptyQueryError,
    InvalidQueryError,
    QueryTooLongError,
    RewriteRejected,
    RouterConfigurationError,
    RouterError,
    RouterProtocolError,
    RouterUnavailableError,
)
from retrieval.b1.memory import InMemoryConversationStore, commit_turn
from retrieval.b1.pipeline import build_jev_router, route_for_b2
from retrieval.b1.types import (
    ConversationTurn,
    QueryRequest,
    RoutedTask,
    RouterConfig,
    RouterObservation,
    RouterResult,
    SystemHandler,
)
from retrieval.b1_jev.router import JevQueryRouter

__all__ = [
    "MAX_ROUTER_MEMORY_TOKENS",
    "ROUTER_CONTEXT_LIMIT",
    "ROUTER_CONTRACT_VERSION",
    "ROUTER_MAX_NEW_TOKENS",
    "ROUTER_NON_HISTORY_RESERVE",
    "ROUTER_OUTPUT_AND_SAFETY_RESERVE",
    "ROUTER_PROMPT_VERSION",
    "ROUTER_SCHEMA_VERSION",
    "ConversationTurn",
    "EmptyQueryError",
    "InMemoryConversationStore",
    "InvalidQueryError",
    "JevQueryRouter",
    "QueryRequest",
    "QueryTooLongError",
    "RewriteRejected",
    "RoutedTask",
    "RouterConfig",
    "RouterConfigurationError",
    "RouterError",
    "RouterObservation",
    "RouterProtocolError",
    "RouterResult",
    "RouterUnavailableError",
    "SystemHandler",
    "build_jev_router",
    "commit_turn",
    "dispatch_knowledge_tasks",
    "knowledge_processing_requests",
    "load_router_config",
    "route_for_b2",
    "router_mode_user_reply",
    "system_task_ids",
    "unsupported_user_reply",
]
