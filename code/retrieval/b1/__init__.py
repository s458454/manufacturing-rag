"""B1 Query Router."""

from retrieval.b1.client import VLLMRouterClient
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
from retrieval.b1.router import QueryRouter
from retrieval.b1.types import (
    ConversationTurn,
    QueryRequest,
    RoutedTask,
    RouterConfig,
    RouterObservation,
    RouterResult,
    SystemHandler,
)

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
    "QueryRequest",
    "QueryRouter",
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
    "VLLMRouterClient",
    "commit_turn",
    "dispatch_knowledge_tasks",
    "knowledge_processing_requests",
    "load_router_config",
    "router_mode_user_reply",
    "system_task_ids",
    "unsupported_user_reply",
]
