"""B1 fail-closed errors. User-facing replies are frozen contract text."""

from __future__ import annotations

from retrieval.b1.constants import (
    USER_REPLY_EMPTY_QUERY,
    USER_REPLY_INVALID_QUERY,
    USER_REPLY_QUERY_TOO_LONG,
    USER_REPLY_REWRITE_REJECTED,
    USER_REPLY_ROUTER_PROTOCOL,
    USER_REPLY_ROUTER_UNAVAILABLE,
)


class RouterError(Exception):
    """Base B1 error."""

    error_type: str = "router_error"
    user_reply: str | None = None

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.error_type)


class InvalidQueryError(RouterError):
    error_type = "invalid_query"
    user_reply = USER_REPLY_INVALID_QUERY


class EmptyQueryError(RouterError):
    error_type = "empty_query"
    user_reply = USER_REPLY_EMPTY_QUERY


class QueryTooLongError(RouterError):
    error_type = "query_too_long"
    user_reply = USER_REPLY_QUERY_TOO_LONG


class RouterUnavailableError(RouterError):
    error_type = "router_unavailable"
    user_reply = USER_REPLY_ROUTER_UNAVAILABLE


class RouterProtocolError(RouterError):
    error_type = "router_protocol_error"
    user_reply = USER_REPLY_ROUTER_PROTOCOL


class RouterConfigurationError(RouterError):
    error_type = "router_configuration_error"
    user_reply = None


class RewriteRejected(Exception):
    """User rejected a pending context rewrite. Not a Router model error."""

    user_reply = USER_REPLY_REWRITE_REJECTED

    def __init__(self) -> None:
        super().__init__(self.user_reply)
