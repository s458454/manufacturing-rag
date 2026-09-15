"""B1.0 input validation. The only allowed text normalization is strip()."""

from __future__ import annotations

import unicodedata

from retrieval.b1.constants import MAX_QUERY_LENGTH
from retrieval.b1.errors import EmptyQueryError, InvalidQueryError, QueryTooLongError
from retrieval.b1.types import QueryRequest


def query_length(text: str) -> int:
    total = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ("W", "F"):
            total += 4
        else:
            total += 1
    return total


def contains_invalid_control_char(text: str) -> bool:
    for char in text:
        code = ord(char)
        if code == 0x7F:
            return True
        if code < 0x20 and char not in ("\t", "\n", "\r"):
            return True
    return False


def validate_query_text(query: object) -> str:
    if not isinstance(query, str):
        raise InvalidQueryError("query must be a str")
    query_text = query.strip()
    if query_text == "":
        raise EmptyQueryError()
    if contains_invalid_control_char(query_text):
        raise InvalidQueryError("query contains invalid control characters")
    if query_length(query_text) > MAX_QUERY_LENGTH:
        raise QueryTooLongError()
    return query_text


def validate_query_request(request: QueryRequest) -> str:
    if not isinstance(request.request_id, str) or request.request_id == "":
        raise InvalidQueryError("request_id must be a non-empty str")
    if not isinstance(request.conversation_id, str) or request.conversation_id == "":
        raise InvalidQueryError("conversation_id must be a non-empty str")
    return validate_query_text(request.query)
