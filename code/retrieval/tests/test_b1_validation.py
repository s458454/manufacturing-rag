"""B1.0 query validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.constants import (  # noqa: E402
    MAX_QUERY_LENGTH,
    USER_REPLY_EMPTY_QUERY,
    USER_REPLY_INVALID_QUERY,
    USER_REPLY_QUERY_TOO_LONG,
)
from retrieval.b1.errors import (  # noqa: E402
    EmptyQueryError,
    InvalidQueryError,
    QueryTooLongError,
)
from retrieval.b1.types import QueryRequest  # noqa: E402
from retrieval.b1.validation import query_length, validate_query_request  # noqa: E402


def test_strip_is_the_only_normalization() -> None:
    text = validate_query_request(
        QueryRequest(request_id="r", conversation_id="c", query="  NASA STD 5006A  ")
    )
    assert text == "NASA STD 5006A"


def test_empty_query_has_frozen_reply() -> None:
    with pytest.raises(EmptyQueryError) as exc:
        validate_query_request(QueryRequest(request_id="r", conversation_id="c", query="   "))
    assert exc.value.user_reply == USER_REPLY_EMPTY_QUERY
    assert exc.value.error_type == "empty_query"


def test_non_str_query_is_invalid() -> None:
    with pytest.raises(InvalidQueryError) as exc:
        validate_query_request(
            QueryRequest(request_id="r", conversation_id="c", query=None)  # type: ignore[arg-type]
        )
    assert exc.value.user_reply == USER_REPLY_INVALID_QUERY


def test_illegal_control_characters_are_rejected() -> None:
    with pytest.raises(InvalidQueryError):
        validate_query_request(
            QueryRequest(request_id="r", conversation_id="c", query="hello\x00world")
        )
    with pytest.raises(InvalidQueryError):
        validate_query_request(
            QueryRequest(request_id="r", conversation_id="c", query="hello\x7f")
        )
    text = validate_query_request(
        QueryRequest(request_id="r", conversation_id="c", query="line1\nline2\tkeep")
    )
    assert text == "line1\nline2\tkeep"


def test_weighted_length_boundary() -> None:
    allowed = "x" * MAX_QUERY_LENGTH
    assert query_length(allowed) == MAX_QUERY_LENGTH
    assert (
        validate_query_request(
            QueryRequest(request_id="r", conversation_id="c", query=allowed)
        )
        == allowed
    )
    with pytest.raises(QueryTooLongError) as exc:
        validate_query_request(
            QueryRequest(request_id="r", conversation_id="c", query=allowed + "y")
        )
    assert exc.value.user_reply == USER_REPLY_QUERY_TOO_LONG


def test_cjk_wide_character_weight_is_four() -> None:
    assert query_length("焊") == 4
    too_long = "焊" * 501
    assert query_length(too_long) == 2004
    with pytest.raises(QueryTooLongError):
        validate_query_request(
            QueryRequest(request_id="r", conversation_id="c", query=too_long)
        )
