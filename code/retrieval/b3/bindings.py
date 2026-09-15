"""A6 Leaf field bindings consumed by B3/B4. This is not a second schema."""

from __future__ import annotations

from knowledge_base.a6_collection import (
    DENSE_METRIC_TYPE,
    FIELD_CHUNK_ID,
    FIELD_CHUNK_INDEX,
    FIELD_CONTENT,
    FIELD_DENSE_VECTOR,
    FIELD_DOCUMENT_ID,
    FIELD_PAGE_END,
    FIELD_PAGE_START,
    FIELD_SECTION_ID,
    FIELD_SPARSE_VECTOR,
    SPARSE_METRIC_TYPE,
)

LEAF_OUTPUT_FIELDS: tuple[str, ...] = (
    FIELD_CHUNK_ID,
    FIELD_DOCUMENT_ID,
    FIELD_SECTION_ID,
    FIELD_CHUNK_INDEX,
    FIELD_PAGE_START,
    FIELD_PAGE_END,
    FIELD_CONTENT,
)

FORBIDDEN_SEARCH_FILTER_KEYS: frozenset[str] = frozenset(
    {"filter", "expr", "status", "expr_params"}
)
