"""Deterministic section_id / chunk_id for A3.2 Leaf identity."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_hex(canonical: str) -> str:
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def make_section_id(
    *,
    document_id: str,
    kind: str,
    heading_level: int | None,
    source_start: int,
) -> str:
    canonical = _canonical_json(
        {
            "document_id": document_id,
            "heading_level": heading_level,
            "kind": kind,
            "source_start": source_start,
        }
    )
    return "sec_" + _sha256_hex(canonical)


def make_chunk_id(
    *,
    document_id: str,
    section_id: str,
    chunk_index: int,
    page_start: int,
    page_end: int,
    content: str,
) -> str:
    canonical = _canonical_json(
        {
            "chunk_index": chunk_index,
            "content": content,
            "document_id": document_id,
            "page_end": page_end,
            "page_start": page_start,
            "section_id": section_id,
        }
    )
    return "chk_" + _sha256_hex(canonical)
