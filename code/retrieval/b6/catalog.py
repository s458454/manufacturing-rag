"""Leaf bodies for B6, projected from the A6 rows JSONL.

``dense_vector`` is discarded. This file is a production dependency of B6,
not a disposable build scratch file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from retrieval.b6.errors import LeafCatalogError

_LEAF_FIELDS = (
    "chunk_id",
    "document_id",
    "section_id",
    "chunk_index",
    "page_start",
    "page_end",
    "content",
)


@dataclass(frozen=True)
class CatalogLeaf:
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str


class LeafCatalog:
    """In-memory leaf index used for identity checks and ±1 windows."""

    def __init__(self, leaves: tuple[CatalogLeaf, ...]) -> None:
        self.leaves = leaves
        by_id: dict[str, CatalogLeaf] = {}
        by_section: dict[tuple[str, str], dict[int, CatalogLeaf]] = {}
        for leaf in leaves:
            if leaf.chunk_id in by_id:
                raise LeafCatalogError(f"duplicate chunk_id: {leaf.chunk_id}")
            by_id[leaf.chunk_id] = leaf
            section_key = (leaf.document_id, leaf.section_id)
            indexes = by_section.setdefault(section_key, {})
            if leaf.chunk_index in indexes:
                raise LeafCatalogError(
                    "duplicate chunk_index "
                    f"{leaf.chunk_index} in section_id={leaf.section_id}"
                )
            indexes[leaf.chunk_index] = leaf
        self._by_id = by_id
        self._by_section = by_section

    def get(self, chunk_id: str) -> CatalogLeaf:
        leaf = self._by_id.get(chunk_id)
        if leaf is None:
            raise LeafCatalogError(f"chunk_id is not in the leaf catalog: {chunk_id}")
        return leaf

    def window_chunks(self, leaf: CatalogLeaf) -> tuple[CatalogLeaf, ...]:
        """Same-section chunk_index ± 1, including the matched leaf."""

        indexes = self._by_section.get((leaf.document_id, leaf.section_id), {})
        chosen: list[CatalogLeaf] = []
        for chunk_index in (leaf.chunk_index - 1, leaf.chunk_index, leaf.chunk_index + 1):
            neighbor = indexes.get(chunk_index)
            if neighbor is not None:
                chosen.append(neighbor)
        return tuple(chosen)


def load_leaf_catalog(path: Path) -> LeafCatalog:
    if not path.is_file():
        raise LeafCatalogError(f"rows jsonl does not exist: {path}")
    leaves: list[CatalogLeaf] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LeafCatalogError(f"invalid JSON in {path} line {line_number}") from exc
        if not isinstance(row, dict):
            raise LeafCatalogError(f"leaf row is not an object in line {line_number}")
        missing = [name for name in _LEAF_FIELDS if name not in row]
        if missing:
            raise LeafCatalogError(
                f"leaf row line {line_number} is missing {missing}"
            )
        chunk_index = row["chunk_index"]
        page_start = row["page_start"]
        page_end = row["page_end"]
        if (
            isinstance(chunk_index, bool)
            or isinstance(page_start, bool)
            or isinstance(page_end, bool)
            or not isinstance(chunk_index, int)
            or not isinstance(page_start, int)
            or not isinstance(page_end, int)
        ):
            raise LeafCatalogError(f"leaf row line {line_number} has a non-integer index or page")
        content = row["content"]
        if not isinstance(content, str):
            raise LeafCatalogError(f"leaf content must be a string in line {line_number}")
        for name in ("chunk_id", "document_id", "section_id"):
            if not isinstance(row[name], str) or not row[name]:
                raise LeafCatalogError(
                    f"{name} must be a non-empty string in line {line_number}"
                )
        leaves.append(
            CatalogLeaf(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                section_id=row["section_id"],
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                content=content,
            )
        )
    return LeafCatalog(tuple(leaves))
