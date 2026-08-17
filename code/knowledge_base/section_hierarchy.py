"""A4 hierarchical Parent-Child Section tree.

Reuses A3.1 ``ParsedMarkdownDocument`` and A3.2 ``SectionRef`` / ``Leaf``.
Does not re-parse Markdown, rewrite A3 identity, or select B6 ancestors.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_base.chunking_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_TOKENS,
)
from knowledge_base.leaf_chunker import (
    A3ChunkingError,
    ChunkingResult,
    Leaf,
    SectionRef,
    chunk_parsed_documents,
)
from knowledge_base.markdown_loader import (
    LoadedMarkdownDocument,
    MarkdownLoadingError,
    load_markdown_documents,
)
from knowledge_base.structure_parser import (
    Heading,
    ParsedMarkdownDocument,
    line_starts,
    page_range_for_span,
    parse_markdown_document,
    section_semantic_end,
    strip_page_markers,
)
from knowledge_base.token_count import (
    SectionProfileError,
    load_tokenizer,
    transformers_version,
)

SECTION_NODE_FIELD_NAMES = (
    "section_id",
    "document_id",
    "parent_section_id",
    "kind",
    "heading_level",
    "heading",
    "source_start",
    "source_end",
    "page_start",
    "page_end",
)

_FORMAL_PAGE_MARKER = re.compile(r"^<!-- PDF page ([1-9][0-9]*) -->$")
_HEADING_LEVELS = set(range(1, 7))


class A4HierarchyError(Exception):
    """Fail-fast error for A4 hierarchy / recovery."""


@dataclass(frozen=True)
class SectionNode:
    section_id: str
    document_id: str
    parent_section_id: str | None
    kind: str
    heading_level: int | None
    heading: str | None
    source_start: int
    source_end: int
    page_start: int
    page_end: int


@dataclass(frozen=True)
class _DocumentRecoveryView:
    document_id: str
    line_count: int
    marker_lines: frozenset[int]


class SectionHierarchy:
    """In-memory Section tree. Not a production persistence format."""

    def __init__(
        self,
        nodes: tuple[SectionNode, ...],
        children_by_parent: dict[tuple[str, str | None], tuple[str, ...]],
        recovery_by_document: dict[str, _DocumentRecoveryView],
    ) -> None:
        self.nodes = nodes
        self._nodes_by_id = {node.section_id: node for node in nodes}
        self._children_by_parent = children_by_parent
        self._recovery_by_document = recovery_by_document

    def get_section(self, section_id: str) -> SectionNode:
        node = self._nodes_by_id.get(section_id)
        if node is None:
            raise A4HierarchyError(f"Unknown section_id: {section_id}")
        return node

    def get_parent(self, section_id: str) -> SectionNode | None:
        node = self.get_section(section_id)
        if node.parent_section_id is None:
            return None
        return self.get_section(node.parent_section_id)

    def get_ancestors(self, section_id: str) -> tuple[SectionNode, ...]:
        self.get_section(section_id)
        ancestors: list[SectionNode] = []
        seen: set[str] = set()
        current_id = self.get_section(section_id).parent_section_id
        while current_id is not None:
            if current_id in seen:
                raise A4HierarchyError(
                    f"hierarchy cycle involving section_id={current_id}"
                )
            seen.add(current_id)
            parent = self.get_section(current_id)
            ancestors.append(parent)
            current_id = parent.parent_section_id
        return tuple(ancestors)

    def child_section_ids(
        self, document_id: str, parent_section_id: str | None
    ) -> tuple[str, ...]:
        return self._children_by_parent.get((document_id, parent_section_id), ())

    def recover_section_text(
        self,
        section_id: str,
        documents: list[LoadedMarkdownDocument],
    ) -> str:
        node = self.get_section(section_id)
        document = _document_by_id(documents, node.document_id)
        view = self._recovery_by_document.get(node.document_id)
        if view is None:
            raise A4HierarchyError(
                "No recovery view for "
                f"document_id={node.document_id}"
            )
        starts = line_starts(document.content)
        if node.source_start < 0 or node.source_end > len(starts):
            raise A4HierarchyError(
                "source span exceeds document in "
                f"section_id={node.section_id} "
                f"span=[{node.source_start}, {node.source_end}) "
                f"line_count={len(starts)}"
            )
        return strip_page_markers(
            document.content,
            starts,
            node.source_start,
            node.source_end,
            set(view.marker_lines),
        )


def build_section_hierarchy(
    documents: list[LoadedMarkdownDocument],
    parsed_documents: list[ParsedMarkdownDocument],
    chunking_result: ChunkingResult,
) -> SectionHierarchy:
    """Build a multi-level Section hierarchy from A3 identity + parse."""

    if not documents:
        raise A4HierarchyError("No documents provided")
    if not parsed_documents:
        raise A4HierarchyError("No parsed documents provided")
    if not chunking_result.leaves or not chunking_result.sections:
        raise A4HierarchyError("Empty chunking_result")
    if len(documents) != len(parsed_documents):
        raise A4HierarchyError(
            "documents and parsed_documents length mismatch: "
            f"{len(documents)} != {len(parsed_documents)}"
        )

    documents_by_id = _unique_documents(documents)
    parsed_by_id = _unique_parsed(parsed_documents)
    if set(documents_by_id) != set(parsed_by_id):
        raise A4HierarchyError(
            "document_id set mismatch between documents and parsed_documents"
        )
    for document_id, document in documents_by_id.items():
        parsed = parsed_by_id[document_id]
        if len(line_starts(document.content)) != parsed.line_count:
            raise A4HierarchyError(
                "parsed line_count does not match document content in "
                f"document_id={document_id}"
            )

    section_doc_ids = {ref.document_id for ref in chunking_result.sections}
    leaf_doc_ids = {leaf.document_id for leaf in chunking_result.leaves}
    if section_doc_ids != set(documents_by_id):
        raise A4HierarchyError(
            "SectionRef.document_id set does not match documents"
        )
    if not leaf_doc_ids <= set(documents_by_id):
        raise A4HierarchyError("Leaf.document_id is not in documents")

    refs_by_id: dict[str, SectionRef] = {}
    refs_by_document: dict[str, list[SectionRef]] = {
        document_id: [] for document_id in documents_by_id
    }
    for ref in chunking_result.sections:
        if ref.section_id in refs_by_id:
            raise A4HierarchyError(
                f"Duplicate section_id: {ref.section_id}"
            )
        if ref.document_id not in documents_by_id:
            raise A4HierarchyError(
                "SectionRef.document_id does not exist: "
                f"{ref.document_id}"
            )
        refs_by_id[ref.section_id] = ref
        refs_by_document[ref.document_id].append(ref)

    nodes: list[SectionNode] = []
    children_acc: dict[tuple[str, str | None], list[str]] = {}
    recovery_by_document: dict[str, _DocumentRecoveryView] = {}

    for document in documents:
        parsed = parsed_by_id[document.document_id]
        doc_nodes = _nodes_for_document(
            parsed, refs_by_document[document.document_id]
        )
        recovery_by_document[document.document_id] = _DocumentRecoveryView(
            document_id=document.document_id,
            line_count=parsed.line_count,
            marker_lines=frozenset(
                marker.line for marker in parsed.page_markers
            ),
        )
        for node in doc_nodes:
            nodes.append(node)
            key = (node.document_id, node.parent_section_id)
            children_acc.setdefault(key, []).append(node.section_id)

    if len(nodes) != len(chunking_result.sections):
        raise A4HierarchyError(
            "A4 node count does not match A3 SectionRef count: "
            f"{len(nodes)} != {len(chunking_result.sections)}"
        )

    nodes_by_id = {node.section_id: node for node in nodes}
    _validate_parent_links(nodes_by_id)
    _validate_leaves(chunking_result.leaves, nodes_by_id)

    return SectionHierarchy(
        nodes=tuple(nodes),
        children_by_parent={
            key: tuple(value) for key, value in children_acc.items()
        },
        recovery_by_document=recovery_by_document,
    )


def _unique_documents(
    documents: list[LoadedMarkdownDocument],
) -> dict[str, LoadedMarkdownDocument]:
    by_id: dict[str, LoadedMarkdownDocument] = {}
    for document in documents:
        if document.document_id in by_id:
            raise A4HierarchyError(
                f"Duplicate document_id: {document.document_id}"
            )
        by_id[document.document_id] = document
    return by_id


def _unique_parsed(
    parsed_documents: list[ParsedMarkdownDocument],
) -> dict[str, ParsedMarkdownDocument]:
    by_id: dict[str, ParsedMarkdownDocument] = {}
    for parsed in parsed_documents:
        if parsed.document_id in by_id:
            raise A4HierarchyError(
                f"Duplicate parsed document_id: {parsed.document_id}"
            )
        by_id[parsed.document_id] = parsed
    return by_id


def _document_by_id(
    documents: list[LoadedMarkdownDocument], document_id: str
) -> LoadedMarkdownDocument:
    matches = [
        document
        for document in documents
        if document.document_id == document_id
    ]
    if not matches:
        raise A4HierarchyError(
            f"document_id not found for recovery: {document_id}"
        )
    if len(matches) > 1:
        raise A4HierarchyError(f"Duplicate document_id: {document_id}")
    return matches[0]


def _nodes_for_document(
    parsed: ParsedMarkdownDocument,
    refs: list[SectionRef],
) -> list[SectionNode]:
    heading_refs = [ref for ref in refs if ref.kind == "heading"]
    root_refs = [ref for ref in refs if ref.kind == "document_root"]
    other = [
        ref for ref in refs if ref.kind not in {"heading", "document_root"}
    ]
    if other:
        kinds = sorted({ref.kind for ref in other})
        raise A4HierarchyError(
            f"Unsupported SectionRef.kind in document_id={parsed.document_id}: "
            f"{kinds}"
        )
    if len(root_refs) > 1:
        raise A4HierarchyError(
            "Multiple document_root SectionRef in "
            f"document_id={parsed.document_id}"
        )
    if len(heading_refs) != len(parsed.headings):
        raise A4HierarchyError(
            "heading SectionRef count does not match parsed headings in "
            f"document_id={parsed.document_id}: "
            f"{len(heading_refs)} != {len(parsed.headings)}"
        )

    heading_by_start = _heading_index(parsed.headings)
    for ref in heading_refs:
        heading = heading_by_start.get(ref.source_start)
        if (
            heading is None
            or heading.level != ref.heading_level
            or heading.text != ref.heading_text
        ):
            raise A4HierarchyError(
                "SectionRef does not join parsed heading in "
                f"document_id={parsed.document_id} "
                f"section_id={ref.section_id} source_start={ref.source_start}"
            )
    parents = _heading_parent_ids(parsed, heading_refs, heading_by_start)
    nodes_by_ref_id: dict[str, SectionNode] = {}

    if root_refs:
        nodes_by_ref_id[root_refs[0].section_id] = _document_root_node(
            parsed, root_refs[0]
        )

    for ref in heading_refs:
        heading = heading_by_start.get(ref.source_start)
        if heading is None:
            raise A4HierarchyError(
                "SectionRef does not join parsed heading in "
                f"document_id={parsed.document_id} "
                f"section_id={ref.section_id} source_start={ref.source_start}"
            )
        if heading.level != ref.heading_level or heading.text != ref.heading_text:
            raise A4HierarchyError(
                "SectionRef heading identity mismatch in "
                f"document_id={parsed.document_id} "
                f"section_id={ref.section_id}"
            )
        index = parsed.headings.index(heading)
        source_start = heading.start_line
        source_end = section_semantic_end(
            parsed.headings, index, parsed.line_count
        )
        _validate_span(parsed, ref.section_id, source_start, source_end)
        page_start, page_end = _section_pages(
            parsed, ref.section_id, source_start, source_end
        )
        if ref.heading_level not in _HEADING_LEVELS:
            raise A4HierarchyError(
                "Illegal heading_level in "
                f"document_id={parsed.document_id} "
                f"section_id={ref.section_id}: {ref.heading_level}"
            )
        if ref.heading_text is None:
            raise A4HierarchyError(
                "heading kind requires heading text in "
                f"section_id={ref.section_id}"
            )
        nodes_by_ref_id[ref.section_id] = SectionNode(
            section_id=ref.section_id,
            document_id=parsed.document_id,
            parent_section_id=parents[ref.section_id],
            kind="heading",
            heading_level=ref.heading_level,
            heading=ref.heading_text,
            source_start=source_start,
            source_end=source_end,
            page_start=page_start,
            page_end=page_end,
        )

    return [nodes_by_ref_id[ref.section_id] for ref in refs]


def _heading_index(headings: tuple[Heading, ...]) -> dict[int, Heading]:
    by_start: dict[int, Heading] = {}
    for heading in headings:
        if heading.start_line in by_start:
            raise A4HierarchyError(
                "Duplicate heading start_line "
                f"{heading.start_line} text={heading.text!r}"
            )
        by_start[heading.start_line] = heading
    return by_start


def _heading_parent_ids(
    parsed: ParsedMarkdownDocument,
    heading_refs: list[SectionRef],
    heading_by_start: dict[int, Heading],
) -> dict[str, str | None]:
    ordered = sorted(
        heading_refs,
        key=lambda ref: heading_by_start[ref.source_start].start_line,
    )
    stack: list[tuple[int, str]] = []
    parents: dict[str, str | None] = {}
    for ref in ordered:
        heading = heading_by_start[ref.source_start]
        if heading.level not in _HEADING_LEVELS:
            raise A4HierarchyError(
                "Illegal heading_level in "
                f"document_id={parsed.document_id}: {heading.level}"
            )
        while stack and stack[-1][0] >= heading.level:
            stack.pop()
        parents[ref.section_id] = stack[-1][1] if stack else None
        stack.append((heading.level, ref.section_id))
    return parents


def _document_root_node(
    parsed: ParsedMarkdownDocument, ref: SectionRef
) -> SectionNode:
    if ref.heading_level is not None:
        raise A4HierarchyError(
            "document_root heading_level must be None in "
            f"section_id={ref.section_id}"
        )
    if ref.heading_text not in {"", None}:
        raise A4HierarchyError(
            "document_root must not have heading text in "
            f"section_id={ref.section_id}"
        )
    if ref.source_start != 0:
        raise A4HierarchyError(
            "document_root source_start must be 0 in "
            f"section_id={ref.section_id}"
        )
    if parsed.unheaded_document_body or not parsed.headings:
        source_end = parsed.line_count
    else:
        source_end = parsed.headings[0].start_line
    _validate_span(parsed, ref.section_id, 0, source_end)
    page_start, page_end = _section_pages(parsed, ref.section_id, 0, source_end)
    return SectionNode(
        section_id=ref.section_id,
        document_id=parsed.document_id,
        parent_section_id=None,
        kind="document_root",
        heading_level=None,
        heading=None,
        source_start=0,
        source_end=source_end,
        page_start=page_start,
        page_end=page_end,
    )


def _validate_span(
    parsed: ParsedMarkdownDocument,
    section_id: str,
    source_start: int,
    source_end: int,
) -> None:
    if source_start < 0:
        raise A4HierarchyError(
            f"source_start < 0 in section_id={section_id}: {source_start}"
        )
    if source_end <= source_start:
        raise A4HierarchyError(
            "invalid source span in "
            f"section_id={section_id}: [{source_start}, {source_end})"
        )
    if source_end > parsed.line_count:
        raise A4HierarchyError(
            "source span exceeds document in "
            f"section_id={section_id}: [{source_start}, {source_end}) "
            f"line_count={parsed.line_count}"
        )


def _section_pages(
    parsed: ParsedMarkdownDocument,
    section_id: str,
    source_start: int,
    source_end: int,
) -> tuple[int, int]:
    if not parsed.page_markers:
        raise A4HierarchyError(
            "page provenance cannot be derived for "
            f"section_id={section_id}: no PDF page markers"
        )
    try:
        page_start, page_end = page_range_for_span(
            parsed.page_markers, source_start, source_end
        )
    except (IndexError, ValueError) as exc:
        raise A4HierarchyError(
            "page provenance cannot be derived for "
            f"section_id={section_id}"
        ) from exc
    if page_start < 1 or page_end < page_start:
        raise A4HierarchyError(
            "invalid page provenance for "
            f"section_id={section_id}: {page_start}-{page_end}"
        )
    return page_start, page_end


def _validate_parent_links(nodes_by_id: dict[str, SectionNode]) -> None:
    for node in nodes_by_id.values():
        if node.parent_section_id is None:
            continue
        parent = nodes_by_id.get(node.parent_section_id)
        if parent is None:
            raise A4HierarchyError(
                "parent_section_id does not exist: "
                f"{node.parent_section_id} (child={node.section_id})"
            )
        if parent.document_id != node.document_id:
            raise A4HierarchyError(
                "cross-document parent in "
                f"child={node.section_id} parent={parent.section_id}"
            )
        if (
            node.kind == "heading"
            and parent.kind == "heading"
            and parent.heading_level is not None
            and node.heading_level is not None
            and parent.heading_level >= node.heading_level
        ):
            raise A4HierarchyError(
                "parent heading_level >= child heading_level in "
                f"child={node.section_id} parent={parent.section_id}"
            )
    for node in nodes_by_id.values():
        seen: set[str] = set()
        current_id: str | None = node.section_id
        while current_id is not None:
            if current_id in seen:
                raise A4HierarchyError(
                    f"hierarchy cycle involving section_id={current_id}"
                )
            seen.add(current_id)
            current = nodes_by_id.get(current_id)
            if current is None:
                raise A4HierarchyError(
                    f"parent_section_id does not exist: {current_id}"
                )
            current_id = current.parent_section_id


def _validate_leaves(
    leaves: tuple[Leaf, ...],
    nodes_by_id: dict[str, SectionNode],
) -> None:
    for leaf in leaves:
        node = nodes_by_id.get(leaf.section_id)
        if node is None:
            raise A4HierarchyError(
                "Leaf.section_id has no SectionNode: "
                f"{leaf.section_id} chunk_id={leaf.chunk_id}"
            )
        if node.document_id != leaf.document_id:
            raise A4HierarchyError(
                "Leaf/Section document_id mismatch: "
                f"chunk_id={leaf.chunk_id} leaf={leaf.document_id} "
                f"section={node.document_id}"
            )


def parse_loaded_documents(
    documents: list[LoadedMarkdownDocument],
) -> list[ParsedMarkdownDocument]:
    """Parse each loaded document once for shared A3/A4 use."""

    if not documents:
        raise A4HierarchyError("No documents provided")
    parsed_documents: list[ParsedMarkdownDocument] = []
    seen: set[str] = set()
    try:
        for document in documents:
            if document.document_id in seen:
                raise A4HierarchyError(
                    f"Duplicate document_id: {document.document_id}"
                )
            seen.add(document.document_id)
            parsed_documents.append(
                parse_markdown_document(
                    document.content, document_id=document.document_id
                )
            )
    except SectionProfileError as exc:
        raise A4HierarchyError(str(exc)) from exc
    return parsed_documents


def build_hierarchy_from_documents(
    documents: list[LoadedMarkdownDocument],
    tokenizer: Any,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> tuple[
    list[ParsedMarkdownDocument],
    ChunkingResult,
    SectionHierarchy,
]:
    parsed_documents = parse_loaded_documents(documents)
    try:
        chunking_result = chunk_parsed_documents(
            documents,
            parsed_documents,
            tokenizer,
            chunk_size=chunk_size,
            overlap_tokens=overlap_tokens,
        )
    except A3ChunkingError as exc:
        raise A4HierarchyError(str(exc)) from exc
    hierarchy = build_section_hierarchy(
        documents, parsed_documents, chunking_result
    )
    return parsed_documents, chunking_result, hierarchy


def _percentile(values: list[int], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percent / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return round(ordered[low] * (1.0 - weight) + ordered[high] * weight, 6)


def _text_preview(text: str, limit: int = 160) -> str:
    preview = text.strip().replace("\n", " ")
    if len(preview) > limit:
        return preview[:limit] + "..."
    return preview


def _has_pdf_page_marker(text: str) -> bool:
    for line in text.splitlines():
        if _FORMAL_PAGE_MARKER.fullmatch(line.strip()) is not None:
            return True
    return False


def _leaf_counts(leaves: tuple[Leaf, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for leaf in leaves:
        counts[leaf.section_id] = counts.get(leaf.section_id, 0) + 1
    return counts


def _node_payload(
    node: SectionNode,
    *,
    direct_leaf_count: int | None = None,
    child_count: int | None = None,
    text_preview: str | None = None,
) -> dict[str, Any]:
    payload = {name: getattr(node, name) for name in SECTION_NODE_FIELD_NAMES}
    if direct_leaf_count is not None:
        payload["direct_leaf_count"] = direct_leaf_count
    if child_count is not None:
        payload["child_count"] = child_count
    if text_preview is not None:
        payload["text_preview"] = text_preview
    return payload


def _select_spotcheck_nodes(
    hierarchy: SectionHierarchy,
    leaves: tuple[Leaf, ...],
) -> list[SectionNode]:
    leaf_counts = _leaf_counts(leaves)
    selected: list[SectionNode] = []
    seen: set[str] = set()

    def add(candidates: list[SectionNode], limit: int) -> None:
        added = 0
        for node in candidates:
            if node.section_id in seen:
                continue
            selected.append(node)
            seen.add(node.section_id)
            added += 1
            if added >= limit:
                return

    top_level = [
        node
        for node in hierarchy.nodes
        if node.parent_section_id is None
    ]
    h2h3h4 = [
        node
        for node in hierarchy.nodes
        if node.kind == "heading"
        and node.heading_level == 4
        and {ancestor.heading_level for ancestor in hierarchy.get_ancestors(node.section_id)}
        >= {2, 3}
    ]
    jumps = [
        node
        for node in hierarchy.nodes
        if node.kind == "heading"
        and hierarchy.get_parent(node.section_id) is not None
        and hierarchy.get_parent(node.section_id).kind == "heading"
        and hierarchy.get_parent(node.section_id).heading_level
        < node.heading_level - 1
    ]
    cross_page = [
        node
        for node in hierarchy.nodes
        if node.page_end > node.page_start
    ]
    empty_parent = [
        node
        for node in hierarchy.nodes
        if node.kind == "heading"
        and leaf_counts.get(node.section_id, 0) == 0
        and hierarchy.child_section_ids(node.document_id, node.section_id)
    ]
    roots = [node for node in hierarchy.nodes if node.kind == "document_root"]

    add(top_level, 2)
    add(h2h3h4, 2)
    add(jumps, 2)
    add(cross_page, 2)
    add(empty_parent, 1)
    add(roots, 1)
    add(list(hierarchy.nodes), 10)
    return selected[:10]


def _manual_recovery_target(
    hierarchy: SectionHierarchy,
) -> SectionNode | None:
    for node in hierarchy.nodes:
        if node.kind != "heading" or node.heading_level != 2:
            continue
        child_ids = hierarchy.child_section_ids(
            node.document_id, node.section_id
        )
        h3 = [
            hierarchy.get_section(child_id)
            for child_id in child_ids
            if hierarchy.get_section(child_id).kind == "heading"
            and hierarchy.get_section(child_id).heading_level == 3
        ]
        if len(h3) >= 2:
            return node
    return None


def _next_heading_text(hierarchy: SectionHierarchy, node: SectionNode) -> str | None:
    same_doc = [
        other
        for other in hierarchy.nodes
        if other.document_id == node.document_id
        and other.kind == "heading"
        and other.source_start >= node.source_end
        and other.heading_level is not None
        and node.heading_level is not None
        and other.heading_level <= node.heading_level
    ]
    if not same_doc:
        return None
    return same_doc[0].heading


def build_hierarchy_report(
    documents: list[LoadedMarkdownDocument],
    parsed_documents: list[ParsedMarkdownDocument],
    chunking_result: ChunkingResult,
    hierarchy: SectionHierarchy,
    *,
    tokenizer_id: str,
    chunk_size: int,
    overlap_tokens: int,
) -> dict[str, Any]:
    leaf_counts = _leaf_counts(chunking_result.leaves)
    heading_nodes = [node for node in hierarchy.nodes if node.kind == "heading"]
    root_nodes = [
        node for node in hierarchy.nodes if node.kind == "document_root"
    ]
    heading_refs = [
        ref for ref in chunking_result.sections if ref.kind == "heading"
    ]
    root_refs = [
        ref for ref in chunking_result.sections if ref.kind == "document_root"
    ]
    level_counts = Counter(
        node.heading_level for node in heading_nodes if node.heading_level
    )
    depths = [
        1 + len(hierarchy.get_ancestors(node.section_id))
        for node in hierarchy.nodes
    ]
    parent_links = [
        node for node in hierarchy.nodes if node.parent_section_id is not None
    ]
    missing_parent = 0
    cross_document_parent = 0
    invalid_level = 0
    for node in parent_links:
        parent = hierarchy._nodes_by_id.get(node.parent_section_id or "")
        if parent is None:
            missing_parent += 1
            continue
        if parent.document_id != node.document_id:
            cross_document_parent += 1
        if (
            node.kind == "heading"
            and parent.kind == "heading"
            and parent.heading_level is not None
            and node.heading_level is not None
            and parent.heading_level >= node.heading_level
        ):
            invalid_level += 1

    leaf_resolution_failures = sum(
        1
        for leaf in chunking_result.leaves
        if leaf.section_id not in hierarchy._nodes_by_id
        or hierarchy._nodes_by_id[leaf.section_id].document_id != leaf.document_id
    )
    page_invalid = sum(
        1
        for node in hierarchy.nodes
        if node.page_start < 1 or node.page_end < node.page_start
    )
    span_invalid = sum(
        1
        for node in hierarchy.nodes
        if node.source_start < 0 or node.source_end <= node.source_start
    )

    recovered_count = 0
    marker_violations = 0
    recovered_previews: dict[str, str] = {}
    for node in hierarchy.nodes:
        text = hierarchy.recover_section_text(node.section_id, documents)
        recovered_count += 1
        recovered_previews[node.section_id] = text
        if _has_pdf_page_marker(text):
            marker_violations += 1

    cycle_count = 0
    for node in hierarchy.nodes:
        seen: set[str] = set()
        current_id: str | None = node.section_id
        while current_id is not None:
            if current_id in seen:
                cycle_count += 1
                break
            seen.add(current_id)
            current = hierarchy._nodes_by_id.get(current_id)
            current_id = None if current is None else current.parent_section_id

    spot_nodes = _select_spotcheck_nodes(hierarchy, chunking_result.leaves)
    spotcheck = []
    for node in spot_nodes:
        text = recovered_previews[node.section_id]
        spotcheck.append(
            _node_payload(
                node,
                direct_leaf_count=leaf_counts.get(node.section_id, 0),
                child_count=len(
                    hierarchy.child_section_ids(
                        node.document_id, node.section_id
                    )
                ),
                text_preview=_text_preview(text),
            )
        )

    manual_node = _manual_recovery_target(hierarchy)
    manual: dict[str, Any] | None = None
    if manual_node is not None:
        recovered = recovered_previews[manual_node.section_id]
        next_heading = _next_heading_text(hierarchy, manual_node)
        child_headings = [
            hierarchy.get_section(child_id).heading
            for child_id in hierarchy.child_section_ids(
                manual_node.document_id, manual_node.section_id
            )
            if hierarchy.get_section(child_id).heading
        ]
        manual = {
            **_node_payload(
                manual_node,
                direct_leaf_count=leaf_counts.get(manual_node.section_id, 0),
                child_count=len(
                    hierarchy.child_section_ids(
                        manual_node.document_id, manual_node.section_id
                    )
                ),
            ),
            "recovered_text": recovered,
            "contains_own_heading": bool(
                manual_node.heading and manual_node.heading in recovered
            ),
            "contains_child_headings": all(
                heading in recovered for heading in child_headings
            ),
            "next_same_or_higher_heading": next_heading,
            "contains_next_same_or_higher_heading": (
                False
                if next_heading is None
                else any(
                    line.lstrip().startswith("#") and next_heading in line
                    for line in recovered.splitlines()
                )
            ),
            "contains_pdf_page_marker": _has_pdf_page_marker(recovered),
        }

    stats = {
        "document_count": len(documents),
        "section_node_count": len(hierarchy.nodes),
        "heading_section_count": len(heading_nodes),
        "document_root_count": len(root_nodes),
        "a3_heading_section_count": len(heading_refs),
        "a3_document_root_count": len(root_refs),
        "heading_level_distribution": {
            str(level): level_counts[level] for level in sorted(level_counts)
        },
        "top_level_section_count": sum(
            1 for node in hierarchy.nodes if node.parent_section_id is None
        ),
        "leaf_count": len(chunking_result.leaves),
        "leaf_section_resolution_failures": leaf_resolution_failures,
        "parent_link_count": len(parent_links),
        "missing_parent_count": missing_parent,
        "cross_document_parent_count": cross_document_parent,
        "hierarchy_cycle_count": cycle_count,
        "invalid_heading_level_relation_count": invalid_level,
        "section_page_invalid_count": page_invalid,
        "section_span_invalid_count": span_invalid,
        "recovered_section_count": recovered_count,
        "recovered_pdf_marker_violation_count": marker_violations,
        "max_hierarchy_depth": max(depths) if depths else None,
        "hierarchy_depth_p50": _percentile(depths, 50),
        "hierarchy_depth_p95": _percentile(depths, 95),
        "parsed_document_count": len(parsed_documents),
        "section_node_field_names": list(SECTION_NODE_FIELD_NAMES),
    }
    return {
        "tokenizer": tokenizer_id,
        "transformers_version": transformers_version(),
        "chunk_size": chunk_size,
        "overlap_tokens": overlap_tokens,
        "stats": stats,
        "spotcheck": spotcheck,
        "manual_recovery": manual,
        "document_root_present_in_corpus": bool(root_nodes),
        "sections": [
            _node_payload(node) for node in hierarchy.nodes
        ],
    }


def format_hierarchy_summary(report: dict[str, Any]) -> str:
    stats = report["stats"]
    levels = stats["heading_level_distribution"]
    level_text = ",".join(
        f"{level}:{count}" for level, count in levels.items()
    )
    lines = [
        f"tokenizer={report['tokenizer']}",
        f"transformers_version={report['transformers_version']}",
        f"chunk_size={report['chunk_size']}",
        f"overlap_tokens={report['overlap_tokens']}",
        f"document_count={stats['document_count']}",
        f"section_node_count={stats['section_node_count']}",
        f"heading_section_count={stats['heading_section_count']}",
        f"document_root_count={stats['document_root_count']}",
        f"heading_level_distribution={level_text}",
        f"top_level_section_count={stats['top_level_section_count']}",
        f"leaf_count={stats['leaf_count']}",
        f"leaf_section_resolution_failures={stats['leaf_section_resolution_failures']}",
        f"parent_link_count={stats['parent_link_count']}",
        f"missing_parent_count={stats['missing_parent_count']}",
        f"cross_document_parent_count={stats['cross_document_parent_count']}",
        f"hierarchy_cycle_count={stats['hierarchy_cycle_count']}",
        f"invalid_heading_level_relation_count={stats['invalid_heading_level_relation_count']}",
        f"section_page_invalid_count={stats['section_page_invalid_count']}",
        f"section_span_invalid_count={stats['section_span_invalid_count']}",
        f"recovered_section_count={stats['recovered_section_count']}",
        f"recovered_pdf_marker_violation_count={stats['recovered_pdf_marker_violation_count']}",
        f"max_hierarchy_depth={stats['max_hierarchy_depth']}",
        f"hierarchy_depth_p50={stats['hierarchy_depth_p50']}",
        f"hierarchy_depth_p95={stats['hierarchy_depth_p95']}",
        "spotcheck:",
    ]
    for row in report["spotcheck"]:
        lines.append(
            "  "
            f"document_id={row['document_id']} "
            f"kind={row['kind']} "
            f"level={row['heading_level']} "
            f"heading={row['heading']!r} "
            f"parent={row['parent_section_id']} "
            f"span=[{row['source_start']},{row['source_end']}) "
            f"page={row['page_start']}-{row['page_end']} "
            f"leaves={row['direct_leaf_count']} "
            f"children={row['child_count']} "
            f"preview={row['text_preview']!r}"
        )
    manual = report.get("manual_recovery")
    if manual is None:
        lines.append("manual_recovery: absent")
    else:
        lines.append(
            "manual_recovery: "
            f"document_id={manual['document_id']} "
            f"heading={manual['heading']!r} "
            f"contains_own_heading={manual['contains_own_heading']} "
            f"contains_child_headings={manual['contains_child_headings']} "
            f"contains_next_same_or_higher_heading="
            f"{manual['contains_next_same_or_higher_heading']} "
            f"contains_pdf_page_marker={manual['contains_pdf_page_marker']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A4 Section hierarchy builder. Reuses one A3.1 parse for A3.2 "
            "Leaf chunking and A4 hierarchy. Does not embed, index, or "
            "select B6 ancestors."
        ),
        epilog=(
            "From the repository root, put code/ on PYTHONPATH, for example:\n"
            '  PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" '
            "python -m knowledge_base.section_hierarchy "
            "--canonical-root <ABS_PATH> "
            "--tokenizer Qwen/Qwen3-Embedding-4B "
            "--output /tmp/a4-sections.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Formal A0 output root containing <document_id>/document.md",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Hugging Face tokenizer id or local tokenizer directory",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Soft token limit for normal Leafs (default {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
        help=(
            "Overlap budget in tokens "
            f"(default {DEFAULT_OVERLAP_TOKENS})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON debug/acceptance artifact path; not a production store",
    )
    args = parser.parse_args(argv)

    try:
        documents = load_markdown_documents(args.canonical_root)
        tokenizer = load_tokenizer(args.tokenizer)
        parsed_documents, chunking_result, hierarchy = (
            build_hierarchy_from_documents(
                documents,
                tokenizer,
                chunk_size=args.chunk_size,
                overlap_tokens=args.overlap,
            )
        )
        report = build_hierarchy_report(
            documents,
            parsed_documents,
            chunking_result,
            hierarchy,
            tokenizer_id=args.tokenizer,
            chunk_size=args.chunk_size,
            overlap_tokens=args.overlap,
        )
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (
        MarkdownLoadingError,
        A3ChunkingError,
        A4HierarchyError,
        SectionProfileError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"canonical_root={args.canonical_root.resolve()}")
    print(f"output={args.output.resolve()}")
    print(format_hierarchy_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
