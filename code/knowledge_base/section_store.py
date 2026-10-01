"""Production Section Store: JSONL sidecar plus in-memory reload.

Writes the A4 ``SectionNode`` fields already produced for retrievable
documents. Does not build a second hierarchy, does not persist section
body text, and does not add a build fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_base.markdown_loader import LoadedMarkdownDocument
from knowledge_base.section_hierarchy import (
    SECTION_NODE_FIELD_NAMES,
    A4HierarchyError,
    SectionHierarchy,
    SectionNode,
    _section_pages,
    _validate_parent_links,
    _validate_span,
)
from knowledge_base.structure_parser import (
    ParsedMarkdownDocument,
    line_starts,
    parse_markdown_document,
    strip_page_markers,
)

_INT_FIELDS = ("source_start", "source_end", "page_start", "page_end")


class SectionStoreError(Exception):
    """Fail-fast error for a damaged or unbound Section Store."""


class SectionStore:
    """In-memory section index loaded from ``sections.jsonl``."""

    def __init__(self, nodes: tuple[SectionNode, ...]) -> None:
        self.nodes = nodes
        self._nodes_by_id = {node.section_id: node for node in nodes}
        children: dict[tuple[str, str | None], list[str]] = {}
        for node in nodes:
            key = (node.document_id, node.parent_section_id)
            children.setdefault(key, []).append(node.section_id)
        self._children = {key: tuple(value) for key, value in children.items()}
        self._parsed: dict[str, ParsedMarkdownDocument] = {}
        self._documents: dict[str, LoadedMarkdownDocument] = {}
        self._marker_lines: dict[str, set[int]] = {}

    def get_section(self, section_id: str) -> SectionNode:
        node = self._nodes_by_id.get(section_id)
        if node is None:
            raise SectionStoreError(f"Unknown section_id: {section_id}")
        return node

    def child_section_ids(
        self, document_id: str, parent_section_id: str | None
    ) -> tuple[str, ...]:
        return self._children.get((document_id, parent_section_id), ())

    def section_path(self, section_id: str) -> tuple[str, ...]:
        chain: list[SectionNode] = []
        seen: set[str] = set()
        current = self.get_section(section_id)
        while True:
            if current.section_id in seen:
                raise SectionStoreError(
                    f"hierarchy cycle involving section_id={current.section_id}"
                )
            seen.add(current.section_id)
            chain.append(current)
            if current.parent_section_id is None:
                break
            current = self.get_section(current.parent_section_id)
        chain.reverse()
        return tuple(node.heading for node in chain if node.heading)

    def bind_documents(self, documents: list[LoadedMarkdownDocument]) -> None:
        """Parse each referenced document once and check A4 span / page invariants."""

        by_id: dict[str, LoadedMarkdownDocument] = {}
        for document in documents:
            if document.document_id in by_id:
                raise SectionStoreError(
                    f"Duplicate document_id: {document.document_id}"
                )
            by_id[document.document_id] = document
        parsed: dict[str, ParsedMarkdownDocument] = {}
        markers: dict[str, set[int]] = {}
        for node in self.nodes:
            document = by_id.get(node.document_id)
            if document is None:
                raise SectionStoreError(
                    "section document_id is not in the canonical documents: "
                    f"{node.document_id}"
                )
            if node.document_id not in parsed:
                parsed[node.document_id] = parse_markdown_document(
                    document.content, document_id=document.document_id
                )
                markers[node.document_id] = {
                    marker.line for marker in parsed[node.document_id].page_markers
                }
            self._check_span_and_pages(node, parsed[node.document_id])
        self._documents = by_id
        self._parsed = parsed
        self._marker_lines = markers

    def recover_section(self, section_id: str) -> str:
        node = self.get_section(section_id)
        return self.recover_span(node.document_id, node.source_start, node.source_end)

    def recover_span(self, document_id: str, start: int, end: int) -> str:
        document = self._documents.get(document_id)
        parsed = self._parsed.get(document_id)
        if document is None or parsed is None:
            raise SectionStoreError(
                f"documents are not bound for document_id={document_id}"
            )
        try:
            _validate_span(parsed, document_id, start, end)
        except A4HierarchyError as exc:
            raise SectionStoreError(str(exc)) from exc
        return strip_page_markers(
            document.content,
            line_starts(document.content),
            start,
            end,
            self._marker_lines[document_id],
        )

    def _check_span_and_pages(
        self, node: SectionNode, parsed: ParsedMarkdownDocument
    ) -> None:
        if node.source_start < 0 or node.source_end <= node.source_start:
            raise SectionStoreError(
                "invalid source span in "
                f"section_id={node.section_id}: "
                f"[{node.source_start}, {node.source_end})"
            )
        if node.page_start < 1 or node.page_end < node.page_start:
            raise SectionStoreError(
                "invalid page range in "
                f"section_id={node.section_id}: "
                f"{node.page_start}-{node.page_end}"
            )
        try:
            _validate_span(parsed, node.section_id, node.source_start, node.source_end)
            derived = _section_pages(
                parsed, node.section_id, node.source_start, node.source_end
            )
        except A4HierarchyError as exc:
            raise SectionStoreError(str(exc)) from exc
        if (node.page_start, node.page_end) != derived:
            raise SectionStoreError(
                "stored page range does not match document provenance in "
                f"section_id={node.section_id}: "
                f"stored={node.page_start}-{node.page_end} derived={derived[0]}-{derived[1]}"
            )


def write_sections_jsonl(hierarchy: SectionHierarchy, path: Path) -> None:
    """Write one ``SectionNode`` per line. Body text is not included."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for node in hierarchy.nodes:
        payload = {name: getattr(node, name) for name in SECTION_NODE_FIELD_NAMES}
        lines.append(json.dumps(payload, ensure_ascii=False))
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8", newline="\n")


def load_sections_jsonl(path: Path) -> SectionStore:
    """Load the sidecar and reject graphs that violate A4 parent invariants."""

    if not path.is_file():
        raise SectionStoreError(f"sections.jsonl does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    nodes: list[SectionNode] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SectionStoreError(
                f"invalid JSON in {path} line {line_number}"
            ) from exc
        if not isinstance(payload, dict):
            raise SectionStoreError(f"section row is not an object in line {line_number}")
        node = _node_from_payload(payload, line_number)
        if node.section_id in seen:
            raise SectionStoreError(f"duplicate section_id: {node.section_id}")
        seen.add(node.section_id)
        nodes.append(node)
    nodes_by_id = {node.section_id: node for node in nodes}
    try:
        _validate_parent_links(nodes_by_id)
    except A4HierarchyError as exc:
        raise SectionStoreError(str(exc)) from exc
    return SectionStore(tuple(nodes))


def _node_from_payload(payload: dict[str, Any], line_number: int) -> SectionNode:
    missing = [name for name in SECTION_NODE_FIELD_NAMES if name not in payload]
    if missing:
        raise SectionStoreError(
            f"section row line {line_number} is missing {missing}"
        )
    extra = [name for name in payload if name not in SECTION_NODE_FIELD_NAMES]
    if extra:
        raise SectionStoreError(
            f"section row line {line_number} has unknown fields {extra}"
        )
    section_id = _required_str(payload, "section_id", line_number)
    document_id = _required_str(payload, "document_id", line_number)
    kind = _required_str(payload, "kind", line_number)
    if kind not in {"heading", "document_root"}:
        raise SectionStoreError(
            f"invalid kind {kind!r} in section_id={section_id}"
        )
    parent_section_id = _nullable_str(payload, "parent_section_id", line_number)
    heading = _nullable_str(payload, "heading", line_number)
    heading_level = _optional_int(payload, "heading_level", line_number)
    if kind == "heading" and heading_level is None:
        raise SectionStoreError(
            f"heading section is missing heading_level: {section_id}"
        )
    if kind == "document_root" and heading_level is not None:
        raise SectionStoreError(
            f"document_root must not set heading_level: {section_id}"
        )
    numbers = {
        name: _required_int(payload, name, line_number) for name in _INT_FIELDS
    }
    return SectionNode(
        section_id=section_id,
        document_id=document_id,
        parent_section_id=parent_section_id,
        kind=kind,
        heading_level=heading_level,
        heading=heading,
        source_start=numbers["source_start"],
        source_end=numbers["source_end"],
        page_start=numbers["page_start"],
        page_end=numbers["page_end"],
    )


def _required_str(payload: dict[str, Any], name: str, line_number: int) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value:
        raise SectionStoreError(
            f"{name} must be a non-empty string in line {line_number}"
        )
    return value


def _nullable_str(payload: dict[str, Any], name: str, line_number: int) -> str | None:
    value = payload[name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise SectionStoreError(f"{name} must be a string or null in line {line_number}")
    return value


def _optional_int(payload: dict[str, Any], name: str, line_number: int) -> int | None:
    value = payload[name]
    if value is None:
        return None
    return _as_int(value, name, line_number)


def _required_int(payload: dict[str, Any], name: str, line_number: int) -> int:
    return _as_int(payload[name], name, line_number)


def _as_int(value: Any, name: str, line_number: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SectionStoreError(f"{name} must be an integer in line {line_number}")
    return value
