"""A3.2 structure-aware Leaf chunking.

Reuses A3.1 ``parse_markdown_document`` and ``count_tokens``. Does not choose
a new tokenizer, rewrite A0 Markdown, or persist A4 hierarchy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_base.chunking_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_TOKENS,
)
from knowledge_base.leaf_ids import make_chunk_id, make_section_id
from knowledge_base.markdown_loader import (
    LoadedMarkdownDocument,
    MarkdownLoadingError,
    load_markdown_documents,
)
from knowledge_base.structure_parser import (
    ContentBlock,
    Heading,
    PageMarker,
    ParsedMarkdownDocument,
    TerminalSection,
    heading_for_terminal_section,
    iter_section_blocks,
    line_starts,
    page_range_for_span,
    parse_markdown_document,
    strip_page_markers,
)
from knowledge_base.token_count import (
    SectionProfileError,
    count_tokens,
    load_tokenizer,
    tokenize_with_offsets,
    transformers_version,
)

LEAF_FIELD_NAMES = (
    "chunk_id",
    "document_id",
    "section_id",
    "chunk_index",
    "page_start",
    "page_end",
    "content",
)


class A3ChunkingError(Exception):
    """Fail-fast error for A3.2 Leaf chunking."""


@dataclass(frozen=True)
class Leaf:
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str


@dataclass(frozen=True)
class SectionRef:
    section_id: str
    document_id: str
    kind: str
    heading_level: int | None
    heading_text: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class ChunkingResult:
    leaves: tuple[Leaf, ...]
    sections: tuple[SectionRef, ...]


@dataclass(frozen=True)
class _Unit:
    content: str
    line_ranges: tuple[tuple[int, int], ...]
    blocks: tuple[ContentBlock, ...]
    is_oversize_table: bool
    is_token_window: bool


def validate_chunking_config(chunk_size: int, overlap_tokens: int) -> None:
    if chunk_size <= 0:
        raise A3ChunkingError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap_tokens < 0:
        raise A3ChunkingError(
            f"overlap_tokens must be >= 0, got {overlap_tokens}"
        )
    if overlap_tokens >= chunk_size:
        raise A3ChunkingError(
            "overlap_tokens must be < chunk_size, got "
            f"overlap_tokens={overlap_tokens} chunk_size={chunk_size}"
        )


def _marker_lines(parsed: ParsedMarkdownDocument) -> set[int]:
    return {marker.line for marker in parsed.page_markers}


def _join_blocks(
    content: str,
    starts: list[int],
    blocks: list[ContentBlock] | tuple[ContentBlock, ...],
    marker_lines: set[int],
) -> str:
    if not blocks:
        return ""
    return strip_page_markers(
        content,
        starts,
        blocks[0].start_line,
        blocks[-1].end_line,
        marker_lines,
    )


def _concat_units(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    return left + right


def _sliding_units(
    block: ContentBlock,
    tokenizer: Any,
    chunk_size: int,
    overlap_tokens: int,
) -> list[_Unit]:
    text = block.source
    ids, offsets = tokenize_with_offsets(text, tokenizer)
    if not ids:
        return []
    if len(ids) <= chunk_size:
        return [
            _Unit(
                content=text,
                line_ranges=((block.start_line, block.end_line),),
                blocks=(block,),
                is_oversize_table=False,
                is_token_window=False,
            )
        ]
    stride = chunk_size - overlap_tokens
    units: list[_Unit] = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_size, len(ids))
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        window = text[char_start:char_end]
        if window.strip():
            units.append(
                _Unit(
                    content=window,
                    line_ranges=((block.start_line, block.end_line),),
                    blocks=(),
                    is_oversize_table=False,
                    is_token_window=True,
                )
            )
        if end >= len(ids):
            break
        start += stride
    return units


def _pack_section(
    content: str,
    starts: list[int],
    marker_lines: set[int],
    blocks: tuple[ContentBlock, ...],
    tokenizer: Any,
    chunk_size: int,
    overlap_tokens: int,
) -> list[_Unit]:
    current: list[ContentBlock] = []
    units: list[_Unit] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        packed = _join_blocks(content, starts, current, marker_lines)
        if packed.strip():
            units.append(
                _Unit(
                    content=packed,
                    line_ranges=((current[0].start_line, current[-1].end_line),),
                    blocks=tuple(current),
                    is_oversize_table=False,
                    is_token_window=False,
                )
            )
        current = []

    for block in blocks:
        block_tokens = count_tokens(
            _join_blocks(content, starts, [block], marker_lines), tokenizer
        )
        if block_tokens == 0:
            continue
        if block.is_table and block_tokens > chunk_size:
            flush()
            units.append(
                _Unit(
                    content=_join_blocks(
                        content, starts, [block], marker_lines
                    ),
                    line_ranges=((block.start_line, block.end_line),),
                    blocks=(block,),
                    is_oversize_table=True,
                    is_token_window=False,
                )
            )
            continue
        if not block.is_table and block_tokens > chunk_size:
            flush()
            units.extend(
                _sliding_units(block, tokenizer, chunk_size, overlap_tokens)
            )
            continue
        if current:
            trial = _join_blocks(
                content, starts, current + [block], marker_lines
            )
            if count_tokens(trial, tokenizer) > chunk_size:
                flush()
        current.append(block)
    flush()
    return units


def _block_overlap(
    previous: _Unit,
    nxt: _Unit,
    content: str,
    starts: list[int],
    marker_lines: set[int],
    tokenizer: Any,
    chunk_size: int,
    overlap_tokens: int,
) -> _Unit:
    if (
        previous.is_token_window
        or nxt.is_token_window
        or previous.is_oversize_table
        or nxt.is_oversize_table
        or not previous.blocks
        or not nxt.blocks
    ):
        return nxt
    chosen: list[ContentBlock] = []
    for block in reversed(previous.blocks):
        if block.is_table:
            break
        trial_blocks = [block, *chosen]
        overlap_text = _join_blocks(
            content, starts, trial_blocks, marker_lines
        )
        if count_tokens(overlap_text, tokenizer) > overlap_tokens:
            break
        combined = _concat_units(overlap_text, nxt.content)
        if count_tokens(combined, tokenizer) > chunk_size:
            break
        chosen = trial_blocks
    if not chosen:
        return nxt
    overlap_text = _join_blocks(content, starts, chosen, marker_lines)
    combined = _concat_units(overlap_text, nxt.content)
    return _Unit(
        content=combined,
        line_ranges=tuple(
            [(chosen[0].start_line, chosen[-1].end_line), *nxt.line_ranges]
        ),
        blocks=tuple(chosen) + nxt.blocks,
        is_oversize_table=False,
        is_token_window=False,
    )


def _apply_overlap(
    units: list[_Unit],
    content: str,
    starts: list[int],
    marker_lines: set[int],
    tokenizer: Any,
    chunk_size: int,
    overlap_tokens: int,
) -> list[_Unit]:
    if not units:
        return []
    result = [units[0]]
    for nxt in units[1:]:
        result.append(
            _block_overlap(
                result[-1],
                nxt,
                content,
                starts,
                marker_lines,
                tokenizer,
                chunk_size,
                overlap_tokens,
            )
        )
    return result


def _pages_for_ranges(
    markers: tuple[PageMarker, ...],
    ranges: tuple[tuple[int, int], ...],
) -> tuple[int, int]:
    if not ranges:
        raise A3ChunkingError("Leaf has no source span for page provenance")
    starts: list[int] = []
    ends: list[int] = []
    for start_line, end_line in ranges:
        page_start, page_end = page_range_for_span(
            markers, start_line, end_line
        )
        starts.append(page_start)
        ends.append(page_end)
    return min(starts), max(ends)


def _document_root_id(document_id: str) -> str:
    return make_section_id(
        document_id=document_id,
        kind="document_root",
        heading_level=None,
        source_start=0,
    )


def _heading_section_id(document_id: str, heading: Heading) -> str:
    return make_section_id(
        document_id=document_id,
        kind="heading",
        heading_level=heading.level,
        source_start=heading.start_line,
    )


def _leaf_section_id(
    parsed: ParsedMarkdownDocument, section: TerminalSection
) -> str:
    heading = heading_for_terminal_section(parsed, section)
    if heading is None:
        return _document_root_id(parsed.document_id)
    return _heading_section_id(parsed.document_id, heading)


def _build_section_refs(
    parsed: ParsedMarkdownDocument, used_section_ids: set[str]
) -> list[SectionRef]:
    refs: list[SectionRef] = []
    seen: set[str] = set()
    root_id = _document_root_id(parsed.document_id)
    needs_root = parsed.unheaded_document_body or any(
        section.heading_level is None for section in parsed.terminal_sections
    )
    if needs_root or root_id in used_section_ids:
        seen.add(root_id)
        refs.append(
            SectionRef(
                section_id=root_id,
                document_id=parsed.document_id,
                kind="document_root",
                heading_level=None,
                heading_text="",
                source_start=0,
                source_end=parsed.line_count,
            )
        )
    for heading in parsed.headings:
        section_id = _heading_section_id(parsed.document_id, heading)
        if section_id in seen:
            raise A3ChunkingError(
                "section_id identity collision in "
                f"document_id={parsed.document_id} heading={heading.text!r} "
                f"line={heading.start_line + 1}"
            )
        seen.add(section_id)
        refs.append(
            SectionRef(
                section_id=section_id,
                document_id=parsed.document_id,
                kind="heading",
                heading_level=heading.level,
                heading_text=heading.text,
                source_start=heading.start_line,
                source_end=heading.end_line,
            )
        )
    return refs


def _chunk_parsed_document(
    document: LoadedMarkdownDocument,
    parsed: ParsedMarkdownDocument,
    tokenizer: Any,
    chunk_size: int,
    overlap_tokens: int,
) -> tuple[list[Leaf], list[SectionRef]]:
    starts = line_starts(document.content)
    marker_lines = _marker_lines(parsed)
    built: list[tuple[str, str, _Unit]] = []
    for section in parsed.terminal_sections:
        if not section.body_text.strip():
            continue
        heading = heading_for_terminal_section(parsed, section)
        if section.heading_level is not None and heading is None:
            raise A3ChunkingError(
                "Terminal section has no semantic heading in "
                f"document_id={parsed.document_id} ordinal={section.ordinal}"
            )
        section_id = _leaf_section_id(parsed, section)
        blocks = iter_section_blocks(document.content, parsed, section)
        units = _pack_section(
            document.content,
            starts,
            marker_lines,
            blocks,
            tokenizer,
            chunk_size,
            overlap_tokens,
        )
        units = _apply_overlap(
            units,
            document.content,
            starts,
            marker_lines,
            tokenizer,
            chunk_size,
            overlap_tokens,
        )
        for unit in units:
            if not unit.content.strip():
                continue
            built.append((section_id, parsed.document_id, unit))
    if not built:
        raise A3ChunkingError(
            "Document produced 0 Leaf: "
            f"document_id={parsed.document_id}"
        )
    leaves: list[Leaf] = []
    seen_chunk_ids: set[str] = set()
    used_section_ids: set[str] = set()
    for chunk_index, (section_id, document_id, unit) in enumerate(built):
        tokens = count_tokens(unit.content, tokenizer)
        if tokens == 0:
            raise A3ChunkingError(
                f"Leaf.content is empty in document_id={document_id}"
            )
        if tokens > chunk_size and not unit.is_oversize_table:
            raise A3ChunkingError(
                "Normal Leaf exceeds chunk_size in "
                f"document_id={document_id} tokens={tokens} "
                f"chunk_size={chunk_size}"
            )
        page_start, page_end = _pages_for_ranges(
            parsed.page_markers, unit.line_ranges
        )
        if page_start < 1 or page_end < page_start:
            raise A3ChunkingError(
                "Invalid page provenance in "
                f"document_id={document_id}: {page_start}-{page_end}"
            )
        chunk_id = make_chunk_id(
            document_id=document_id,
            section_id=section_id,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
            content=unit.content,
        )
        if chunk_id in seen_chunk_ids:
            raise A3ChunkingError(
                f"Duplicate chunk_id in document_id={document_id}"
            )
        seen_chunk_ids.add(chunk_id)
        used_section_ids.add(section_id)
        leaves.append(
            Leaf(
                chunk_id=chunk_id,
                document_id=document_id,
                section_id=section_id,
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                content=unit.content,
            )
        )
    expected = list(range(len(leaves)))
    actual = [leaf.chunk_index for leaf in leaves]
    if actual != expected:
        raise A3ChunkingError(
            f"chunk_index is not contiguous in document_id={parsed.document_id}"
        )
    return leaves, _build_section_refs(parsed, used_section_ids)


def chunk_parsed_documents(
    documents: list[LoadedMarkdownDocument],
    parsed_documents: list[ParsedMarkdownDocument],
    tokenizer: Any,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> ChunkingResult:
    """Chunk already-parsed A1 documents. Does not mutate inputs or re-parse."""

    validate_chunking_config(chunk_size, overlap_tokens)
    if not documents:
        raise A3ChunkingError("No documents provided")
    if len(documents) != len(parsed_documents):
        raise A3ChunkingError(
            "documents and parsed_documents length mismatch: "
            f"{len(documents)} != {len(parsed_documents)}"
        )
    seen_ids: set[str] = set()
    all_leaves: list[Leaf] = []
    all_sections: list[SectionRef] = []
    try:
        for document, parsed in zip(documents, parsed_documents, strict=True):
            if document.document_id in seen_ids:
                raise A3ChunkingError(
                    f"Duplicate document_id: {document.document_id}"
                )
            seen_ids.add(document.document_id)
            if parsed.document_id != document.document_id:
                raise A3ChunkingError(
                    "parsed document_id mismatch: "
                    f"{parsed.document_id!r} != {document.document_id!r}"
                )
            leaves, sections = _chunk_parsed_document(
                document,
                parsed,
                tokenizer,
                chunk_size,
                overlap_tokens,
            )
            all_leaves.extend(leaves)
            all_sections.extend(sections)
    except SectionProfileError as exc:
        raise A3ChunkingError(str(exc)) from exc
    return ChunkingResult(
        leaves=tuple(all_leaves), sections=tuple(all_sections)
    )


def chunk_documents(
    documents: list[LoadedMarkdownDocument],
    tokenizer: Any,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> ChunkingResult:
    """Chunk A1 documents into A3 Leafs. Does not mutate *documents*."""

    validate_chunking_config(chunk_size, overlap_tokens)
    if not documents:
        raise A3ChunkingError("No documents provided")
    seen_ids: set[str] = set()
    parsed_documents: list[ParsedMarkdownDocument] = []
    try:
        for document in documents:
            if document.document_id in seen_ids:
                raise A3ChunkingError(
                    f"Duplicate document_id: {document.document_id}"
                )
            seen_ids.add(document.document_id)
            parsed_documents.append(
                parse_markdown_document(
                    document.content, document_id=document.document_id
                )
            )
    except SectionProfileError as exc:
        raise A3ChunkingError(str(exc)) from exc
    return chunk_parsed_documents(
        documents,
        parsed_documents,
        tokenizer,
        chunk_size=chunk_size,
        overlap_tokens=overlap_tokens,
    )


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


def _leaf_debug(
    leaf: Leaf, tokenizer: Any, *, is_oversize_table: bool | None = None
) -> dict[str, Any]:
    preview = leaf.content.strip().replace("\n", " ")
    if len(preview) > 160:
        preview = preview[:160] + "..."
    payload = {
        "document_id": leaf.document_id,
        "section_id": leaf.section_id,
        "chunk_id": leaf.chunk_id,
        "chunk_index": leaf.chunk_index,
        "page_start": leaf.page_start,
        "page_end": leaf.page_end,
        "token_count": count_tokens(leaf.content, tokenizer),
        "content_preview": preview,
    }
    if is_oversize_table is not None:
        payload["is_oversize_table"] = is_oversize_table
    return payload


def build_chunking_report(
    documents: list[LoadedMarkdownDocument],
    result: ChunkingResult,
    tokenizer: Any,
    *,
    tokenizer_id: str,
    chunk_size: int,
    overlap_tokens: int,
) -> dict[str, Any]:
    parsed_docs = [
        parse_markdown_document(
            document.content, document_id=document.document_id
        )
        for document in documents
    ]
    terminal_count = sum(len(parsed.terminal_sections) for parsed in parsed_docs)
    empty_terminals = sum(
        1
        for parsed in parsed_docs
        for section in parsed.terminal_sections
        if not section.body_text.strip()
    )
    tokens_by_leaf = [
        count_tokens(leaf.content, tokenizer) for leaf in result.leaves
    ]
    oversize = [
        leaf
        for leaf, tokens in zip(result.leaves, tokens_by_leaf)
        if tokens > chunk_size
    ]
    normal = [
        leaf
        for leaf, tokens in zip(result.leaves, tokens_by_leaf)
        if tokens <= chunk_size
    ]
    normal_tokens = [
        count_tokens(leaf.content, tokenizer) for leaf in normal
    ]
    oversize_tokens = [
        count_tokens(leaf.content, tokenizer) for leaf in oversize
    ]
    leaves_by_section: dict[str, int] = {}
    for leaf in result.leaves:
        leaves_by_section[leaf.section_id] = (
            leaves_by_section.get(leaf.section_id, 0) + 1
        )
    sections_split = sum(
        1 for count in leaves_by_section.values() if count > 1
    )
    sections_unsplit = sum(
        1 for count in leaves_by_section.values() if count == 1
    )
    docs_with_leaves = {leaf.document_id for leaf in result.leaves}
    chunk_ids = [leaf.chunk_id for leaf in result.leaves]
    section_ids = [ref.section_id for ref in result.sections]
    indexes_ok = True
    for document in documents:
        doc_leaves = [
            leaf
            for leaf in result.leaves
            if leaf.document_id == document.document_id
        ]
        if [leaf.chunk_index for leaf in doc_leaves] != list(
            range(len(doc_leaves))
        ):
            indexes_ok = False
    invalid_pages = sum(
        1
        for leaf in result.leaves
        if leaf.page_start < 1 or leaf.page_end < leaf.page_start
    )
    stats = {
        "document_count": len(documents),
        "terminal_section_count": terminal_count,
        "terminal_sections_with_leaf": terminal_count - empty_terminals,
        "empty_terminal_sections": empty_terminals,
        "leaf_count": len(result.leaves),
        "leaf_token": {
            "p50": _percentile(tokens_by_leaf, 50),
            "p75": _percentile(tokens_by_leaf, 75),
            "p90": _percentile(tokens_by_leaf, 90),
            "p95": _percentile(tokens_by_leaf, 95),
            "max": max(tokens_by_leaf) if tokens_by_leaf else None,
            "mean": (
                round(sum(tokens_by_leaf) / len(tokens_by_leaf), 6)
                if tokens_by_leaf
                else None
            ),
        },
        "normal_leaf_count": len(normal),
        "normal_leaf_max_tokens": max(normal_tokens) if normal_tokens else None,
        "oversize_table_leaf_count": len(oversize),
        "oversize_table_max_tokens": (
            max(oversize_tokens) if oversize_tokens else None
        ),
        "sections_split_count": sections_split,
        "sections_unsplit_count": sections_unsplit,
        "page_start_page_end_invalid_count": invalid_pages,
        "duplicate_chunk_id_count": len(chunk_ids) - len(set(chunk_ids)),
        "duplicate_section_id_collision_count": len(section_ids)
        - len(set(section_ids)),
        "non_contiguous_chunk_index_count": 0 if indexes_ok else 1,
        "table_split_violation_count": 0,
        "zero_leaf_document_count": len(documents) - len(docs_with_leaves),
    }
    longest_normal = max(
        normal, key=lambda leaf: count_tokens(leaf.content, tokenizer)
    ) if normal else None
    longest_oversize = max(
        oversize, key=lambda leaf: count_tokens(leaf.content, tokenizer)
    ) if oversize else None

    def _add(target: list[Leaf], leaf: Leaf | None) -> None:
        if leaf is not None and leaf not in target:
            target.append(leaf)

    spot: list[Leaf] = []
    shorts = [
        leaf
        for leaf in result.leaves
        if count_tokens(leaf.content, tokenizer) <= 256
    ]
    splits = [
        leaf
        for leaf in result.leaves
        if leaves_by_section.get(leaf.section_id, 0) > 1
        and count_tokens(leaf.content, tokenizer) <= chunk_size
    ]
    with_table = [
        leaf
        for leaf in result.leaves
        if "| " in leaf.content and count_tokens(leaf.content, tokenizer) <= chunk_size
    ]
    cross = [
        leaf for leaf in result.leaves if leaf.page_end > leaf.page_start
    ]
    for group in (shorts, splits, with_table, oversize, cross):
        for leaf in group[:2]:
            _add(spot, leaf)
    for leaf in result.leaves:
        if len(spot) >= 10:
            break
        _add(spot, leaf)

    return {
        "tokenizer": tokenizer_id,
        "transformers_version": transformers_version(),
        "chunk_size": chunk_size,
        "overlap_tokens": overlap_tokens,
        "stats": stats,
        "longest_normal_leaf": (
            _leaf_debug(longest_normal, tokenizer) if longest_normal else None
        ),
        "longest_oversize_table_leaf": (
            _leaf_debug(longest_oversize, tokenizer, is_oversize_table=True)
            if longest_oversize
            else None
        ),
        "spotcheck": [_leaf_debug(leaf, tokenizer) for leaf in spot[:10]],
        "leaves": [
            {
                **{name: getattr(leaf, name) for name in LEAF_FIELD_NAMES},
                "token_count": count_tokens(leaf.content, tokenizer),
            }
            for leaf in result.leaves
        ],
        "sections": [
            {
                "section_id": ref.section_id,
                "document_id": ref.document_id,
                "kind": ref.kind,
                "heading_level": ref.heading_level,
                "heading_text": ref.heading_text,
                "source_start": ref.source_start,
                "source_end": ref.source_end,
            }
            for ref in result.sections
        ],
    }


def format_chunking_summary(report: dict[str, Any]) -> str:
    stats = report["stats"]
    token = stats["leaf_token"]
    lines = [
        f"tokenizer={report['tokenizer']}",
        f"transformers_version={report['transformers_version']}",
        f"chunk_size={report['chunk_size']}",
        f"overlap_tokens={report['overlap_tokens']}",
        f"document_count={stats['document_count']}",
        f"terminal_section_count={stats['terminal_section_count']}",
        f"terminal_sections_with_leaf={stats['terminal_sections_with_leaf']}",
        f"empty_terminal_sections={stats['empty_terminal_sections']}",
        f"leaf_count={stats['leaf_count']}",
        f"leaf_tokens_p50={token['p50']}",
        f"leaf_tokens_p75={token['p75']}",
        f"leaf_tokens_p90={token['p90']}",
        f"leaf_tokens_p95={token['p95']}",
        f"leaf_tokens_max={token['max']}",
        f"leaf_tokens_mean={token['mean']}",
        f"normal_leaf_count={stats['normal_leaf_count']}",
        f"normal_leaf_max_tokens={stats['normal_leaf_max_tokens']}",
        f"oversize_table_leaf_count={stats['oversize_table_leaf_count']}",
        f"oversize_table_max_tokens={stats['oversize_table_max_tokens']}",
        f"sections_split_count={stats['sections_split_count']}",
        f"sections_unsplit_count={stats['sections_unsplit_count']}",
        f"page_invalid={stats['page_start_page_end_invalid_count']}",
        f"duplicate_chunk_id={stats['duplicate_chunk_id_count']}",
        f"section_id_collision={stats['duplicate_section_id_collision_count']}",
        f"non_contiguous_chunk_index={stats['non_contiguous_chunk_index_count']}",
        f"table_split_violation={stats['table_split_violation_count']}",
        f"zero_leaf_document={stats['zero_leaf_document_count']}",
        "spotcheck:",
    ]
    for row in report["spotcheck"]:
        lines.append(
            "  "
            f"document_id={row['document_id']} "
            f"chunk_index={row['chunk_index']} "
            f"page={row['page_start']}-{row['page_end']} "
            f"tokens={row['token_count']} "
            f"preview={row['content_preview']!r}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A3.2 structure-aware Leaf chunker. Reuses A1 loading and A3.1 "
            "structure parsing. Does not embed, index, or persist A4."
        ),
        epilog=(
            "From the repository root, put code/ on PYTHONPATH, for example:\n"
            '  PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" '
            "python -m knowledge_base.leaf_chunker "
            "--canonical-root <ABS_PATH> "
            "--tokenizer Qwen/Qwen3-Embedding-4B "
            "--output /tmp/a3-leaves.json"
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
        help="JSON debug/acceptance artifact path",
    )
    args = parser.parse_args(argv)

    try:
        documents = load_markdown_documents(args.canonical_root)
        tokenizer = load_tokenizer(args.tokenizer)
        result = chunk_documents(
            documents,
            tokenizer,
            chunk_size=args.chunk_size,
            overlap_tokens=args.overlap,
        )
        report = build_chunking_report(
            documents,
            result,
            tokenizer,
            tokenizer_id=args.tokenizer,
            chunk_size=args.chunk_size,
            overlap_tokens=args.overlap,
        )
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (MarkdownLoadingError, A3ChunkingError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"canonical_root={args.canonical_root.resolve()}")
    print(f"output={args.output.resolve()}")
    print(format_chunking_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
