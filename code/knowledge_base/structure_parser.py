"""A3.1 Markdown structure parser: headings, terminal spans, tables, page markers.

[CURRENT IMPLEMENTATION] uses markdown-it-py for block tokens and source maps.
This is not a permanent architecture constraint. The parser never rewrites
``LoadedMarkdownDocument.content`` and never reads A0 audit JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from knowledge_base.token_count import SectionProfileError

_FORMAL_PAGE_MARKER = re.compile(r"^<!-- PDF page ([1-9][0-9]*) -->$")
_BROKEN_PAGE_MARKER = re.compile(r"<!--\s*PDF\s*page\b", re.IGNORECASE)


@dataclass(frozen=True)
class Heading:
    level: int
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class MarkdownTableSpan:
    start_line: int
    end_line: int
    source: str


@dataclass(frozen=True)
class PageMarker:
    line: int
    page: int


@dataclass(frozen=True)
class TerminalSection:
    ordinal: int
    heading_level: int | None
    heading_text: str
    body_start_line: int
    body_end_line: int
    page_start: int
    page_end: int
    body_text: str
    is_unheaded_document: bool


@dataclass(frozen=True)
class ParsedMarkdownDocument:
    document_id: str
    headings: tuple[Heading, ...]
    tables: tuple[MarkdownTableSpan, ...]
    page_markers: tuple[PageMarker, ...]
    terminal_sections: tuple[TerminalSection, ...]
    unheaded_document_body: bool
    line_count: int = field(compare=False)


@dataclass(frozen=True)
class ContentBlock:
    start_line: int
    end_line: int
    source: str
    is_table: bool
    token_type: str


def _markdown_it() -> Any:
    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise SectionProfileError(
            "markdown-it-py is required for A3.1 structure parsing"
        ) from exc
    return MarkdownIt("commonmark", {"html": True}).enable("table")


def line_starts(content: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(content):
        if char == "\n":
            starts.append(index + 1)
    return starts


def slice_lines(content: str, starts: list[int], start: int, end: int) -> str:
    if start >= end or start >= len(starts):
        return ""
    char_start = starts[start]
    char_end = starts[end] if end < len(starts) else len(content)
    return content[char_start:char_end]


def _line_text(content: str, starts: list[int], line: int) -> str:
    return slice_lines(content, starts, line, line + 1)


def _is_blank_line(raw_line: str) -> bool:
    return raw_line.strip() == ""


def collect_page_markers(content: str, *, document_id: str) -> list[PageMarker]:
    starts = line_starts(content)
    markers: list[PageMarker] = []
    for line_no in range(len(starts)):
        raw = _line_text(content, starts, line_no)
        stripped = raw.strip()
        matched = _FORMAL_PAGE_MARKER.fullmatch(stripped)
        if matched is not None:
            markers.append(PageMarker(line=line_no, page=int(matched.group(1))))
            continue
        if _BROKEN_PAGE_MARKER.search(raw) is not None:
            raise SectionProfileError(
                "Malformed PDF page marker in "
                f"document_id={document_id} line={line_no + 1}: {stripped!r}"
            )
    if not markers:
        raise SectionProfileError(
            f"No PDF page marker in document_id={document_id}"
        )

    first_marker_line = markers[0].line
    for line_no in range(first_marker_line):
        raw = _line_text(content, starts, line_no)
        if not _is_blank_line(raw):
            raise SectionProfileError(
                "Body text appears before the first PDF page marker in "
                f"document_id={document_id} line={line_no + 1}"
            )

    previous_page = markers[0].page
    for marker in markers[1:]:
        if marker.page < previous_page:
            raise SectionProfileError(
                "PDF page marker goes backwards in "
                f"document_id={document_id} line={marker.line + 1}: "
                f"{previous_page} -> {marker.page}"
            )
        previous_page = marker.page

    marker_lines = {marker.line for marker in markers}
    for line_no in range(len(starts)):
        raw = _line_text(content, starts, line_no)
        if _is_blank_line(raw) or line_no in marker_lines:
            continue
        if _active_page(markers, line_no) is None:
            raise SectionProfileError(
                "Body text has no active PDF page marker in "
                f"document_id={document_id} line={line_no + 1}"
            )
    return markers


def _active_page(markers: list[PageMarker], line: int) -> int | None:
    current: int | None = None
    for marker in markers:
        if marker.line <= line:
            current = marker.page
        else:
            break
    return current


def _page_range(
    markers: list[PageMarker],
    body_start: int,
    body_end: int,
) -> tuple[int, int]:
    page_start = _active_page(markers, body_start)
    if page_start is None and body_start > 0:
        page_start = _active_page(markers, body_start - 1)
    if page_start is None:
        page_start = markers[0].page
    page_end = page_start
    for marker in markers:
        if body_start <= marker.line < body_end:
            page_end = marker.page
    return page_start, page_end


def _strip_markers(
    content: str,
    starts: list[int],
    body_start: int,
    body_end: int,
    marker_lines: set[int],
) -> str:
    parts: list[str] = []
    last = min(body_end, len(starts))
    for line_no in range(body_start, last):
        if line_no in marker_lines:
            continue
        parts.append(_line_text(content, starts, line_no))
    return "".join(parts)


def _has_non_blank_body(body_text: str) -> bool:
    return body_text.strip() != ""


def strip_page_markers(
    content: str,
    starts: list[int],
    start_line: int,
    end_line: int,
    marker_lines: set[int],
) -> str:
    """Return source ``[start_line, end_line)`` with PDF page markers removed."""

    return _strip_markers(content, starts, start_line, end_line, marker_lines)


def page_range_for_span(
    markers: tuple[PageMarker, ...] | list[PageMarker],
    start_line: int,
    end_line: int,
) -> tuple[int, int]:
    """Page range covering a source span, using A3.1 marker rules."""

    return _page_range(list(markers), start_line, end_line)


def heading_for_terminal_section(
    parsed: ParsedMarkdownDocument, section: TerminalSection
) -> Heading | None:
    """Return the semantic Markdown heading that owns *section*, if any.

    Preface spans bind to the parent heading, not to a synthetic section.
    Lead / unheaded document-root spans return None.
    """

    if section.heading_level is None:
        return None
    for heading in parsed.headings:
        if (
            heading.end_line == section.body_start_line
            and heading.level == section.heading_level
        ):
            return heading
    return None


def iter_section_blocks(
    content: str,
    parsed: ParsedMarkdownDocument,
    section: TerminalSection,
) -> tuple[ContentBlock, ...]:
    """List Markdown blocks inside a terminal Section source span.

    Page-marker html blocks and blank-only spans are omitted. Headings are
    outside the body span by construction. This does not change
    ``parse_markdown_document`` behavior.
    """

    starts = line_starts(content)
    marker_lines = {marker.line for marker in parsed.page_markers}
    tokens = _markdown_it().parse(content)
    blocks: list[ContentBlock] = []
    for token in tokens:
        if token.map is None:
            continue
        start_line, end_line = token.map
        if end_line <= section.body_start_line or start_line >= section.body_end_line:
            continue
        if token.level != 0:
            continue
        is_open = token.nesting == 1
        is_self = token.nesting == 0 and token.type in {
            "fence",
            "html_block",
            "hr",
            "code_block",
        }
        if not is_open and not is_self:
            continue
        if token.type == "heading_open":
            continue
        clipped_start = max(start_line, section.body_start_line)
        clipped_end = min(end_line, section.body_end_line)
        if token.type == "table_open":
            for table in parsed.tables:
                if table.start_line == start_line:
                    clipped_end = min(table.end_line, section.body_end_line)
                    break
        source = strip_page_markers(
            content, starts, clipped_start, clipped_end, marker_lines
        )
        if not _has_non_blank_body(source):
            continue
        stripped = source.strip()
        if _FORMAL_PAGE_MARKER.fullmatch(stripped) is not None:
            continue
        blocks.append(
            ContentBlock(
                start_line=clipped_start,
                end_line=clipped_end,
                source=source,
                is_table=token.type == "table_open",
                token_type=token.type,
            )
        )
    return tuple(blocks)


def _heading_text(tokens: list[Any], index: int) -> str:
    if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
        return tokens[index + 1].content.strip()
    return ""


def _collect_headings(tokens: list[Any]) -> list[Heading]:
    headings: list[Heading] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        tag = token.tag or ""
        if len(tag) != 2 or tag[0] != "h" or tag[1] not in "123456":
            continue
        start_line, end_line = token.map
        headings.append(
            Heading(
                level=int(tag[1]),
                text=_heading_text(tokens, index),
                start_line=start_line,
                end_line=end_line,
            )
        )
    return headings


def _collect_tables(
    tokens: list[Any], content: str, starts: list[int]
) -> list[MarkdownTableSpan]:
    tables: list[MarkdownTableSpan] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type != "table_open":
            index += 1
            continue
        depth = 1
        close_index = index + 1
        while close_index < len(tokens) and depth:
            if tokens[close_index].type == "table_open":
                depth += 1
            elif tokens[close_index].type == "table_close":
                depth -= 1
            close_index += 1
        start_line = token.map[0] if token.map is not None else None
        end_line = token.map[1] if token.map is not None else None
        close_token = tokens[close_index - 1] if close_index > index else None
        if (
            close_token is not None
            and close_token.map is not None
            and (end_line is None or close_token.map[1] > end_line)
        ):
            end_line = close_token.map[1]
        if start_line is None or end_line is None:
            index = close_index
            continue
        tables.append(
            MarkdownTableSpan(
                start_line=start_line,
                end_line=end_line,
                source=slice_lines(content, starts, start_line, end_line),
            )
        )
        index = close_index
    return tables


def _section_end(headings: list[Heading], index: int, line_count: int) -> int:
    current = headings[index]
    for later in headings[index + 1 :]:
        if later.level <= current.level:
            return later.start_line
    return line_count


def section_semantic_end(
    headings: tuple[Heading, ...] | list[Heading],
    index: int,
    line_count: int,
) -> int:
    """Half-open line end of the heading at *index*.

    End is the next heading with ``level <= current``, or ``line_count``.
    This is the A3.1 / A4 semantic Section span rule, not an identity anchor.
    """

    return _section_end(list(headings), index, line_count)


def _child_headings(
    headings: list[Heading], index: int, section_end: int
) -> list[Heading]:
    current = headings[index]
    children: list[Heading] = []
    for later in headings[index + 1 :]:
        if later.start_line >= section_end:
            break
        if later.level > current.level:
            children.append(later)
    return children


def _emit_terminal(
    *,
    ordinal: int,
    heading_level: int | None,
    heading_text: str,
    body_start: int,
    body_end: int,
    content: str,
    starts: list[int],
    markers: list[PageMarker],
    marker_lines: set[int],
    is_unheaded_document: bool,
) -> TerminalSection:
    page_start, page_end = _page_range(markers, body_start, body_end)
    body_text = _strip_markers(
        content, starts, body_start, body_end, marker_lines
    )
    return TerminalSection(
        ordinal=ordinal,
        heading_level=heading_level,
        heading_text=heading_text,
        body_start_line=body_start,
        body_end_line=body_end,
        page_start=page_start,
        page_end=page_end,
        body_text=body_text,
        is_unheaded_document=is_unheaded_document,
    )


def parse_markdown_document(
    content: str,
    *,
    document_id: str,
) -> ParsedMarkdownDocument:
    """Parse one Markdown document into headings, tables, and terminal spans.

    Heading level jumps do not invent missing intermediate headings. Duplicate
    heading text is allowed. Documents without headings become one
    document-root terminal span.
    """

    markers = collect_page_markers(content, document_id=document_id)
    starts = line_starts(content)
    line_count = len(starts)
    marker_lines = {marker.line for marker in markers}
    tokens = _markdown_it().parse(content)
    headings = _collect_headings(tokens)
    tables = _collect_tables(tokens, content, starts)

    terminals: list[TerminalSection] = []
    ordinal = 0

    if not headings:
        terminals.append(
            _emit_terminal(
                ordinal=ordinal,
                heading_level=None,
                heading_text="",
                body_start=0,
                body_end=line_count,
                content=content,
                starts=starts,
                markers=markers,
                marker_lines=marker_lines,
                is_unheaded_document=True,
            )
        )
        return ParsedMarkdownDocument(
            document_id=document_id,
            headings=tuple(headings),
            tables=tuple(tables),
            page_markers=tuple(markers),
            terminal_sections=tuple(terminals),
            unheaded_document_body=True,
            line_count=line_count,
        )

    first_heading_start = headings[0].start_line
    lead_text = _strip_markers(
        content, starts, 0, first_heading_start, marker_lines
    )
    if _has_non_blank_body(lead_text):
        terminals.append(
            _emit_terminal(
                ordinal=ordinal,
                heading_level=None,
                heading_text="",
                body_start=0,
                body_end=first_heading_start,
                content=content,
                starts=starts,
                markers=markers,
                marker_lines=marker_lines,
                is_unheaded_document=False,
            )
        )
        ordinal += 1

    for index, heading in enumerate(headings):
        section_end = _section_end(headings, index, line_count)
        children = _child_headings(headings, index, section_end)
        if children:
            preface_end = children[0].start_line
            preface_text = _strip_markers(
                content,
                starts,
                heading.end_line,
                preface_end,
                marker_lines,
            )
            if _has_non_blank_body(preface_text):
                terminals.append(
                    _emit_terminal(
                        ordinal=ordinal,
                        heading_level=heading.level,
                        heading_text=heading.text,
                        body_start=heading.end_line,
                        body_end=preface_end,
                        content=content,
                        starts=starts,
                        markers=markers,
                        marker_lines=marker_lines,
                        is_unheaded_document=False,
                    )
                )
                ordinal += 1
            continue
        terminals.append(
            _emit_terminal(
                ordinal=ordinal,
                heading_level=heading.level,
                heading_text=heading.text,
                body_start=heading.end_line,
                body_end=section_end,
                content=content,
                starts=starts,
                markers=markers,
                marker_lines=marker_lines,
                is_unheaded_document=False,
            )
        )
        ordinal += 1

    return ParsedMarkdownDocument(
        document_id=document_id,
        headings=tuple(headings),
        tables=tuple(tables),
        page_markers=tuple(markers),
        terminal_sections=tuple(terminals),
        unheaded_document_body=False,
        line_count=line_count,
    )
