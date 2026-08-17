"""A4 hierarchical Parent-Child Section tree."""

from __future__ import annotations

import json
import sys
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.chunking_config import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_TOKENS,
)
from knowledge_base.leaf_chunker import (  # noqa: E402
    ChunkingResult,
    Leaf,
    SectionRef,
    chunk_parsed_documents,
)
from knowledge_base.markdown_loader import (  # noqa: E402
    LoadedMarkdownDocument,
)
from knowledge_base.section_hierarchy import (  # noqa: E402
    SECTION_NODE_FIELD_NAMES,
    A4HierarchyError,
    SectionNode,
    build_hierarchy_from_documents,
    build_section_hierarchy,
)
from knowledge_base.structure_parser import (  # noqa: E402
    Heading,
    PageMarker,
    ParsedMarkdownDocument,
    parse_markdown_document,
)
from knowledge_base.token_count import count_tokens  # noqa: E402


class StubTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> list[int]:
        assert add_special_tokens is False
        assert truncation is False
        return [ord(char) for char in text]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list]:
        ids = self.encode(text, add_special_tokens, truncation)
        offsets = [(index, index + 1) for index in range(len(text))]
        assert return_offsets_mapping is True
        return {"input_ids": ids, "offset_mapping": offsets}


STUB = StubTokenizer()


def _write_document(root: Path, document_id: str, content: str) -> Path:
    folder = root / document_id
    folder.mkdir(parents=True)
    path = folder / "document.md"
    path.write_bytes(content.encode("utf-8"))
    return path


def _loaded(root: Path, document_id: str, content: str) -> LoadedMarkdownDocument:
    path = _write_document(root, document_id, content)
    return LoadedMarkdownDocument(
        document_id=document_id,
        content=content,
        path=path.resolve(),
    )


def _para(token_count: int, fill: str = "a") -> str:
    return fill * token_count


def _build(
    tmp_path: Path,
    specs: list[tuple[str, str]],
):
    documents = [
        _loaded(tmp_path, document_id, content)
        for document_id, content in specs
    ]
    parsed = [
        parse_markdown_document(
            document.content, document_id=document.document_id
        )
        for document in documents
    ]
    result = chunk_parsed_documents(documents, parsed, STUB)
    hierarchy = build_section_hierarchy(documents, parsed, result)
    return documents, parsed, result, hierarchy


def _heading(hierarchy, text: str, *, index: int = 0) -> SectionNode:
    matches = [
        node
        for node in hierarchy.nodes
        if node.kind == "heading" and node.heading == text
    ]
    return matches[index]


def _root(hierarchy, document_id: str | None = None) -> SectionNode:
    matches = [
        node
        for node in hierarchy.nodes
        if node.kind == "document_root"
        and (document_id is None or node.document_id == document_id)
    ]
    assert matches
    return matches[0]


def test_t48_simple_hierarchy(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## H2\n"
        "h2 body\n"
        "### H3\n"
        "h3 body\n"
        "#### H4\n"
        "h4 body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    h2 = _heading(hierarchy, "H2")
    h3 = _heading(hierarchy, "H3")
    h4 = _heading(hierarchy, "H4")
    assert h3.parent_section_id == h2.section_id
    assert h4.parent_section_id == h3.section_id
    assert hierarchy.get_parent(h3.section_id) == h2
    assert hierarchy.get_parent(h4.section_id) == h3


def test_t49_siblings_share_parent(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## H2\n"
        "### A\n"
        "a body\n"
        "### B\n"
        "b body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    h2 = _heading(hierarchy, "H2")
    assert _heading(hierarchy, "A").parent_section_id == h2.section_id
    assert _heading(hierarchy, "B").parent_section_id == h2.section_id
    children = hierarchy.child_section_ids("doc", h2.section_id)
    assert children == (
        _heading(hierarchy, "A").section_id,
        _heading(hierarchy, "B").section_id,
    )


def test_t50_stack_pop_sibling_h3(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## H2\n"
        "### H3A\n"
        "#### H4\n"
        "h4 body\n"
        "### H3B\n"
        "h3b body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    h2 = _heading(hierarchy, "H2")
    h4 = _heading(hierarchy, "H4")
    h3b = _heading(hierarchy, "H3B")
    assert h3b.parent_section_id == h2.section_id
    assert h3b.parent_section_id != h4.section_id
    assert hierarchy.get_parent(h3b.section_id) == h2


def test_t51_level_jump_has_no_synthetic_h3(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "#### Material\n"
        "material body\n"
    )
    _documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    h2 = _heading(hierarchy, "Welding")
    h4 = _heading(hierarchy, "Material")
    assert h4.parent_section_id == h2.section_id
    assert {node.heading_level for node in hierarchy.nodes if node.kind == "heading"} == {
        2,
        4,
    }
    assert len([node for node in hierarchy.nodes if node.kind == "heading"]) == 2
    assert len([ref for ref in result.sections if ref.kind == "heading"]) == 2


def test_t52_no_h1_top_level_h2(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Scope\n"
        "scope body\n"
        "### Details\n"
        "detail body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    scope = _heading(hierarchy, "Scope")
    assert scope.heading_level == 2
    assert scope.parent_section_id is None
    assert hierarchy.get_parent(scope.section_id) is None
    assert hierarchy.get_ancestors(scope.section_id) == ()


def test_t53_duplicate_heading_text(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Parent\n"
        "### General\n"
        "first general\n"
        "### General\n"
        "second general\n"
    )
    _documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    first = _heading(hierarchy, "General", index=0)
    second = _heading(hierarchy, "General", index=1)
    assert first.section_id != second.section_id
    assert first.source_start != second.source_start
    heading_ids = [
        ref.section_id for ref in result.sections if ref.heading_text == "General"
    ]
    assert heading_ids == [first.section_id, second.section_id]


def test_t54_multiple_top_level_headings(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Alpha\n"
        "alpha body\n"
        "## Beta\n"
        "beta body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    alpha = _heading(hierarchy, "Alpha")
    beta = _heading(hierarchy, "Beta")
    assert alpha.parent_section_id is None
    assert beta.parent_section_id is None
    top = hierarchy.child_section_ids("doc", None)
    assert top == (alpha.section_id, beta.section_id)


def test_t55_empty_parent_heading_preserved(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "### General\n"
        "child body\n"
    )
    documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    welding = _heading(hierarchy, "Welding")
    general = _heading(hierarchy, "General")
    assert welding.section_id in {ref.section_id for ref in result.sections}
    assert all(leaf.section_id != welding.section_id for leaf in result.leaves)
    assert any(leaf.section_id == general.section_id for leaf in result.leaves)
    recovered = hierarchy.recover_section_text(welding.section_id, documents)
    assert "## Welding" in recovered
    assert "### General" in recovered


def test_t56_parent_preface_leaf_unchanged(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "preface body\n"
        "### General\n"
        "child body\n"
    )
    _documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    welding = _heading(hierarchy, "Welding")
    general = _heading(hierarchy, "General")
    preface = next(leaf for leaf in result.leaves if "preface body" in leaf.content)
    child = next(leaf for leaf in result.leaves if "child body" in leaf.content)
    assert preface.section_id == welding.section_id
    assert child.section_id == general.section_id
    assert hierarchy.get_section(preface.section_id) == welding


def test_t57_lead_document_root_is_sibling(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "lead text before scope\n"
        "\n"
        "## Scope\n"
        "scope body\n"
    )
    _documents, parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    root = _root(hierarchy, "doc")
    scope = _heading(hierarchy, "Scope")
    lead = next(leaf for leaf in result.leaves if "lead text before scope" in leaf.content)
    assert lead.section_id == root.section_id
    assert root.parent_section_id is None
    assert scope.parent_section_id is None
    assert root.heading is None
    assert root.heading_level is None
    assert root.source_start == 0
    assert root.source_end == parsed[0].headings[0].start_line
    assert root.source_end != [ref.source_end for ref in result.sections if ref.kind == "document_root"][0]
    top = hierarchy.child_section_ids("doc", None)
    assert top == (root.section_id, scope.section_id)


def test_t58_unheaded_document_root(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\njust body\n"
    _documents, parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    assert len(hierarchy.nodes) == 1
    root = _root(hierarchy)
    assert root.kind == "document_root"
    assert root.parent_section_id is None
    assert root.source_start == 0
    assert root.source_end == parsed[0].line_count
    assert result.leaves[0].section_id == root.section_id


def test_t59_semantic_h2_span(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## First\n"
        "### ChildA\n"
        "a body\n"
        "### ChildB\n"
        "b body\n"
        "## Second\n"
        "second body\n"
    )
    _documents, parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    first = _heading(hierarchy, "First")
    second = _heading(hierarchy, "Second")
    headings = {heading.text: heading for heading in parsed[0].headings}
    assert first.source_start == headings["First"].start_line
    assert first.source_end == headings["Second"].start_line
    assert second.source_start == headings["Second"].start_line
    assert second.source_end == parsed[0].line_count


def test_t60_semantic_h3_span_stops_at_next_h3(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Parent\n"
        "### ChildA\n"
        "#### Deep\n"
        "deep body\n"
        "### ChildB\n"
        "b body\n"
        "## Next\n"
        "next body\n"
    )
    _documents, parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    child_a = _heading(hierarchy, "ChildA")
    child_b = _heading(hierarchy, "ChildB")
    headings = {heading.text: heading for heading in parsed[0].headings}
    assert child_a.source_start == headings["ChildA"].start_line
    assert child_a.source_end == headings["ChildB"].start_line
    assert child_a.source_end > headings["Deep"].start_line
    assert _heading(hierarchy, "Parent").source_end == headings["Next"].start_line


def test_t61_half_open_source_span(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## FirstUniqueH2\n"
        "first body\n"
        "## ZZZNextSection\n"
        "second body\n"
    )
    documents, parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    first = _heading(hierarchy, "FirstUniqueH2")
    nxt = _heading(hierarchy, "ZZZNextSection")
    assert first.source_end == nxt.source_start
    recovered = hierarchy.recover_section_text(first.section_id, documents)
    assert "## FirstUniqueH2" in recovered
    assert "## ZZZNextSection" not in recovered
    assert "second body" not in recovered
    assert nxt.source_start == parsed[0].headings[1].start_line


def test_t62_recover_keeps_own_heading(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "intro\n"
    )
    documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    text = hierarchy.recover_section_text(
        _heading(hierarchy, "Welding").section_id, documents
    )
    assert "## Welding" in text


def test_t63_recover_keeps_child_headings(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "intro\n"
        "### General\n"
        "A\n"
        "#### Material\n"
        "deep\n"
        "### Inspection\n"
        "B\n"
    )
    documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    text = hierarchy.recover_section_text(
        _heading(hierarchy, "Welding").section_id, documents
    )
    assert "## Welding" in text
    assert "### General" in text
    assert "#### Material" in text
    assert "### Inspection" in text
    assert "intro" in text
    assert "\nA\n" in text or text.endswith("A\n") or "\nA" in text
    assert "B" in text


def test_t64_recover_removes_pdf_page_marker(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 25 -->\n"
        "\n"
        "## Welding\n"
        "Text A\n"
        "<!-- PDF page 26 -->\n"
        "Text B\n"
    )
    documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    text = hierarchy.recover_section_text(
        _heading(hierarchy, "Welding").section_id, documents
    )
    assert "<!-- PDF page 25 -->" not in text
    assert "<!-- PDF page 26 -->" not in text
    assert "Text A" in text
    assert "Text B" in text


def test_t65_preserves_normal_html_comments(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "<!-- TABLE caption=demo -->\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "<!-- /TABLE -->\n"
        "note\n"
    )
    documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    text = hierarchy.recover_section_text(
        _heading(hierarchy, "Welding").section_id, documents
    )
    assert "<!-- TABLE caption=demo -->" in text
    assert "<!-- /TABLE -->" in text
    assert "<!-- PDF page 1 -->" not in text
    assert "| 1 | 2 |" in text


def test_t66_section_page_single_page(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 7 -->\n"
        "\n"
        "## OnlySeven\n"
        "body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    node = _heading(hierarchy, "OnlySeven")
    assert node.page_start == 7
    assert node.page_end == 7


def test_t67_section_page_cross_page(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 25 -->\n"
        "\n"
        "## Welding\n"
        "start\n"
        "<!-- PDF page 26 -->\n"
        "middle\n"
        "<!-- PDF page 27 -->\n"
        "<!-- PDF page 28 -->\n"
        "end\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    node = _heading(hierarchy, "Welding")
    assert node.page_start == 25
    assert node.page_end == 28


def test_t68_empty_direct_body_covers_descendant_pages(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 25 -->\n"
        "\n"
        "## Welding\n"
        "### General\n"
        "A\n"
        "<!-- PDF page 26 -->\n"
        "### Inspection\n"
        "B\n"
        "<!-- PDF page 27 -->\n"
        "more\n"
        "<!-- PDF page 28 -->\n"
        "end\n"
    )
    _documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    welding = _heading(hierarchy, "Welding")
    assert all(leaf.section_id != welding.section_id for leaf in result.leaves)
    assert welding.page_start == 25
    assert welding.page_end == 28


def test_t69_every_leaf_section_id_resolves(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "lead text\n"
        "## H2\n"
        "preface\n"
        "### H3\n"
        "child\n"
        "#### H4\n"
        "deep\n"
    )
    _documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    for leaf in result.leaves:
        node = hierarchy.get_section(leaf.section_id)
        assert node.document_id == leaf.document_id


def test_t70_parent_exists(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## H2\n"
        "### H3\n"
        "#### H4\n"
        "body\n"
        "## Other\n"
        "other\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    for node in hierarchy.nodes:
        if node.parent_section_id is None:
            continue
        parent = hierarchy.get_section(node.parent_section_id)
        assert parent.section_id == node.parent_section_id


def test_t71_parent_has_lower_heading_level(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## H2\n"
        "#### H4\n"
        "body\n"
        "### H3\n"
        "h3 body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    for node in hierarchy.nodes:
        parent = hierarchy.get_parent(node.section_id)
        if parent is None or node.kind != "heading" or parent.kind != "heading":
            continue
        assert parent.heading_level < node.heading_level


def test_t72_no_hierarchy_cycles(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## A\n"
        "### B\n"
        "#### C\n"
        "body\n"
        "### D\n"
        "d body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    for node in hierarchy.nodes:
        seen: set[str] = set()
        current: SectionNode | None = node
        while current is not None:
            assert current.section_id not in seen
            seen.add(current.section_id)
            current = hierarchy.get_parent(current.section_id)


def test_t73_same_document_relationship(tmp_path: Path) -> None:
    first = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Alpha\n"
        "### AlphaChild\n"
        "a\n"
    )
    second = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Beta\n"
        "### BetaChild\n"
        "b\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path,
        [("doc-a", first), ("doc-b", second)],
    )
    for node in hierarchy.nodes:
        parent = hierarchy.get_parent(node.section_id)
        if parent is not None:
            assert parent.document_id == node.document_id
    alpha_top = hierarchy.child_section_ids("doc-a", None)
    beta_top = hierarchy.child_section_ids("doc-b", None)
    assert _heading(hierarchy, "Alpha").section_id in alpha_top
    assert _heading(hierarchy, "Beta").section_id not in alpha_top
    assert _heading(hierarchy, "Beta").section_id in beta_top
    assert _heading(hierarchy, "Alpha").section_id not in beta_top


def test_t74_deterministic_build(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "### General\n"
        "#### Material\n"
        "body\n"
        "### Inspection\n"
        "inspect\n"
        "## Brazing\n"
        "braze\n"
    )
    first = _build(tmp_path / "one", [("doc", content)])
    second = _build(tmp_path / "two", [("doc", content)])
    assert first[3].nodes == second[3].nodes
    for left, right in zip(first[3].nodes, second[3].nodes, strict=True):
        assert left.parent_section_id == right.parent_section_id
        assert left.source_start == right.source_start
        assert left.source_end == right.source_end
        assert left.page_start == right.page_start
        assert left.page_end == right.page_end


def test_t75_a3_immutable(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "preface body\n"
        "### General\n"
        "child body\n"
    )
    documents = [_loaded(tmp_path, "doc", content)]
    parsed = [
        parse_markdown_document(
            documents[0].content, document_id=documents[0].document_id
        )
    ]
    result = chunk_parsed_documents(documents, parsed, STUB)
    leaf_snapshot = tuple(result.leaves)
    section_snapshot = tuple(result.sections)
    leaf_fields = [
        (
            leaf.chunk_id,
            leaf.document_id,
            leaf.section_id,
            leaf.chunk_index,
            leaf.page_start,
            leaf.page_end,
            leaf.content,
        )
        for leaf in result.leaves
    ]
    section_fields = [
        (
            ref.section_id,
            ref.document_id,
            ref.kind,
            ref.heading_level,
            ref.heading_text,
            ref.source_start,
            ref.source_end,
        )
        for ref in result.sections
    ]
    original_content = documents[0].content
    hierarchy = build_section_hierarchy(documents, parsed, result)
    assert result.leaves == leaf_snapshot
    assert result.sections == section_snapshot
    assert [
        (
            leaf.chunk_id,
            leaf.document_id,
            leaf.section_id,
            leaf.chunk_index,
            leaf.page_start,
            leaf.page_end,
            leaf.content,
        )
        for leaf in result.leaves
    ] == leaf_fields
    assert [
        (
            ref.section_id,
            ref.document_id,
            ref.kind,
            ref.heading_level,
            ref.heading_text,
            ref.source_start,
            ref.source_end,
        )
        for ref in result.sections
    ] == section_fields
    assert documents[0].content == original_content
    assert {item.name for item in fields(SectionNode)} == set(
        SECTION_NODE_FIELD_NAMES
    )
    assert "content" not in SECTION_NODE_FIELD_NAMES
    assert "child_section_ids" not in SECTION_NODE_FIELD_NAMES
    assert hierarchy.get_section(result.leaves[0].section_id).section_id == (
        result.leaves[0].section_id
    )


def test_t76_no_leaf_reconstruction(tmp_path: Path) -> None:
    marker = "UNIQUE_OVERLAP_BLOCK_TOKEN" + "x" * 20
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Heading\n"
        f"{_para(700, 'a')}\n"
        "\n"
        f"{marker}\n"
        "<!-- PDF page 2 -->\n"
        f"{_para(700, 'c')}\n"
    )
    documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    assert len(result.leaves) >= 2
    concat = "".join(leaf.content for leaf in result.leaves)
    recovered = hierarchy.recover_section_text(
        _heading(hierarchy, "Heading").section_id, documents
    )
    assert content.count(marker) == 1
    assert recovered.count(marker) == 1
    assert concat.count(marker) >= 1
    assert recovered.count(marker) <= concat.count(marker)
    assert "UNIQUE_OVERLAP_BLOCK_TOKEN" in recovered


def test_t77_invalid_section_identity(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n## Heading\nbody\n"
    documents, parsed, result, _hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    bad_leaf = replace(result.leaves[0], section_id="sec_does_not_exist")
    bad_result = ChunkingResult(
        leaves=(bad_leaf,) + result.leaves[1:],
        sections=result.sections,
    )
    with pytest.raises(A4HierarchyError, match="no SectionNode"):
        build_section_hierarchy(documents, parsed, bad_result)


def test_t78_invalid_source_span(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n"
    document = _loaded(tmp_path, "doc", content)
    parsed = ParsedMarkdownDocument(
        document_id="doc",
        headings=(Heading(level=2, text="Heading", start_line=2, end_line=3),),
        tables=(),
        page_markers=(PageMarker(line=0, page=1),),
        terminal_sections=(),
        unheaded_document_body=False,
        line_count=2,
    )
    ref = SectionRef(
        section_id="sec_invalid_span",
        document_id="doc",
        kind="heading",
        heading_level=2,
        heading_text="Heading",
        source_start=2,
        source_end=3,
    )
    leaf = Leaf(
        chunk_id="chk_invalid",
        document_id="doc",
        section_id="sec_invalid_span",
        chunk_index=0,
        page_start=1,
        page_end=1,
        content="body",
    )
    with pytest.raises(A4HierarchyError, match="invalid source span"):
        build_section_hierarchy(
            [document],
            [parsed],
            ChunkingResult(leaves=(leaf,), sections=(ref,)),
        )


def test_t79_invalid_page_provenance(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n## Heading\nbody\n"
    documents = [_loaded(tmp_path, "doc", content)]
    parsed = parse_markdown_document(content, document_id="doc")
    result = chunk_parsed_documents(documents, [parsed], STUB)
    bad_parsed = ParsedMarkdownDocument(
        document_id=parsed.document_id,
        headings=parsed.headings,
        tables=parsed.tables,
        page_markers=(PageMarker(line=parsed.page_markers[0].line, page=0),),
        terminal_sections=parsed.terminal_sections,
        unheaded_document_body=parsed.unheaded_document_body,
        line_count=parsed.line_count,
    )
    with pytest.raises(A4HierarchyError, match="invalid page provenance"):
        build_section_hierarchy(documents, [bad_parsed], result)


def test_t80_full_regression_collects_a1_to_a4() -> None:
    tests_dir = Path(__file__).resolve().parent
    assert (tests_dir / "test_markdown_loader.py").is_file()
    assert (tests_dir / "test_document_registry.py").is_file()
    assert (tests_dir / "test_section_profile.py").is_file()
    assert (tests_dir / "test_leaf_chunker.py").is_file()
    assert (tests_dir / "test_section_hierarchy.py").is_file()


def test_empty_input_fails(tmp_path: Path) -> None:
    with pytest.raises(A4HierarchyError, match="No documents"):
        build_section_hierarchy([], [], ChunkingResult(leaves=(), sections=()))


def test_section_node_span_is_not_identity_anchor(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "intro\n"
        "### General\n"
        "A\n"
        "## Brazing\n"
        "B\n"
    )
    _documents, _parsed, result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    welding_ref = next(
        ref for ref in result.sections if ref.heading_text == "Welding"
    )
    welding = _heading(hierarchy, "Welding")
    brazing = _heading(hierarchy, "Brazing")
    assert welding.source_start == welding_ref.source_start
    assert welding.source_end != welding_ref.source_end
    assert welding.source_end == brazing.source_start


def test_get_ancestors_nearest_first(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# H1\n"
        "## H2\n"
        "### H3\n"
        "#### H4\n"
        "body\n"
    )
    _documents, _parsed, _result, hierarchy = _build(
        tmp_path, [("doc", content)]
    )
    h4 = _heading(hierarchy, "H4")
    ancestors = hierarchy.get_ancestors(h4.section_id)
    assert [node.heading for node in ancestors] == ["H3", "H2", "H1"]
    assert h4 not in ancestors


def test_cli_writes_debug_artifact(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(
        canonical,
        "doc",
        (
            "<!-- PDF page 1 -->\n"
            "\n"
            "## Welding\n"
            "intro\n"
            "### General\n"
            "A\n"
            "### Inspection\n"
            "B\n"
        ),
    )
    output = tmp_path / "a4-sections.json"
    from knowledge_base.section_hierarchy import main

    with patch(
        "knowledge_base.section_hierarchy.load_tokenizer",
        return_value=STUB,
    ):
        exit_code = main(
            [
                "--canonical-root",
                str(canonical),
                "--tokenizer",
                "Qwen/Qwen3-Embedding-4B",
                "--output",
                str(output),
            ]
        )
    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["chunk_size"] == DEFAULT_CHUNK_SIZE
    assert payload["overlap_tokens"] == DEFAULT_OVERLAP_TOKENS
    assert payload["stats"]["document_count"] == 1
    assert payload["stats"]["leaf_section_resolution_failures"] == 0
    assert payload["stats"]["missing_parent_count"] == 0
    assert payload["manual_recovery"]["contains_own_heading"] is True
    assert payload["manual_recovery"]["contains_child_headings"] is True
    assert payload["manual_recovery"]["contains_pdf_page_marker"] is False


def test_pipeline_helper_matches_build(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n## Heading\nbody\n"
    documents = [_loaded(tmp_path, "doc", content)]
    parsed, result, hierarchy = build_hierarchy_from_documents(documents, STUB)
    rebuilt = build_section_hierarchy(documents, parsed, result)
    assert rebuilt.nodes == hierarchy.nodes
    assert count_tokens(result.leaves[0].content, STUB) > 0
