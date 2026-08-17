"""A3.2 structure-aware Leaf chunking."""

from __future__ import annotations

import json
import random
import sys
import time
import uuid
from dataclasses import fields
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
    A3ChunkingError,
    LEAF_FIELD_NAMES,
    Leaf,
    chunk_documents,
    chunk_parsed_documents,
)
from knowledge_base.markdown_loader import (  # noqa: E402
    LoadedMarkdownDocument,
    load_markdown_documents,
)
from knowledge_base.structure_parser import parse_markdown_document  # noqa: E402
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


def _chunk(
    documents: list[LoadedMarkdownDocument],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
):
    return chunk_documents(
        documents,
        STUB,
        chunk_size=chunk_size,
        overlap_tokens=overlap_tokens,
    )


def _para(token_count: int, fill: str = "a") -> str:
    return fill * token_count


def test_t19_short_terminal_section_is_one_leaf(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# UNIQUEHEADINGXYZ\n"
        "\n"
        "short body\n"
    )
    document = _loaded(tmp_path, "doc", content)
    result = _chunk([document])
    assert len(result.leaves) == 1
    leaf = result.leaves[0]
    assert "short body" in leaf.content
    assert "UNIQUEHEADINGXYZ" not in leaf.content
    assert "<!-- PDF page" not in leaf.content


def test_t20_exactly_chunk_size_stays_one_leaf(tmp_path: Path) -> None:
    body = _para(DEFAULT_CHUNK_SIZE)
    content = f"<!-- PDF page 1 -->\n\n# Heading\n{body}"
    parsed = parse_markdown_document(content, document_id="doc")
    assert count_tokens(parsed.terminal_sections[0].body_text, STUB) == (
        DEFAULT_CHUNK_SIZE
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) == 1
    assert count_tokens(result.leaves[0].content, STUB) == DEFAULT_CHUNK_SIZE


def test_t21_multi_block_packing_does_not_split_block_c(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(250)}\n"
        "\n"
        f"{_para(300, 'b')}\n"
        "\n"
        f"{_para(350, 'c')}\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) == 2
    assert "c" * 50 in result.leaves[1].content
    assert "c" * 50 not in result.leaves[0].content
    assert "a" * 50 in result.leaves[0].content
    assert "b" * 50 in result.leaves[0].content


def test_t22_oversized_paragraph_uses_token_windows(tmp_path: Path) -> None:
    body = _para(800)
    content = f"<!-- PDF page 1 -->\n\n# Heading\n{body}"
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) >= 2
    for leaf in result.leaves:
        assert count_tokens(leaf.content, STUB) <= DEFAULT_CHUNK_SIZE
        assert "<!-- PDF page" not in leaf.content
    first = result.leaves[0].content
    second = result.leaves[1].content
    assert first[-DEFAULT_OVERLAP_TOKENS:] == second[:DEFAULT_OVERLAP_TOKENS]
    recovered = first + second[DEFAULT_OVERLAP_TOKENS:]
    assert body in recovered or recovered.replace("\n", "") == body


def test_t23_overlap_does_not_cross_section(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# First\n"
        "alpha-only-body\n"
        "# Second\n"
        "beta-only-body\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) == 2
    assert "alpha-only-body" in result.leaves[0].content
    assert "beta-only-body" in result.leaves[1].content
    assert "alpha-only-body" not in result.leaves[1].content


def test_t24_block_overlap_does_not_split_paragraph(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(500, 'a')}\n"
        "\n"
        f"{_para(80, 'b')}\n"
        "\n"
        f"{_para(700, 'c')}\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) >= 2
    for leaf in result.leaves:
        assert count_tokens(leaf.content, STUB) <= DEFAULT_CHUNK_SIZE
    last = result.leaves[-1].content
    if "b" * 10 in last and "c" * 10 in last:
        assert count_tokens(last, STUB) <= DEFAULT_CHUNK_SIZE


def test_t25_table_at_or_under_limit_stays_atomic(tmp_path: Path) -> None:
    table = (
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
    )
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(120)}\n"
        "\n"
        f"{table}"
        f"{_para(100, 'z')}\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    table_leaves = [leaf for leaf in result.leaves if "| A | B |" in leaf.content]
    assert len(table_leaves) == 1
    assert "| 1 | 2 |" in table_leaves[0].content
    assert table.count("| 1 | 2 |") == table_leaves[0].content.count("| 1 | 2 |")


def test_t26_table_that_does_not_fit_flushes_prose(tmp_path: Path) -> None:
    rows = "".join(f"| {index:03d} | value |\n" for index in range(40))
    table = "| A | B |\n|---|---|\n" + rows
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(400, 'p')}\n"
        "\n"
        f"{table}"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) >= 2
    assert "p" * 50 in result.leaves[0].content
    assert "| A | B |" not in result.leaves[0].content
    assert "| A | B |" in result.leaves[1].content
    assert "p" * 50 not in result.leaves[1].content


def test_t27_oversize_table_is_one_complete_leaf(tmp_path: Path) -> None:
    rows = "".join(f"| {index:04d} | {index:04d} |\n" for index in range(80))
    table = "| A | B |\n|---|---|\n" + rows
    content = f"<!-- PDF page 1 -->\n\n# Heading\n\n{table}"
    table_tokens = count_tokens(table, STUB)
    assert table_tokens > DEFAULT_CHUNK_SIZE
    result = _chunk([_loaded(tmp_path, "doc", content)])
    table_leaves = [leaf for leaf in result.leaves if "| A | B |" in leaf.content]
    assert len(table_leaves) == 1
    leaf = table_leaves[0]
    assert count_tokens(leaf.content, STUB) > DEFAULT_CHUNK_SIZE
    assert f"| {79:04d} | {79:04d} |" in leaf.content
    assert leaf.content.count("| A | B |") == 1


def test_t28_oversize_table_is_isolated_from_prose(tmp_path: Path) -> None:
    rows = "".join(f"| {index:04d} | {index:04d} |\n" for index in range(80))
    table = "| A | B |\n|---|---|\n" + rows
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        "before table\n"
        "\n"
        f"{table}"
        "\n"
        "after table\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) >= 3
    table_leaf = next(
        leaf for leaf in result.leaves if "| A | B |" in leaf.content
    )
    assert "before table" not in table_leaf.content
    assert "after table" not in table_leaf.content
    assert any("before table" in leaf.content for leaf in result.leaves)
    assert any("after table" in leaf.content for leaf in result.leaves)


def test_t29_table_is_not_copied_as_overlap(tmp_path: Path) -> None:
    table = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(700, 'a')}\n"
        "\n"
        f"{table}"
        f"{_para(700, 'z')}\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    table_hits = [
        index
        for index, leaf in enumerate(result.leaves)
        if "| A | B |" in leaf.content
    ]
    assert len(table_hits) == 1


def test_t30_single_page_leaf(tmp_path: Path) -> None:
    content = "<!-- PDF page 10 -->\n\n# Heading\nonly page ten\n"
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert result.leaves[0].page_start == 10
    assert result.leaves[0].page_end == 10


def test_t31_cross_page_leaf(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 10 -->\n"
        "\n"
        "# Heading\n"
        "page ten\n"
        "<!-- PDF page 11 -->\n"
        "page eleven\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) == 1
    assert result.leaves[0].page_start == 10
    assert result.leaves[0].page_end == 11
    assert "<!-- PDF page" not in result.leaves[0].content


def test_t32_overlap_expands_page_range(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 10 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(700, 'a')}\n"
        "\n"
        f"{_para(50, 'b')}\n"
        "<!-- PDF page 11 -->\n"
        f"{_para(700, 'c')}\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) >= 2
    later = result.leaves[-1]
    if "b" * 10 in later.content and "c" * 10 in later.content:
        assert later.page_start == 10
        assert later.page_end == 11


def test_t33_chunk_index_is_document_local_and_contiguous(
    tmp_path: Path,
) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# First\n"
        "one\n"
        "# Second\n"
        "two\n"
        "# Third\n"
        "three\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert [leaf.chunk_index for leaf in result.leaves] == [0, 1, 2]


def test_t34_second_document_resets_chunk_index(tmp_path: Path) -> None:
    first = "<!-- PDF page 1 -->\n\n# A\none\n# B\ntwo\n"
    second = "<!-- PDF page 1 -->\n\n# C\nthree\n"
    result = _chunk(
        [
            _loaded(tmp_path / "a", "doc-a", first),
            _loaded(tmp_path / "b", "doc-b", second),
        ]
    )
    a_leaves = [leaf for leaf in result.leaves if leaf.document_id == "doc-a"]
    b_leaves = [leaf for leaf in result.leaves if leaf.document_id == "doc-b"]
    assert [leaf.chunk_index for leaf in a_leaves] == [0, 1]
    assert [leaf.chunk_index for leaf in b_leaves] == [0]


def test_t35_section_id_is_deterministic(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Heading\nbody\n"
    first = _chunk([_loaded(tmp_path / "a", "doc", content)])
    second = _chunk([_loaded(tmp_path / "b", "doc", content)])
    assert first.leaves[0].section_id == second.leaves[0].section_id
    heading_refs = [ref for ref in first.sections if ref.kind == "heading"]
    assert heading_refs[0].section_id == first.leaves[0].section_id


def test_t36_duplicate_heading_text_gets_distinct_section_ids(
    tmp_path: Path,
) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Same\n"
        "first body\n"
        "# Same\n"
        "second body\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert result.leaves[0].section_id != result.leaves[1].section_id
    heading_refs = [ref for ref in result.sections if ref.kind == "heading"]
    assert len(heading_refs) == 2
    assert heading_refs[0].section_id != heading_refs[1].section_id


def test_t37_chunk_id_is_deterministic(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Heading\nbody\n"
    first = _chunk([_loaded(tmp_path / "a", "doc", content)])
    second = _chunk([_loaded(tmp_path / "b", "doc", content)])
    assert first.leaves[0].chunk_id == second.leaves[0].chunk_id


def test_t38_content_change_changes_chunk_id(tmp_path: Path) -> None:
    first = _chunk(
        [_loaded(tmp_path / "a", "doc", "<!-- PDF page 1 -->\n\n# H\none\n")]
    )
    second = _chunk(
        [_loaded(tmp_path / "b", "doc", "<!-- PDF page 1 -->\n\n# H\ntwo\n")]
    )
    assert first.leaves[0].chunk_id != second.leaves[0].chunk_id


def test_t39_chunk_config_change_changes_identity_when_boundaries_move(
    tmp_path: Path,
) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(500, 'a')}\n"
        "\n"
        f"{_para(500, 'b')}\n"
    )
    wide = _chunk(
        [_loaded(tmp_path / "a", "doc", content)],
        chunk_size=DEFAULT_CHUNK_SIZE,
        overlap_tokens=DEFAULT_OVERLAP_TOKENS,
    )
    narrow = _chunk(
        [_loaded(tmp_path / "b", "doc", content)],
        chunk_size=400,
        overlap_tokens=40,
    )
    assert [leaf.chunk_id for leaf in wide.leaves] != [
        leaf.chunk_id for leaf in narrow.leaves
    ]


def test_t40_ids_ignore_time_uuid_and_random(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Heading\nbody\n"
    with (
        patch("time.time", return_value=1),
        patch("uuid.uuid4", return_value=uuid.UUID(int=1)),
        patch("random.random", return_value=0.1),
    ):
        first = _chunk([_loaded(tmp_path / "a", "doc", content)])
    with (
        patch("time.time", return_value=999),
        patch("uuid.uuid4", return_value=uuid.UUID(int=99)),
        patch("random.random", return_value=0.9),
    ):
        second = _chunk([_loaded(tmp_path / "b", "doc", content)])
    assert first.leaves[0].chunk_id == second.leaves[0].chunk_id
    assert first.leaves[0].section_id == second.leaves[0].section_id
    assert time.time and uuid.uuid4 and random.random


def test_t41_empty_terminal_section_is_skipped(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Empty\n"
        "# HasBody\n"
        "visible\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) == 1
    assert "visible" in result.leaves[0].content
    heading_refs = [ref for ref in result.sections if ref.kind == "heading"]
    assert {ref.heading_text for ref in heading_refs} == {"Empty", "HasBody"}


def test_t42_whole_document_zero_leaf_fails(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Empty\n"
    with pytest.raises(A3ChunkingError, match="0 Leaf"):
        _chunk([_loaded(tmp_path, "doc", content)])


@pytest.mark.parametrize(
    "chunk_size, overlap_tokens",
    [(0, 0), (768, -1), (768, 768), (768, 900)],
)
def test_t43_invalid_configuration_fails_fast(
    tmp_path: Path, chunk_size: int, overlap_tokens: int
) -> None:
    content = "<!-- PDF page 1 -->\n\n# Heading\nbody\n"
    with pytest.raises(A3ChunkingError):
        _chunk(
            [_loaded(tmp_path, "doc", content)],
            chunk_size=chunk_size,
            overlap_tokens=overlap_tokens,
        )


def test_t44_no_silent_truncation(tmp_path: Path) -> None:
    body = _para(2000, "q")
    content = f"<!-- PDF page 1 -->\n\n# Heading\n{body}"
    result = _chunk([_loaded(tmp_path, "doc", content)])
    recovered = result.leaves[0].content
    for leaf in result.leaves[1:]:
        overlap = min(DEFAULT_OVERLAP_TOKENS, len(recovered), len(leaf.content))
        recovered = recovered + leaf.content[overlap:]
    assert "q" * 2000 in recovered.replace("\n", "")


def test_t45_a1_content_is_unchanged(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Heading\nbody\n"
    document = _loaded(tmp_path, "doc", content)
    original = document.content
    _chunk([document])
    assert document.content == original
    assert document.content is original


def test_t46_leaf_schema_is_minimal(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Heading\nbody\n"
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert [item.name for item in fields(Leaf)] == list(LEAF_FIELD_NAMES)
    payload = result.leaves[0].__dict__
    assert set(payload) == set(LEAF_FIELD_NAMES)
    assert "heading" not in payload
    assert "token_count" not in payload
    assert "is_table" not in payload
    assert "document_title" not in payload


def test_t47_knowledge_base_tests_are_present() -> None:
    tests_dir = Path(__file__).resolve().parent
    assert (tests_dir / "test_markdown_loader.py").is_file()
    assert (tests_dir / "test_document_registry.py").is_file()
    assert (tests_dir / "test_section_profile.py").is_file()
    assert (tests_dir / "test_leaf_chunker.py").is_file()


def test_preface_leaf_binds_to_parent_heading(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Welding\n"
        "preface body\n"
        "### General\n"
        "child body\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    heading_refs = {ref.heading_text: ref for ref in result.sections if ref.kind == "heading"}
    assert set(heading_refs) == {"Welding", "General"}
    assert all(ref.kind in {"heading", "document_root"} for ref in result.sections)
    preface = next(leaf for leaf in result.leaves if "preface body" in leaf.content)
    child = next(leaf for leaf in result.leaves if "child body" in leaf.content)
    assert preface.section_id == heading_refs["Welding"].section_id
    assert child.section_id == heading_refs["General"].section_id
    assert preface.section_id != child.section_id


def test_lead_and_unheaded_bind_to_document_root(tmp_path: Path) -> None:
    lead = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "lead text\n"
        "\n"
        "# Title\n"
        "titled body\n"
    )
    unheaded = "<!-- PDF page 1 -->\n\njust body\n"
    result = _chunk(
        [
            _loaded(tmp_path / "a", "with-headings", lead),
            _loaded(tmp_path / "b", "no-headings", unheaded),
        ]
    )
    lead_leaf = next(
        leaf for leaf in result.leaves if "lead text" in leaf.content
    )
    title_leaf = next(
        leaf for leaf in result.leaves if "titled body" in leaf.content
    )
    root_leaf = next(
        leaf for leaf in result.leaves if "just body" in leaf.content
    )
    lead_root = next(
        ref
        for ref in result.sections
        if ref.document_id == "with-headings" and ref.kind == "document_root"
    )
    unheaded_root = next(
        ref
        for ref in result.sections
        if ref.document_id == "no-headings" and ref.kind == "document_root"
    )
    assert lead_leaf.section_id == lead_root.section_id
    assert title_leaf.section_id != lead_leaf.section_id
    assert root_leaf.section_id == unheaded_root.section_id
    assert "preface" not in {ref.kind for ref in result.sections}
    assert "lead" not in {ref.kind for ref in result.sections}


def test_cli_reuses_a1_loader(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(
        canonical,
        "doc",
        "<!-- PDF page 1 -->\n\n# Heading\nbody\n",
    )
    output = tmp_path / "leaves.json"
    from knowledge_base.leaf_chunker import main

    with patch(
        "knowledge_base.leaf_chunker.load_tokenizer",
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
    assert payload["stats"]["leaf_count"] >= 1


def test_chunk_parsed_documents_matches_chunk_documents(tmp_path: Path) -> None:
    headed = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "lead text before title\n"
        "\n"
        "# Title\n"
        "titled body\n"
        "## Child\n"
        "child body\n"
        "### Grandchild\n"
        "deep body\n"
        "## Sibling\n"
        "sibling body\n"
    )
    unheaded = "<!-- PDF page 2 -->\n\njust body\n"
    nested = (
        "<!-- PDF page 3 -->\n"
        "\n"
        "## Welding\n"
        "preface body\n"
        "### General\n"
        "child body\n"
        "#### Material\n"
        "leaf body\n"
        "### General\n"
        "duplicate heading body\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
    )
    documents = [
        _loaded(tmp_path / "a", "with-lead", headed),
        _loaded(tmp_path / "b", "no-headings", unheaded),
        _loaded(tmp_path / "c", "nested", nested),
    ]
    via_old = chunk_documents(documents, STUB)
    parsed = [
        parse_markdown_document(
            document.content, document_id=document.document_id
        )
        for document in documents
    ]
    via_new = chunk_parsed_documents(documents, parsed, STUB)
    assert via_old == via_new
    assert via_old.leaves == via_new.leaves
    assert via_old.sections == via_new.sections
    assert len(via_old.leaves) == len(via_new.leaves)
    for left, right in zip(via_old.leaves, via_new.leaves, strict=True):
        assert left.chunk_id == right.chunk_id
        assert left.document_id == right.document_id
        assert left.section_id == right.section_id
        assert left.chunk_index == right.chunk_index
        assert left.page_start == right.page_start
        assert left.page_end == right.page_end
        assert left.content == right.content
    for left, right in zip(via_old.sections, via_new.sections, strict=True):
        assert left.section_id == right.section_id
        assert left.document_id == right.document_id
        assert left.kind == right.kind
        assert left.heading_level == right.heading_level
        assert left.heading_text == right.heading_text
        assert left.source_start == right.source_start
        assert left.source_end == right.source_end


def test_overlap_yields_when_it_would_exceed_chunk_size(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        f"{_para(80, 'a')}\n"
        "\n"
        f"{_para(740, 'b')}\n"
    )
    result = _chunk([_loaded(tmp_path, "doc", content)])
    assert len(result.leaves) == 2
    assert "b" * 20 in result.leaves[1].content
    assert "a" * 20 not in result.leaves[1].content
    for leaf in result.leaves:
        assert count_tokens(leaf.content, STUB) <= DEFAULT_CHUNK_SIZE
