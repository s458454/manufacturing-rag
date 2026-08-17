"""A3.1 structure parser and terminal Section profiler."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.markdown_loader import (  # noqa: E402
    LoadedMarkdownDocument,
    load_markdown_documents,
)
from knowledge_base.section_profile import (  # noqa: E402
    format_profile_summary,
    main,
    profile_documents,
)
from knowledge_base.structure_parser import (  # noqa: E402
    parse_markdown_document,
)
from knowledge_base.token_count import (  # noqa: E402
    SectionProfileError,
    count_tokens,
)


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


STUB = StubTokenizer()
TOKENIZER_ID = "stub-tokenizer"


def _write_document(root: Path, document_id: str, content: str) -> Path:
    folder = root / document_id
    folder.mkdir(parents=True)
    path = folder / "document.md"
    path.write_bytes(content.encode("utf-8"))
    return path


def _loaded(
    root: Path, document_id: str, content: str
) -> LoadedMarkdownDocument:
    path = _write_document(root, document_id, content)
    return LoadedMarkdownDocument(
        document_id=document_id,
        content=content,
        path=path.resolve(),
    )


def _profile(
    documents: list[LoadedMarkdownDocument],
) -> dict[str, object]:
    return profile_documents(documents, TOKENIZER_ID, tokenizer=STUB)


def test_t1_nested_headings_identify_terminal_section(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Alpha\n"
        "## Beta\n"
        "### Gamma\n"
        "gamma body\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    assert [heading.level for heading in parsed.headings] == [1, 2, 3]
    assert [section.heading_text for section in parsed.terminal_sections] == [
        "Gamma"
    ]
    assert parsed.terminal_sections[0].heading_level == 3
    assert "gamma body" in parsed.terminal_sections[0].body_text
    assert "Alpha" not in parsed.terminal_sections[0].body_text
    assert "Beta" not in parsed.terminal_sections[0].body_text


def test_t2_sibling_h2_under_one_h1(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Parent\n"
        "## First\n"
        "first body\n"
        "## Second\n"
        "second body\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    terminals = parsed.terminal_sections
    assert [section.heading_text for section in terminals] == ["First", "Second"]
    assert [section.heading_level for section in terminals] == [2, 2]
    assert "first body" in terminals[0].body_text
    assert "second body" in terminals[1].body_text
    assert "second body" not in terminals[0].body_text


def test_t3_heading_level_jump_does_not_invent_h2(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Alpha\n"
        "### Gamma\n"
        "jumped body\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    assert [heading.level for heading in parsed.headings] == [1, 3]
    assert 2 not in {heading.level for heading in parsed.headings}
    assert [section.heading_text for section in parsed.terminal_sections] == [
        "Gamma"
    ]
    assert parsed.terminal_sections[0].heading_level == 3


def test_t4_duplicate_heading_text_does_not_conflict(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Same\n"
        "first\n"
        "# Same\n"
        "second\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    terminals = parsed.terminal_sections
    assert len(terminals) == 2
    assert terminals[0].heading_text == terminals[1].heading_text == "Same"
    assert terminals[0].ordinal != terminals[1].ordinal
    assert "first" in terminals[0].body_text
    assert "second" in terminals[1].body_text


def test_t5_unheaded_document_is_document_root_span(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\njust body\n"
    parsed = parse_markdown_document(content, document_id="doc")

    assert parsed.headings == ()
    assert parsed.unheaded_document_body is True
    assert len(parsed.terminal_sections) == 1
    section = parsed.terminal_sections[0]
    assert section.heading_level is None
    assert section.heading_text == ""
    assert "just body" in section.body_text
    profile = _profile([_loaded(tmp_path, "doc", content)])
    assert profile["anomalies"]["unheaded_document_body"] == [
        {"document_id": "doc"}
    ]


def test_t6_page_marker_is_excluded_from_body_text_and_tokens(
    tmp_path: Path,
) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        "hello\n"
        "<!-- PDF page 2 -->\n"
        "world\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")
    body = parsed.terminal_sections[0].body_text

    assert "<!-- PDF page" not in body
    assert "hello" in body
    assert "world" in body
    stripped_tokens = count_tokens(body, STUB)
    with_markers = count_tokens(
        "hello\n<!-- PDF page 2 -->\nworld\n",
        STUB,
    )
    assert stripped_tokens < with_markers
    assert stripped_tokens == count_tokens("hello\nworld\n", STUB)


def test_t7_single_page_section_has_equal_page_range(tmp_path: Path) -> None:
    content = "<!-- PDF page 4 -->\n\n# Heading\nbody\n"
    parsed = parse_markdown_document(content, document_id="doc")
    section = parsed.terminal_sections[0]

    assert section.page_start == section.page_end == 4


def test_t8_cross_page_section_records_page_start_and_end(
    tmp_path: Path,
) -> None:
    content = (
        "<!-- PDF page 4 -->\n"
        "\n"
        "# Heading\n"
        "page four\n"
        "<!-- PDF page 5 -->\n"
        "page five\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")
    section = parsed.terminal_sections[0]

    assert section.page_start == 4
    assert section.page_end == 5


def test_t9_body_before_first_page_marker_fails_fast(tmp_path: Path) -> None:
    content = "leaked body\n\n<!-- PDF page 1 -->\n\n# Heading\nbody\n"

    with pytest.raises(SectionProfileError, match="before the first PDF page"):
        parse_markdown_document(content, document_id="doc")


def test_t10_missing_page_marker_fails_fast(tmp_path: Path) -> None:
    content = "# Heading\n\nbody\n"

    with pytest.raises(SectionProfileError, match="No PDF page marker"):
        parse_markdown_document(content, document_id="doc")


def test_t11_page_number_going_backwards_fails_fast(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 2 -->\n"
        "\n"
        "# Heading\n"
        "<!-- PDF page 1 -->\n"
        "body\n"
    )

    with pytest.raises(SectionProfileError, match="goes backwards"):
        parse_markdown_document(content, document_id="doc")


def test_t12_markdown_table_is_atomic_source_span(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        "\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n"
        "\n"
        "after table\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    assert len(parsed.tables) == 1
    source = parsed.tables[0].source
    assert "| A | B |" in source
    assert "| 3 | 4 |" in source
    assert "after table" not in source
    assert source.strip().startswith("| A | B |")
    assert source.strip().endswith("| 3 | 4 |")


def test_t13_pipes_in_prose_are_not_tables(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        "\n"
        "The ratio a | b | c is prose.\n"
        "\n"
        "```\n"
        "| not | a | table |\n"
        "```\n"
        "\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    assert len(parsed.tables) == 1
    assert "| A | B |" in parsed.tables[0].source
    assert "The ratio a | b | c is prose." not in parsed.tables[0].source
    assert "| not | a | table |" not in parsed.tables[0].source


def test_t14_heading_text_is_excluded_from_body_profiling(
    tmp_path: Path,
) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# UNIQUEHEADINGXYZ\n"
        "\n"
        "only body\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")
    body = parsed.terminal_sections[0].body_text

    assert "UNIQUEHEADINGXYZ" not in body
    assert "only body" in body
    assert parsed.terminal_sections[0].heading_text == "UNIQUEHEADINGXYZ"
    assert count_tokens(body, STUB) == count_tokens("\nonly body\n", STUB)


def test_t15_a1_content_is_not_modified(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Heading\nbody\n"
    document = _loaded(tmp_path, "doc", content)
    original = document.content

    _profile([document])

    assert document.content == original
    assert document.content == content
    assert document.content is original


def test_t16_profile_statistics_are_deterministic(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        "hello table\n"
        "\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
    )
    first = _profile([_loaded(tmp_path / "a", "doc", content)])
    second = _profile([_loaded(tmp_path / "b", "doc", content)])
    first["transformers_version"] = "pinned"
    second["transformers_version"] = "pinned"

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_t17_audit_json_is_not_read_as_body(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n# Trusted\ntrusted body\n"
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    document_dir = canonical / "real-doc"
    _write_document(canonical, "real-doc", content)
    (document_dir / "document.json").write_bytes(
        b"# Audit Heading\n\n| leak | table |\n|---|---|\n| 9 | 9 |\n"
    )
    (document_dir / "regions.json").write_bytes(b'{"ocr_text": "AUDIT"}')
    (document_dir / "quality_report.json").write_bytes(b'{"ocr_text": "AUDIT"}')

    documents = load_markdown_documents(canonical)
    parsed = parse_markdown_document(
        documents[0].content, document_id=documents[0].document_id
    )
    profile = _profile(documents)

    assert "Audit Heading" not in documents[0].content
    assert "trusted body" in parsed.terminal_sections[0].body_text
    assert parsed.terminal_sections[0].heading_text == "Trusted"
    assert all(
        "AUDIT" not in json.dumps(profile)
        for _ in [0]
    )
    assert "Audit Heading" not in json.dumps(profile)
    assert "leak" not in parsed.tables[0].source if parsed.tables else True
    assert parsed.tables == ()


def test_t18_existing_a1_a2_tests_remain_collectable() -> None:
    tests_dir = Path(__file__).resolve().parent
    assert (tests_dir / "test_markdown_loader.py").is_file()
    assert (tests_dir / "test_document_registry.py").is_file()


def test_parent_preface_is_profiled_as_terminal_span() -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Parent\n"
        "preface body\n"
        "## Child\n"
        "child body\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    assert [section.heading_text for section in parsed.terminal_sections] == [
        "Parent",
        "Child",
    ]
    assert "preface body" in parsed.terminal_sections[0].body_text
    assert "child body" in parsed.terminal_sections[1].body_text


def test_table_wrapper_comment_is_not_a_page_marker() -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "# Heading\n"
        "\n"
        "<!-- TABLE id=doc-p0001-t001 page=1 source=native -->\n"
        "\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "<!-- /TABLE -->\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    assert [marker.page for marker in parsed.page_markers] == [1]
    assert len(parsed.tables) == 1


def test_repeated_same_page_marker_is_not_backwards() -> None:
    content = (
        "<!-- PDF page 3 -->\n"
        "\n"
        "# Heading\n"
        "<!-- PDF page 3 -->\n"
        "still page three\n"
    )
    parsed = parse_markdown_document(content, document_id="doc")

    assert parsed.terminal_sections[0].page_start == 3
    assert parsed.terminal_sections[0].page_end == 3


def test_malformed_page_marker_fails_fast() -> None:
    content = "<!-- pdf page 1 -->\n\n# Heading\nbody\n"

    with pytest.raises(SectionProfileError, match="Malformed PDF page marker"):
        parse_markdown_document(content, document_id="doc")


def test_zero_page_marker_is_malformed() -> None:
    content = "<!-- PDF page 0 -->\n\n# Heading\nbody\n"

    with pytest.raises(SectionProfileError, match="Malformed PDF page marker"):
        parse_markdown_document(content, document_id="doc")


def test_cli_reuses_a1_loader_and_writes_json(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(
        canonical,
        "doc",
        "<!-- PDF page 1 -->\n\n# Heading\nbody\n",
    )
    output = tmp_path / "profile.json"

    with patch(
        "knowledge_base.section_profile.load_tokenizer",
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
    assert payload["tokenizer"] == "Qwen/Qwen3-Embedding-4B"
    assert payload["document_count"] == 1
    assert payload["terminal_section_count"] == 1
    summary = format_profile_summary(payload)
    assert "body_tokens_p50=" in summary


def test_empty_document_list_fails_fast() -> None:
    with pytest.raises(SectionProfileError, match="No documents provided"):
        _profile([])
