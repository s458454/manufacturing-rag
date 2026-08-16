"""A1 Markdown loading: discovery, lossless UTF-8 read, and fail-fast errors."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.markdown_loader import (  # noqa: E402
    MarkdownLoadingError,
    load_markdown_documents,
)

LOSSLESS_MARKDOWN = (
    "\n"
    "<!-- PDF page 25 -->\n"
    "\n"
    "# 4 Welding\n"
    "## 4.2 Welding Requirements\n"
    "\n"
    "Paragraph A.\n"
    "\n"
    "- item 1\n"
    "- item 2\n"
    "\n"
    "Figure 4—Fillet Welds\n"
    "\n"
    "| A | B |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    "\n"
    "<!-- PDF page 26 -->\n"
    "\n"
    "中英混排: welding 要求\n"
)


def _write_document(root: Path, document_id: str, content: str | bytes) -> Path:
    document_dir = root / document_id
    document_dir.mkdir(parents=True)
    path = document_dir / "document.md"
    raw = content if isinstance(content, bytes) else content.encode("utf-8")
    path.write_bytes(raw)
    return path


def _symlink_or_skip(target: Path, link: Path, *, target_is_directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        pytest.skip(f"symlinks are not available in this environment: {exc}")


def test_t1_batch_load_is_sorted_by_document_id(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "doc-b", "b\n")
    _write_document(canonical, "doc-a", "a\n")

    loaded = load_markdown_documents(canonical)

    assert [document.document_id for document in loaded] == ["doc-a", "doc-b"]
    assert loaded[0].content == "a\n"
    assert loaded[1].content == "b\n"


def test_t2_lossless_markdown_is_byte_identical(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    path = _write_document(canonical, "weld-doc", LOSSLESS_MARKDOWN)
    original_bytes = path.read_bytes()

    loaded = load_markdown_documents(canonical)

    assert len(loaded) == 1
    assert loaded[0].content == LOSSLESS_MARKDOWN
    assert loaded[0].content.encode("utf-8") == original_bytes
    assert loaded[0].path == path.resolve()


def test_t3_does_not_recurse_into_nested_document_md(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "real-doc", "real\n")
    nested_parent = canonical / "nested"
    nested_parent.mkdir()
    _write_document(nested_parent, "other", "nested\n")

    loaded = load_markdown_documents(canonical)

    assert [document.document_id for document in loaded] == ["real-doc"]
    assert loaded[0].content == "real\n"


def test_t4_does_not_load_a0_audit_json(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    document_dir = canonical / "real-doc"
    markdown_path = _write_document(canonical, "real-doc", "# trusted body\n")
    (document_dir / "document.json").write_bytes(b'{"not": "markdown"}')
    (document_dir / "regions.json").write_bytes(b"[]")
    (document_dir / "quality_report.json").write_bytes(b"{}")
    tables = document_dir / "tables"
    tables.mkdir()
    (tables / "index.json").write_bytes(b'{"tables": []}')
    (tables / "doc-p0001-t001.json").write_bytes(b'{"cells": []}')

    loaded = load_markdown_documents(canonical)

    assert len(loaded) == 1
    assert loaded[0].document_id == "real-doc"
    assert loaded[0].content == "# trusted body\n"
    assert loaded[0].content != (document_dir / "document.json").read_text(
        encoding="utf-8"
    )
    assert loaded[0].path == markdown_path.resolve()


def test_t5_does_not_enter_failed_or_staging_trees(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "real-doc", "real\n")
    _write_document(canonical / ".failed", "failed-doc", "failed\n")
    _write_document(canonical / ".staging", "staging-doc", "staging\n")

    loaded = load_markdown_documents(canonical)

    assert [document.document_id for document in loaded] == ["real-doc"]


def test_t6_invalid_utf8_fails_the_entire_load(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "bad-doc", b"\xff\xfe not utf-8")

    with pytest.raises(MarkdownLoadingError, match="Invalid UTF-8"):
        load_markdown_documents(canonical)


def test_t6_invalid_utf8_does_not_return_partial_corpus(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "doc-a", "ok\n")
    _write_document(canonical, "doc-b", b"ok \x80 bad")

    with pytest.raises(MarkdownLoadingError, match="Invalid UTF-8"):
        load_markdown_documents(canonical)


def test_t7_empty_corpus_fails_fast(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()

    with pytest.raises(MarkdownLoadingError, match="No document.md found"):
        load_markdown_documents(canonical)


def test_t7_nested_only_corpus_is_empty(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical / ".failed", "failed-doc", "failed\n")

    with pytest.raises(MarkdownLoadingError, match="No document.md found"):
        load_markdown_documents(canonical)


def test_t8_directory_symlink_fails_fast(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_document(outside, "escaped", "secret\n")
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "real-doc", "real\n")
    _symlink_or_skip(
        outside / "escaped",
        canonical / "fake-doc",
        target_is_directory=True,
    )

    with pytest.raises(MarkdownLoadingError, match="symlinked document directory"):
        load_markdown_documents(canonical)


def test_t8_document_md_symlink_fails_fast(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "real-doc", "real\n")
    target = tmp_path / "elsewhere.md"
    target.write_bytes(b"escaped body\n")
    document_dir = canonical / "linked-doc"
    document_dir.mkdir()
    _symlink_or_skip(
        target,
        document_dir / "document.md",
        target_is_directory=False,
    )

    with pytest.raises(MarkdownLoadingError, match="symlinked document.md"):
        load_markdown_documents(canonical)


def test_missing_canonical_root_fails_fast(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(MarkdownLoadingError, match="does not exist"):
        load_markdown_documents(missing)


def test_canonical_root_enumeration_oserror_is_wrapped(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "real-doc", "ok\n")

    with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
        with pytest.raises(MarkdownLoadingError, match="Failed to enumerate") as caught:
            load_markdown_documents(canonical)

    assert isinstance(caught.value.__cause__, PermissionError)


def test_canonical_root_file_fails_fast(tmp_path: Path) -> None:
    root_file = tmp_path / "not-a-dir"
    root_file.write_bytes(b"nope")

    with pytest.raises(MarkdownLoadingError, match="not a directory"):
        load_markdown_documents(root_file)


def test_root_level_document_md_is_ignored(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "document.md").write_bytes(b"not a document_id child\n")

    with pytest.raises(MarkdownLoadingError, match="No document.md found"):
        load_markdown_documents(canonical)


def test_empty_document_md_is_loaded(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _write_document(canonical, "empty-doc", b"")

    loaded = load_markdown_documents(canonical)

    assert len(loaded) == 1
    assert loaded[0].document_id == "empty-doc"
    assert loaded[0].content == ""
