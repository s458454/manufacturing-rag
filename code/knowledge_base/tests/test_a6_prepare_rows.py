"""A6 row preparation: torch-free filtering logic + torch-gated orchestration.

The module under test defers its ``torch``/``transformers`` imports (via A3
``chunk_documents`` and A5 ``embed_leaves``) to inside ``prepare_rows()``, so
it can be imported and partially tested on a machine with neither installed
(this Windows dev machine). The orchestration test that actually calls
``prepare_rows()`` is gated with ``pytest.importorskip("torch")`` and stubs
out ``chunk_documents``/``embed_leaves`` so it does not need real Qwen
weights either; it is expected to run for real on a GPU machine (e.g.
``shiyuan``), and to skip (not fail) here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.a6_document_metadata import (  # noqa: E402
    A6MetadataError,
    DocumentMetadata,
    STATUS_ACTIVE,
    STATUS_DUPLICATE,
    STATUS_HISTORICAL,
    STATUS_VERSION_CONFLICT,
)
from knowledge_base.a6_prepare_rows import (  # noqa: E402
    select_retrievable_documents,
    write_rows_jsonl,
)
from knowledge_base.markdown_loader import LoadedMarkdownDocument  # noqa: E402


def _doc(document_id: str) -> LoadedMarkdownDocument:
    return LoadedMarkdownDocument(
        document_id=document_id,
        content=f"# {document_id}\nbody\n",
        path=Path(f"/canonical/{document_id}/document.md"),
    )


def _metadata(document_id: str, *, status: str) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        file_name=f"{document_id}.pdf",
        document_title=f"Title {document_id}",
        document_number=None,
        document_version=None,
        source_sha256="a" * 64,
        document_content_hash="b" * 64,
        finalized_at=None,
        effective_from=None,
        effective_to=None,
        ingested_at=1,
        status=status,
    )


# ---------------------------------------------------------------------------
# Torch-free: select_retrievable_documents
# ---------------------------------------------------------------------------


def test_select_retrievable_documents_filters_to_active_and_conflict() -> None:
    documents = [_doc("a"), _doc("b"), _doc("c"), _doc("d")]
    catalog = [
        _metadata("a", status=STATUS_ACTIVE),
        _metadata("b", status=STATUS_VERSION_CONFLICT),
        _metadata("c", status=STATUS_HISTORICAL),
        _metadata("d", status=STATUS_DUPLICATE),
    ]
    docs, metadata = select_retrievable_documents(documents, catalog)
    assert {d.document_id for d in docs} == {"a", "b"}
    assert set(metadata) == {"a", "b"}


def test_select_retrievable_documents_empty_catalog_fails() -> None:
    with pytest.raises(A6MetadataError, match="empty"):
        select_retrievable_documents([_doc("a")], [])


def test_select_retrievable_documents_no_retrievable_fails() -> None:
    catalog = [_metadata("a", status=STATUS_DUPLICATE)]
    with pytest.raises(A6MetadataError, match="nothing to embed"):
        select_retrievable_documents([_doc("a")], catalog)


def test_select_retrievable_documents_missing_under_canonical_root_fails() -> None:
    catalog = [_metadata("a", status=STATUS_ACTIVE), _metadata("missing-doc", status=STATUS_ACTIVE)]
    with pytest.raises(A6MetadataError, match="missing-doc"):
        select_retrievable_documents([_doc("a")], catalog)


# ---------------------------------------------------------------------------
# write_rows_jsonl (torch-free)
# ---------------------------------------------------------------------------


def test_write_rows_jsonl_roundtrip(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [{"chunk_id": "c1", "n": 1}, {"chunk_id": "c2", "n": 2}]
    path = tmp_path / "rows.jsonl"
    write_rows_jsonl(path, rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == rows


def test_write_rows_jsonl_empty_writes_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_rows_jsonl(path, [])
    assert path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# Torch-gated: prepare_rows orchestration (stubs chunk_documents/embed_leaves)
# ---------------------------------------------------------------------------


def test_prepare_rows_wires_chunk_embed_and_join(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")  # noqa: F841 -- gate only; not used directly
    import numpy as np

    from knowledge_base import a6_prepare_rows
    from knowledge_base.a6_document_metadata import write_document_metadata_jsonl
    from knowledge_base.leaf_chunker import ChunkingResult, Leaf
    from knowledge_base.embedding_config import EMBEDDING_DIMENSION

    canonical_root = tmp_path / "canonical"
    (canonical_root / "doc-a").mkdir(parents=True)
    (canonical_root / "doc-a" / "document.md").write_text("# Doc A\nbody\n", encoding="utf-8")
    (canonical_root / "doc-dup").mkdir(parents=True)
    (canonical_root / "doc-dup" / "document.md").write_text("# Dup\nbody\n", encoding="utf-8")

    catalog = [
        _metadata("doc-a", status=STATUS_ACTIVE),
        _metadata("doc-dup", status=STATUS_DUPLICATE),
    ]
    metadata_path = tmp_path / "document_metadata.jsonl"
    write_document_metadata_jsonl(metadata_path, catalog)

    leaf = Leaf(
        chunk_id="doc-a-chk-0",
        document_id="doc-a",
        section_id="doc-a-sec-0",
        chunk_index=0,
        page_start=1,
        page_end=1,
        content="hello world",
    )
    chunking_calls: list[list[str]] = []

    def fake_chunk_documents(documents, tokenizer, *, chunk_size, overlap_tokens):
        chunking_calls.append([d.document_id for d in documents])
        return ChunkingResult(leaves=(leaf,), sections=())

    class FakeDenseResult:
        chunk_ids = ("doc-a-chk-0",)
        vectors = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)

    embed_calls: list[list[str]] = []

    def fake_embed_leaves(leaves, *, config):
        embed_calls.append([leaf.chunk_id for leaf in leaves])
        return FakeDenseResult()

    monkeypatch.setattr("knowledge_base.leaf_chunker.chunk_documents", fake_chunk_documents)
    monkeypatch.setattr("knowledge_base.token_count.load_tokenizer", lambda tokenizer_id: object())
    monkeypatch.setattr("knowledge_base.dense_embedding.embed_leaves", fake_embed_leaves)

    rows, document_count = a6_prepare_rows.prepare_rows(
        canonical_root,
        metadata_path,
        tokenizer_id="stub-tokenizer",
    )

    assert document_count == 1
    assert chunking_calls == [["doc-a"]]  # doc-dup must never reach A3/A5
    assert embed_calls == [["doc-a-chk-0"]]
    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "doc-a-chk-0"
    assert rows[0]["document_id"] == "doc-a"
    assert "sparse_vector" not in rows[0]
