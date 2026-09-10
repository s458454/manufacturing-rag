"""A6 production row preparation: A1 load -> A3 chunk -> A5 embed -> A6 join.

Runs on a GPU machine with Qwen3-Embedding-4B weights (e.g. ``shiyuan``), NOT
on the Milvus-only machine: ``chunk_documents``/``embed_leaves`` are imported
lazily inside ``prepare_rows`` (deferred import) so this module itself stays
importable, and its torch-free helper functions stay testable, on a machine
that has neither ``torch`` nor ``transformers`` installed. The module-level
surface is intentionally the same shape as ``a6_document_metadata.py`` and
``a6_collection.py``: no hard torch/transformers dependency at import time.

Output is a rows JSONL artifact consumed by
``knowledge_base.a6_collection build/ingest-update/rebuild`` on whichever
machine can reach Milvus. Does not modify A0-A5 contracts and does not
re-run A6 document metadata governance: it reads an already-produced
``document_metadata.jsonl`` (from ``knowledge_base.a6_document_metadata
extract``) as the source of truth for which documents are retrievable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from knowledge_base.a6_collection import (
    A6CollectionError,
    build_collection_rows,
    join_leaves_with_embeddings,
    select_retrievable_metadata,
)
from knowledge_base.a6_document_metadata import (
    A6MetadataError,
    DocumentMetadata,
    load_document_metadata_jsonl,
)
from knowledge_base.chunking_config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP_TOKENS
from knowledge_base.embedding_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MODEL_NAME_OR_PATH,
)
from knowledge_base.markdown_loader import (
    LoadedMarkdownDocument,
    MarkdownLoadingError,
    load_markdown_documents,
)


def select_retrievable_documents(
    documents: Sequence[LoadedMarkdownDocument],
    catalog: Sequence[DocumentMetadata],
) -> tuple[list[LoadedMarkdownDocument], dict[str, DocumentMetadata]]:
    """Torch-free: which A1 documents actually need A3/A5 work this run.

    Restricting to ACTIVE/VERSION_CONFLICT documents *before* tokenizing and
    embedding avoids wasting GPU time on DUPLICATE/HISTORICAL documents that
    will never be inserted (SS35).
    """

    if not catalog:
        raise A6MetadataError("document_metadata catalog is empty; nothing to prepare")
    retrievable_metadata = select_retrievable_metadata(catalog)
    if not retrievable_metadata:
        raise A6MetadataError(
            "No ACTIVE/VERSION_CONFLICT documents in the metadata catalog; "
            "nothing to embed"
        )
    by_id = {document.document_id: document for document in documents}
    missing = set(retrievable_metadata) - set(by_id)
    if missing:
        raise A6MetadataError(
            "document_metadata.jsonl references document_id(s) not found "
            f"under canonical_root: {sorted(missing)}"
        )
    retrievable_documents = [by_id[document_id] for document_id in retrievable_metadata]
    return retrievable_documents, retrievable_metadata


def prepare_rows(
    canonical_root: Path,
    metadata_path: Path,
    *,
    tokenizer_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    model_name_or_path: str = DEFAULT_MODEL_NAME_OR_PATH,
    device: str = DEFAULT_DEVICE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
) -> tuple[list[dict[str, Any]], int]:
    """Full A1->A3->A5->A6-join pipeline. Requires torch/transformers and
    (for a real model) a GPU; only reached once the torch-free filtering in
    ``select_retrievable_documents`` has already succeeded. Returns
    (rows, retrievable_document_count)."""

    # Deferred imports: keep this module importable without torch/transformers.
    from knowledge_base.dense_embedding import DenseEmbeddingConfig, embed_leaves
    from knowledge_base.leaf_chunker import chunk_documents
    from knowledge_base.token_count import load_tokenizer

    documents = load_markdown_documents(canonical_root)
    catalog = load_document_metadata_jsonl(metadata_path)
    retrievable_documents, retrievable_metadata = select_retrievable_documents(
        documents, catalog
    )

    tokenizer = load_tokenizer(tokenizer_id)
    chunking = chunk_documents(
        retrievable_documents,
        tokenizer,
        chunk_size=chunk_size,
        overlap_tokens=overlap_tokens,
    )
    embedding_config = DenseEmbeddingConfig(
        model_name_or_path=model_name_or_path,
        max_input_tokens=max_input_tokens,
        batch_size=batch_size,
        device=device,
    )
    dense_result = embed_leaves(chunking.leaves, config=embedding_config)
    dense_by_chunk_id = join_leaves_with_embeddings(chunking.leaves, dense_result)
    rows = build_collection_rows(chunking.leaves, dense_by_chunk_id, retrievable_metadata)
    return rows, len(retrievable_metadata)


def write_rows_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A6 row preparation: A1 load -> A3 chunk -> A5 embed -> A6 "
            "strict join. Requires torch/transformers + Qwen3-Embedding-4B "
            "weights (GPU machine, e.g. shiyuan). Reads an already-produced "
            "document_metadata.jsonl (knowledge_base.a6_document_metadata "
            "extract) to decide which documents to embed."
        ),
        epilog=(
            "From the repository root, put code/ on PYTHONPATH, for example:\n"
            '  PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" '
            "python -m knowledge_base.a6_prepare_rows "
            "--canonical-root <ABS_PATH> "
            "--metadata-jsonl outputs/index_metadata/document_metadata.jsonl "
            "--tokenizer Qwen/Qwen3-Embedding-4B "
            "--device cuda "
            "--output outputs/index_metadata/a6_rows.jsonl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        required=True,
        help="Output of `python -m knowledge_base.a6_document_metadata extract`",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    expected_errors: tuple[type[BaseException], ...] = (
        MarkdownLoadingError,
        A6MetadataError,
        A6CollectionError,
        OSError,
    )
    try:
        from knowledge_base.dense_embedding import A5EmbeddingError
        from knowledge_base.leaf_chunker import A3ChunkingError

        expected_errors = expected_errors + (A5EmbeddingError, A3ChunkingError)
    except ImportError as exc:
        print(
            f"error: missing dependency for row preparation (needs torch/transformers): {exc}",
            file=sys.stderr,
        )
        return 2

    try:
        rows, document_count = prepare_rows(
            args.canonical_root,
            args.metadata_jsonl,
            tokenizer_id=args.tokenizer,
            chunk_size=args.chunk_size,
            overlap_tokens=args.overlap,
            model_name_or_path=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
        )
        write_rows_jsonl(args.output, rows)
    except expected_errors as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"canonical_root={args.canonical_root.resolve()}")
    print(f"retrievable_document_count={document_count}")
    print(f"row_count={len(rows)}")
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
