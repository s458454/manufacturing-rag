"""Startup assembly for B6. Request handling only reads this memory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge_base.markdown_loader import load_markdown_documents
from knowledge_base.section_store import SectionStore, load_sections_jsonl
from retrieval.b5.types import RerankingResult
from retrieval.b6.catalog import LeafCatalog, load_leaf_catalog
from retrieval.b6.constants import NOMINAL_EVIDENCE_BUDGET
from retrieval.b6.errors import RecoveryConfigurationError
from retrieval.b6.recover import recover
from retrieval.b6.tokens import load_generation_tokenizer
from retrieval.b6.types import RecoveryResult


@dataclass(frozen=True)
class RecoveryContext:
    store: SectionStore
    catalog: LeafCatalog
    tokenizer: Any


def load_recovery_context(
    *,
    sections_jsonl: Path,
    rows_jsonl: Path,
    canonical_root: Path,
    tokenizer_name_or_path: str,
) -> RecoveryContext:
    """Load section store, leaf catalog, documents, and the tokenizer once."""

    sections_path = Path(sections_jsonl)
    rows_path = Path(rows_jsonl)
    root = Path(canonical_root)
    if not sections_path.is_file():
        raise RecoveryConfigurationError(f"sections.jsonl does not exist: {sections_path}")
    if not rows_path.is_file():
        raise RecoveryConfigurationError(
            f"a6 rows jsonl does not exist: {rows_path}. "
            "This file is a B6 production dependency."
        )
    if not root.is_dir():
        raise RecoveryConfigurationError(f"canonical root does not exist: {root}")
    store = load_sections_jsonl(sections_path)
    store.bind_documents(load_markdown_documents(root))
    catalog = load_leaf_catalog(rows_path)
    tokenizer = load_generation_tokenizer(tokenizer_name_or_path)
    return RecoveryContext(store=store, catalog=catalog, tokenizer=tokenizer)


def recover_request(
    context: RecoveryContext,
    result: RerankingResult,
    *,
    budget: int = NOMINAL_EVIDENCE_BUDGET,
) -> RecoveryResult:
    return recover(
        result,
        context.store,
        context.catalog,
        context.tokenizer,
        budget=budget,
    )
