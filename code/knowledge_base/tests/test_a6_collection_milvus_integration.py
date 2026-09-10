"""Mechanism-level integration test for the production A6 Milvus Collection.

Uses a small synthetic dataset (no real corpus, no Qwen weights available on
this machine) against the real local Milvus 2.5.14 instance to verify the
actual schema/BM25 Function/index/lifecycle mechanics work end-to-end. Real
production ingestion of the full ver1 corpus is blocked on this machine by
the missing ``document.md`` corpus and Qwen3-Embedding-4B weights/torch; see
docs/proceeding/2026-09-08-a6-production-collection.md for that block record.

Skips gracefully (does not fail the suite) if Milvus is unreachable, so CI
without a live Milvus still passes the rest of the knowledge_base suite.
"""

from __future__ import annotations

import random
import sys
import uuid
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

pymilvus = pytest.importorskip("pymilvus")

from knowledge_base.a6_collection import (  # noqa: E402
    MilvusCollectionStore,
    build_a6_report,
    build_collection,
    build_collection_rows,
    bm25_smoke_test,
    dense_smoke_test,
    document_ids_to_delete_for_update,
    ingest_update_collection,
    rebuild_collection,
    select_retrievable_metadata,
    verify_collection,
)
from knowledge_base.a6_document_metadata import (  # noqa: E402
    STATUS_ACTIVE,
    STATUS_HISTORICAL,
    DocumentMetadata,
)
from knowledge_base.embedding_config import EMBEDDING_DIMENSION  # noqa: E402
from knowledge_base.leaf_chunker import Leaf  # noqa: E402

MILVUS_URI = "http://localhost:19530"


@pytest.fixture(scope="module")
def store() -> MilvusCollectionStore:
    candidate = MilvusCollectionStore(MILVUS_URI)
    try:
        candidate.client.list_collections()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Milvus not reachable at {MILVUS_URI}: {exc}")
    return candidate


def _unit_vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    raw = [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIMENSION)]
    norm = sum(value * value for value in raw) ** 0.5
    return [value / norm for value in raw]


_DOC_A_SENTENCES = [
    "Titanium 6Al-4V requires solution treatment before aging for aerospace fasteners.",
    "Welding procedures for aluminum 6061 must control heat input to avoid cracking.",
    "Nondestructive evaluation uses radiographic inspection to detect internal porosity.",
]
_DOC_B_SENTENCES = [
    "Additive manufacturing systems require calibrated laser power for consistent layers.",
    "Crimp wiring terminations must meet pull-force requirements per NASA standards.",
]
_DOC_C_SENTENCES = [
    "Electrical bonding straps reduce resistance between structural joints.",
]


def _metadata(document_id: str, *, status: str, ingested_at: int) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        file_name=f"{document_id}.pdf",
        document_title=f"Synthetic {document_id}",
        document_number=f"IT-{document_id.upper()}",
        document_version="V1",
        source_sha256="1" * 64,
        document_content_hash="2" * 64,
        finalized_at=1_700_000_000_000,
        effective_from=1_700_000_000_000,
        effective_to=None,
        ingested_at=ingested_at,
        status=status,
    )


def _leaves_and_dense(document_id: str, sentences: list[str], seed_base: int):
    leaves = []
    dense = {}
    for index, sentence in enumerate(sentences):
        chunk_id = f"{document_id}-chk-{index}"
        leaves.append(
            Leaf(
                chunk_id=chunk_id,
                document_id=document_id,
                section_id=f"{document_id}-sec-{index}",
                chunk_index=index,
                page_start=index + 1,
                page_end=index + 1,
                content=sentence,
            )
        )
        dense[chunk_id] = _unit_vector(seed_base + index)
    return leaves, dense


def test_full_lifecycle_against_real_milvus(store: MilvusCollectionStore) -> None:
    collection_name = f"a6_it_{uuid.uuid4().hex[:12]}"
    try:
        # ---- Build: doc-a (ACTIVE) + doc-b (ACTIVE) ----
        leaves_a, dense_a = _leaves_and_dense("docA", _DOC_A_SENTENCES, seed_base=1)
        leaves_b, dense_b = _leaves_and_dense("docB", _DOC_B_SENTENCES, seed_base=100)
        catalog_v1 = [
            _metadata("docA", status=STATUS_ACTIVE, ingested_at=1),
            _metadata("docB", status=STATUS_ACTIVE, ingested_at=2),
        ]
        retrievable_v1 = select_retrievable_metadata(catalog_v1)
        rows = build_collection_rows(
            leaves_a + leaves_b, {**dense_a, **dense_b}, retrievable_v1
        )
        build_collection(store, collection_name, rows)

        verification = verify_collection(
            store,
            collection_name,
            expected_document_ids=["docA", "docB"],
            expected_row_count=len(_DOC_A_SENTENCES) + len(_DOC_B_SENTENCES),
        )
        assert verification.passed, verification

        dense_result = dense_smoke_test(store, collection_name, _unit_vector(1), limit=5)
        assert dense_result.passed
        assert dense_result.hit_count > 0

        bm25_result = bm25_smoke_test(
            store, collection_name, "titanium aerospace fasteners", limit=5
        )
        assert bm25_result.passed
        assert bm25_result.hit_count > 0

        report_v1 = build_a6_report(
            collection_name=collection_name,
            catalog=catalog_v1,
            leaf_count=len(rows),
            verification=verification,
            dense_smoke=dense_result,
            bm25_smoke=bm25_result,
            join_missing_count=0,
            join_extra_count=0,
        )
        assert report_v1["row_count"] == len(rows)
        assert report_v1["verification"]["passed"] is True

        # ---- Ingest-update: doc-a superseded by doc-a-v2 (version replace) ----
        leaves_a2, dense_a2 = _leaves_and_dense("docA2", _DOC_A_SENTENCES[:2], seed_base=1)
        catalog_v2 = [
            _metadata("docA", status=STATUS_HISTORICAL, ingested_at=1),
            _metadata("docA2", status=STATUS_ACTIVE, ingested_at=3),
            _metadata("docB", status=STATUS_ACTIVE, ingested_at=2),
        ]
        to_delete = document_ids_to_delete_for_update(catalog_v1, catalog_v2)
        assert to_delete == ("docA",)
        retrievable_v2 = select_retrievable_metadata(catalog_v2)
        # Only docA2's leaves are new in this batch; docB is unchanged and
        # already in the Collection, so it must not appear in this join.
        new_document_metadata = {"docA2": retrievable_v2["docA2"]}
        new_rows = build_collection_rows(leaves_a2, dense_a2, new_document_metadata)
        ingest_update_collection(
            store, collection_name, delete_document_ids=to_delete, rows=new_rows
        )

        verification2 = verify_collection(
            store,
            collection_name,
            expected_document_ids=["docA2", "docB"],
            expected_row_count=2 + len(_DOC_B_SENTENCES),
        )
        assert verification2.passed, verification2

        # ---- Rebuild: doc-c only, explicit confirmation required ----
        leaves_c, dense_c = _leaves_and_dense("docC", _DOC_C_SENTENCES, seed_base=200)
        catalog_v3 = [_metadata("docC", status=STATUS_ACTIVE, ingested_at=4)]
        retrievable_v3 = select_retrievable_metadata(catalog_v3)
        rebuild_rows = build_collection_rows(leaves_c, dense_c, retrievable_v3)
        rebuild_collection(
            store, collection_name, rebuild_rows, confirm_rebuild=True
        )
        verification3 = verify_collection(
            store,
            collection_name,
            expected_document_ids=["docC"],
            expected_row_count=len(_DOC_C_SENTENCES),
        )
        assert verification3.passed, verification3
    finally:
        store.drop_collection(collection_name)
        assert not store.has_collection(collection_name)
