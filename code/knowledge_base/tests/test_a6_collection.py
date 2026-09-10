"""Production A6 Milvus Leaf Collection: strict join, schema, lifecycle."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.a6_collection import (  # noqa: E402
    CHUNK_ID_MAX_BYTES,
    CONTENT_MAX_BYTES,
    DOCUMENT_ID_MAX_BYTES,
    FIELD_CHUNK_ID,
    FIELD_DENSE_VECTOR,
    FIELD_DOCUMENT_ID,
    FIELD_SPARSE_VECTOR,
    ICU_ANALYZER_PARAMS,
    A6CollectionError,
    CollectionStore,
    RankedHit,
    ROW_FIELD_NAMES,
    SPARSE_INDEX_TYPE,
    SPARSE_METRIC_TYPE,
    _cmd_report,
    bm25_index_params,
    build_a6_report,
    build_collection,
    build_collection_rows,
    build_parser,
    bm25_smoke_test,
    dense_smoke_test,
    derive_bm25_query,
    document_ids_to_delete_for_update,
    format_a6_report_summary,
    ingest_update_collection,
    join_leaves_with_embeddings,
    join_leaves_with_metadata,
    rebuild_collection,
    select_retrievable_metadata,
    validate_dense_vectors,
    validate_row_varchar_lengths,
    verify_collection,
)
from knowledge_base.a6_document_metadata import (  # noqa: E402
    STATUS_ACTIVE,
    STATUS_DUPLICATE,
    STATUS_HISTORICAL,
    STATUS_VERSION_CONFLICT,
    DocumentMetadata,
)
from knowledge_base.a6_profile_config import (  # noqa: E402
    ANALYZER_PARAMS,
    BM25_B,
    BM25_INDEX_PARAMS,
    BM25_INVERTED_INDEX_ALGO,
    BM25_K1,
    CHUNK_ID_MAX_BYTES as PROFILE_CHUNK_ID_MAX_BYTES,
    DOCUMENT_ID_MAX_BYTES as PROFILE_DOCUMENT_ID_MAX_BYTES,
    PROFILE_EN_ICU,
    VARCHAR_CONTENT_MAX_BYTES,
)
from knowledge_base.embedding_config import EMBEDDING_DIMENSION  # noqa: E402
from knowledge_base.leaf_chunker import Leaf  # noqa: E402


def _leaf(
    chunk_id: str,
    document_id: str = "doc-a",
    *,
    section_id: str = "sec_x",
    chunk_index: int = 0,
    page_start: int = 1,
    page_end: int = 1,
    content: str = "hello world",
) -> Leaf:
    return Leaf(
        chunk_id=chunk_id,
        document_id=document_id,
        section_id=section_id,
        chunk_index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        content=content,
    )


def _metadata(
    document_id: str = "doc-a",
    *,
    status: str = STATUS_ACTIVE,
    file_name: str = "doc.pdf",
    document_title: str = "Doc Title",
) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=document_id,
        file_name=file_name,
        document_title=document_title,
        document_number=None,
        document_version=None,
        source_sha256="a" * 64,
        document_content_hash="b" * 64,
        finalized_at=None,
        effective_from=None,
        effective_to=None,
        ingested_at=1000,
        status=status,
    )


@dataclass
class DenseResultStub:
    chunk_ids: tuple[str, ...]
    vectors: list[list[float]]


def _vector(fill: float = 0.1) -> list[float]:
    return [fill] * EMBEDDING_DIMENSION


# ---------------------------------------------------------------------------
# Anti-drift: A6 constants must never silently diverge from A6.0 frozen values
# ---------------------------------------------------------------------------


def test_chunk_id_and_document_id_lengths_match_a6_0_frozen_values() -> None:
    assert CHUNK_ID_MAX_BYTES == PROFILE_CHUNK_ID_MAX_BYTES
    assert DOCUMENT_ID_MAX_BYTES == PROFILE_DOCUMENT_ID_MAX_BYTES


def test_content_max_bytes_matches_a6_0_frozen_value() -> None:
    assert CONTENT_MAX_BYTES == VARCHAR_CONTENT_MAX_BYTES


def test_icu_analyzer_params_match_a6_0_frozen_recommendation() -> None:
    assert ICU_ANALYZER_PARAMS == ANALYZER_PARAMS[PROFILE_EN_ICU]


def test_bm25_index_params_match_a6_0_frozen_baseline() -> None:
    params = bm25_index_params()
    assert params == BM25_INDEX_PARAMS
    assert params["params"]["bm25_k1"] == BM25_K1 == 1.2
    assert params["params"]["bm25_b"] == BM25_B == 0.75
    assert params["params"]["inverted_index_algo"] == BM25_INVERTED_INDEX_ALGO == "DAAT_MAXSCORE"
    assert SPARSE_INDEX_TYPE == "SPARSE_INVERTED_INDEX"
    assert SPARSE_METRIC_TYPE == "BM25"


def test_bm25_index_params_is_a_copy_not_shared_mutable_state() -> None:
    params = bm25_index_params()
    params["params"]["bm25_k1"] = 999.0
    assert bm25_index_params()["params"]["bm25_k1"] == 1.2


# ---------------------------------------------------------------------------
# SS31 strict join: Dense
# ---------------------------------------------------------------------------


def test_join_leaves_with_embeddings_success() -> None:
    leaves = [_leaf("c1"), _leaf("c2")]
    result = DenseResultStub(chunk_ids=("c1", "c2"), vectors=[_vector(0.1), _vector(0.2)])
    joined = join_leaves_with_embeddings(leaves, result)
    assert set(joined) == {"c1", "c2"}


def test_join_leaves_with_embeddings_missing_fails() -> None:
    leaves = [_leaf("c1"), _leaf("c2")]
    result = DenseResultStub(chunk_ids=("c1",), vectors=[_vector()])
    with pytest.raises(A6CollectionError, match="missing=1"):
        join_leaves_with_embeddings(leaves, result)


def test_join_leaves_with_embeddings_extra_fails() -> None:
    leaves = [_leaf("c1")]
    result = DenseResultStub(chunk_ids=("c1", "c2"), vectors=[_vector(), _vector()])
    with pytest.raises(A6CollectionError, match="extra=1"):
        join_leaves_with_embeddings(leaves, result)


def test_join_leaves_with_embeddings_duplicate_leaf_fails() -> None:
    leaves = [_leaf("c1"), _leaf("c1")]
    result = DenseResultStub(chunk_ids=("c1",), vectors=[_vector()])
    with pytest.raises(A6CollectionError, match="duplicate_leaf=1"):
        join_leaves_with_embeddings(leaves, result)


def test_join_leaves_with_embeddings_duplicate_embedding_fails() -> None:
    leaves = [_leaf("c1")]
    result = DenseResultStub(chunk_ids=("c1", "c1"), vectors=[_vector(), _vector()])
    with pytest.raises(A6CollectionError, match="duplicate_embedding=1"):
        join_leaves_with_embeddings(leaves, result)


def test_join_leaves_with_embeddings_requires_duck_typed_attrs() -> None:
    with pytest.raises(A6CollectionError):
        join_leaves_with_embeddings([_leaf("c1")], object())


# ---------------------------------------------------------------------------
# SS32 dense vector validation (dim + finite; no re-normalize)
# ---------------------------------------------------------------------------


def test_validate_dense_vectors_accepts_correct_dimension() -> None:
    validate_dense_vectors({"c1": _vector()})


def test_validate_dense_vectors_rejects_wrong_dimension() -> None:
    with pytest.raises(A6CollectionError, match="dimension mismatch"):
        validate_dense_vectors({"c1": [0.1, 0.2]})


def test_validate_dense_vectors_rejects_nan() -> None:
    bad = _vector()
    bad[0] = float("nan")
    with pytest.raises(A6CollectionError, match="non-finite"):
        validate_dense_vectors({"c1": bad})


def test_validate_dense_vectors_rejects_inf() -> None:
    bad = _vector()
    bad[0] = float("inf")
    with pytest.raises(A6CollectionError, match="non-finite"):
        validate_dense_vectors({"c1": bad})


# ---------------------------------------------------------------------------
# SS35 retrievable-set selection + SS31 metadata join
# ---------------------------------------------------------------------------


def test_select_retrievable_metadata_keeps_active_and_conflict_only() -> None:
    catalog = [
        _metadata("a", status=STATUS_ACTIVE),
        _metadata("b", status=STATUS_VERSION_CONFLICT),
        _metadata("c", status=STATUS_HISTORICAL),
        _metadata("d", status=STATUS_DUPLICATE),
    ]
    retrievable = select_retrievable_metadata(catalog)
    assert set(retrievable) == {"a", "b"}


def test_join_leaves_with_metadata_exact_equality_success() -> None:
    leaves = [_leaf("c1", "a"), _leaf("c2", "b")]
    metadata = {"a": _metadata("a"), "b": _metadata("b")}
    join_leaves_with_metadata(leaves, metadata)  # no raise


def test_join_leaves_with_metadata_missing_fails() -> None:
    leaves = [_leaf("c1", "a"), _leaf("c2", "b")]
    metadata = {"a": _metadata("a")}
    with pytest.raises(A6CollectionError, match=r"missing=\['b'\]"):
        join_leaves_with_metadata(leaves, metadata)


def test_join_leaves_with_metadata_extra_fails() -> None:
    leaves = [_leaf("c1", "a")]
    metadata = {"a": _metadata("a"), "b": _metadata("b")}
    with pytest.raises(A6CollectionError, match=r"extra=\['b'\]"):
        join_leaves_with_metadata(leaves, metadata)


# ---------------------------------------------------------------------------
# Row construction + VARCHAR guard (SS28-SS30)
# ---------------------------------------------------------------------------


def test_build_collection_rows_shape_and_no_sparse_vector_key() -> None:
    leaves = [_leaf("c1", "a", section_id="sec_1", content="hello")]
    dense = {"c1": _vector(0.5)}
    metadata = {"a": _metadata("a")}
    rows = build_collection_rows(leaves, dense, metadata)
    assert len(rows) == 1
    row = rows[0]
    assert FIELD_SPARSE_VECTOR not in row
    assert set(row) == set(ROW_FIELD_NAMES)
    assert row[FIELD_CHUNK_ID] == "c1"
    assert row[FIELD_DOCUMENT_ID] == "a"
    assert row[FIELD_DENSE_VECTOR] == _vector(0.5)


def test_build_collection_rows_rejects_oversize_content() -> None:
    leaves = [_leaf("c1", "a", content="x" * (VARCHAR_CONTENT_MAX_BYTES + 1))]
    dense = {"c1": _vector()}
    metadata = {"a": _metadata("a")}
    with pytest.raises(A6CollectionError, match="content"):
        build_collection_rows(leaves, dense, metadata)


def test_validate_row_varchar_lengths_rejects_oversize_file_name() -> None:
    row = {
        FIELD_CHUNK_ID: "c1",
        FIELD_DOCUMENT_ID: "a",
        "file_name": "x" * 2000,
        "document_title": "t",
        "document_number": None,
        "document_version": None,
        "section_id": "sec_1",
        "status": STATUS_ACTIVE,
        "content": "ok",
    }
    with pytest.raises(A6CollectionError, match="file_name"):
        validate_row_varchar_lengths(row)


def test_build_collection_rows_never_truncates_it_fails_instead() -> None:
    """SS30/SS45: forbidden to silently truncate; must FAIL."""

    leaves = [_leaf("c1", "a", content="y" * (VARCHAR_CONTENT_MAX_BYTES + 10))]
    dense = {"c1": _vector()}
    metadata = {"a": _metadata("a")}
    with pytest.raises(A6CollectionError):
        rows = build_collection_rows(leaves, dense, metadata)
        assert len(rows[0][FIELD_CHUNK_ID]) == 0  # unreachable; documents intent


# ---------------------------------------------------------------------------
# SS26/SS35 delete-on-transition
# ---------------------------------------------------------------------------


def test_document_ids_to_delete_active_to_historical() -> None:
    previous = [_metadata("old", status=STATUS_ACTIVE)]
    updated = [_metadata("old", status=STATUS_HISTORICAL), _metadata("new", status=STATUS_ACTIVE)]
    assert document_ids_to_delete_for_update(previous, updated) == ("old",)


def test_document_ids_to_delete_conflict_transition_keeps_leaves() -> None:
    previous = [_metadata("a", status=STATUS_ACTIVE)]
    updated = [_metadata("a", status=STATUS_VERSION_CONFLICT), _metadata("b", status=STATUS_VERSION_CONFLICT)]
    assert document_ids_to_delete_for_update(previous, updated) == ()


def test_document_ids_to_delete_new_document_not_included() -> None:
    previous: list[DocumentMetadata] = []
    updated = [_metadata("brand-new", status=STATUS_ACTIVE)]
    assert document_ids_to_delete_for_update(previous, updated) == ()


def test_document_ids_to_delete_active_to_duplicate() -> None:
    previous = [_metadata("a", status=STATUS_ACTIVE)]
    updated = [_metadata("a", status=STATUS_DUPLICATE)]
    assert document_ids_to_delete_for_update(previous, updated) == ("a",)


# ---------------------------------------------------------------------------
# Fake CollectionStore for lifecycle / verification / smoke tests
# ---------------------------------------------------------------------------


@dataclass
class _FakeCollection:
    analyzer_params: dict[str, Any]
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    loaded: bool = False


class FakeCollectionStore(CollectionStore):
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}
        self.dense_hits: dict[str, list[RankedHit]] = {}
        self.sparse_hits: dict[str, list[RankedHit]] = {}
        self.create_calls: list[str] = []
        self.drop_calls: list[str] = []

    def has_collection(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, name: str, *, analyzer_params: dict[str, Any]) -> None:
        self.create_calls.append(name)
        self.collections[name] = _FakeCollection(analyzer_params=dict(analyzer_params))

    def insert_rows(self, name: str, rows: Sequence[dict[str, Any]]) -> None:
        collection = self.collections[name]
        for row in rows:
            collection.rows[row[FIELD_CHUNK_ID]] = dict(row)

    def delete_by_document_ids(self, name: str, document_ids: Sequence[str]) -> int:
        collection = self.collections[name]
        to_remove = [
            chunk_id
            for chunk_id, row in collection.rows.items()
            if row[FIELD_DOCUMENT_ID] in set(document_ids)
        ]
        for chunk_id in to_remove:
            del collection.rows[chunk_id]
        return len(to_remove)

    def load_collection(self, name: str) -> None:
        self.collections[name].loaded = True

    def release_collection(self, name: str) -> None:
        self.collections[name].loaded = False

    def drop_collection(self, name: str) -> None:
        self.drop_calls.append(name)
        self.collections.pop(name, None)

    def row_count(self, name: str) -> int:
        return len(self.collections[name].rows)

    def query(
        self,
        name: str,
        filter_expr: str,
        output_fields: Sequence[str],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = list(self.collections[name].rows.values())
        if limit is not None:
            rows = rows[:limit]
        return [{field: row.get(field) for field in output_fields} for row in rows]

    def query_all(
        self,
        name: str,
        filter_expr: str,
        output_fields: Sequence[str],
        *,
        batch_size: int = 1000,
    ) -> list[dict[str, Any]]:
        return self.query(name, filter_expr, output_fields, limit=None)

    def search_dense(self, name: str, vector: Sequence[float], *, limit: int) -> list[RankedHit]:
        return self.dense_hits.get(name, [])[:limit]

    def search_sparse(self, name: str, query_text: str, *, limit: int) -> list[RankedHit]:
        return self.sparse_hits.get(name, [])[:limit]

    def describe_index(self, name: str, field_name: str) -> dict[str, Any]:
        if field_name == FIELD_SPARSE_VECTOR:
            return {"field_name": field_name, "index_type": SPARSE_INDEX_TYPE}
        if field_name == FIELD_DENSE_VECTOR:
            return {"field_name": field_name, "index_type": "FLAT"}
        return {}


def _sample_rows(document_id: str = "a", n: int = 2) -> list[dict[str, Any]]:
    chunk_ids = [f"{document_id}-c{i}" for i in range(n)]
    leaves = [_leaf(chunk_id, document_id) for chunk_id in chunk_ids]
    dense = {chunk_id: _vector(0.1 * i) for i, chunk_id in enumerate(chunk_ids)}
    metadata = {document_id: _metadata(document_id)}
    return build_collection_rows(leaves, dense, metadata)


# ---------------------------------------------------------------------------
# SS34 lifecycle
# ---------------------------------------------------------------------------


def test_build_collection_creates_and_inserts() -> None:
    store = FakeCollectionStore()
    rows = _sample_rows("a", 3)
    build_collection(store, "coll", rows)
    assert store.row_count("coll") == 3
    assert store.collections["coll"].loaded is True
    assert store.collections["coll"].analyzer_params == ICU_ANALYZER_PARAMS


def test_build_collection_fails_if_already_exists() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 1))
    with pytest.raises(A6CollectionError, match="already exists"):
        build_collection(store, "coll", _sample_rows("a", 1))


def test_ingest_update_requires_existing_collection() -> None:
    store = FakeCollectionStore()
    with pytest.raises(A6CollectionError, match="run build first"):
        ingest_update_collection(store, "coll", delete_document_ids=(), rows=_sample_rows())


def test_ingest_update_deletes_old_and_inserts_new_document() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("old", 2))
    new_rows = _sample_rows("new", 2)
    ingest_update_collection(
        store, "coll", delete_document_ids=["old"], rows=new_rows
    )
    remaining_docs = {row[FIELD_DOCUMENT_ID] for row in store.collections["coll"].rows.values()}
    assert remaining_docs == {"new"}


def test_ingest_update_conflict_flow_keeps_old_and_adds_new() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("old", 2))
    new_rows = _sample_rows("new", 2)
    ingest_update_collection(store, "coll", delete_document_ids=[], rows=new_rows)
    remaining_docs = {row[FIELD_DOCUMENT_ID] for row in store.collections["coll"].rows.values()}
    assert remaining_docs == {"old", "new"}


def test_rebuild_requires_explicit_confirmation() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 1))
    with pytest.raises(A6CollectionError, match="confirm_rebuild"):
        rebuild_collection(store, "coll", _sample_rows("b", 1), confirm_rebuild=False)


def test_rebuild_drops_and_recreates() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 5))
    rebuild_collection(store, "coll", _sample_rows("b", 2), confirm_rebuild=True)
    assert store.drop_calls == ["coll"]
    assert store.row_count("coll") == 2
    remaining_docs = {row[FIELD_DOCUMENT_ID] for row in store.collections["coll"].rows.values()}
    assert remaining_docs == {"b"}


# ---------------------------------------------------------------------------
# SS37 verification
# ---------------------------------------------------------------------------


def test_verify_collection_passes_for_healthy_data() -> None:
    store = FakeCollectionStore()
    rows = _sample_rows("a", 3)
    build_collection(store, "coll", rows)
    result = verify_collection(
        store, "coll", expected_document_ids=["a"], expected_row_count=3
    )
    assert result.passed is True
    assert result.row_count == 3
    assert result.duplicate_chunk_id_count == 0
    assert result.document_id_coverage_missing == ()
    assert result.dense_dimension_ok is True
    assert result.sparse_present is True


def test_verify_collection_detects_missing_document_id() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 2))
    result = verify_collection(store, "coll", expected_document_ids=["a", "b"])
    assert result.passed is False
    assert result.document_id_coverage_missing == ("b",)


def test_verify_collection_detects_extra_document_id() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 2))
    result = verify_collection(store, "coll", expected_document_ids=[])
    assert result.passed is False
    assert result.document_id_coverage_extra == ("a",)


def test_verify_collection_detects_wrong_row_count() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 2))
    result = verify_collection(
        store, "coll", expected_document_ids=["a"], expected_row_count=999
    )
    assert result.passed is False


def test_verify_collection_detects_bad_dense_dimension() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 1))
    # Corrupt the stored row to simulate a bad dimension slipping through.
    for row in store.collections["coll"].rows.values():
        row[FIELD_DENSE_VECTOR] = [0.1, 0.2]
    result = verify_collection(store, "coll", expected_document_ids=["a"])
    assert result.dense_dimension_ok is False
    assert result.passed is False


# ---------------------------------------------------------------------------
# SS38/SS39 smoke tests
# ---------------------------------------------------------------------------


def test_dense_smoke_test_pass() -> None:
    store = FakeCollectionStore()
    store.dense_hits["coll"] = [RankedHit("c1", "a", 1, 0.9)]
    result = dense_smoke_test(store, "coll", _vector(), limit=10)
    assert result.passed is True
    assert result.anns_field == FIELD_DENSE_VECTOR


def test_dense_smoke_test_fail_when_empty() -> None:
    store = FakeCollectionStore()
    result = dense_smoke_test(store, "coll", _vector(), limit=10)
    assert result.passed is False


def test_bm25_smoke_test_pass() -> None:
    store = FakeCollectionStore()
    store.sparse_hits["coll"] = [RankedHit("c1", "a", 1, 5.0)]
    result = bm25_smoke_test(store, "coll", "query text", limit=10)
    assert result.passed is True
    assert result.anns_field == FIELD_SPARSE_VECTOR


def test_bm25_smoke_test_fail_when_empty() -> None:
    store = FakeCollectionStore()
    result = bm25_smoke_test(store, "coll", "query text", limit=10)
    assert result.passed is False


# ---------------------------------------------------------------------------
# SS44 build report
# ---------------------------------------------------------------------------


def test_build_a6_report_contains_required_fields() -> None:
    store = FakeCollectionStore()
    build_collection(store, "coll", _sample_rows("a", 2))
    verification = verify_collection(store, "coll", expected_document_ids=["a"], expected_row_count=2)
    dense = dense_smoke_test(store, "coll", _vector(), limit=5)
    catalog = [_metadata("a", status=STATUS_ACTIVE)]
    report = build_a6_report(
        collection_name="coll",
        catalog=catalog,
        leaf_count=2,
        verification=verification,
        dense_smoke=dense,
        bm25_smoke=None,
        join_missing_count=0,
        join_extra_count=0,
    )
    required = {
        "collection_name",
        "build_timestamp",
        "document_count",
        "leaf_count",
        "dense_vector_count",
        "sparse_vector_count",
        "active_document_count",
        "historical_document_count",
        "duplicate_document_count",
        "version_conflict_document_count",
        "metadata_null_counts",
        "dense_index",
        "sparse_index",
        "analyzer",
        "join_missing_count",
        "join_extra_count",
        "row_count",
        "dense_smoke",
        "bm25_smoke",
    }
    assert required <= set(report)
    assert report["active_document_count"] == 1
    assert report["analyzer"] == ICU_ANALYZER_PARAMS
    summary = format_a6_report_summary(report)
    assert "collection_name=coll" in summary


# ---------------------------------------------------------------------------
# CLI: `report` subcommand (torch-free; argparse wiring + pure helper only,
# no live Milvus call here -- that is covered by
# test_a6_collection_milvus_integration.py)
# ---------------------------------------------------------------------------


def test_derive_bm25_query_takes_first_n_words() -> None:
    content = "one two three four five six seven eight nine ten"
    assert derive_bm25_query(content, word_count=3) == "one two three"


def test_derive_bm25_query_default_word_count_is_eight() -> None:
    content = "one two three four five six seven eight nine ten"
    assert derive_bm25_query(content) == "one two three four five six seven eight"


def test_derive_bm25_query_handles_short_content() -> None:
    assert derive_bm25_query("only two") == "only two"
    assert derive_bm25_query("") == ""


def test_report_subcommand_parses_required_and_default_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "report",
            "--collection",
            "a6_leaf_v1",
            "--metadata-jsonl",
            "meta.jsonl",
            "--rows-jsonl",
            "rows.jsonl",
        ]
    )
    assert args.command == "report"
    assert args.uri == "http://localhost:19530"
    assert args.collection == "a6_leaf_v1"
    assert args.metadata_jsonl == Path("meta.jsonl")
    assert args.rows_jsonl == Path("rows.jsonl")
    assert args.bm25_query is None
    assert args.limit == 10
    assert args.report_out is None
    assert args.func is _cmd_report


def test_report_subcommand_accepts_bm25_query_and_report_out_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "report",
            "--uri",
            "http://example:19530",
            "--collection",
            "a6_leaf_v1",
            "--metadata-jsonl",
            "meta.jsonl",
            "--rows-jsonl",
            "rows.jsonl",
            "--bm25-query",
            "welding requirements",
            "--limit",
            "5",
            "--report-out",
            "report.json",
        ]
    )
    assert args.uri == "http://example:19530"
    assert args.bm25_query == "welding requirements"
    assert args.limit == 5
    assert args.report_out == Path("report.json")


def test_cmd_report_fails_fast_when_metadata_catalog_is_empty(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text("", encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text(json.dumps({"document_id": "a"}) + "\n", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(
        [
            "report",
            "--collection",
            "a6_leaf_v1",
            "--metadata-jsonl",
            str(metadata_path),
            "--rows-jsonl",
            str(rows_path),
        ]
    )
    assert _cmd_report(args) == 2


def test_cmd_report_fails_fast_when_rows_jsonl_is_empty(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    catalog = [_metadata("a", status=STATUS_ACTIVE)]
    metadata_path.write_text(
        "\n".join(json.dumps(entry.to_json_dict(), sort_keys=True) for entry in catalog) + "\n",
        encoding="utf-8",
    )
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(
        [
            "report",
            "--collection",
            "a6_leaf_v1",
            "--metadata-jsonl",
            str(metadata_path),
            "--rows-jsonl",
            str(rows_path),
        ]
    )
    assert _cmd_report(args) == 2
