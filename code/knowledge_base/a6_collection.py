"""A6 production Milvus Leaf Collection: strict join, schema, BM25, lifecycle.

Reuses A3 ``Leaf`` identity and the A6.0-frozen Native BM25 analyzer/index
params (``a6_profile_config.py``). Does not import ``dense_embedding.py`` or
``a6_analyzer_profile.py`` directly: this module must be importable and
testable on a Milvus-only machine with no ``torch``/``transformers`` installed.
Dense vectors are accepted as plain ``(chunk_ids, vectors)`` data, not as the
``DenseEmbeddingResult`` dataclass, to avoid a hard torch dependency here.

Does not re-embed, re-normalize, choose a new Analyzer, or re-implement A0-A5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from knowledge_base.a6_document_metadata import (
    STATUS_ACTIVE,
    STATUS_VERSION_CONFLICT,
    DocumentMetadata,
    load_document_metadata_jsonl,
)
from knowledge_base.a6_profile_config import (
    ANALYZER_PARAMS as _PROFILE_ANALYZER_PARAMS,
    BM25_B,
    BM25_INDEX_PARAMS as _PROFILE_BM25_INDEX_PARAMS,
    BM25_INVERTED_INDEX_ALGO,
    BM25_K1,
    BM25_METRIC_TYPE,
    CHUNK_ID_MAX_BYTES,
    DOCUMENT_ID_MAX_BYTES,
    PROFILE_EN_ICU,
    VARCHAR_CONTENT_MAX_BYTES,
)
from knowledge_base.embedding_config import EMBEDDING_DIMENSION
from knowledge_base.leaf_chunker import Leaf

# ---------------------------------------------------------------------------
# Frozen schema constants (SS28, SS30, SS33)
# ---------------------------------------------------------------------------

FIELD_CHUNK_ID = "chunk_id"
FIELD_DOCUMENT_ID = "document_id"
FIELD_FILE_NAME = "file_name"
FIELD_DOCUMENT_TITLE = "document_title"
FIELD_DOCUMENT_NUMBER = "document_number"
FIELD_DOCUMENT_VERSION = "document_version"
FIELD_FINALIZED_AT = "finalized_at"
FIELD_EFFECTIVE_FROM = "effective_from"
FIELD_EFFECTIVE_TO = "effective_to"
FIELD_INGESTED_AT = "ingested_at"
FIELD_STATUS = "status"
FIELD_SECTION_ID = "section_id"
FIELD_CHUNK_INDEX = "chunk_index"
FIELD_PAGE_START = "page_start"
FIELD_PAGE_END = "page_end"
FIELD_CONTENT = "content"
FIELD_DENSE_VECTOR = "dense_vector"
FIELD_SPARSE_VECTOR = "sparse_vector"

ROW_FIELD_NAMES: tuple[str, ...] = (
    FIELD_CHUNK_ID,
    FIELD_DOCUMENT_ID,
    FIELD_FILE_NAME,
    FIELD_DOCUMENT_TITLE,
    FIELD_DOCUMENT_NUMBER,
    FIELD_DOCUMENT_VERSION,
    FIELD_FINALIZED_AT,
    FIELD_EFFECTIVE_FROM,
    FIELD_EFFECTIVE_TO,
    FIELD_INGESTED_AT,
    FIELD_STATUS,
    FIELD_SECTION_ID,
    FIELD_CHUNK_INDEX,
    FIELD_PAGE_START,
    FIELD_PAGE_END,
    FIELD_CONTENT,
    FIELD_DENSE_VECTOR,
)

# VARCHAR safe-length constants, centralized per SS30. chunk_id/document_id
# and BM25 params are imported (not re-declared) from the A6.0-frozen
# a6_profile_config.py so the two can never silently drift; see
# test_a6_collection.py::test_frozen_constants_match_a6_profile_config.
FILE_NAME_MAX_BYTES = 1024
DOCUMENT_TITLE_MAX_BYTES = 1024
DOCUMENT_NUMBER_MAX_BYTES = 256
DOCUMENT_VERSION_MAX_BYTES = 128
SECTION_ID_MAX_BYTES = 128
STATUS_MAX_BYTES = 32
CONTENT_MAX_BYTES = VARCHAR_CONTENT_MAX_BYTES

ICU_ANALYZER_PARAMS: dict[str, Any] = json.loads(
    json.dumps(_PROFILE_ANALYZER_PARAMS[PROFILE_EN_ICU])
)

DENSE_INDEX_TYPE = "FLAT"
DENSE_METRIC_TYPE = "IP"

SPARSE_INDEX_TYPE = _PROFILE_BM25_INDEX_PARAMS["index_type"]
SPARSE_METRIC_TYPE = _PROFILE_BM25_INDEX_PARAMS["metric_type"]


def bm25_index_params() -> dict[str, Any]:
    return json.loads(json.dumps(_PROFILE_BM25_INDEX_PARAMS))


BM25_FUNCTION_NAME = "a6_bm25"

RETRIEVABLE_STATUSES = (STATUS_ACTIVE, STATUS_VERSION_CONFLICT)


class A6CollectionError(Exception):
    """Fail-fast error for the production A6 Milvus Collection."""


# ---------------------------------------------------------------------------
# Duck-typed dense embedding input (no torch import in this module)
# ---------------------------------------------------------------------------


@runtime_checkable
class DenseEmbeddingLike(Protocol):
    chunk_ids: Sequence[str]
    vectors: Any


def _dense_fields(embedding_result: Any) -> tuple[tuple[str, ...], Any]:
    chunk_ids = getattr(embedding_result, "chunk_ids", None)
    vectors = getattr(embedding_result, "vectors", None)
    if chunk_ids is None or vectors is None:
        raise A6CollectionError(
            "embedding_result must expose .chunk_ids and .vectors "
            "(A5 DenseEmbeddingResult shape)"
        )
    return tuple(chunk_ids), vectors


# ---------------------------------------------------------------------------
# SS31/SS32 strict join + vector validation
# ---------------------------------------------------------------------------


def join_leaves_with_embeddings(
    leaves: Sequence[Leaf], embedding_result: Any
) -> dict[str, Any]:
    """SS31 Dense binding: Leaf.chunk_id <-> DenseEmbeddingResult.chunk_id.

    Requires missing == extra == duplicate == 0. Returns chunk_id -> vector.
    """

    embedding_chunk_ids, vectors = _dense_fields(embedding_result)
    if len(embedding_chunk_ids) != len(vectors):
        raise A6CollectionError(
            "embedding_result chunk_ids/vectors length mismatch: "
            f"{len(embedding_chunk_ids)} != {len(vectors)}"
        )
    leaf_ids = [leaf.chunk_id for leaf in leaves]
    leaf_id_set = set(leaf_ids)
    embed_id_set = set(embedding_chunk_ids)
    missing = leaf_id_set - embed_id_set
    extra = embed_id_set - leaf_id_set
    duplicate_leaf = len(leaf_ids) - len(leaf_id_set)
    duplicate_embedding = len(embedding_chunk_ids) - len(embed_id_set)
    if missing or extra or duplicate_leaf or duplicate_embedding:
        raise A6CollectionError(
            "A3/A5 chunk_id strict join failed: "
            f"missing={len(missing)} extra={len(extra)} "
            f"duplicate_leaf={duplicate_leaf} duplicate_embedding={duplicate_embedding}"
        )
    return {
        chunk_id: vector for chunk_id, vector in zip(embedding_chunk_ids, vectors)
    }


def validate_dense_vectors(dense_by_chunk_id: dict[str, Any]) -> None:
    """SS32: dim=2560, finite. Does not re-normalize or re-check L2 norm
    (A5 already normalized; A6 must not repeat or second-guess that step)."""

    import math

    for chunk_id, vector in dense_by_chunk_id.items():
        values = list(vector)
        if len(values) != EMBEDDING_DIMENSION:
            raise A6CollectionError(
                f"dense_vector dimension mismatch for chunk_id={chunk_id}: "
                f"expected {EMBEDDING_DIMENSION}, got {len(values)}"
            )
        for value in values:
            fvalue = float(value)
            if math.isnan(fvalue) or math.isinf(fvalue):
                raise A6CollectionError(
                    f"non-finite dense_vector value for chunk_id={chunk_id}"
                )


def select_retrievable_metadata(
    catalog: Sequence[DocumentMetadata],
) -> dict[str, DocumentMetadata]:
    """SS35: only ACTIVE + VERSION_CONFLICT documents are ever in the
    default online Collection."""

    return {
        entry.document_id: entry
        for entry in catalog
        if entry.status in RETRIEVABLE_STATUSES
    }


def join_leaves_with_metadata(
    leaves: Sequence[Leaf], metadata_by_document_id: dict[str, DocumentMetadata]
) -> None:
    """SS31 metadata binding: A3 document_id set == metadata document_id set.

    *metadata_by_document_id* must already be scoped to exactly the documents
    intended for this batch (see ``select_retrievable_metadata``); this
    function enforces the symmetric equality the contract requires, not a
    one-sided subset check.
    """

    leaf_doc_ids = {leaf.document_id for leaf in leaves}
    metadata_ids = set(metadata_by_document_id)
    missing = leaf_doc_ids - metadata_ids
    extra = metadata_ids - leaf_doc_ids
    if missing or extra:
        raise A6CollectionError(
            "A3 document_id / metadata document_id strict join failed: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )


# ---------------------------------------------------------------------------
# SS30 VARCHAR guard (no silent truncation, ever)
# ---------------------------------------------------------------------------


def _check_varchar(value: str, max_bytes: int, field_name: str, chunk_id: str) -> None:
    length = len(value.encode("utf-8"))
    if length > max_bytes:
        raise A6CollectionError(
            f"{field_name} exceeds VARCHAR max {max_bytes} UTF-8 bytes "
            f"(got {length}) for chunk_id={chunk_id}; truncation is forbidden"
        )


def validate_row_varchar_lengths(row: dict[str, Any]) -> None:
    chunk_id = row[FIELD_CHUNK_ID]
    _check_varchar(chunk_id, CHUNK_ID_MAX_BYTES, FIELD_CHUNK_ID, chunk_id)
    _check_varchar(row[FIELD_DOCUMENT_ID], DOCUMENT_ID_MAX_BYTES, FIELD_DOCUMENT_ID, chunk_id)
    _check_varchar(row[FIELD_FILE_NAME], FILE_NAME_MAX_BYTES, FIELD_FILE_NAME, chunk_id)
    _check_varchar(
        row[FIELD_DOCUMENT_TITLE], DOCUMENT_TITLE_MAX_BYTES, FIELD_DOCUMENT_TITLE, chunk_id
    )
    if row[FIELD_DOCUMENT_NUMBER] is not None:
        _check_varchar(
            row[FIELD_DOCUMENT_NUMBER],
            DOCUMENT_NUMBER_MAX_BYTES,
            FIELD_DOCUMENT_NUMBER,
            chunk_id,
        )
    if row[FIELD_DOCUMENT_VERSION] is not None:
        _check_varchar(
            row[FIELD_DOCUMENT_VERSION],
            DOCUMENT_VERSION_MAX_BYTES,
            FIELD_DOCUMENT_VERSION,
            chunk_id,
        )
    _check_varchar(row[FIELD_SECTION_ID], SECTION_ID_MAX_BYTES, FIELD_SECTION_ID, chunk_id)
    _check_varchar(row[FIELD_STATUS], STATUS_MAX_BYTES, FIELD_STATUS, chunk_id)
    _check_varchar(row[FIELD_CONTENT], CONTENT_MAX_BYTES, FIELD_CONTENT, chunk_id)


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def build_collection_rows(
    leaves: Sequence[Leaf],
    dense_by_chunk_id: dict[str, Any],
    metadata_by_document_id: dict[str, DocumentMetadata],
) -> list[dict[str, Any]]:
    """SS28/SS29: only schema fields. sparse_vector is generated by the Milvus
    BM25 Function from ``content`` and must never be supplied here."""

    join_leaves_with_metadata(leaves, metadata_by_document_id)
    validate_dense_vectors(
        {leaf.chunk_id: dense_by_chunk_id[leaf.chunk_id] for leaf in leaves}
    )
    rows: list[dict[str, Any]] = []
    for leaf in leaves:
        meta = metadata_by_document_id[leaf.document_id]
        vector = dense_by_chunk_id[leaf.chunk_id]
        vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        row = {
            FIELD_CHUNK_ID: leaf.chunk_id,
            FIELD_DOCUMENT_ID: leaf.document_id,
            FIELD_FILE_NAME: meta.file_name,
            FIELD_DOCUMENT_TITLE: meta.document_title,
            FIELD_DOCUMENT_NUMBER: meta.document_number,
            FIELD_DOCUMENT_VERSION: meta.document_version,
            FIELD_FINALIZED_AT: meta.finalized_at,
            FIELD_EFFECTIVE_FROM: meta.effective_from,
            FIELD_EFFECTIVE_TO: meta.effective_to,
            FIELD_INGESTED_AT: meta.ingested_at,
            FIELD_STATUS: meta.status,
            FIELD_SECTION_ID: leaf.section_id,
            FIELD_CHUNK_INDEX: leaf.chunk_index,
            FIELD_PAGE_START: leaf.page_start,
            FIELD_PAGE_END: leaf.page_end,
            FIELD_CONTENT: leaf.content,
            FIELD_DENSE_VECTOR: vector_list,
        }
        validate_row_varchar_lengths(row)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# SS26/SS35: which document_ids must lose their Leaves on this update
# ---------------------------------------------------------------------------


def document_ids_to_delete_for_update(
    previous_catalog: Sequence[DocumentMetadata],
    updated_catalog: Sequence[DocumentMetadata],
) -> tuple[str, ...]:
    """Documents that just left the retrievable set (ACTIVE/VERSION_CONFLICT)
    must have their existing Leaves removed from the default Collection."""

    previous_by_id = {entry.document_id: entry for entry in previous_catalog}
    to_delete: list[str] = []
    for entry in updated_catalog:
        previous = previous_by_id.get(entry.document_id)
        if previous is None:
            continue
        was_retrievable = previous.status in RETRIEVABLE_STATUSES
        is_retrievable = entry.status in RETRIEVABLE_STATUSES
        if was_retrievable and not is_retrievable:
            to_delete.append(entry.document_id)
    return tuple(to_delete)


# ---------------------------------------------------------------------------
# Store abstraction (SS34 lifecycle uses this; tests inject a fake)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    document_id: str
    rank: int
    score: float


class CollectionStore:
    """Minimal Milvus surface used by production A6. Tests inject a fake."""

    def has_collection(self, name: str) -> bool:
        raise NotImplementedError

    def create_collection(self, name: str, *, analyzer_params: dict[str, Any]) -> None:
        raise NotImplementedError

    def insert_rows(self, name: str, rows: Sequence[dict[str, Any]]) -> None:
        raise NotImplementedError

    def delete_by_document_ids(self, name: str, document_ids: Sequence[str]) -> int:
        raise NotImplementedError

    def load_collection(self, name: str) -> None:
        raise NotImplementedError

    def release_collection(self, name: str) -> None:
        raise NotImplementedError

    def drop_collection(self, name: str) -> None:
        raise NotImplementedError

    def row_count(self, name: str) -> int:
        raise NotImplementedError

    def query(
        self,
        name: str,
        filter_expr: str,
        output_fields: Sequence[str],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def query_all(
        self,
        name: str,
        filter_expr: str,
        output_fields: Sequence[str],
        *,
        batch_size: int = 1000,
    ) -> list[dict[str, Any]]:
        """Unbounded scan for verification. Milvus caps a single ``query``
        offset+limit window at 16384, so this must page/iterate instead of
        relying on a single large ``limit``."""

        raise NotImplementedError

    def search_dense(
        self, name: str, vector: Sequence[float], *, limit: int
    ) -> list[RankedHit]:
        raise NotImplementedError

    def search_sparse(
        self, name: str, query_text: str, *, limit: int
    ) -> list[RankedHit]:
        raise NotImplementedError

    def describe_index(self, name: str, field_name: str) -> dict[str, Any]:
        raise NotImplementedError


def _hits_from_raw(raw: Any) -> list[RankedHit]:
    hits = raw[0] if raw else []
    ranked: list[RankedHit] = []
    for index, hit in enumerate(hits, start=1):
        entity = hit.get("entity", hit)
        chunk_id = entity.get(FIELD_CHUNK_ID, hit.get("id"))
        document_id = entity.get(FIELD_DOCUMENT_ID)
        score = float(hit.get("distance", hit.get("score", 0.0)))
        ranked.append(
            RankedHit(
                chunk_id=str(chunk_id),
                document_id=str(document_id),
                rank=index,
                score=score,
            )
        )
    return ranked


class MilvusCollectionStore(CollectionStore):
    """Real Milvus 2.5.14 implementation via pymilvus MilvusClient."""

    def __init__(self, uri: str) -> None:
        try:
            from pymilvus import DataType, Function, FunctionType, MilvusClient
        except ImportError as exc:
            raise A6CollectionError(
                "pymilvus is required for the production A6 Milvus store"
            ) from exc
        self._DataType = DataType
        self._Function = Function
        self._FunctionType = FunctionType
        self.uri = uri
        self.client = MilvusClient(uri=uri)

    def has_collection(self, name: str) -> bool:
        return bool(self.client.has_collection(name))

    def create_collection(self, name: str, *, analyzer_params: dict[str, Any]) -> None:
        DataType = self._DataType
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=FIELD_CHUNK_ID,
            datatype=DataType.VARCHAR,
            max_length=CHUNK_ID_MAX_BYTES,
            is_primary=True,
        )
        schema.add_field(
            field_name=FIELD_DOCUMENT_ID, datatype=DataType.VARCHAR, max_length=DOCUMENT_ID_MAX_BYTES
        )
        schema.add_field(
            field_name=FIELD_FILE_NAME, datatype=DataType.VARCHAR, max_length=FILE_NAME_MAX_BYTES
        )
        schema.add_field(
            field_name=FIELD_DOCUMENT_TITLE,
            datatype=DataType.VARCHAR,
            max_length=DOCUMENT_TITLE_MAX_BYTES,
        )
        schema.add_field(
            field_name=FIELD_DOCUMENT_NUMBER,
            datatype=DataType.VARCHAR,
            max_length=DOCUMENT_NUMBER_MAX_BYTES,
            nullable=True,
        )
        schema.add_field(
            field_name=FIELD_DOCUMENT_VERSION,
            datatype=DataType.VARCHAR,
            max_length=DOCUMENT_VERSION_MAX_BYTES,
            nullable=True,
        )
        schema.add_field(field_name=FIELD_FINALIZED_AT, datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name=FIELD_EFFECTIVE_FROM, datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name=FIELD_EFFECTIVE_TO, datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name=FIELD_INGESTED_AT, datatype=DataType.INT64)
        schema.add_field(
            field_name=FIELD_STATUS, datatype=DataType.VARCHAR, max_length=STATUS_MAX_BYTES
        )
        schema.add_field(
            field_name=FIELD_SECTION_ID, datatype=DataType.VARCHAR, max_length=SECTION_ID_MAX_BYTES
        )
        schema.add_field(field_name=FIELD_CHUNK_INDEX, datatype=DataType.INT64)
        schema.add_field(field_name=FIELD_PAGE_START, datatype=DataType.INT64)
        schema.add_field(field_name=FIELD_PAGE_END, datatype=DataType.INT64)
        schema.add_field(
            field_name=FIELD_CONTENT,
            datatype=DataType.VARCHAR,
            max_length=CONTENT_MAX_BYTES,
            enable_analyzer=True,
            analyzer_params=analyzer_params,
        )
        schema.add_field(field_name=FIELD_DENSE_VECTOR, datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
        schema.add_field(field_name=FIELD_SPARSE_VECTOR, datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            self._Function(
                name=BM25_FUNCTION_NAME,
                input_field_names=[FIELD_CONTENT],
                output_field_names=[FIELD_SPARSE_VECTOR],
                function_type=self._FunctionType.BM25,
            )
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=FIELD_DENSE_VECTOR,
            index_type=DENSE_INDEX_TYPE,
            metric_type=DENSE_METRIC_TYPE,
        )
        sparse = bm25_index_params()
        index_params.add_index(
            field_name=FIELD_SPARSE_VECTOR,
            index_type=sparse["index_type"],
            metric_type=sparse["metric_type"],
            params=sparse["params"],
        )
        try:
            self.client.create_collection(
                collection_name=name, schema=schema, index_params=index_params
            )
        except Exception as exc:
            raise A6CollectionError(
                f"failed to create production collection {name}: {exc}"
            ) from exc

    def insert_rows(self, name: str, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        self.client.insert(name, list(rows))
        self.client.flush(name)

    def delete_by_document_ids(self, name: str, document_ids: Sequence[str]) -> int:
        if not document_ids:
            return 0
        quoted = ", ".join(json.dumps(doc_id) for doc_id in document_ids)
        expr = f"{FIELD_DOCUMENT_ID} in [{quoted}]"
        result = self.client.delete(collection_name=name, filter=expr)
        self.client.flush(name)
        if isinstance(result, dict):
            return int(result.get("delete_count", 0))
        return 0

    def load_collection(self, name: str) -> None:
        self.client.load_collection(name)

    def release_collection(self, name: str) -> None:
        self.client.release_collection(name)

    def drop_collection(self, name: str) -> None:
        if self.client.has_collection(name):
            self.client.drop_collection(name)

    def row_count(self, name: str) -> int:
        stats = self.client.get_collection_stats(name)
        return int(stats.get("row_count", 0))

    def query(
        self,
        name: str,
        filter_expr: str,
        output_fields: Sequence[str],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = limit
        return self.client.query(
            collection_name=name, filter=filter_expr, output_fields=list(output_fields), **kwargs
        )

    def query_all(
        self,
        name: str,
        filter_expr: str,
        output_fields: Sequence[str],
        *,
        batch_size: int = 1000,
    ) -> list[dict[str, Any]]:
        iterator = self.client.query_iterator(
            collection_name=name,
            filter=filter_expr,
            output_fields=list(output_fields),
            batch_size=batch_size,
            limit=-1,
        )
        rows: list[dict[str, Any]] = []
        try:
            while True:
                batch = iterator.next()
                if not batch:
                    break
                rows.extend(batch)
        finally:
            iterator.close()
        return rows

    def search_dense(
        self, name: str, vector: Sequence[float], *, limit: int
    ) -> list[RankedHit]:
        raw = self.client.search(
            collection_name=name,
            data=[list(vector)],
            anns_field=FIELD_DENSE_VECTOR,
            search_params={"metric_type": DENSE_METRIC_TYPE},
            output_fields=[FIELD_CHUNK_ID, FIELD_DOCUMENT_ID],
            limit=limit,
        )
        return _hits_from_raw(raw)

    def search_sparse(
        self, name: str, query_text: str, *, limit: int
    ) -> list[RankedHit]:
        raw = self.client.search(
            collection_name=name,
            data=[query_text],
            anns_field=FIELD_SPARSE_VECTOR,
            output_fields=[FIELD_CHUNK_ID, FIELD_DOCUMENT_ID],
            limit=limit,
        )
        return _hits_from_raw(raw)

    def describe_index(self, name: str, field_name: str) -> dict[str, Any]:
        try:
            return dict(self.client.describe_index(collection_name=name, index_name=field_name))
        except Exception:
            names = self.client.list_indexes(collection_name=name)
            for index_name in names:
                described = dict(
                    self.client.describe_index(collection_name=name, index_name=index_name)
                )
                if described.get("field_name") == field_name:
                    return described
            raise


# ---------------------------------------------------------------------------
# SS34 lifecycle
# ---------------------------------------------------------------------------


def build_collection(
    store: CollectionStore,
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    analyzer_params: dict[str, Any] | None = None,
) -> None:
    """SS34.1: first build. FAILs if the Collection already exists."""

    if store.has_collection(name):
        raise A6CollectionError(
            f"Collection already exists, refusing to silently overwrite: {name}"
        )
    store.create_collection(name, analyzer_params=analyzer_params or ICU_ANALYZER_PARAMS)
    store.insert_rows(name, rows)
    store.load_collection(name)


def ingest_update_collection(
    store: CollectionStore,
    name: str,
    *,
    delete_document_ids: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    """SS34.2/SS26/SS27: only touches the affected documents. Never drops
    or rebuilds the whole Collection."""

    if not store.has_collection(name):
        raise A6CollectionError(
            f"Collection does not exist; run build first (not ingest-update): {name}"
        )
    if delete_document_ids:
        store.delete_by_document_ids(name, delete_document_ids)
    if rows:
        store.insert_rows(name, rows)
    store.load_collection(name)


def rebuild_collection(
    store: CollectionStore,
    name: str,
    rows: Sequence[dict[str, Any]],
    *,
    analyzer_params: dict[str, Any] | None = None,
    confirm_rebuild: bool,
) -> None:
    """SS34.3: explicit full rebuild only. Requires an explicit confirmation
    flag so a normal update path can never silently fall into this branch."""

    if not confirm_rebuild:
        raise A6CollectionError(
            "rebuild_collection requires confirm_rebuild=True (explicit --rebuild)"
        )
    store.drop_collection(name)
    store.create_collection(name, analyzer_params=analyzer_params or ICU_ANALYZER_PARAMS)
    store.insert_rows(name, rows)
    store.load_collection(name)


# ---------------------------------------------------------------------------
# SS37 verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    row_count: int
    stats_row_count: int
    expected_row_count: int | None
    chunk_id_count: int
    duplicate_chunk_id_count: int
    document_id_coverage_missing: tuple[str, ...]
    document_id_coverage_extra: tuple[str, ...]
    dense_dimension_ok: bool
    dense_finite_ok: bool
    sparse_present: bool
    metadata_fields_readable: bool
    passed: bool


def verify_collection(
    store: CollectionStore,
    name: str,
    *,
    expected_document_ids: Sequence[str],
    expected_row_count: int | None = None,
    batch_size: int = 1000,
) -> VerificationResult:
    """SS37: row_count, chunk_id uniqueness, document_id coverage, dense
    dimension/finite, sparse presence, metadata readability."""

    import math

    stats_row_count = store.row_count(name)
    rows = store.query_all(
        name,
        f'{FIELD_CHUNK_ID} != ""',
        [
            FIELD_CHUNK_ID,
            FIELD_DOCUMENT_ID,
            FIELD_FILE_NAME,
            FIELD_DOCUMENT_TITLE,
            FIELD_STATUS,
            FIELD_INGESTED_AT,
            FIELD_DENSE_VECTOR,
        ],
        batch_size=batch_size,
    )
    chunk_ids = [row.get(FIELD_CHUNK_ID) for row in rows]
    row_count = len(chunk_ids)
    duplicate_chunk_id_count = len(chunk_ids) - len(set(chunk_ids))
    seen_document_ids = {row.get(FIELD_DOCUMENT_ID) for row in rows}
    expected_ids = set(expected_document_ids)
    missing = tuple(sorted(expected_ids - seen_document_ids))
    extra = tuple(sorted(seen_document_ids - expected_ids))

    dense_dimension_ok = True
    dense_finite_ok = True
    metadata_fields_readable = True
    for row in rows:
        vector = row.get(FIELD_DENSE_VECTOR)
        if vector is None or len(vector) != EMBEDDING_DIMENSION:
            dense_dimension_ok = False
        elif not all(math.isfinite(float(value)) for value in vector):
            dense_finite_ok = False
        if row.get(FIELD_FILE_NAME) is None or row.get(FIELD_DOCUMENT_TITLE) is None:
            metadata_fields_readable = False
        if row.get(FIELD_STATUS) is None or row.get(FIELD_INGESTED_AT) is None:
            metadata_fields_readable = False

    sparse_index = store.describe_index(name, FIELD_SPARSE_VECTOR)
    sparse_present = bool(sparse_index)

    row_count_ok = expected_row_count is None or row_count == expected_row_count
    passed = (
        row_count_ok
        and duplicate_chunk_id_count == 0
        and not missing
        and not extra
        and dense_dimension_ok
        and dense_finite_ok
        and sparse_present
        and metadata_fields_readable
    )
    return VerificationResult(
        row_count=row_count,
        stats_row_count=stats_row_count,
        expected_row_count=expected_row_count,
        chunk_id_count=len(chunk_ids),
        duplicate_chunk_id_count=duplicate_chunk_id_count,
        document_id_coverage_missing=missing,
        document_id_coverage_extra=extra,
        dense_dimension_ok=dense_dimension_ok,
        dense_finite_ok=dense_finite_ok,
        sparse_present=sparse_present,
        metadata_fields_readable=metadata_fields_readable,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# SS38/SS39 smoke tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeResult:
    anns_field: str
    metric_type: str
    hit_count: int
    top_chunk_ids: tuple[str, ...]
    passed: bool


def dense_smoke_test(
    store: CollectionStore, name: str, query_vector: Sequence[float], *, limit: int = 10
) -> SmokeResult:
    """SS38: Dense-only. Never invokes RRF/Reranker to compensate."""

    hits = store.search_dense(name, query_vector, limit=limit)
    return SmokeResult(
        anns_field=FIELD_DENSE_VECTOR,
        metric_type=DENSE_METRIC_TYPE,
        hit_count=len(hits),
        top_chunk_ids=tuple(hit.chunk_id for hit in hits),
        passed=len(hits) > 0,
    )


def bm25_smoke_test(
    store: CollectionStore, name: str, query_text: str, *, limit: int = 10
) -> SmokeResult:
    """SS39: BM25-only. Never invokes Dense/RRF/Reranker to compensate."""

    hits = store.search_sparse(name, query_text, limit=limit)
    return SmokeResult(
        anns_field=FIELD_SPARSE_VECTOR,
        metric_type=SPARSE_METRIC_TYPE,
        hit_count=len(hits),
        top_chunk_ids=tuple(hit.chunk_id for hit in hits),
        passed=len(hits) > 0,
    )


# ---------------------------------------------------------------------------
# SS44 build report
# ---------------------------------------------------------------------------


def build_a6_report(
    *,
    collection_name: str,
    catalog: Sequence[DocumentMetadata],
    leaf_count: int,
    verification: VerificationResult,
    dense_smoke: SmokeResult | None,
    bm25_smoke: SmokeResult | None,
    join_missing_count: int,
    join_extra_count: int,
    build_timestamp: str | None = None,
) -> dict[str, Any]:
    from knowledge_base.a6_document_metadata import (
        STATUS_DUPLICATE,
        STATUS_HISTORICAL,
    )

    status_counts: dict[str, int] = {}
    null_counts = {
        FIELD_DOCUMENT_NUMBER: 0,
        FIELD_DOCUMENT_VERSION: 0,
        FIELD_FINALIZED_AT: 0,
        FIELD_EFFECTIVE_FROM: 0,
        FIELD_EFFECTIVE_TO: 0,
    }
    for entry in catalog:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
        if entry.document_number is None:
            null_counts[FIELD_DOCUMENT_NUMBER] += 1
        if entry.document_version is None:
            null_counts[FIELD_DOCUMENT_VERSION] += 1
        if entry.finalized_at is None:
            null_counts[FIELD_FINALIZED_AT] += 1
        if entry.effective_from is None:
            null_counts[FIELD_EFFECTIVE_FROM] += 1
        if entry.effective_to is None:
            null_counts[FIELD_EFFECTIVE_TO] += 1

    return {
        "collection_name": collection_name,
        "build_timestamp": build_timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "document_count": len(catalog),
        "leaf_count": leaf_count,
        "dense_vector_count": verification.row_count,
        "sparse_vector_count": verification.row_count if verification.sparse_present else 0,
        "active_document_count": status_counts.get(STATUS_ACTIVE, 0),
        "historical_document_count": status_counts.get(STATUS_HISTORICAL, 0),
        "duplicate_document_count": status_counts.get(STATUS_DUPLICATE, 0),
        "version_conflict_document_count": status_counts.get(STATUS_VERSION_CONFLICT, 0),
        "metadata_null_counts": null_counts,
        "dense_index": {"index_type": DENSE_INDEX_TYPE, "metric_type": DENSE_METRIC_TYPE},
        "sparse_index": bm25_index_params(),
        "analyzer": ICU_ANALYZER_PARAMS,
        "join_missing_count": join_missing_count,
        "join_extra_count": join_extra_count,
        "row_count": verification.row_count,
        "verification": {
            "expected_row_count": verification.expected_row_count,
            "duplicate_chunk_id_count": verification.duplicate_chunk_id_count,
            "document_id_coverage_missing": list(verification.document_id_coverage_missing),
            "document_id_coverage_extra": list(verification.document_id_coverage_extra),
            "dense_dimension_ok": verification.dense_dimension_ok,
            "dense_finite_ok": verification.dense_finite_ok,
            "sparse_present": verification.sparse_present,
            "metadata_fields_readable": verification.metadata_fields_readable,
            "passed": verification.passed,
        },
        "dense_smoke": (
            {
                "anns_field": dense_smoke.anns_field,
                "metric_type": dense_smoke.metric_type,
                "hit_count": dense_smoke.hit_count,
                "passed": dense_smoke.passed,
            }
            if dense_smoke is not None
            else None
        ),
        "bm25_smoke": (
            {
                "anns_field": bm25_smoke.anns_field,
                "metric_type": bm25_smoke.metric_type,
                "hit_count": bm25_smoke.hit_count,
                "passed": bm25_smoke.passed,
            }
            if bm25_smoke is not None
            else None
        ),
    }


def format_a6_report_summary(report: dict[str, Any]) -> str:
    verification = report["verification"]
    lines = [
        f"collection_name={report['collection_name']}",
        f"build_timestamp={report['build_timestamp']}",
        f"document_count={report['document_count']}",
        f"leaf_count={report['leaf_count']}",
        f"row_count={report['row_count']}",
        f"active={report['active_document_count']} "
        f"historical={report['historical_document_count']} "
        f"duplicate={report['duplicate_document_count']} "
        f"version_conflict={report['version_conflict_document_count']}",
        f"metadata_null_counts={report['metadata_null_counts']}",
        f"join_missing_count={report['join_missing_count']}",
        f"join_extra_count={report['join_extra_count']}",
        f"verification_passed={verification['passed']}",
        f"dense_smoke={report['dense_smoke']}",
        f"bm25_smoke={report['bm25_smoke']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: operates on a pre-joined rows JSONL artifact (no torch dependency)
# ---------------------------------------------------------------------------


def _read_rows_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _cmd_build(args: argparse.Namespace) -> int:
    rows = _read_rows_jsonl(args.rows_jsonl)
    store = MilvusCollectionStore(args.uri)
    build_collection(store, args.collection, rows, analyzer_params=ICU_ANALYZER_PARAMS)
    print(f"collection={args.collection}")
    print(f"row_count={store.row_count(args.collection)}")
    return 0


def _cmd_ingest_update(args: argparse.Namespace) -> int:
    rows = _read_rows_jsonl(args.rows_jsonl)
    delete_ids = (
        [line.strip() for line in args.delete_document_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.delete_document_ids_file is not None
        else []
    )
    store = MilvusCollectionStore(args.uri)
    ingest_update_collection(
        store, args.collection, delete_document_ids=delete_ids, rows=rows
    )
    print(f"collection={args.collection}")
    print(f"row_count={store.row_count(args.collection)}")
    return 0


def _cmd_rebuild(args: argparse.Namespace) -> int:
    rows = _read_rows_jsonl(args.rows_jsonl)
    store = MilvusCollectionStore(args.uri)
    rebuild_collection(
        store,
        args.collection,
        rows,
        analyzer_params=ICU_ANALYZER_PARAMS,
        confirm_rebuild=args.rebuild,
    )
    print(f"collection={args.collection}")
    print(f"row_count={store.row_count(args.collection)}")
    return 0


def derive_bm25_query(content: str, *, word_count: int = 8) -> str:
    """Fallback BM25 smoke-test query: first few words of a real row's
    content, so the smoke test exercises real corpus text end-to-end when
    the caller does not supply --bm25-query explicitly."""

    words = content.split()
    return " ".join(words[:word_count])


def _cmd_report(args: argparse.Namespace) -> int:
    """SS37/SS38/SS39/SS44: run verify + dense/bm25 smoke tests against an
    already-built production Collection and print/save the acceptance
    report. Does not build/modify the Collection; read-only against Milvus.

    Sources a real dense query vector and a real BM25 query text from the
    first row of --rows-jsonl (the same rows.jsonl used to build/ingest the
    Collection), so smoke tests exercise real corpus content rather than
    synthetic vectors.
    """

    catalog = load_document_metadata_jsonl(args.metadata_jsonl)
    if not catalog:
        print("error: document_metadata catalog is empty; nothing to report", file=sys.stderr)
        return 2

    rows = _read_rows_jsonl(args.rows_jsonl)
    if not rows:
        print("error: rows_jsonl is empty; nothing to report", file=sys.stderr)
        return 2

    retrievable = select_retrievable_metadata(catalog)
    expected_document_ids = sorted(retrievable.keys())
    document_ids_in_rows = {row[FIELD_DOCUMENT_ID] for row in rows}
    join_missing_count = len(document_ids_in_rows - set(expected_document_ids))
    join_extra_count = len(set(expected_document_ids) - document_ids_in_rows)

    store = MilvusCollectionStore(args.uri)
    verification = verify_collection(
        store,
        args.collection,
        expected_document_ids=expected_document_ids,
        expected_row_count=len(rows),
    )

    query_vector = rows[0][FIELD_DENSE_VECTOR]
    dense_result = dense_smoke_test(store, args.collection, query_vector, limit=args.limit)

    bm25_query = args.bm25_query or derive_bm25_query(rows[0].get(FIELD_CONTENT, ""))
    bm25_result = bm25_smoke_test(store, args.collection, bm25_query, limit=args.limit)

    report = build_a6_report(
        collection_name=args.collection,
        catalog=catalog,
        leaf_count=len(rows),
        verification=verification,
        dense_smoke=dense_result,
        bm25_smoke=bm25_result,
        join_missing_count=join_missing_count,
        join_extra_count=join_extra_count,
    )
    print(f"bm25_query={bm25_query!r}")
    print(format_a6_report_summary(report))

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        print(f"report_out={args.report_out}")

    all_passed = (
        verification.passed and dense_result.passed and bm25_result.passed
    )
    return 0 if all_passed else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Production A6 Milvus Leaf Collection lifecycle. Consumes a "
            "pre-joined rows JSONL artifact (chunk_id/document_id/.../"
            "dense_vector); does not embed, tokenize, or run A0-A5 itself."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="SS34.1 first build; FAILs if collection exists")
    build.add_argument("--uri", default="http://localhost:19530")
    build.add_argument("--collection", required=True)
    build.add_argument("--rows-jsonl", type=Path, required=True)
    build.set_defaults(func=_cmd_build)

    ingest = sub.add_parser("ingest-update", help="SS34.2 document-level add/replace")
    ingest.add_argument("--uri", default="http://localhost:19530")
    ingest.add_argument("--collection", required=True)
    ingest.add_argument("--rows-jsonl", type=Path, required=True)
    ingest.add_argument("--delete-document-ids-file", type=Path, default=None)
    ingest.set_defaults(func=_cmd_ingest_update)

    rebuild = sub.add_parser("rebuild", help="SS34.3 explicit full rebuild")
    rebuild.add_argument("--uri", default="http://localhost:19530")
    rebuild.add_argument("--collection", required=True)
    rebuild.add_argument("--rows-jsonl", type=Path, required=True)
    rebuild.add_argument("--rebuild", action="store_true", help="Explicit confirmation")
    rebuild.set_defaults(func=_cmd_rebuild)

    report = sub.add_parser(
        "report",
        help="SS37/SS38/SS39/SS44 verify + dense/bm25 smoke tests against an already-built Collection",
    )
    report.add_argument("--uri", default="http://localhost:19530")
    report.add_argument("--collection", required=True)
    report.add_argument(
        "--metadata-jsonl",
        type=Path,
        required=True,
        help="Full document_metadata.jsonl catalog (all statuses, not just retrievable)",
    )
    report.add_argument(
        "--rows-jsonl",
        type=Path,
        required=True,
        help="The rows.jsonl used to build/ingest the Collection; sources the smoke-test query vector/text",
    )
    report.add_argument(
        "--bm25-query",
        default=None,
        help="Override BM25 smoke-test query text (default: derived from the first row's content)",
    )
    report.add_argument("--limit", type=int, default=10)
    report.add_argument("--report-out", type=Path, default=None, help="Optional path to write the JSON report")
    report.set_defaults(func=_cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (A6CollectionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
