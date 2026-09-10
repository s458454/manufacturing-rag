"""A6.0 Native BM25 analyzer profiling.

Rebuilds current A3 Leaves via A1 + A3, or reuses a frozen leaf_catalog.jsonl
so local BM25 runs do not load the A3 Qwen tokenizer. Compares five temporary
BM25 profiles. Does not implement the official A6 collection, Dense, Hybrid,
RRF, or query rewrite. Does not call A2 or A5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from knowledge_base.a6_profile_config import (
    ALL_PROFILES,
    ANALYZER_PARAMS,
    BM25_B,
    BM25_INDEX_PARAMS,
    BM25_INDEX_TYPE,
    BM25_INVERTED_INDEX_ALGO,
    BM25_K1,
    BM25_METRIC_TYPE,
    CHUNK_ID_MAX_BYTES,
    CHN_CATEGORY_MINIMA,
    CHN_DOCUMENT_IDS,
    CHN_PROFILES,
    CHN_QUERIES_PER_DOC_MAX,
    CHN_QUERIES_PER_DOC_MIN,
    COLLECTION_FIELD_CHUNK_ID,
    COLLECTION_FIELD_CONTENT,
    COLLECTION_FIELD_DOCUMENT_ID,
    COLLECTION_FIELD_SPARSE,
    DEFAULT_A3_TOKENIZER,
    DOCUMENT_ID_MAX_BYTES,
    EN_CATEGORY_MINIMA,
    EN_DOCUMENT_IDS,
    EN_MIN_DOCUMENT_COVERAGE,
    EN_PROFILES,
    EXPECTED_CHN_DOCUMENT_COUNT,
    EXPECTED_CHN_QUERY_COUNT,
    EXPECTED_DOCUMENT_COUNT,
    EXPECTED_EN_DOCUMENT_COUNT,
    EXPECTED_EN_QUERY_COUNT,
    EXPECTED_LEAF_COUNT,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_LANGUAGE_SPECIFIC,
    OUTCOME_UNIFIED_ICU,
    PROFILE_CHN_CHINESE,
    PROFILE_CHN_ICU,
    PROFILE_CHN_JIEBA,
    PROFILE_EN_ICU,
    PROFILE_EN_STANDARD,
    QUERY_META_PURPOSE,
    SEARCH_ANNS_FIELD,
    SPOTCHECK_EN,
    SPOTCHECK_ZH,
    TOP_K,
    VARCHAR_CONTENT_MAX_BYTES,
)
from knowledge_base.chunking_config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP_TOKENS,
)
from knowledge_base.leaf_chunker import (
    A3ChunkingError,
    Leaf,
    chunk_documents,
)
from knowledge_base.markdown_loader import (
    MarkdownLoadingError,
    load_markdown_documents,
)
from knowledge_base.token_count import load_tokenizer


class A6ProfileError(Exception):
    """Fail-fast error for A6.0 analyzer profiling."""


@dataclass(frozen=True)
class ProfileQuery:
    query_id: str
    language: str
    query: str
    gold_document_id: str
    gold_chunk_ids: tuple[str, ...]
    evidence_text: str
    category: tuple[str, ...]


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    query: str
    gold_chunk_ids: tuple[str, ...]
    top20_chunk_ids: tuple[str, ...]
    gold_rank: int | None
    hit_at_5: bool
    hit_at_20: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class ProfileMetrics:
    profile: str
    query_count: int
    recall_at_5: float
    recall_at_20: float
    mrr_at_20: float
    miss_count: int
    miss_query_ids: tuple[str, ...]
    results: tuple[QueryResult, ...]


@dataclass(frozen=True)
class PairwiseRecall:
    both_hit: int
    left_only: int
    right_only: int
    both_miss: int


@dataclass(frozen=True)
class CohitRankStats:
    cohit_count: int
    icu_rank_better_count: int
    icu_rank_equal_count: int
    icu_rank_worse_count: int
    median_rank_delta: float | None


@dataclass(frozen=True)
class CorpusBinding:
    leaves: tuple[Leaf, ...]
    en_document_ids: frozenset[str]
    chn_document_ids: frozenset[str]
    en_leaves: tuple[Leaf, ...]
    chn_leaves: tuple[Leaf, ...]
    max_content_bytes: int


@dataclass
class CollectionSession:
    name: str
    dropped: bool = False


def utf8_byte_length(text: str) -> int:
    return len(text.encode("utf-8"))


def sha256_file_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def analyzer_params_for(profile: str) -> dict[str, Any]:
    try:
        params = ANALYZER_PARAMS[profile]
    except KeyError as exc:
        raise A6ProfileError(f"unknown profile: {profile}") from exc
    return json.loads(json.dumps(params))


def bm25_index_params() -> dict[str, Any]:
    return json.loads(json.dumps(BM25_INDEX_PARAMS))


def assert_only_analyzer_differs(profiles: Sequence[str] = ALL_PROFILES) -> None:
    """P11/P12: BM25 index knobs are identical; only analyzer_params change."""

    index = bm25_index_params()
    if index["index_type"] != BM25_INDEX_TYPE:
        raise A6ProfileError("index_type drifted from SPARSE_INVERTED_INDEX")
    if index["metric_type"] != BM25_METRIC_TYPE:
        raise A6ProfileError("metric_type drifted from BM25")
    params = index["params"]
    if params["inverted_index_algo"] != BM25_INVERTED_INDEX_ALGO:
        raise A6ProfileError("inverted_index_algo drifted from DAAT_MAXSCORE")
    if params["bm25_k1"] != BM25_K1 or params["bm25_b"] != BM25_B:
        raise A6ProfileError("BM25 k1/b drifted from frozen baseline")
    for profile in profiles:
        analyzer_params_for(profile)
    en_std = json.dumps(analyzer_params_for(PROFILE_EN_STANDARD), sort_keys=True)
    en_icu = json.dumps(analyzer_params_for(PROFILE_EN_ICU), sort_keys=True)
    if en_std == en_icu:
        raise A6ProfileError("EN-STD and EN-ICU analyzer_params must differ")
    chn_params = [
        json.dumps(analyzer_params_for(name), sort_keys=True)
        for name in CHN_PROFILES
    ]
    if len(set(chn_params)) != 3:
        raise A6ProfileError("Chinese profiles must use distinct analyzer_params")
    if analyzer_params_for(PROFILE_EN_ICU) != analyzer_params_for(PROFILE_CHN_ICU):
        raise A6ProfileError("EN-ICU and CHN-ICU must share the same ICU pipeline")


def language_for_document_id(
    document_id: str,
    *,
    en_ids: frozenset[str] = EN_DOCUMENT_IDS,
    chn_ids: frozenset[str] = CHN_DOCUMENT_IDS,
) -> str:
    if document_id in en_ids:
        return "en"
    if document_id in chn_ids:
        return "zh"
    raise A6ProfileError(f"document_id is not in EN/CHN allowlists: {document_id}")


def bind_corpus(
    leaves: Sequence[Leaf],
    *,
    en_ids: frozenset[str] = EN_DOCUMENT_IDS,
    chn_ids: frozenset[str] = CHN_DOCUMENT_IDS,
    expected_leaf_count: int | None = EXPECTED_LEAF_COUNT,
) -> CorpusBinding:
    if not leaves:
        raise A6ProfileError("No leaves provided")
    if en_ids & chn_ids:
        raise A6ProfileError("EN_DOCUMENT_IDS and CHN_DOCUMENT_IDS overlap")
    if len(en_ids) != EXPECTED_EN_DOCUMENT_COUNT:
        raise A6ProfileError(
            f"EN_DOCUMENT_IDS must contain {EXPECTED_EN_DOCUMENT_COUNT} ids, "
            f"got {len(en_ids)}"
        )
    if len(chn_ids) != EXPECTED_CHN_DOCUMENT_COUNT:
        raise A6ProfileError(
            f"CHN_DOCUMENT_IDS must contain {EXPECTED_CHN_DOCUMENT_COUNT} ids, "
            f"got {len(chn_ids)}"
        )

    actual_ids = {leaf.document_id for leaf in leaves}
    expected_ids = en_ids | chn_ids
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        raise A6ProfileError(
            "document_id set does not match frozen EN/CHN allowlists: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )

    en_leaves = tuple(
        leaf for leaf in leaves if leaf.document_id in en_ids
    )
    chn_leaves = tuple(
        leaf for leaf in leaves if leaf.document_id in chn_ids
    )
    if expected_leaf_count is not None and len(leaves) != expected_leaf_count:
        raise A6ProfileError(
            f"leaf_count={len(leaves)} != expected {expected_leaf_count}"
        )
    if len(en_leaves) + len(chn_leaves) != len(leaves):
        raise A6ProfileError("EN+CHN leaves do not cover the full leaf set")

    max_bytes = 0
    for leaf in leaves:
        content_bytes = utf8_byte_length(leaf.content)
        if content_bytes > VARCHAR_CONTENT_MAX_BYTES:
            raise A6ProfileError(
                f"Leaf {leaf.chunk_id} content is {content_bytes} UTF-8 bytes, "
                f"exceeds VARCHAR max {VARCHAR_CONTENT_MAX_BYTES}; truncation "
                "is forbidden"
            )
        if content_bytes > max_bytes:
            max_bytes = content_bytes
        if utf8_byte_length(leaf.chunk_id) > CHUNK_ID_MAX_BYTES:
            raise A6ProfileError(
                f"chunk_id exceeds {CHUNK_ID_MAX_BYTES} UTF-8 bytes: "
                f"{leaf.chunk_id!r}"
            )
        if utf8_byte_length(leaf.document_id) > DOCUMENT_ID_MAX_BYTES:
            raise A6ProfileError(
                f"document_id exceeds {DOCUMENT_ID_MAX_BYTES} UTF-8 bytes: "
                f"{leaf.document_id!r}"
            )

    return CorpusBinding(
        leaves=tuple(leaves),
        en_document_ids=en_ids,
        chn_document_ids=chn_ids,
        en_leaves=en_leaves,
        chn_leaves=chn_leaves,
        max_content_bytes=max_bytes,
    )


def load_bound_corpus(
    canonical_root: Path,
    tokenizer: Any,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    en_ids: frozenset[str] = EN_DOCUMENT_IDS,
    chn_ids: frozenset[str] = CHN_DOCUMENT_IDS,
    expected_leaf_count: int | None = EXPECTED_LEAF_COUNT,
) -> CorpusBinding:
    documents = load_markdown_documents(canonical_root)
    chunking = chunk_documents(
        documents,
        tokenizer,
        chunk_size=chunk_size,
        overlap_tokens=overlap_tokens,
    )
    actual_doc_count = len({doc.document_id for doc in documents})
    if actual_doc_count != EXPECTED_DOCUMENT_COUNT:
        raise A6ProfileError(
            f"document_count={actual_doc_count} != {EXPECTED_DOCUMENT_COUNT}"
        )
    return bind_corpus(
        chunking.leaves,
        en_ids=en_ids,
        chn_ids=chn_ids,
        expected_leaf_count=expected_leaf_count,
    )


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise A6ProfileError(f"query field {key!r} must be a non-empty string")
    return value


def parse_query_record(payload: dict[str, Any]) -> ProfileQuery:
    query_id = _require_string(payload, "query_id")
    language = _require_string(payload, "language")
    if language not in {"en", "zh"}:
        raise A6ProfileError(
            f"{query_id}: language must be 'en' or 'zh', got {language!r}"
        )
    query = _require_string(payload, "query")
    gold_document_id = _require_string(payload, "gold_document_id")
    gold_raw = payload.get("gold_chunk_ids")
    if not isinstance(gold_raw, list) or not gold_raw:
        raise A6ProfileError(f"{query_id}: gold_chunk_ids must be a non-empty list")
    gold_chunk_ids: list[str] = []
    for item in gold_raw:
        if not isinstance(item, str) or not item.strip():
            raise A6ProfileError(f"{query_id}: gold_chunk_ids items must be strings")
        gold_chunk_ids.append(item)
    evidence_text = _require_string(payload, "evidence_text")
    category_raw = payload.get("category")
    if not isinstance(category_raw, list) or not category_raw:
        raise A6ProfileError(f"{query_id}: category must be a non-empty list")
    categories: list[str] = []
    for item in category_raw:
        if not isinstance(item, str) or not item.strip():
            raise A6ProfileError(f"{query_id}: category items must be strings")
        categories.append(item)
    if query.strip() == evidence_text.strip():
        raise A6ProfileError(
            f"{query_id}: query must not be a full copy of evidence_text"
        )
    return ProfileQuery(
        query_id=query_id,
        language=language,
        query=query,
        gold_document_id=gold_document_id,
        gold_chunk_ids=tuple(gold_chunk_ids),
        evidence_text=evidence_text,
        category=tuple(categories),
    )


def load_query_set(path: Path) -> tuple[ProfileQuery, ...]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise A6ProfileError(f"query set is empty: {path}")
    queries: list[ProfileQuery] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            raise A6ProfileError(
                f"{path}:{line_no}: JSONL comments are forbidden; "
                "put metadata in query_set.meta.json"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise A6ProfileError(
                f"{path}:{line_no}: invalid JSON object"
            ) from exc
        if not isinstance(payload, dict):
            raise A6ProfileError(f"{path}:{line_no}: each line must be a JSON object")
        queries.append(parse_query_record(payload))
    return tuple(queries)


def _leaf_index(leaves: Sequence[Leaf]) -> dict[str, Leaf]:
    index: dict[str, Leaf] = {}
    for leaf in leaves:
        if leaf.chunk_id in index:
            raise A6ProfileError(f"duplicate chunk_id in corpus: {leaf.chunk_id}")
        index[leaf.chunk_id] = leaf
    return index


def _count_categories(queries: Sequence[ProfileQuery]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for query in queries:
        for name in set(query.category):
            counts[name] += 1
    return counts


def validate_query_set(
    queries: Sequence[ProfileQuery],
    binding: CorpusBinding,
) -> None:
    if not queries:
        raise A6ProfileError("query set is empty")
    seen_ids: set[str] = set()
    en_queries: list[ProfileQuery] = []
    zh_queries: list[ProfileQuery] = []
    en_index = _leaf_index(binding.en_leaves)
    zh_index = _leaf_index(binding.chn_leaves)

    for query in queries:
        if query.query_id in seen_ids:
            raise A6ProfileError(f"duplicate query_id: {query.query_id}")
        seen_ids.add(query.query_id)
        expected_prefix = "EN" if query.language == "en" else "CHN"
        if not query.query_id.startswith(expected_prefix):
            raise A6ProfileError(
                f"{query.query_id}: language={query.language} does not match "
                f"query_id prefix (expected {expected_prefix})"
            )
        if query.language == "en":
            allow_ids = binding.en_document_ids
            leaf_index = en_index
            en_queries.append(query)
        else:
            allow_ids = binding.chn_document_ids
            leaf_index = zh_index
            zh_queries.append(query)
        if query.gold_document_id not in allow_ids:
            raise A6ProfileError(
                f"{query.query_id}: gold_document_id {query.gold_document_id} "
                f"is not in the {query.language} corpus"
            )
        for chunk_id in query.gold_chunk_ids:
            leaf = leaf_index.get(chunk_id)
            if leaf is None:
                raise A6ProfileError(
                    f"{query.query_id}: gold chunk {chunk_id} is not in the "
                    f"{query.language} corpus"
                )
            if leaf.document_id != query.gold_document_id:
                raise A6ProfileError(
                    f"{query.query_id}: gold chunk {chunk_id} belongs to "
                    f"{leaf.document_id}, not {query.gold_document_id}"
                )

    if len(en_queries) != EXPECTED_EN_QUERY_COUNT:
        raise A6ProfileError(
            f"english query count={len(en_queries)} != {EXPECTED_EN_QUERY_COUNT}"
        )
    if len(zh_queries) != EXPECTED_CHN_QUERY_COUNT:
        raise A6ProfileError(
            f"chinese query count={len(zh_queries)} != {EXPECTED_CHN_QUERY_COUNT}"
        )

    en_docs = {query.gold_document_id for query in en_queries}
    if len(en_docs) < EN_MIN_DOCUMENT_COVERAGE:
        raise A6ProfileError(
            f"english queries cover {len(en_docs)} documents, "
            f"need >= {EN_MIN_DOCUMENT_COVERAGE}"
        )
    en_cat = _count_categories(en_queries)
    for name, minimum in EN_CATEGORY_MINIMA.items():
        if en_cat[name] < minimum:
            raise A6ProfileError(
                f"english category {name}={en_cat[name]} < {minimum}"
            )
    zh_cat = _count_categories(zh_queries)
    for name, minimum in CHN_CATEGORY_MINIMA.items():
        if zh_cat[name] < minimum:
            raise A6ProfileError(
                f"chinese category {name}={zh_cat[name]} < {minimum}"
            )
    per_doc = Counter(query.gold_document_id for query in zh_queries)
    if set(per_doc) != set(binding.chn_document_ids):
        raise A6ProfileError(
            "chinese queries must cover all three CHN documents: "
            f"got {sorted(per_doc)}"
        )
    for document_id, count in per_doc.items():
        if count < CHN_QUERIES_PER_DOC_MIN:
            raise A6ProfileError(
                f"chinese document {document_id} has {count} queries; "
                f"need >= {CHN_QUERIES_PER_DOC_MIN} or mark INCONCLUSIVE"
            )
        if count > CHN_QUERIES_PER_DOC_MAX:
            raise A6ProfileError(
                f"chinese document {document_id} has {count} queries; "
                f"max {CHN_QUERIES_PER_DOC_MAX}"
            )


def build_query_meta(
    query_set_path: Path,
    *,
    sha256: str | None = None,
) -> dict[str, Any]:
    digest = sha256 or sha256_file_bytes(query_set_path)
    return {
        "purpose": QUERY_META_PURPOSE,
        "is_d1_gold": False,
        "gold_chunk_id_build_specific": True,
        "query_count_en": EXPECTED_EN_QUERY_COUNT,
        "query_count_zh": EXPECTED_CHN_QUERY_COUNT,
        "query_set_sha256": digest,
    }


def write_query_meta(path: Path, query_set_path: Path) -> dict[str, Any]:
    meta = build_query_meta(query_set_path)
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return meta


def load_query_meta(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A6ProfileError(f"failed to read query meta: {path}") from exc
    if not isinstance(payload, dict):
        raise A6ProfileError(f"query meta must be a JSON object: {path}")
    return payload


def assert_query_freeze_gate(query_set_path: Path, meta_path: Path) -> str:
    meta = load_query_meta(meta_path)
    expected = meta.get("query_set_sha256")
    if not isinstance(expected, str) or not expected:
        raise A6ProfileError("query_set.meta.json is missing query_set_sha256")
    if meta.get("is_d1_gold") is True:
        raise A6ProfileError("A6.0 query set must not be marked as D1 Gold")
    actual = sha256_file_bytes(query_set_path)
    if actual != expected:
        raise A6ProfileError(
            "query_set.jsonl SHA256 does not match query_set.meta.json: "
            f"file={actual} meta={expected}"
        )
    return actual


def gold_rank(gold_chunk_ids: Sequence[str], ranked: Sequence[str]) -> int | None:
    gold = set(gold_chunk_ids)
    for index, chunk_id in enumerate(ranked, start=1):
        if chunk_id in gold:
            return index
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None, *, cutoff: int = TOP_K) -> float:
    if rank is None or rank > cutoff:
        return 0.0
    return 1.0 / rank


def score_query(
    query: ProfileQuery, ranked_chunk_ids: Sequence[str]
) -> QueryResult:
    top20 = tuple(ranked_chunk_ids[:TOP_K])
    rank = gold_rank(query.gold_chunk_ids, top20)
    return QueryResult(
        query_id=query.query_id,
        query=query.query,
        gold_chunk_ids=query.gold_chunk_ids,
        top20_chunk_ids=top20,
        gold_rank=rank,
        hit_at_5=hit_at_k(rank, 5),
        hit_at_20=hit_at_k(rank, 20),
        reciprocal_rank=reciprocal_rank(rank),
    )


def aggregate_metrics(
    profile: str, queries: Sequence[ProfileQuery], ranked: Sequence[Sequence[str]]
) -> ProfileMetrics:
    if len(queries) != len(ranked):
        raise A6ProfileError("ranked lists must align with queries")
    results = tuple(
        score_query(query, ranks) for query, ranks in zip(queries, ranked)
    )
    n = len(results)
    recall_at_5 = sum(item.hit_at_5 for item in results) / n
    recall_at_20 = sum(item.hit_at_20 for item in results) / n
    mrr_at_20 = sum(item.reciprocal_rank for item in results) / n
    misses = tuple(item.query_id for item in results if not item.hit_at_20)
    return ProfileMetrics(
        profile=profile,
        query_count=n,
        recall_at_5=recall_at_5,
        recall_at_20=recall_at_20,
        mrr_at_20=mrr_at_20,
        miss_count=len(misses),
        miss_query_ids=misses,
        results=results,
    )


def pairwise_recall20(
    left: ProfileMetrics, right: ProfileMetrics
) -> PairwiseRecall:
    left_hits = {item.query_id: item.hit_at_20 for item in left.results}
    right_hits = {item.query_id: item.hit_at_20 for item in right.results}
    if left_hits.keys() != right_hits.keys():
        raise A6ProfileError("pairwise metrics cover different query ids")
    both_hit = left_only = right_only = both_miss = 0
    for query_id in left_hits:
        l_hit = left_hits[query_id]
        r_hit = right_hits[query_id]
        if l_hit and r_hit:
            both_hit += 1
        elif l_hit:
            left_only += 1
        elif r_hit:
            right_only += 1
        else:
            both_miss += 1
    return PairwiseRecall(
        both_hit=both_hit,
        left_only=left_only,
        right_only=right_only,
        both_miss=both_miss,
    )


def cohit_rank_stats(
    reference: ProfileMetrics, icu: ProfileMetrics
) -> CohitRankStats:
    ref_ranks = {item.query_id: item.gold_rank for item in reference.results}
    icu_ranks = {item.query_id: item.gold_rank for item in icu.results}
    deltas: list[int] = []
    better = equal = worse = 0
    for query_id, ref_rank in ref_ranks.items():
        icu_rank = icu_ranks.get(query_id)
        if ref_rank is None or icu_rank is None:
            continue
        delta = icu_rank - ref_rank
        deltas.append(delta)
        if delta < 0:
            better += 1
        elif delta == 0:
            equal += 1
        else:
            worse += 1
    median = float(statistics.median(deltas)) if deltas else None
    return CohitRankStats(
        cohit_count=len(deltas),
        icu_rank_better_count=better,
        icu_rank_equal_count=equal,
        icu_rank_worse_count=worse,
        median_rank_delta=median,
    )


def recall20_loss_queries(reference: ProfileMetrics, other: ProfileMetrics) -> int:
    pair = pairwise_recall20(reference, other)
    return pair.left_only


def icu_recall_acceptable(reference: ProfileMetrics, icu: ProfileMetrics) -> bool:
    return recall20_loss_queries(reference, icu) <= 1


def choose_chn_specialist(
    chinese: ProfileMetrics, jieba: ProfileMetrics
) -> ProfileMetrics:
    if chinese.recall_at_20 > jieba.recall_at_20:
        return chinese
    if jieba.recall_at_20 > chinese.recall_at_20:
        return jieba
    if chinese.mrr_at_20 >= jieba.mrr_at_20:
        return chinese
    return jieba


def recommend_outcome(
    en_std: ProfileMetrics,
    en_icu: ProfileMetrics,
    chn_specialist: ProfileMetrics,
    chn_icu: ProfileMetrics,
    *,
    inconclusive_reason: str | None = None,
) -> dict[str, Any]:
    if inconclusive_reason:
        outcome = OUTCOME_INCONCLUSIVE
    else:
        en_ok = icu_recall_acceptable(en_std, en_icu)
        chn_ok = icu_recall_acceptable(chn_specialist, chn_icu)
        if en_ok and chn_ok:
            outcome = OUTCOME_UNIFIED_ICU
        else:
            outcome = OUTCOME_LANGUAGE_SPECIFIC
    return {
        "recommendation": outcome,
        "inconclusive_reason": inconclusive_reason,
        "icu_en_recall_acceptable": icu_recall_acceptable(en_std, en_icu),
        "icu_chn_recall_acceptable": icu_recall_acceptable(
            chn_specialist, chn_icu
        ),
        "chn_specialist_profile": chn_specialist.profile,
        "en_cohit": asdict(cohit_rank_stats(en_std, en_icu)),
        "chn_cohit": asdict(cohit_rank_stats(chn_specialist, chn_icu)),
        "en_pairwise": asdict(pairwise_recall20(en_std, en_icu)),
        "chn_pairwise": asdict(pairwise_recall20(chn_specialist, chn_icu)),
    }


def corpus_summary(binding: CorpusBinding) -> dict[str, Any]:
    return {
        "corpus_document_count": len(
            binding.en_document_ids | binding.chn_document_ids
        ),
        "corpus_leaf_count": len(binding.leaves),
        "english_document_count": len(binding.en_document_ids),
        "english_leaf_count": len(binding.en_leaves),
        "chinese_document_count": len(binding.chn_document_ids),
        "chinese_leaf_count": len(binding.chn_leaves),
        "max_content_bytes": binding.max_content_bytes,
        "a2_runtime": False,
        "a5_runtime": False,
    }


def write_leaf_catalog(path: Path, binding: CorpusBinding) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for leaf in binding.leaves:
        language = language_for_document_id(
            leaf.document_id,
            en_ids=binding.en_document_ids,
            chn_ids=binding.chn_document_ids,
        )
        lines.append(
            json.dumps(
                {
                    "chunk_id": leaf.chunk_id,
                    "document_id": leaf.document_id,
                    "page_start": leaf.page_start,
                    "page_end": leaf.page_end,
                    "language": language,
                    "content": leaf.content,
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_leaf_catalog(
    path: Path,
    *,
    en_ids: frozenset[str] = EN_DOCUMENT_IDS,
    chn_ids: frozenset[str] = CHN_DOCUMENT_IDS,
    expected_leaf_count: int | None = EXPECTED_LEAF_COUNT,
) -> CorpusBinding:
    """Bind Leaves from a frozen catalog. Does not load a tokenizer or A0 Markdown."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise A6ProfileError(f"failed to read leaf catalog: {path}") from exc
    if not text.strip():
        raise A6ProfileError(f"leaf catalog is empty: {path}")
    leaves: list[Leaf] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            raise A6ProfileError(
                f"{path}:{line_no}: JSONL comments are forbidden"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise A6ProfileError(
                f"{path}:{line_no}: invalid JSON object"
            ) from exc
        if not isinstance(payload, dict):
            raise A6ProfileError(f"{path}:{line_no}: each line must be a JSON object")
        chunk_id = payload.get("chunk_id")
        document_id = payload.get("document_id")
        content = payload.get("content")
        language = payload.get("language")
        page_start = payload.get("page_start")
        page_end = payload.get("page_end")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise A6ProfileError(f"{path}:{line_no}: chunk_id must be a string")
        if not isinstance(document_id, str) or not document_id.strip():
            raise A6ProfileError(f"{path}:{line_no}: document_id must be a string")
        if not isinstance(content, str):
            raise A6ProfileError(f"{path}:{line_no}: content must be a string")
        if not isinstance(language, str) or language not in {"en", "zh"}:
            raise A6ProfileError(f"{path}:{line_no}: language must be 'en' or 'zh'")
        if not isinstance(page_start, int) or not isinstance(page_end, int):
            raise A6ProfileError(f"{path}:{line_no}: page_start/page_end must be ints")
        expected_language = language_for_document_id(
            document_id, en_ids=en_ids, chn_ids=chn_ids
        )
        if language != expected_language:
            raise A6ProfileError(
                f"{path}:{line_no}: language={language!r} does not match "
                f"document_id allowlist ({expected_language})"
            )
        leaves.append(
            Leaf(
                chunk_id=chunk_id,
                document_id=document_id,
                section_id="catalog",
                chunk_index=len(leaves),
                page_start=page_start,
                page_end=page_end,
                content=content,
            )
        )
    return bind_corpus(
        leaves,
        en_ids=en_ids,
        chn_ids=chn_ids,
        expected_leaf_count=expected_leaf_count,
    )


def load_binding_for_cli(args: argparse.Namespace) -> CorpusBinding:
    catalog = getattr(args, "leaf_catalog", None)
    root = getattr(args, "canonical_root", None)
    if catalog and root:
        raise A6ProfileError(
            "use either --leaf-catalog or --canonical-root, not both"
        )
    if catalog:
        return load_leaf_catalog(catalog)
    if root is None:
        raise A6ProfileError(
            "need --leaf-catalog (frozen catalog, no Qwen) or "
            "--canonical-root (A1+A3 rebuild)"
        )
    tokenizer = load_tokenizer(args.tokenizer)
    return load_bound_corpus(
        root,
        tokenizer,
        chunk_size=args.chunk_size,
        overlap_tokens=args.overlap,
    )


def profiling_schema_fields() -> tuple[str, ...]:
    return (
        COLLECTION_FIELD_CHUNK_ID,
        COLLECTION_FIELD_DOCUMENT_ID,
        COLLECTION_FIELD_CONTENT,
        COLLECTION_FIELD_SPARSE,
    )


def assert_search_contract(
    *,
    anns_field: str,
    schema_fields: Sequence[str],
    dense_invoked: bool = False,
    rrf_invoked: bool = False,
    reranker_invoked: bool = False,
) -> None:
    if dense_invoked:
        raise A6ProfileError("A6.0 runtime must not call a dense encoder")
    if rrf_invoked:
        raise A6ProfileError("A6.0 runtime must not call RRF")
    if reranker_invoked:
        raise A6ProfileError("A6.0 runtime must not call a reranker")
    if COLLECTION_FIELD_SPARSE not in schema_fields:
        raise A6ProfileError("profiling collection is missing sparse_vector")
    if "dense_vector" in schema_fields:
        raise A6ProfileError("profiling collection must not include dense_vector")
    if anns_field != SEARCH_ANNS_FIELD:
        raise A6ProfileError(
            f"search anns_field must be {SEARCH_ANNS_FIELD}, got {anns_field!r}"
        )


def collection_name(profile: str, timestamp: str) -> str:
    slug = {
        PROFILE_EN_STANDARD: "en_standard",
        PROFILE_EN_ICU: "en_icu",
        PROFILE_CHN_CHINESE: "zh_chinese",
        PROFILE_CHN_JIEBA: "zh_jieba",
        PROFILE_CHN_ICU: "zh_icu",
    }[profile]
    return f"a6_profile_{slug}_{timestamp}"


def tokens_from_run_analyzer(result: Any) -> list[str]:
    """Normalize pymilvus 2.5 run_analyzer payloads to token strings.

    2.5.14 + pymilvus 2.5.11 returns AnalyzeResult (object with .tokens),
    not a bare list. Token items may be strings or dicts with 'token'.
    """

    if hasattr(result, "tokens"):
        raw = result.tokens
    elif isinstance(result, dict):
        raw = result.get("tokens", result)
    elif isinstance(result, list) and result and isinstance(result[0], dict):
        raw = result[0].get("tokens", result)
    else:
        raw = result
    if not isinstance(raw, list):
        raise A6ProfileError(
            f"run_analyzer returned unexpected type: {type(result)!r}"
        )
    tokens: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            tokens.append(str(item.get("token", item)))
        elif hasattr(item, "token"):
            tokens.append(str(item.token))
        else:
            tokens.append(str(item))
    return tokens


def _lite_run_analyzer(text: str, analyzer_params: dict[str, Any]) -> list[str]:
    """Tokenize via Milvus Lite when the URI is a local .db file.

    Lite 3 does not implement the pymilvus run_analyzer RPC and only
    supports standard/jieba. Missing ICU/chinese must fail fast.
    """

    try:
        from milvus_lite.analyzer.factory import create_analyzer
    except ImportError as exc:
        raise A6ProfileError(
            "local file URI requires milvus-lite in the same environment"
        ) from exc
    try:
        analyzer = create_analyzer(analyzer_params)
    except Exception as exc:
        raise A6ProfileError(
            "this Milvus runtime cannot apply analyzer_params="
            f"{analyzer_params!r}: {exc}. A6.0 requires Milvus 2.5.14 "
            "Native analyzers (standard, icu, chinese, jieba). Milvus Lite 3 "
            "only supports standard and jieba."
        ) from exc
    tokens = analyzer.tokenize(text)
    if not isinstance(tokens, list):
        raise A6ProfileError(
            f"lite analyzer returned unexpected type: {type(tokens)!r}"
        )
    return [str(token) for token in tokens]


class ProfileStore:
    """Minimal Milvus operations used by A6.0. Tests inject a fake."""

    def run_analyzer(
        self, text: str, analyzer_params: dict[str, Any]
    ) -> list[str]:
        raise NotImplementedError

    def create_profile_collection(
        self, name: str, analyzer_params: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    def insert_leaves(self, name: str, leaves: Sequence[Leaf]) -> None:
        raise NotImplementedError

    def search(
        self, name: str, query: str, *, limit: int = TOP_K
    ) -> list[RankedHit]:
        raise NotImplementedError

    def drop_collection(self, name: str) -> None:
        raise NotImplementedError

    def has_collection(self, name: str) -> bool:
        raise NotImplementedError


class MilvusProfileStore(ProfileStore):
    def __init__(self, uri: str) -> None:
        try:
            from pymilvus import (
                DataType,
                Function,
                FunctionType,
                MilvusClient,
            )
        except ImportError as exc:
            raise A6ProfileError(
                "pymilvus is required for real A6.0 Milvus smoke"
            ) from exc
        self._DataType = DataType
        self._Function = Function
        self._FunctionType = FunctionType
        self.uri = uri
        self.local_file = not uri.startswith(("http://", "https://", "tcp://", "unix:"))
        self.client = MilvusClient(uri=uri)

    def run_analyzer(
        self, text: str, analyzer_params: dict[str, Any]
    ) -> list[str]:
        if self.local_file:
            return _lite_run_analyzer(text, analyzer_params)
        result = self.client.run_analyzer(text, analyzer_params)
        return tokens_from_run_analyzer(result)

    def create_profile_collection(
        self, name: str, analyzer_params: dict[str, Any]
    ) -> None:
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name=COLLECTION_FIELD_CHUNK_ID,
            datatype=self._DataType.VARCHAR,
            max_length=CHUNK_ID_MAX_BYTES,
            is_primary=True,
        )
        schema.add_field(
            field_name=COLLECTION_FIELD_DOCUMENT_ID,
            datatype=self._DataType.VARCHAR,
            max_length=DOCUMENT_ID_MAX_BYTES,
        )
        schema.add_field(
            field_name=COLLECTION_FIELD_CONTENT,
            datatype=self._DataType.VARCHAR,
            max_length=VARCHAR_CONTENT_MAX_BYTES,
            enable_analyzer=True,
            analyzer_params=analyzer_params,
        )
        schema.add_field(
            field_name=COLLECTION_FIELD_SPARSE,
            datatype=self._DataType.SPARSE_FLOAT_VECTOR,
        )
        schema.add_function(
            self._Function(
                name="a6_profile_bm25",
                input_field_names=[COLLECTION_FIELD_CONTENT],
                output_field_names=[COLLECTION_FIELD_SPARSE],
                function_type=self._FunctionType.BM25,
            )
        )
        index_params = self.client.prepare_index_params()
        frozen = bm25_index_params()
        index_params.add_index(
            field_name=COLLECTION_FIELD_SPARSE,
            index_type=frozen["index_type"],
            metric_type=frozen["metric_type"],
            params=frozen["params"],
        )
        try:
            self.client.create_collection(
                collection_name=name,
                schema=schema,
                index_params=index_params,
            )
            self.client.load_collection(name)
        except Exception as exc:
            raise A6ProfileError(
                f"failed to create profiling collection {name} "
                f"with analyzer_params={analyzer_params!r}: {exc}"
            ) from exc

    def insert_leaves(self, name: str, leaves: Sequence[Leaf]) -> None:
        rows = [
            {
                COLLECTION_FIELD_CHUNK_ID: leaf.chunk_id,
                COLLECTION_FIELD_DOCUMENT_ID: leaf.document_id,
                COLLECTION_FIELD_CONTENT: leaf.content,
            }
            for leaf in leaves
        ]
        self.client.insert(name, rows)
        self.client.flush(name)
        self.client.load_collection(name)

    def search(
        self, name: str, query: str, *, limit: int = TOP_K
    ) -> list[RankedHit]:
        assert_search_contract(
            anns_field=SEARCH_ANNS_FIELD,
            schema_fields=profiling_schema_fields(),
        )
        raw = self.client.search(
            collection_name=name,
            data=[query],
            anns_field=SEARCH_ANNS_FIELD,
            output_fields=[
                COLLECTION_FIELD_CHUNK_ID,
                COLLECTION_FIELD_DOCUMENT_ID,
            ],
            limit=limit,
        )
        hits = raw[0] if raw else []
        ranked: list[RankedHit] = []
        for index, hit in enumerate(hits, start=1):
            entity = hit.get("entity", hit)
            chunk_id = entity.get(COLLECTION_FIELD_CHUNK_ID, hit.get("id"))
            score = float(hit.get("distance", hit.get("score", 0.0)))
            ranked.append(
                RankedHit(chunk_id=str(chunk_id), rank=index, score=score)
            )
        return ranked

    def drop_collection(self, name: str) -> None:
        if self.client.has_collection(name):
            self.client.drop_collection(name)

    def has_collection(self, name: str) -> bool:
        return bool(self.client.has_collection(name))


def run_spotcheck(
    store: ProfileStore,
    texts: Sequence[str],
    profiles: Sequence[str],
) -> dict[str, dict[str, list[str]]]:
    output: dict[str, dict[str, list[str]]] = {}
    for text in texts:
        output[text] = {}
        for profile in profiles:
            output[text][profile] = store.run_analyzer(
                text, analyzer_params_for(profile)
            )
    return output


def search_queries(
    store: ProfileStore,
    collection: str,
    queries: Sequence[ProfileQuery],
) -> list[list[str]]:
    ranked: list[list[str]] = []
    for query in queries:
        hits = store.search(collection, query.query, limit=TOP_K)
        ranked.append([hit.chunk_id for hit in hits])
    return ranked


def run_one_profile(
    store: ProfileStore,
    profile: str,
    leaves: Sequence[Leaf],
    queries: Sequence[ProfileQuery],
    *,
    timestamp: str,
    keep_collections: bool,
    sessions: list[CollectionSession],
) -> ProfileMetrics:
    name = collection_name(profile, timestamp)
    session = CollectionSession(name=name)
    sessions.append(session)
    store.create_profile_collection(name, analyzer_params_for(profile))
    store.insert_leaves(name, leaves)
    ranked = search_queries(store, name, queries)
    metrics = aggregate_metrics(profile, queries, ranked)
    if not keep_collections:
        store.drop_collection(name)
        session.dropped = True
    return metrics


def drop_sessions(
    store: ProfileStore,
    sessions: Sequence[CollectionSession],
    *,
    keep_collections: bool,
) -> None:
    if keep_collections:
        return
    for session in sessions:
        if not session.dropped:
            store.drop_collection(session.name)
            session.dropped = True


def metrics_to_dict(metrics: ProfileMetrics) -> dict[str, Any]:
    return {
        "profile": metrics.profile,
        "query_count": metrics.query_count,
        "recall_at_5": metrics.recall_at_5,
        "recall_at_20": metrics.recall_at_20,
        "mrr_at_20": metrics.mrr_at_20,
        "miss_count": metrics.miss_count,
        "miss_query_ids": list(metrics.miss_query_ids),
        "results": [
            {
                "query_id": item.query_id,
                "query": item.query,
                "gold_chunk_ids": list(item.gold_chunk_ids),
                "top20_chunk_ids": list(item.top20_chunk_ids),
                "gold_rank": item.gold_rank,
                "hit_at_5": item.hit_at_5,
                "hit_at_20": item.hit_at_20,
                "reciprocal_rank": item.reciprocal_rank,
            }
            for item in metrics.results
        ],
    }


def gold_ranks(metrics: ProfileMetrics) -> dict[str, int | None]:
    return {item.query_id: item.gold_rank for item in metrics.results}


def assert_repeatable(first: ProfileMetrics, second: ProfileMetrics) -> None:
    if gold_ranks(first) != gold_ranks(second):
        raise A6ProfileError(
            f"{first.profile}: repeated run gold_rank mismatch; "
            "report BM25 ties, do not ignore"
        )
    if (
        first.recall_at_5 != second.recall_at_5
        or first.recall_at_20 != second.recall_at_20
        or first.mrr_at_20 != second.mrr_at_20
        or first.miss_count != second.miss_count
    ):
        raise A6ProfileError(
            f"{first.profile}: repeated run summary metrics mismatch"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_profiling(
    binding: CorpusBinding,
    queries: Sequence[ProfileQuery],
    store: ProfileStore,
    output_dir: Path,
    *,
    keep_collections: bool = False,
    timestamp: str | None = None,
    repeat: bool = True,
) -> dict[str, Any]:
    assert_only_analyzer_differs()
    validate_query_set(queries, binding)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or time.strftime("%Y%m%d%H%M%S")
    en_queries = [query for query in queries if query.language == "en"]
    zh_queries = [query for query in queries if query.language == "zh"]
    tokenization_en = run_spotcheck(store, SPOTCHECK_EN, EN_PROFILES)
    tokenization_zh = run_spotcheck(store, SPOTCHECK_ZH, CHN_PROFILES)
    _write_json(output_dir / "tokenization_en.json", tokenization_en)
    _write_json(output_dir / "tokenization_zh.json", tokenization_zh)

    sessions: list[CollectionSession] = []
    metrics: dict[str, ProfileMetrics] = {}
    try:
        for profile in EN_PROFILES:
            metrics[profile] = run_one_profile(
                store,
                profile,
                binding.en_leaves,
                en_queries,
                timestamp=stamp,
                keep_collections=keep_collections,
                sessions=sessions,
            )
        for profile in CHN_PROFILES:
            metrics[profile] = run_one_profile(
                store,
                profile,
                binding.chn_leaves,
                zh_queries,
                timestamp=stamp,
                keep_collections=keep_collections,
                sessions=sessions,
            )
        if repeat:
            repeat_metrics: dict[str, ProfileMetrics] = {}
            repeat_stamp = stamp + "r"
            for profile in EN_PROFILES:
                repeat_metrics[profile] = run_one_profile(
                    store,
                    profile,
                    binding.en_leaves,
                    en_queries,
                    timestamp=repeat_stamp,
                    keep_collections=False,
                    sessions=sessions,
                )
            for profile in CHN_PROFILES:
                repeat_metrics[profile] = run_one_profile(
                    store,
                    profile,
                    binding.chn_leaves,
                    zh_queries,
                    timestamp=repeat_stamp,
                    keep_collections=False,
                    sessions=sessions,
                )
            for profile, first in metrics.items():
                assert_repeatable(first, repeat_metrics[profile])
    except Exception:
        drop_sessions(store, sessions, keep_collections=False)
        raise
    if not keep_collections:
        drop_sessions(store, sessions, keep_collections=False)

    for profile, item in metrics.items():
        slug = {
            PROFILE_EN_STANDARD: "results_en_standard.json",
            PROFILE_EN_ICU: "results_en_icu.json",
            PROFILE_CHN_CHINESE: "results_zh_chinese.json",
            PROFILE_CHN_JIEBA: "results_zh_jieba.json",
            PROFILE_CHN_ICU: "results_zh_icu.json",
        }[profile]
        _write_json(output_dir / slug, metrics_to_dict(item))

    specialist = choose_chn_specialist(
        metrics[PROFILE_CHN_CHINESE], metrics[PROFILE_CHN_JIEBA]
    )
    recommendation = recommend_outcome(
        metrics[PROFILE_EN_STANDARD],
        metrics[PROFILE_EN_ICU],
        specialist,
        metrics[PROFILE_CHN_ICU],
    )
    summary = {
        **corpus_summary(binding),
        "english_query_count": len(en_queries),
        "chinese_query_count": len(zh_queries),
        "profiles": {
            profile: {
                "recall_at_5": item.recall_at_5,
                "recall_at_20": item.recall_at_20,
                "mrr_at_20": item.mrr_at_20,
                "miss_count": item.miss_count,
            }
            for profile, item in metrics.items()
        },
        "pairwise": {
            "en_standard_vs_icu": asdict(
                pairwise_recall20(
                    metrics[PROFILE_EN_STANDARD], metrics[PROFILE_EN_ICU]
                )
            ),
            "chn_chinese_vs_jieba": asdict(
                pairwise_recall20(
                    metrics[PROFILE_CHN_CHINESE], metrics[PROFILE_CHN_JIEBA]
                )
            ),
            "chn_specialist_vs_icu": asdict(
                pairwise_recall20(specialist, metrics[PROFILE_CHN_ICU])
            ),
        },
        "cohit_rank": {
            "en": asdict(
                cohit_rank_stats(
                    metrics[PROFILE_EN_STANDARD], metrics[PROFILE_EN_ICU]
                )
            ),
            "chn": asdict(cohit_rank_stats(specialist, metrics[PROFILE_CHN_ICU])),
        },
        **recommendation,
        "a5_runtime_dependency": "NONE",
        "keep_collections": keep_collections,
        "collection_names": [session.name for session in sessions],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _cmd_bind_corpus(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    binding = load_bound_corpus(
        args.canonical_root,
        tokenizer,
        chunk_size=args.chunk_size,
        overlap_tokens=args.overlap,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_leaf_catalog(args.output_dir / "leaf_catalog.jsonl", binding)
    _write_json(args.output_dir / "corpus_binding.json", corpus_summary(binding))
    print(json.dumps(corpus_summary(binding), ensure_ascii=False, indent=2))
    return 0


def _cmd_validate_queries(args: argparse.Namespace) -> int:
    binding = load_binding_for_cli(args)
    queries = load_query_set(args.query_set)
    validate_query_set(queries, binding)
    meta = write_query_meta(args.meta, args.query_set)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    freeze = assert_query_freeze_gate(args.query_set, args.meta)
    binding = load_binding_for_cli(args)
    queries = load_query_set(args.query_set)
    store = MilvusProfileStore(args.uri)
    summary = run_profiling(
        binding,
        queries,
        store,
        args.output_dir,
        keep_collections=args.keep_collections,
        repeat=not args.skip_repeat,
    )
    summary["query_set_sha256"] = freeze
    summary["corpus_source"] = (
        "leaf_catalog" if getattr(args, "leaf_catalog", None) else "a1_a3"
    )
    summary["milvus_uri"] = args.uri
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(
        {
            "recommendation": summary["recommendation"],
            "query_set_sha256": freeze,
            "corpus_source": summary["corpus_source"],
            "milvus_uri": args.uri,
            "output_dir": str(args.output_dir.resolve()),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def _add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Formal A0 output root containing <document_id>/document.md",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_A3_TOKENIZER,
        help="A3 tokenizer id used only to rebuild Leaves (not A5 encode)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
    )


def _add_binding_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--leaf-catalog",
        type=Path,
        default=None,
        help="Frozen leaf_catalog.jsonl. Skips A1+A3 and does not load Qwen.",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help="A0 output root. Rebuilds Leaves with A1+A3. Mutually exclusive "
        "with --leaf-catalog.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_A3_TOKENIZER,
        help="A3 tokenizer id used only with --canonical-root (not A5 encode)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_TOKENS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "A6.0 BM25 analyzer profiling. Temporary collections only. "
            "Does not implement official A6, Dense, Hybrid, or RRF. "
            "A2/A5 runtime dependency = NONE."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bind = sub.add_parser("bind-corpus", help="P1: A1+A3 language binding")
    _add_corpus_args(bind)
    bind.add_argument("--output-dir", type=Path, required=True)
    bind.set_defaults(func=_cmd_bind_corpus)

    validate = sub.add_parser(
        "validate-queries", help="P3: validate and freeze query_set.meta.json"
    )
    _add_binding_source_args(validate)
    validate.add_argument("--query-set", type=Path, required=True)
    validate.add_argument("--meta", type=Path, required=True)
    validate.set_defaults(func=_cmd_validate_queries)

    run = sub.add_parser("run", help="P4-P8: tokenize + 5 BM25 profiles")
    _add_binding_source_args(run)
    run.add_argument("--query-set", type=Path, required=True)
    run.add_argument("--meta", type=Path, required=True)
    run.add_argument("--uri", default="http://localhost:19530")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--keep-collections",
        action="store_true",
        help="Keep temporary profiling collections after the run",
    )
    run.add_argument(
        "--skip-repeat",
        action="store_true",
        help="Skip the second retrieval pass (debug only)",
    )
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (
        MarkdownLoadingError,
        A3ChunkingError,
        A6ProfileError,
        OSError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
