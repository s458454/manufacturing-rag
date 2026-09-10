"""A6.0 analyzer profiling unit tests. Ordinary pytest does not connect to Milvus."""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.a6_analyzer_profile import (  # noqa: E402
    A6ProfileError,
    PairwiseRecall,
    ProfileMetrics,
    ProfileQuery,
    ProfileStore,
    QueryResult,
    RankedHit,
    aggregate_metrics,
    analyzer_params_for,
    assert_only_analyzer_differs,
    assert_query_freeze_gate,
    assert_repeatable,
    assert_search_contract,
    bind_corpus,
    bm25_index_params,
    build_query_meta,
    choose_chn_specialist,
    cohit_rank_stats,
    gold_rank,
    icu_recall_acceptable,
    language_for_document_id,
    load_leaf_catalog,
    load_query_set,
    tokens_from_run_analyzer,
    pairwise_recall20,
    parse_query_record,
    profiling_schema_fields,
    recommend_outcome,
    reciprocal_rank,
    run_profiling,
    score_query,
    utf8_byte_length,
    validate_query_set,
    write_leaf_catalog,
    write_query_meta,
)
from knowledge_base.a6_profile_config import (  # noqa: E402
    ALL_PROFILES,
    ANALYZER_PARAMS,
    BM25_B,
    BM25_INDEX_TYPE,
    BM25_INVERTED_INDEX_ALGO,
    BM25_K1,
    BM25_METRIC_TYPE,
    CHN_CATEGORY_MINIMA,
    CHN_DOCUMENT_IDS,
    COLLECTION_FIELD_SPARSE,
    EN_CATEGORY_MINIMA,
    EN_DOCUMENT_IDS,
    OUTCOME_LANGUAGE_SPECIFIC,
    OUTCOME_UNIFIED_ICU,
    PROFILE_CHN_CHINESE,
    PROFILE_CHN_ICU,
    PROFILE_CHN_JIEBA,
    PROFILE_EN_ICU,
    PROFILE_EN_STANDARD,
    SEARCH_ANNS_FIELD,
    VARCHAR_CONTENT_MAX_BYTES,
)
from knowledge_base.leaf_chunker import Leaf  # noqa: E402


@pytest.fixture
def work_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as name:
        yield Path(name)


def _leaf(
    document_id: str,
    chunk_id: str,
    content: str = "leaf body",
    *,
    chunk_index: int = 0,
) -> Leaf:
    return Leaf(
        chunk_id=chunk_id,
        document_id=document_id,
        section_id=f"sec_{chunk_id}",
        chunk_index=chunk_index,
        page_start=1,
        page_end=1,
        content=content,
    )


def _tiny_binding() -> object:
    leaves = []
    for index, document_id in enumerate(sorted(EN_DOCUMENT_IDS)):
        leaves.append(
            _leaf(document_id, f"chk_en_{index}", f"english leaf {index} NASA-STD")
        )
    for index, document_id in enumerate(sorted(CHN_DOCUMENT_IDS)):
        leaves.append(
            _leaf(document_id, f"chk_zh_{index}", f"中文叶子 {index} 避雷器")
        )
    return bind_corpus(leaves, expected_leaf_count=17)


def _query(
    query_id: str,
    language: str,
    document_id: str,
    chunk_id: str,
    *,
    text: str | None = None,
    evidence: str | None = None,
    category: list[str] | None = None,
) -> ProfileQuery:
    if language == "en":
        cats = category or list(EN_CATEGORY_MINIMA)
        default_q = f"What is the {query_id} requirement in NASA-STD?"
        default_ev = f"The {query_id} requirement is stated in the standard."
    else:
        cats = category or list(CHN_CATEGORY_MINIMA)
        default_q = f"{query_id} 的额定电压要求是多少"
        default_ev = f"{query_id} 的额定电压要求写在技术条件书中。"
    return ProfileQuery(
        query_id=query_id,
        language=language,
        query=text or default_q,
        gold_document_id=document_id,
        gold_chunk_ids=(chunk_id,),
        evidence_text=evidence or default_ev,
        category=tuple(cats),
    )


def _valid_queries(binding) -> list[ProfileQuery]:
    en_docs = sorted(binding.en_document_ids)[:10]
    zh_docs = sorted(binding.chn_document_ids)
    en_chunks = {
        leaf.document_id: leaf.chunk_id
        for leaf in binding.en_leaves
        if leaf.document_id in set(en_docs)
    }
    zh_chunks = {
        leaf.document_id: leaf.chunk_id for leaf in binding.chn_leaves
    }
    queries: list[ProfileQuery] = []
    for index in range(30):
        document_id = en_docs[index % 10]
        queries.append(
            _query(
                f"EN{index:03d}",
                "en",
                document_id,
                en_chunks[document_id],
            )
        )
    for doc_index, document_id in enumerate(zh_docs):
        for offset in range(10):
            index = doc_index * 10 + offset
            queries.append(
                _query(
                    f"CHN{index:03d}",
                    "zh",
                    document_id,
                    zh_chunks[document_id],
                )
            )
    return queries


def _metrics(
    profile: str,
    ranks: dict[str, int | None],
    queries: list[ProfileQuery] | None = None,
) -> ProfileMetrics:
    if queries is None:
        queries = [
            _query(query_id, "en", "doc", "gold") for query_id in ranks
        ]
    results = []
    for query in queries:
        rank = ranks[query.query_id]
        top20 = ["gold"] + [f"other_{i}" for i in range(19)]
        if rank is not None:
            top20 = ["other"] * (rank - 1) + ["gold"] + ["other"] * 20
            top20 = top20[:20]
        else:
            top20 = [f"other_{i}" for i in range(20)]
        results.append(
            QueryResult(
                query_id=query.query_id,
                query=query.query,
                gold_chunk_ids=query.gold_chunk_ids,
                top20_chunk_ids=tuple(top20),
                gold_rank=rank,
                hit_at_5=rank is not None and rank <= 5,
                hit_at_20=rank is not None and rank <= 20,
                reciprocal_rank=0.0 if rank is None or rank > 20 else 1.0 / rank,
            )
        )
    n = len(results)
    return ProfileMetrics(
        profile=profile,
        query_count=n,
        recall_at_5=sum(item.hit_at_5 for item in results) / n,
        recall_at_20=sum(item.hit_at_20 for item in results) / n,
        mrr_at_20=sum(item.reciprocal_rank for item in results) / n,
        miss_count=sum(not item.hit_at_20 for item in results),
        miss_query_ids=tuple(
            item.query_id for item in results if not item.hit_at_20
        ),
        results=tuple(results),
    )


class FakeProfileStore(ProfileStore):
    def __init__(self, query_ranks: dict[str, list[str]] | None = None) -> None:
        self.query_ranks = query_ranks or {}
        self.created: list[str] = []
        self.dropped: list[str] = []
        self.analyzer_calls: list[tuple[str, dict]] = []
        self.search_calls: list[dict[str, object]] = []
        self.inserted: dict[str, int] = {}

    def run_analyzer(self, text: str, analyzer_params: dict) -> list[str]:
        self.analyzer_calls.append((text, analyzer_params))
        return [text]

    def create_profile_collection(self, name: str, analyzer_params: dict) -> None:
        self.created.append(name)

    def insert_leaves(self, name: str, leaves) -> None:
        self.inserted[name] = len(list(leaves))

    def search(self, name: str, query: str, *, limit: int = 20) -> list[RankedHit]:
        assert_search_contract(
            anns_field=SEARCH_ANNS_FIELD,
            schema_fields=profiling_schema_fields(),
        )
        self.search_calls.append(
            {
                "collection": name,
                "query": query,
                "anns_field": SEARCH_ANNS_FIELD,
                "limit": limit,
            }
        )
        ids = self.query_ranks.get(query, [])
        return [
            RankedHit(chunk_id=chunk_id, rank=index, score=1.0)
            for index, chunk_id in enumerate(ids[:limit], start=1)
        ]

    def drop_collection(self, name: str) -> None:
        self.dropped.append(name)

    def has_collection(self, name: str) -> bool:
        return name in self.created and name not in self.dropped


def test_p01_query_set_schema_validation() -> None:
    record = parse_query_record(
        {
            "query_id": "EN000",
            "language": "en",
            "query": "NASA-STD-4003A electrical bonding class",
            "gold_document_id": "doc",
            "gold_chunk_ids": ["chk_1"],
            "evidence_text": "Class S electrical bonding shall ...",
            "category": ["technical_term"],
        }
    )
    assert record.query_id == "EN000"
    with pytest.raises(A6ProfileError, match="full copy"):
        parse_query_record(
            {
                "query_id": "EN000",
                "language": "en",
                "query": "q",
                "gold_document_id": "doc",
                "gold_chunk_ids": ["chk_1"],
                "evidence_text": "q",
                "category": ["technical_term"],
            }
        )


def test_p02_duplicate_query_id_rejected() -> None:
    binding = _tiny_binding()
    queries = _valid_queries(binding)
    queries[1] = ProfileQuery(
        query_id=queries[0].query_id,
        language=queries[1].language,
        query=queries[1].query,
        gold_document_id=queries[1].gold_document_id,
        gold_chunk_ids=queries[1].gold_chunk_ids,
        evidence_text=queries[1].evidence_text,
        category=queries[1].category,
    )
    with pytest.raises(A6ProfileError, match="duplicate query_id"):
        validate_query_set(queries, binding)


def test_p03_wrong_language_query_rejected() -> None:
    binding = _tiny_binding()
    queries = _valid_queries(binding)
    zh_doc = sorted(binding.chn_document_ids)[0]
    zh_chunk = next(
        leaf.chunk_id
        for leaf in binding.chn_leaves
        if leaf.document_id == zh_doc
    )
    queries[0] = _query("EN000", "en", zh_doc, zh_chunk)
    with pytest.raises(A6ProfileError, match="not in the en corpus"):
        validate_query_set(queries, binding)


def test_p04_missing_gold_rejected() -> None:
    with pytest.raises(A6ProfileError, match="gold_chunk_ids"):
        parse_query_record(
            {
                "query_id": "EN000",
                "language": "en",
                "query": "what is the bonding class",
                "gold_document_id": "doc",
                "gold_chunk_ids": [],
                "evidence_text": "evidence",
                "category": ["technical_term"],
            }
        )


def test_p05_gold_chunk_not_in_target_corpus_rejected() -> None:
    binding = _tiny_binding()
    queries = _valid_queries(binding)
    en_doc = queries[0].gold_document_id
    queries[0] = ProfileQuery(
        query_id=queries[0].query_id,
        language="en",
        query=queries[0].query,
        gold_document_id=en_doc,
        gold_chunk_ids=("chk_missing",),
        evidence_text=queries[0].evidence_text,
        category=queries[0].category,
    )
    with pytest.raises(A6ProfileError, match="not in the en corpus"):
        validate_query_set(queries, binding)


def test_p06_recall_at_5() -> None:
    queries = [_query("EN000", "en", "doc", "gold")]
    ranked = [["gold"] + [f"x{i}" for i in range(19)]]
    metrics = aggregate_metrics("EN-STD", queries, ranked)
    assert metrics.recall_at_5 == 1.0
    ranked_miss = [[f"x{i}" for i in range(20)]]
    assert aggregate_metrics("EN-STD", queries, ranked_miss).recall_at_5 == 0.0


def test_p07_recall_at_20() -> None:
    queries = [_query("EN000", "en", "doc", "gold")]
    ranked = [[f"x{i}" for i in range(19)] + ["gold"]]
    metrics = aggregate_metrics("EN-STD", queries, ranked)
    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_20 == 1.0


def test_p08_mrr_at_20() -> None:
    queries = [
        _query("EN000", "en", "doc", "gold"),
        _query("EN001", "en", "doc", "gold"),
    ]
    ranked = [
        ["gold"] + [f"x{i}" for i in range(19)],
        [f"x{i}" for i in range(4)] + ["gold"] + [f"y{i}" for i in range(15)],
    ]
    metrics = aggregate_metrics("EN-STD", queries, ranked)
    assert metrics.mrr_at_20 == pytest.approx((1.0 + 0.2) / 2)


def test_p09_multi_gold_hit_semantics() -> None:
    query = ProfileQuery(
        query_id="EN000",
        language="en",
        query="bonding class requirement",
        gold_document_id="doc",
        gold_chunk_ids=("chk_a", "chk_b"),
        evidence_text="Class S bonding",
        category=("technical_term",),
    )
    result = score_query(query, ["noise", "chk_b"] + [f"x{i}" for i in range(18)])
    assert result.hit_at_5 is True
    assert result.gold_rank == 2


def test_p10_top20_miss_rr_zero() -> None:
    assert reciprocal_rank(None) == 0.0
    assert gold_rank(["gold"], [f"x{i}" for i in range(20)]) is None
    query = _query("EN000", "en", "doc", "gold")
    result = score_query(query, [f"x{i}" for i in range(20)])
    assert result.reciprocal_rank == 0.0
    assert result.hit_at_20 is False


def test_p11_analyzer_config_exactness() -> None:
    assert analyzer_params_for(PROFILE_EN_STANDARD) == {"type": "standard"}
    assert analyzer_params_for(PROFILE_EN_ICU) == {
        "tokenizer": "icu",
        "filter": ["lowercase", "removepunct"],
    }
    assert analyzer_params_for(PROFILE_CHN_CHINESE) == {"type": "chinese"}
    assert analyzer_params_for(PROFILE_CHN_JIEBA) == {
        "tokenizer": "jieba",
        "filter": ["removepunct"],
    }
    assert analyzer_params_for(PROFILE_CHN_ICU) == analyzer_params_for(
        PROFILE_EN_ICU
    )
    assert ANALYZER_PARAMS[PROFILE_CHN_JIEBA]["tokenizer"] == "jieba"


def test_p12_only_analyzer_differs_between_profiles() -> None:
    assert_only_analyzer_differs(ALL_PROFILES)
    index = bm25_index_params()
    assert index["index_type"] == BM25_INDEX_TYPE
    assert index["metric_type"] == BM25_METRIC_TYPE
    assert index["params"]["inverted_index_algo"] == BM25_INVERTED_INDEX_ALGO
    assert index["params"]["bm25_k1"] == BM25_K1
    assert index["params"]["bm25_b"] == BM25_B


def test_p13_dense_not_invoked() -> None:
    with pytest.raises(A6ProfileError, match="dense encoder"):
        assert_search_contract(
            anns_field=SEARCH_ANNS_FIELD,
            schema_fields=profiling_schema_fields(),
            dense_invoked=True,
        )


def test_p14_rrf_not_invoked() -> None:
    with pytest.raises(A6ProfileError, match="RRF"):
        assert_search_contract(
            anns_field=SEARCH_ANNS_FIELD,
            schema_fields=profiling_schema_fields(),
            rrf_invoked=True,
        )


def test_p15_reranker_not_invoked() -> None:
    with pytest.raises(A6ProfileError, match="reranker"):
        assert_search_contract(
            anns_field=SEARCH_ANNS_FIELD,
            schema_fields=profiling_schema_fields(),
            reranker_invoked=True,
        )
    with pytest.raises(A6ProfileError, match="dense_vector"):
        assert_search_contract(
            anns_field=SEARCH_ANNS_FIELD,
            schema_fields=(*profiling_schema_fields(), "dense_vector"),
        )
    with pytest.raises(A6ProfileError, match="anns_field"):
        assert_search_contract(
            anns_field="dense_vector",
            schema_fields=profiling_schema_fields(),
        )
    assert_search_contract(
        anns_field=SEARCH_ANNS_FIELD,
        schema_fields=profiling_schema_fields(),
    )
    assert "dense_vector" not in profiling_schema_fields()
    assert COLLECTION_FIELD_SPARSE in profiling_schema_fields()


def test_p16_temporary_collection_cleanup(work_dir: Path) -> None:
    binding = _tiny_binding()
    queries = _valid_queries(binding)
    ranks = {query.query: [query.gold_chunk_ids[0]] for query in queries}
    store = FakeProfileStore(ranks)
    run_profiling(
        binding,
        queries,
        store,
        work_dir,
        keep_collections=False,
        timestamp="t1",
        repeat=False,
    )
    assert store.created
    assert set(store.dropped) == set(store.created)
    keep_store = FakeProfileStore(ranks)
    run_profiling(
        binding,
        queries,
        keep_store,
        work_dir / "keep",
        keep_collections=True,
        timestamp="t2",
        repeat=False,
    )
    assert keep_store.created
    assert keep_store.dropped == []


def test_p17_repeated_run_summary_consistency() -> None:
    first = _metrics(PROFILE_EN_STANDARD, {"EN000": 2, "EN001": None})
    second = _metrics(PROFILE_EN_STANDARD, {"EN000": 2, "EN001": None})
    assert_repeatable(first, second)
    drifted = _metrics(PROFILE_EN_STANDARD, {"EN000": 3, "EN001": None})
    with pytest.raises(A6ProfileError, match="gold_rank mismatch"):
        assert_repeatable(first, drifted)


def test_utf8_byte_length_not_python_len() -> None:
    assert len("避雷器") == 3
    assert utf8_byte_length("避雷器") == 9


def test_oversize_utf8_content_rejected() -> None:
    leaves = []
    for index, document_id in enumerate(sorted(EN_DOCUMENT_IDS)):
        content = "a" if index else "x" * (VARCHAR_CONTENT_MAX_BYTES + 1)
        leaves.append(_leaf(document_id, f"chk_en_{index}", content))
    for index, document_id in enumerate(sorted(CHN_DOCUMENT_IDS)):
        leaves.append(_leaf(document_id, f"chk_zh_{index}", "中文"))
    with pytest.raises(A6ProfileError, match="UTF-8 bytes"):
        bind_corpus(leaves, expected_leaf_count=17)


def test_jsonl_comments_rejected(work_dir: Path) -> None:
    path = work_dir / "query_set.jsonl"
    path.write_text("# A6.0 profiling-only\n{}\n", encoding="utf-8")
    with pytest.raises(A6ProfileError, match="comments are forbidden"):
        load_query_set(path)


def test_query_freeze_gate(work_dir: Path) -> None:
    query_path = work_dir / "query_set.jsonl"
    query_path.write_text(
        json.dumps(
            {
                "query_id": "EN000",
                "language": "en",
                "query": "NASA-STD-4003A class",
                "gold_document_id": "doc",
                "gold_chunk_ids": ["chk"],
                "evidence_text": "Class S",
                "category": ["technical_term"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    meta_path = work_dir / "query_set.meta.json"
    write_query_meta(meta_path, query_path)
    digest = assert_query_freeze_gate(query_path, meta_path)
    assert digest == build_query_meta(query_path)["query_set_sha256"]
    query_path.write_text(query_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(A6ProfileError, match="SHA256"):
        assert_query_freeze_gate(query_path, meta_path)


def test_language_binding_uses_document_id_not_sha256() -> None:
    binding = _tiny_binding()
    assert len(binding.en_document_ids) == 14
    assert len(binding.chn_document_ids) == 3
    assert language_for_document_id(sorted(EN_DOCUMENT_IDS)[0]) == "en"
    assert language_for_document_id(sorted(CHN_DOCUMENT_IDS)[0]) == "zh"


def test_cohit_rank_and_recommendation() -> None:
    queries = [_query(f"EN{i:03d}", "en", "doc", "gold") for i in range(3)]
    std = _metrics(
        PROFILE_EN_STANDARD,
        {"EN000": 2, "EN001": 4, "EN002": 3},
        queries,
    )
    icu = _metrics(
        PROFILE_EN_ICU,
        {"EN000": 1, "EN001": 4, "EN002": 5},
        queries,
    )
    stats = cohit_rank_stats(std, icu)
    assert stats.cohit_count == 3
    assert stats.icu_rank_better_count == 1
    assert stats.icu_rank_equal_count == 1
    assert stats.icu_rank_worse_count == 1
    assert stats.median_rank_delta == 0.0
    pair = pairwise_recall20(std, icu)
    assert pair == PairwiseRecall(both_hit=3, left_only=0, right_only=0, both_miss=0)
    specialist = _metrics(PROFILE_CHN_JIEBA, {"CHN000": 1}, [_query("CHN000", "zh", "d", "g")])
    chn_icu = _metrics(PROFILE_CHN_ICU, {"CHN000": 1}, [_query("CHN000", "zh", "d", "g")])
    payload = recommend_outcome(std, icu, specialist, chn_icu)
    assert payload["recommendation"] == OUTCOME_UNIFIED_ICU
    worse_icu = _metrics(
        PROFILE_EN_ICU,
        {"EN000": None, "EN001": None, "EN002": 3},
        queries,
    )
    assert icu_recall_acceptable(std, worse_icu) is False
    payload = recommend_outcome(std, worse_icu, specialist, chn_icu)
    assert payload["recommendation"] == OUTCOME_LANGUAGE_SPECIFIC


def test_choose_chn_specialist_prefers_better_recall() -> None:
    chinese = _metrics(PROFILE_CHN_CHINESE, {"CHN000": 1, "CHN001": None})
    jieba = _metrics(PROFILE_CHN_JIEBA, {"CHN000": 1, "CHN001": 2})
    assert choose_chn_specialist(chinese, jieba).profile == PROFILE_CHN_JIEBA


def test_valid_query_set_passes() -> None:
    binding = _tiny_binding()
    validate_query_set(_valid_queries(binding), binding)


def test_run_profiling_writes_summary(work_dir: Path) -> None:
    binding = _tiny_binding()
    queries = _valid_queries(binding)
    ranks = {query.query: [query.gold_chunk_ids[0]] for query in queries}
    store = FakeProfileStore(ranks)
    summary = run_profiling(
        binding,
        queries,
        store,
        work_dir,
        keep_collections=False,
        timestamp="s1",
        repeat=True,
    )
    assert summary["recommendation"] in {
        OUTCOME_UNIFIED_ICU,
        OUTCOME_LANGUAGE_SPECIFIC,
    }
    assert summary["a5_runtime_dependency"] == "NONE"
    assert summary["english_query_count"] == 30
    assert (work_dir / "summary.json").is_file()
    assert (work_dir / "tokenization_zh.json").is_file()
    assert store.search_calls
    assert all(
        call["anns_field"] == SEARCH_ANNS_FIELD for call in store.search_calls
    )
    assert "YH5WR-17/45" in (work_dir / "tokenization_zh.json").read_text(
        encoding="utf-8"
    )


def test_leaf_catalog_roundtrip_does_not_need_tokenizer(work_dir: Path) -> None:
    binding = _tiny_binding()
    path = work_dir / "leaf_catalog.jsonl"
    write_leaf_catalog(path, binding)
    loaded = load_leaf_catalog(path, expected_leaf_count=17)
    assert len(loaded.leaves) == 17
    assert {leaf.chunk_id for leaf in loaded.leaves} == {
        leaf.chunk_id for leaf in binding.leaves
    }
    assert loaded.en_document_ids == binding.en_document_ids
    assert loaded.chn_document_ids == binding.chn_document_ids
    assert all(leaf.section_id == "catalog" for leaf in loaded.leaves)


def test_tokens_from_run_analyzer_accepts_analyze_result() -> None:
    class AnalyzeResult:
        def __init__(self, tokens: list) -> None:
            self.tokens = tokens

    assert tokens_from_run_analyzer(AnalyzeResult(["NASA", "STD"])) == [
        "NASA",
        "STD",
    ]
    assert tokens_from_run_analyzer([{"token": "icu"}, {"token": "tokenizer"}]) == [
        "icu",
        "tokenizer",
    ]
    assert tokens_from_run_analyzer(["plain", "list"]) == ["plain", "list"]
    with pytest.raises(A6ProfileError, match="unexpected type"):
        tokens_from_run_analyzer("not-a-payload")


def test_leaf_catalog_language_mismatch_rejected(work_dir: Path) -> None:
    binding = _tiny_binding()
    path = work_dir / "leaf_catalog.jsonl"
    write_leaf_catalog(path, binding)
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    first["language"] = "zh" if first["language"] == "en" else "en"
    path.write_text(json.dumps(first, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(A6ProfileError, match="does not match"):
        load_leaf_catalog(path, expected_leaf_count=1)
