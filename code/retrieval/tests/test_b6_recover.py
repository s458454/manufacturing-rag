"""B6 recovery: budget, windows, coverage union, and direct parent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.section_hierarchy import SectionNode  # noqa: E402
from retrieval.b5.types import RerankedCandidate, RerankingResult  # noqa: E402
from retrieval.b6.catalog import CatalogLeaf, LeafCatalog, load_leaf_catalog  # noqa: E402
from retrieval.b6.constants import (  # noqa: E402
    GENERATION_TOKENIZER_REVISION,
    NOMINAL_EVIDENCE_BUDGET,
    NOMINAL_EVIDENCE_BUDGET_RATIO,
    GENERATION_CONTEXT_LIMIT,
)
from retrieval.b6.errors import (  # noqa: E402
    CoreLeafExceedsBudgetError,
    RecoveryIdentityError,
    RecoveryInputError,
)
from retrieval.b6.recover import recover, union_source_spans  # noqa: E402
from retrieval.b6.types import RecoveredEvidence  # noqa: E402


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [1] * len(text)


class SpanStore:
    def __init__(
        self,
        nodes: list[SectionNode],
        overrides: dict[str, str] | None = None,
    ) -> None:
        self._nodes = {node.section_id: node for node in nodes}
        self._overrides = overrides or {}

    def get_section(self, section_id: str) -> SectionNode:
        return self._nodes[section_id]

    def recover_span(self, document_id: str, start: int, end: int) -> str:
        return "x" * (end - start)

    def recover_section(self, section_id: str) -> str:
        if section_id in self._overrides:
            return self._overrides[section_id]
        node = self.get_section(section_id)
        return self.recover_span(node.document_id, node.source_start, node.source_end)

    def section_path(self, section_id: str) -> tuple[str, ...]:
        chain: list[SectionNode] = []
        seen: set[str] = set()
        current = self.get_section(section_id)
        while current.section_id not in seen:
            seen.add(current.section_id)
            chain.append(current)
            if current.parent_section_id is None:
                break
            current = self.get_section(current.parent_section_id)
        chain.reverse()
        return tuple(node.heading for node in chain if node.heading)


def _node(
    section_id: str,
    *,
    parent: str | None,
    start: int,
    end: int,
    heading: str | None,
    level: int | None = 2,
    page_start: int = 1,
    page_end: int = 1,
    document_id: str = "doc",
    kind: str = "heading",
) -> SectionNode:
    return SectionNode(
        section_id=section_id,
        document_id=document_id,
        parent_section_id=parent,
        kind=kind,
        heading_level=level,
        heading=heading,
        source_start=start,
        source_end=end,
        page_start=page_start,
        page_end=page_end,
    )


def _leaf(
    chunk_id: str,
    section_id: str,
    index: int,
    content: str,
    *,
    page_start: int = 1,
    page_end: int = 1,
    document_id: str = "doc",
) -> CatalogLeaf:
    return CatalogLeaf(
        chunk_id=chunk_id,
        document_id=document_id,
        section_id=section_id,
        chunk_index=index,
        page_start=page_start,
        page_end=page_end,
        content=content,
    )


def _candidate(leaf: CatalogLeaf, rank: int, *, content: str | None = None) -> RerankedCandidate:
    return RerankedCandidate(
        chunk_id=leaf.chunk_id,
        document_id=leaf.document_id,
        section_id=leaf.section_id,
        chunk_index=leaf.chunk_index,
        page_start=leaf.page_start,
        page_end=leaf.page_end,
        content=leaf.content if content is None else content,
        rrf_score=0.1,
        fused_rank=rank,
        rerank_score=0.9,
        rerank_rank=rank,
    )


def _recover(
    nodes: list[SectionNode],
    leaves: list[CatalogLeaf],
    ranked: list[tuple[CatalogLeaf, int]],
    budget: int,
    overrides: dict[str, str] | None = None,
    extra_candidates: list[RerankedCandidate] | None = None,
):
    catalog = LeafCatalog(tuple(leaves))
    candidates = [_candidate(leaf, rank) for leaf, rank in ranked]
    if extra_candidates:
        candidates.extend(extra_candidates)
    result = recover(
        RerankingResult("req", "task", "query", tuple(candidates)),
        SpanStore(nodes, overrides),
        catalog,
        CharTokenizer(),
        budget=budget,
    )
    return result


def _modes(result) -> list[str]:
    return [item.recovery_mode for item in result.evidence]


def test_union_source_spans_keeps_gaps_and_merges_overlaps() -> None:
    assert union_source_spans([(0, 10), (20, 30)]) == ((0, 10), (20, 30))
    assert union_source_spans([(20, 30), (0, 10)]) == ((0, 10), (20, 30))
    assert union_source_spans([(0, 10), (5, 15)]) == ((0, 15),)
    assert union_source_spans([(0, 10), (10, 20)]) == ((0, 20),)


def test_nominal_budget_is_three_quarters_of_the_context_limit() -> None:
    assert NOMINAL_EVIDENCE_BUDGET_RATIO == 0.75
    assert NOMINAL_EVIDENCE_BUDGET == int(GENERATION_CONTEXT_LIMIT * 0.75)
    assert NOMINAL_EVIDENCE_BUDGET == 24576


def test_tokenizer_revision_matches_the_manifest() -> None:
    manifest = json.loads(
        (CODE_ROOT.parent / "models" / "retrieval-models.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    pinned = next(
        asset["revision"]
        for asset in manifest["assets"]
        if asset["logical_id"] == "Qwen/Qwen2.5-7B-Instruct"
    )
    assert GENERATION_TOKENIZER_REVISION == pinned
    assert pinned == "a09a35458c702b33eeacc393d103063234e8bc28"


def test_top1_over_budget_fails_without_truncation() -> None:
    body = "Z" * 100
    leaf = _leaf("c1", "s", 0, body)
    node = _node("s", parent=None, start=0, end=10, heading="S")
    with pytest.raises(CoreLeafExceedsBudgetError, match="token_count=100") as caught:
        _recover([node], [leaf], [(leaf, 1)], budget=40)
    assert "truncat" not in str(caught.value).lower()
    assert body == leaf.content


def test_dropping_lower_ranks_still_fails_when_top1_alone_exceeds_budget() -> None:
    top = _leaf("top", "s1", 0, "Z" * 80)
    low = _leaf("low", "s2", 0, "a" * 10)
    nodes = [
        _node("s1", parent=None, start=0, end=10, heading="A"),
        _node("s2", parent=None, start=0, end=10, heading="B"),
    ]
    with pytest.raises(CoreLeafExceedsBudgetError):
        _recover(nodes, [top, low], [(top, 1), (low, 5)], budget=50)


def test_core_admission_drops_worst_rank_first() -> None:
    leaves = [
        _leaf("r1", "s1", 0, "a" * 100, page_start=9),
        _leaf("r2", "s2", 0, "b" * 100, page_start=2),
        _leaf("r5", "s3", 0, "c" * 100, page_start=1),
    ]
    nodes = [
        _node(leaf.section_id, parent=None, start=0, end=10000, heading=leaf.section_id)
        for leaf in leaves
    ]
    result = _recover(
        nodes,
        leaves,
        [(leaves[0], 1), (leaves[1], 2), (leaves[2], 5)],
        budget=250,
    )
    assert [item.evidence_priority for item in result.evidence] == [1, 2]
    assert all(item.recovery_mode == "leaf" for item in result.evidence)
    assert "r5" not in {item.matched_chunk_ids[0] for item in result.evidence}


def test_same_section_is_recovered_once_at_best_rank() -> None:
    first = _leaf("c1", "s", 0, "a" * 10, page_start=4)
    second = _leaf("c4", "s", 3, "b" * 10, page_start=1)
    node = _node("s", parent=None, start=0, end=80, heading="Section", page_start=1, page_end=4)
    result = _recover([node], [first, second], [(first, 1), (second, 4)], budget=1000)
    assert len(result.evidence) == 1
    item = result.evidence[0]
    assert item.recovery_mode == "section"
    assert item.evidence_priority == 1
    assert item.matched_chunk_ids == ("c1", "c4")
    assert item.context_chunk_ids == ()
    assert item.supporting_rerank_ranks == (1, 4)
    assert item.section_path == ("Section",)


def test_overlapping_windows_are_priced_once() -> None:
    leaves = [
        _leaf(f"c{index}", "s", index, "q" * 100) for index in (9, 10, 11, 12, 13)
    ]
    node = _node("s", parent=None, start=0, end=10000, heading="S")
    matched = [leaves[1], leaves[3]]
    result = _recover(
        [node],
        leaves,
        [(matched[0], 1), (matched[1], 3)],
        budget=500,
    )
    assert len(result.evidence) == 1
    item = result.evidence[0]
    assert item.recovery_mode == "leaf_window"
    assert item.context_chunk_ids == ("c9", "c11", "c13")
    assert item.matched_chunk_ids == ("c10", "c12")
    assert item.token_count == 500
    assert item.content == "q" * 500
    assert item.content.count("q") == 500


def test_gapped_windows_in_one_section_both_remain() -> None:
    indexes = (9, 10, 11, 24, 25, 26)
    leaves = [_leaf(f"c{index}", "s", index, "z" * 100) for index in indexes]
    node = _node("s", parent=None, start=0, end=100000, heading="S")
    result = _recover(
        [node],
        leaves,
        [(leaves[1], 1), (leaves[4], 4)],
        budget=700,
    )
    assert _modes(result) == ["leaf_window", "leaf_window"]
    assert [item.section_id for item in result.evidence] == ["s", "s"]
    assert result.evidence[0].matched_chunk_ids == ("c10",)
    assert result.evidence[1].matched_chunk_ids == ("c25",)
    ordered = [item.context_chunk_ids for item in result.evidence]
    assert ordered == [("c9", "c11"), ("c24", "c26")]
    flat = [chunk_id for item in result.evidence for chunk_id in item.matched_chunk_ids]
    flat += [chunk_id for item in result.evidence for chunk_id in item.context_chunk_ids]
    assert "c12" not in flat
    assert "c23" not in flat


def test_merged_window_does_not_fall_back_to_a_smaller_window() -> None:
    leaves = [_leaf(f"c{index}", "s", index, "q" * 100) for index in (9, 10, 11, 12, 13)]
    node = _node("s", parent=None, start=0, end=10000, heading="S")
    result = _recover(
        [node],
        leaves,
        [(leaves[1], 1), (leaves[3], 3)],
        budget=350,
    )
    assert _modes(result) == ["leaf", "leaf"]
    assert all(item.context_chunk_ids == () for item in result.evidence)
    assert {item.matched_chunk_ids for item in result.evidence} == {("c10",), ("c12",)}


def test_natural_boundary_keeps_a_one_sided_window() -> None:
    first = _leaf("c0", "s", 0, "a" * 100)
    second = _leaf("c1", "s", 1, "b" * 100)
    node = _node("s", parent=None, start=0, end=10000, heading="S")
    start = _recover([node], [first, second], [(first, 1)], budget=250)
    assert start.evidence[0].recovery_mode == "leaf_window"
    assert start.evidence[0].matched_chunk_ids == ("c0",)
    assert start.evidence[0].context_chunk_ids == ("c1",)
    end = _recover([node], [first, second], [(second, 1)], budget=250)
    assert end.evidence[0].context_chunk_ids == ("c0",)
    assert end.evidence[0].matched_chunk_ids == ("c1",)


def test_full_window_does_not_drop_one_existing_neighbor() -> None:
    leaves = [_leaf(f"c{index}", "s", index, "a" * 100) for index in (0, 1, 2)]
    node = _node("s", parent=None, start=0, end=10000, heading="S")
    result = _recover([node], leaves, [(leaves[1], 1)], budget=150)
    item = result.evidence[0]
    assert item.recovery_mode == "leaf"
    assert item.matched_chunk_ids == ("c1",)
    assert item.context_chunk_ids == ()
    assert item.content == "a" * 100


def test_neighbor_does_not_cross_sections() -> None:
    matched = _leaf("a5", "sec-a", 5, "A" * 50)
    previous = _leaf("b4", "sec-b", 4, "B" * 50)
    nxt = _leaf("b6", "sec-b", 6, "C" * 50)
    nodes = [
        _node("sec-a", parent=None, start=0, end=5000, heading="A"),
        _node("sec-b", parent=None, start=0, end=5000, heading="B"),
    ]
    result = _recover(nodes, [matched, previous, nxt], [(matched, 1)], budget=80)
    item = result.evidence[0]
    assert item.recovery_mode == "leaf"
    assert item.context_chunk_ids == ()
    assert item.content == "A" * 50


def test_overlap_text_is_kept_in_both_chunks() -> None:
    first = _leaf("c0", "s", 0, "HELLO")
    second = _leaf("c1", "s", 1, "HELLO")
    node = _node("s", parent=None, start=0, end=1000, heading="S")
    result = _recover([node], [first, second], [(first, 1)], budget=20)
    assert result.evidence[0].content == "HELLOHELLO"


def test_unmatched_middle_child_is_excluded_from_coverage() -> None:
    nodes = [
        _node("P", parent=None, start=0, end=100, heading="Parent", level=2),
        _node("A", parent="P", start=0, end=10, heading="A", level=3),
        _node("B", parent="P", start=10, end=90, heading="B", level=3),
        _node("C", parent="P", start=90, end=100, heading="C", level=3),
    ]
    first = _leaf("a", "A", 0, "a" * 5)
    second = _leaf("c", "C", 1, "c" * 5)
    result = _recover(nodes, [first, second], [(first, 1), (second, 2)], budget=10000)
    assert _modes(result) == ["section", "section"]
    assert [len(item.content) for item in result.evidence] == [10, 10]


def test_parent_merges_when_true_union_covers_half() -> None:
    nodes = [
        _node("P", parent=None, start=0, end=100, heading="Parent", level=2),
        _node("A", parent="P", start=0, end=30, heading="A", level=3),
        _node("B", parent="P", start=30, end=40, heading="B", level=3),
        _node("C", parent="P", start=40, end=100, heading="C", level=3),
    ]
    first = _leaf("a", "A", 0, "a")
    second = _leaf("c", "C", 1, "c")
    result = _recover(nodes, [first, second], [(first, 1), (second, 3)], budget=100)
    assert len(result.evidence) == 1
    item = result.evidence[0]
    assert item.recovery_mode == "parent_section"
    assert item.section_id == "P"
    assert item.token_count == 100
    assert item.matched_chunk_ids == ("a", "c")
    assert item.context_chunk_ids == ()
    assert item.evidence_priority == 1
    assert "rerank_score" not in RecoveredEvidence.__dataclass_fields__


def test_pass1_finishes_before_a_large_parent() -> None:
    nodes = [
        _node("P", parent=None, start=0, end=4000, heading="Parent"),
        _node("A", parent="P", start=0, end=1500, heading="A", page_start=9),
        _node("B", parent="P", start=1500, end=3000, heading="B"),
        _node("C", parent=None, start=0, end=2000, heading="Other", page_start=1),
    ]
    first = _leaf("a", "A", 0, "a" * 100, page_start=9)
    second = _leaf("b", "B", 1, "b" * 100)
    third = _leaf("c", "C", 2, "c" * 100, page_start=1)
    result = _recover(
        nodes,
        [first, second, third],
        [(first, 1), (second, 2), (third, 3)],
        budget=5000,
    )
    assert _modes(result) == ["section", "section", "section"]
    assert [item.section_id for item in result.evidence] == ["A", "B", "C"]
    assert result.evidence[0].page_start == 9
    assert result.evidence[-1].page_start == 1


def test_parent_candidate_survives_when_one_child_stays_a_window() -> None:
    nodes = [
        _node("P", parent=None, start=0, end=8000, heading="Parent"),
        _node("A", parent="P", start=0, end=1000, heading="A"),
        _node("B", parent="P", start=2000, end=7000, heading="B"),
    ]
    first = _leaf("a", "A", 0, "a" * 100)
    matched_b = _leaf("b", "B", 6, "b" * 100)
    neighbor = _leaf("bn", "B", 5, "n" * 200)
    result = _recover(
        nodes,
        [first, matched_b, neighbor],
        [(first, 1), (matched_b, 2)],
        budget=3200,
        overrides={"P": "P" * 2200},
    )
    assert len(result.evidence) == 1
    item = result.evidence[0]
    assert item.recovery_mode == "parent_section"
    assert item.section_id == "P"
    assert item.matched_chunk_ids == ("a", "b")
    assert item.context_chunk_ids == ()
    assert item.token_count == 2200


def test_direct_parent_stops_at_h2_when_h4_skips_h3() -> None:
    nodes = [
        _node("H1", parent=None, start=0, end=200, heading="H1", level=1),
        _node("H2", parent="H1", start=0, end=100, heading="H2", level=2),
        _node("H4a", parent="H2", start=0, end=40, heading="H4a", level=4),
        _node("H4b", parent="H2", start=40, end=80, heading="H4b", level=4),
    ]
    first = _leaf("a", "H4a", 0, "a")
    second = _leaf("b", "H4b", 1, "b")
    result = _recover(nodes, [first, second], [(first, 1), (second, 2)], budget=500)
    assert result.evidence[0].recovery_mode == "parent_section"
    assert result.evidence[0].section_id == "H2"
    assert result.evidence[0].section_path == ("H1", "H2")
    assert all(item.section_id != "H1" for item in result.evidence)


def test_missing_parent_pointer_does_not_invent_document_root() -> None:
    nodes = [
        _node("root", parent=None, start=0, end=5, heading=None, level=None, kind="document_root"),
        _node("A", parent=None, start=5, end=20, heading="A"),
        _node("B", parent=None, start=20, end=40, heading="B"),
    ]
    first = _leaf("a", "A", 0, "a")
    second = _leaf("b", "B", 1, "b")
    result = _recover(nodes, [first, second], [(first, 1), (second, 2)], budget=1000)
    assert "parent_section" not in _modes(result)


def test_document_root_is_used_only_when_it_is_the_stored_parent() -> None:
    nodes = [
        _node("root", parent=None, start=0, end=100, heading=None, level=None, kind="document_root"),
        _node("A", parent="root", start=0, end=40, heading="A"),
        _node("B", parent="root", start=40, end=80, heading="B"),
    ]
    first = _leaf("a", "A", 0, "a")
    second = _leaf("b", "B", 1, "b")
    result = _recover(nodes, [first, second], [(first, 1), (second, 2)], budget=1000)
    assert result.evidence[0].recovery_mode == "parent_section"
    assert result.evidence[0].section_id == "root"


def test_large_parent_is_allowed_inside_the_nominal_ceiling() -> None:
    nodes = [
        _node("P", parent=None, start=0, end=14000, heading="Parent"),
        _node("A", parent="P", start=0, end=8000, heading="A"),
        _node("B", parent="P", start=8000, end=14000, heading="B"),
    ]
    first = _leaf("a", "A", 0, "a" * 10)
    second = _leaf("b", "B", 1, "b" * 10)
    result = _recover(
        nodes,
        [first, second],
        [(first, 1), (second, 2)],
        budget=NOMINAL_EVIDENCE_BUDGET,
    )
    assert result.evidence[0].recovery_mode == "parent_section"
    assert result.evidence[0].token_count == 14000
    assert result.evidence_token_count == 14000


def test_output_order_follows_rerank_rank_not_page() -> None:
    late_page = _leaf("top", "A", 0, "a", page_start=9, page_end=9)
    early_page = _leaf("low", "B", 0, "b", page_start=1, page_end=1)
    nodes = [
        _node("A", parent=None, start=0, end=20, heading="A", page_start=9, page_end=9),
        _node("B", parent=None, start=0, end=20, heading="B", page_start=1, page_end=1),
    ]
    result = _recover(
        nodes,
        [early_page, late_page],
        [(late_page, 1), (early_page, 3)],
        budget=100,
    )
    assert [item.evidence_priority for item in result.evidence] == [1, 3]
    assert result.evidence[0].page_start == 9
    assert result.evidence[1].page_start == 1


def test_more_than_five_candidates_fail() -> None:
    leaves = [_leaf(f"c{index}", f"s{index}", 0, "a") for index in range(6)]
    nodes = [
        _node(leaf.section_id, parent=None, start=0, end=1, heading=leaf.section_id)
        for leaf in leaves
    ]
    with pytest.raises(RecoveryInputError):
        _recover(nodes, leaves, [(leaf, index + 1) for index, leaf in enumerate(leaves)], budget=100)


def test_empty_candidates_return_no_evidence() -> None:
    result = recover(
        RerankingResult("req", "task", "query", ()),
        SpanStore([]),
        LeafCatalog(()),
        CharTokenizer(),
    )
    assert result.evidence == ()
    assert result.evidence_token_count == 0
    assert result.evidence_budget == NOMINAL_EVIDENCE_BUDGET


def test_missing_catalog_chunk_fails_the_identity_guard() -> None:
    leaf = _leaf("missing", "s", 0, "body")
    node = _node("s", parent=None, start=0, end=10, heading="S")
    catalog = LeafCatalog(())
    with pytest.raises(RecoveryIdentityError, match="missing"):
        recover(
            RerankingResult("req", "task", "query", (_candidate(leaf, 1),)),
            SpanStore([node]),
            catalog,
            CharTokenizer(),
            budget=100,
        )


def test_catalog_identity_mismatch_fails_without_claiming_same_build() -> None:
    stored = _leaf("c1", "other", 0, "body")
    incoming = _leaf("c1", "s", 0, "body")
    node = _node("s", parent=None, start=0, end=10, heading="S")
    with pytest.raises(RecoveryIdentityError, match="identity guard"):
        _recover([node], [stored], [(incoming, 1)], budget=100)


def test_changed_leaf_text_uses_catalog_content() -> None:
    stored = _leaf("c1", "s", 0, "NEW")
    node = _node("s", parent=None, start=0, end=1000, heading="S")
    catalog = LeafCatalog((stored,))
    candidate = _candidate(stored, 1, content="OLD")
    result = recover(
        RerankingResult("req", "task", "query", (candidate,)),
        SpanStore([node]),
        catalog,
        CharTokenizer(),
        budget=3,
    )
    assert result.evidence[0].recovery_mode == "leaf"
    assert result.evidence[0].content == "NEW"


def test_rows_jsonl_projection_drops_dense_vector(tmp_path: Path) -> None:
    path = tmp_path / "a6_rows.jsonl"
    path.write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "document_id": "doc",
                "section_id": "s",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 2,
                "content": "body",
                "dense_vector": [0.1, 0.2],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = load_leaf_catalog(path)
    leaf = catalog.get("c1")
    assert leaf.content == "body"
    assert not hasattr(leaf, "dense_vector")
