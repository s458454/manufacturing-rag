"""Hierarchical context recovery. Pure function: no disk and no model loads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from knowledge_base.section_hierarchy import SectionNode
from retrieval.b5.types import RerankedCandidate, RerankingResult
from retrieval.b6.catalog import CatalogLeaf, LeafCatalog
from retrieval.b6.constants import (
    MAX_RERANK_CANDIDATES,
    MODE_LEAF,
    MODE_LEAF_WINDOW,
    MODE_PARENT,
    MODE_SECTION,
    NOMINAL_EVIDENCE_BUDGET,
)
from retrieval.b6.errors import (
    CoreLeafExceedsBudgetError,
    LeafCatalogError,
    RecoveryConfigurationError,
    RecoveryError,
    RecoveryIdentityError,
    RecoveryInputError,
)
from retrieval.b6.tokens import count_tokens
from retrieval.b6.types import RecoveredEvidence, RecoveryResult


class SectionSource(Protocol):
    def get_section(self, section_id: str) -> SectionNode: ...

    def recover_section(self, section_id: str) -> str: ...

    def recover_span(self, document_id: str, start: int, end: int) -> str: ...

    def section_path(self, section_id: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class _Admitted:
    candidate: RerankedCandidate
    leaf: CatalogLeaf
    token_count: int


@dataclass
class _Piece:
    mode: str
    document_id: str
    section_id: str
    anchor_section_id: str
    content: str
    token_count: int
    page_start: int
    page_end: int
    priority: int
    matched_chunk_ids: tuple[str, ...]
    context_chunk_ids: tuple[str, ...]
    supporting_ranks: tuple[int, ...]
    ordered_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Window:
    chunks: tuple[CatalogLeaf, ...]
    matched: tuple[_Admitted, ...]


def union_source_spans(
    spans: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """True half-open union. A gap stays split; a bounding span is not used."""

    ordered = sorted(spans)
    if not ordered:
        return ()
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def recover(
    result: RerankingResult,
    store: SectionSource,
    catalog: LeafCatalog,
    tokenizer: Any,
    *,
    budget: int = NOMINAL_EVIDENCE_BUDGET,
) -> RecoveryResult:
    if budget < 0:
        raise RecoveryConfigurationError("evidence budget must be >= 0")
    if len(result.candidates) > MAX_RERANK_CANDIDATES:
        raise RecoveryInputError(
            f"B6 accepts at most {MAX_RERANK_CANDIDATES} candidates, "
            f"got {len(result.candidates)}"
        )
    if not result.candidates:
        return RecoveryResult(evidence=(), evidence_budget=budget, evidence_token_count=0)

    admitted = _admit(_guarded(result.candidates, catalog, tokenizer), budget)
    pieces = [_leaf_piece(item) for item in admitted]
    pieces = _pass1(pieces, admitted, store, catalog, tokenizer, budget)
    pieces = _pass2(pieces, admitted, store, tokenizer, budget)
    evidence = _finalize(pieces, store)
    return RecoveryResult(
        evidence=evidence,
        evidence_budget=budget,
        evidence_token_count=sum(item.token_count for item in evidence),
    )


def _guarded(
    candidates: tuple[RerankedCandidate, ...],
    catalog: LeafCatalog,
    tokenizer: Any,
) -> list[_Admitted]:
    chosen: dict[str, RerankedCandidate] = {}
    order: list[str] = []
    for candidate in candidates:
        previous = chosen.get(candidate.chunk_id)
        if previous is None:
            chosen[candidate.chunk_id] = candidate
            order.append(candidate.chunk_id)
        elif candidate.rerank_rank < previous.rerank_rank:
            chosen[candidate.chunk_id] = candidate
    admitted: list[_Admitted] = []
    for chunk_id in order:
        candidate = chosen[chunk_id]
        try:
            leaf = catalog.get(chunk_id)
        except LeafCatalogError as exc:
            raise RecoveryIdentityError(
                f"matched-Leaf runtime identity guard failed for chunk_id={chunk_id}"
            ) from exc
        if (
            leaf.document_id != candidate.document_id
            or leaf.section_id != candidate.section_id
            or leaf.chunk_index != candidate.chunk_index
        ):
            raise RecoveryIdentityError(
                "matched-Leaf runtime identity guard failed for "
                f"chunk_id={chunk_id}: catalog identity does not match the candidate"
            )
        admitted.append(
            _Admitted(
                candidate=candidate,
                leaf=leaf,
                token_count=count_tokens(tokenizer, leaf.content),
            )
        )
    return admitted


def _admit(items: list[_Admitted], budget: int) -> list[_Admitted]:
    admitted = list(items)

    def worst(item: _Admitted) -> tuple[int, str]:
        return (item.candidate.rerank_rank, item.leaf.chunk_id)

    while sum(item.token_count for item in admitted) > budget:
        if len(admitted) == 1:
            raise CoreLeafExceedsBudgetError(
                "highest-priority leaf "
                f"token_count={admitted[0].token_count} exceeds evidence budget {budget}"
            )
        admitted.remove(max(admitted, key=worst))
    return admitted


def _leaf_piece(item: _Admitted) -> _Piece:
    leaf = item.leaf
    return _Piece(
        mode=MODE_LEAF,
        document_id=leaf.document_id,
        section_id=leaf.section_id,
        anchor_section_id=leaf.section_id,
        content=leaf.content,
        token_count=item.token_count,
        page_start=leaf.page_start,
        page_end=leaf.page_end,
        priority=item.candidate.rerank_rank,
        matched_chunk_ids=(leaf.chunk_id,),
        context_chunk_ids=(),
        supporting_ranks=(item.candidate.rerank_rank,),
        ordered_chunk_ids=(leaf.chunk_id,),
    )


def _pass1(
    pieces: list[_Piece],
    admitted: list[_Admitted],
    store: SectionSource,
    catalog: LeafCatalog,
    tokenizer: Any,
    budget: int,
) -> list[_Piece]:
    groups: dict[str, list[_Admitted]] = {}
    for item in admitted:
        groups.setdefault(item.leaf.section_id, []).append(item)
    order = sorted(
        groups,
        key=lambda section_id: (
            min(item.candidate.rerank_rank for item in groups[section_id]),
            section_id,
        ),
    )
    for section_id in order:
        pieces = _upgrade_group(
            pieces, groups[section_id], store, catalog, tokenizer, budget
        )
    return pieces


def _upgrade_group(
    pieces: list[_Piece],
    group: list[_Admitted],
    store: SectionSource,
    catalog: LeafCatalog,
    tokenizer: Any,
    budget: int,
) -> list[_Piece]:
    section_id = group[0].leaf.section_id
    store.get_section(section_id)
    windows = _merge_windows(
        [
            _Window(chunks=catalog.window_chunks(item.leaf), matched=(item,))
            for item in group
        ]
    )
    windows.sort(
        key=lambda window: (
            min(item.candidate.rerank_rank for item in window.matched),
            min(chunk.chunk_index for chunk in window.chunks),
        )
    )
    for window in windows:
        pieces = _try_window(pieces, window, tokenizer, budget)
    return _try_section(pieces, group, store, tokenizer, budget)


def _merge_windows(windows: list[_Window]) -> list[_Window]:
    if not windows:
        return []
    parent = list(range(len(windows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def touches(left: _Window, right: _Window) -> bool:
        left_indexes = {chunk.chunk_index for chunk in left.chunks}
        right_indexes = {chunk.chunk_index for chunk in right.chunks}
        if left_indexes & right_indexes:
            return True
        return min(right_indexes) <= max(left_indexes) + 1 and min(left_indexes) <= (
            max(right_indexes) + 1
        )

    for left_index, left in enumerate(windows):
        for right_index in range(left_index + 1, len(windows)):
            if touches(left, windows[right_index]):
                parent[find(right_index)] = find(left_index)

    grouped: dict[int, list[_Window]] = {}
    for index, window in enumerate(windows):
        grouped.setdefault(find(index), []).append(window)

    merged: list[_Window] = []
    for members in grouped.values():
        by_id: dict[str, CatalogLeaf] = {}
        matched: dict[str, _Admitted] = {}
        for window in members:
            for chunk in window.chunks:
                by_id[chunk.chunk_id] = chunk
            for item in window.matched:
                matched[item.leaf.chunk_id] = item
        chunks = tuple(sorted(by_id.values(), key=lambda chunk: chunk.chunk_index))
        matched_items = tuple(
            sorted(matched.values(), key=lambda item: item.leaf.chunk_index)
        )
        merged.append(_Window(chunks=chunks, matched=matched_items))
    return merged


def _try_window(
    pieces: list[_Piece],
    window: _Window,
    tokenizer: Any,
    budget: int,
) -> list[_Piece]:
    if len(window.chunks) <= 1:
        return pieces
    matched_ids = {item.leaf.chunk_id for item in window.matched}
    content = "".join(chunk.content for chunk in window.chunks)
    new_tokens = count_tokens(tokenizer, content)
    replaced = [
        piece
        for piece in pieces
        if piece.mode == MODE_LEAF and piece.matched_chunk_ids[0] in matched_ids
    ]
    if not replaced:
        return pieces
    incremental = new_tokens - sum(piece.token_count for piece in replaced)
    if _total(pieces) + incremental > budget:
        return pieces
    replaced_ids = {id(piece) for piece in replaced}
    kept = [piece for piece in pieces if id(piece) not in replaced_ids]
    kept.append(_window_piece(window, content, new_tokens))
    return kept


def _window_piece(window: _Window, content: str, token_count: int) -> _Piece:
    matched = tuple(sorted(window.matched, key=lambda item: item.leaf.chunk_index))
    matched_ids = tuple(item.leaf.chunk_id for item in matched)
    matched_set = set(matched_ids)
    ordered_ids = tuple(chunk.chunk_id for chunk in window.chunks)
    context_ids = tuple(
        chunk.chunk_id for chunk in window.chunks if chunk.chunk_id not in matched_set
    )
    ranks = tuple(sorted(item.candidate.rerank_rank for item in matched))
    return _Piece(
        mode=MODE_LEAF_WINDOW,
        document_id=matched[0].leaf.document_id,
        section_id=matched[0].leaf.section_id,
        anchor_section_id=matched[0].leaf.section_id,
        content=content,
        token_count=token_count,
        page_start=min(chunk.page_start for chunk in window.chunks),
        page_end=max(chunk.page_end for chunk in window.chunks),
        priority=ranks[0],
        matched_chunk_ids=matched_ids,
        context_chunk_ids=context_ids,
        supporting_ranks=ranks,
        ordered_chunk_ids=ordered_ids,
    )


def _try_section(
    pieces: list[_Piece],
    group: list[_Admitted],
    store: SectionSource,
    tokenizer: Any,
    budget: int,
) -> list[_Piece]:
    section_id = group[0].leaf.section_id
    node = store.get_section(section_id)
    content = store.recover_section(section_id)
    new_tokens = count_tokens(tokenizer, content)
    current = [piece for piece in pieces if piece.anchor_section_id == section_id]
    incremental = new_tokens - sum(piece.token_count for piece in current)
    if _total(pieces) + incremental > budget:
        return pieces
    matched = tuple(sorted(group, key=lambda item: item.leaf.chunk_index))
    ranks = tuple(sorted(item.candidate.rerank_rank for item in matched))
    kept = [piece for piece in pieces if piece.anchor_section_id != section_id]
    kept.append(
        _Piece(
            mode=MODE_SECTION,
            document_id=node.document_id,
            section_id=section_id,
            anchor_section_id=section_id,
            content=content,
            token_count=new_tokens,
            page_start=node.page_start,
            page_end=node.page_end,
            priority=ranks[0],
            matched_chunk_ids=tuple(item.leaf.chunk_id for item in matched),
            context_chunk_ids=(),
            supporting_ranks=ranks,
            ordered_chunk_ids=tuple(item.leaf.chunk_id for item in matched),
        )
    )
    return kept


def _pass2(
    pieces: list[_Piece],
    admitted: list[_Admitted],
    store: SectionSource,
    tokenizer: Any,
    budget: int,
) -> list[_Piece]:
    for parent_id, child_ids, _priority in _parent_candidates(admitted, store, tokenizer):
        pieces = _try_parent(pieces, parent_id, child_ids, admitted, store, tokenizer, budget)
    return pieces


def _parent_candidates(
    admitted: list[_Admitted],
    store: SectionSource,
    tokenizer: Any,
) -> list[tuple[str, tuple[str, ...], int]]:
    groups: dict[str, list[_Admitted]] = {}
    for item in admitted:
        groups.setdefault(item.leaf.section_id, []).append(item)
    by_parent: dict[str, list[str]] = {}
    for section_id in groups:
        parent_id = store.get_section(section_id).parent_section_id
        if parent_id is None:
            continue
        by_parent.setdefault(parent_id, []).append(section_id)
    candidates: list[tuple[str, tuple[str, ...], int]] = []
    for parent_id, child_ids in by_parent.items():
        unique_children = tuple(dict.fromkeys(child_ids))
        if len(unique_children) < 2:
            continue
        parent = store.get_section(parent_id)
        child_nodes = [store.get_section(child_id) for child_id in unique_children]
        if any(node.document_id != parent.document_id for node in child_nodes):
            raise RecoveryError(
                f"direct parent {parent_id} has children in another document"
            )
        numerator = _coverage_numerator(store, child_nodes, tokenizer)
        denominator = count_tokens(tokenizer, store.recover_section(parent_id))
        if denominator <= 0 or numerator * 2 < denominator:
            continue
        ranks = [
            item.candidate.rerank_rank
            for child_id in unique_children
            for item in groups[child_id]
        ]
        candidates.append((parent_id, unique_children, min(ranks)))
    candidates.sort(key=lambda item: (item[2], item[0]))
    return candidates


def _coverage_numerator(
    store: SectionSource,
    children: Sequence[SectionNode],
    tokenizer: Any,
) -> int:
    document_id = children[0].document_id
    spans = [(node.source_start, node.source_end) for node in children]
    total = 0
    for start, end in union_source_spans(spans):
        total += count_tokens(tokenizer, store.recover_span(document_id, start, end))
    return total


def _try_parent(
    pieces: list[_Piece],
    parent_id: str,
    child_ids: tuple[str, ...],
    admitted: list[_Admitted],
    store: SectionSource,
    tokenizer: Any,
    budget: int,
) -> list[_Piece]:
    child_set = set(child_ids)
    replaced = [piece for piece in pieces if piece.anchor_section_id in child_set]
    present = {piece.anchor_section_id for piece in replaced}
    if present != child_set:
        return pieces
    parent = store.get_section(parent_id)
    content = store.recover_section(parent_id)
    new_tokens = count_tokens(tokenizer, content)
    incremental = new_tokens - sum(piece.token_count for piece in replaced)
    if _total(pieces) + incremental > budget:
        return pieces
    matched = tuple(
        sorted(
            (item for item in admitted if item.leaf.section_id in child_set),
            key=lambda item: item.leaf.chunk_index,
        )
    )
    ranks = tuple(sorted(item.candidate.rerank_rank for item in matched))
    kept = [piece for piece in pieces if piece.anchor_section_id not in child_set]
    kept.append(
        _Piece(
            mode=MODE_PARENT,
            document_id=parent.document_id,
            section_id=parent_id,
            anchor_section_id=parent_id,
            content=content,
            token_count=new_tokens,
            page_start=parent.page_start,
            page_end=parent.page_end,
            priority=ranks[0],
            matched_chunk_ids=tuple(item.leaf.chunk_id for item in matched),
            context_chunk_ids=(),
            supporting_ranks=ranks,
            ordered_chunk_ids=tuple(item.leaf.chunk_id for item in matched),
        )
    )
    return kept


def _finalize(pieces: list[_Piece], store: SectionSource) -> tuple[RecoveredEvidence, ...]:
    kept: dict[tuple[Any, ...], _Piece] = {}
    for piece in pieces:
        key = _identity(piece)
        previous = kept.get(key)
        if previous is None or _prefer(piece, previous):
            kept[key] = piece
    evidence = [
        RecoveredEvidence(
            document_id=piece.document_id,
            recovery_mode=piece.mode,
            section_id=piece.section_id,
            section_path=store.section_path(piece.section_id),
            page_start=piece.page_start,
            page_end=piece.page_end,
            content=piece.content,
            evidence_priority=piece.priority,
            matched_chunk_ids=piece.matched_chunk_ids,
            context_chunk_ids=piece.context_chunk_ids,
            supporting_rerank_ranks=piece.supporting_ranks,
            token_count=piece.token_count,
        )
        for piece in kept.values()
    ]
    evidence.sort(key=lambda item: (item.evidence_priority, item.matched_chunk_ids))
    return tuple(evidence)


def _identity(piece: _Piece) -> tuple[Any, ...]:
    if piece.mode == MODE_PARENT:
        return (MODE_PARENT, piece.section_id)
    if piece.mode == MODE_SECTION:
        return (MODE_SECTION, piece.section_id)
    if piece.mode == MODE_LEAF:
        return (MODE_LEAF, piece.matched_chunk_ids[0])
    return (MODE_LEAF_WINDOW, piece.ordered_chunk_ids)


def _prefer(candidate: _Piece, current: _Piece) -> bool:
    if candidate.priority != current.priority:
        return candidate.priority < current.priority
    return candidate.matched_chunk_ids < current.matched_chunk_ids


def _total(pieces: Sequence[_Piece]) -> int:
    return sum(piece.token_count for piece in pieces)
