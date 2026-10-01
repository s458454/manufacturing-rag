"""Production sections.jsonl reload keeps A4 invariants."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.markdown_loader import LoadedMarkdownDocument  # noqa: E402
from knowledge_base.section_hierarchy import (  # noqa: E402
    SECTION_NODE_FIELD_NAMES,
    SectionNode,
    build_hierarchy_from_documents,
)
from knowledge_base.section_store import (  # noqa: E402
    SectionStoreError,
    load_sections_jsonl,
    write_sections_jsonl,
)
from knowledge_base.structure_parser import parse_markdown_document  # noqa: E402


class StubTokenizer:
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> list[int]:
        return [ord(char) for char in text]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list]:
        ids = self.encode(text, add_special_tokens, truncation)
        offsets = [(index, index + 1) for index in range(len(text))]
        return {"input_ids": ids, "offset_mapping": offsets}


def _loaded(root: Path, document_id: str, content: str) -> LoadedMarkdownDocument:
    folder = root / document_id
    folder.mkdir(parents=True)
    path = folder / "document.md"
    path.write_bytes(content.encode("utf-8"))
    return LoadedMarkdownDocument(
        document_id=document_id,
        content=content,
        path=path.resolve(),
    )


def _hierarchy(tmp_path: Path, content: str):
    document = _loaded(tmp_path, "doc", content)
    parsed, _chunking, hierarchy = build_hierarchy_from_documents(
        [document], StubTokenizer()
    )
    return [document], parsed, hierarchy


def _write_nodes(path: Path, nodes: list[SectionNode]) -> None:
    lines = []
    for node in nodes:
        payload = {name: getattr(node, name) for name in SECTION_NODE_FIELD_NAMES}
        lines.append(json.dumps(payload))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _node(
    section_id: str,
    *,
    document_id: str = "doc",
    parent: str | None = None,
    level: int | None = 2,
    kind: str = "heading",
    start: int = 0,
    end: int = 2,
) -> SectionNode:
    return SectionNode(
        section_id=section_id,
        document_id=document_id,
        parent_section_id=parent,
        kind=kind,
        heading_level=level,
        heading=section_id,
        source_start=start,
        source_end=end,
        page_start=1,
        page_end=1,
    )


def test_roundtrip_recovers_the_same_text_as_a4(tmp_path: Path) -> None:
    content = (
        "<!-- PDF page 1 -->\n"
        "\n"
        "## Parent\n"
        "intro\n"
        "### Child\n"
        "child body\n"
    )
    documents, _parsed, hierarchy = _hierarchy(tmp_path, content)
    path = tmp_path / "sections.jsonl"
    write_sections_jsonl(hierarchy, path)
    raw = path.read_text(encoding="utf-8")
    assert "content" not in json.loads(raw.splitlines()[0])
    assert "fingerprint" not in raw
    store = load_sections_jsonl(path)
    store.bind_documents(documents)
    for node in hierarchy.nodes:
        assert store.recover_section(node.section_id) == hierarchy.recover_section_text(
            node.section_id, documents
        )
        assert "<!-- PDF page" not in store.recover_section(node.section_id)


def test_load_rejects_unknown_fields_duplicates_and_bad_parents(tmp_path: Path) -> None:
    path = tmp_path / "sections.jsonl"
    child = _node("child", parent="parent", level=3, start=1, end=2)
    parent = _node("parent", level=2, start=0, end=2)
    _write_nodes(path, [parent, child, child])
    with pytest.raises(SectionStoreError, match="duplicate"):
        load_sections_jsonl(path)

    missing = {"section_id": "only"}
    path.write_text(json.dumps(missing) + "\n", encoding="utf-8")
    with pytest.raises(SectionStoreError, match="missing"):
        load_sections_jsonl(path)

    payload = {name: getattr(parent, name) for name in SECTION_NODE_FIELD_NAMES}
    payload["fingerprint"] = "nope"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(SectionStoreError, match="unknown"):
        load_sections_jsonl(path)

    orphan = _node("child", parent="missing", level=3)
    _write_nodes(path, [orphan])
    with pytest.raises(SectionStoreError, match="parent_section_id"):
        load_sections_jsonl(path)

    foreign = _node("child", document_id="other", parent="parent", level=3)
    _write_nodes(path, [parent, foreign])
    with pytest.raises(SectionStoreError, match="cross-document"):
        load_sections_jsonl(path)

    deeper = _node("child", parent="parent", level=2)
    shallower = _node("parent", parent="child", level=3)
    _write_nodes(path, [deeper, shallower])
    with pytest.raises(SectionStoreError, match="heading_level"):
        load_sections_jsonl(path)

    left = _node("left", parent="right", level=None, kind="document_root")
    right = _node("right", parent="left", level=None, kind="document_root")
    _write_nodes(path, [left, right])
    with pytest.raises(SectionStoreError, match="cycle"):
        load_sections_jsonl(path)


def test_bind_rejects_span_and_page_that_disagree_with_the_document(tmp_path: Path) -> None:
    content = "<!-- PDF page 1 -->\n\n## Title\nbody\n"
    documents, _parsed, hierarchy = _hierarchy(tmp_path, content)
    path = tmp_path / "sections.jsonl"
    broken_span = [
        replace(node, source_end=node.source_end + 50) for node in hierarchy.nodes
    ]
    _write_nodes(path, broken_span)
    store = load_sections_jsonl(path)
    with pytest.raises(SectionStoreError, match="source span"):
        store.bind_documents(documents)

    broken_page = [replace(node, page_start=9, page_end=9) for node in hierarchy.nodes]
    _write_nodes(path, broken_page)
    store = load_sections_jsonl(path)
    with pytest.raises(SectionStoreError, match="page"):
        store.bind_documents(documents)


def test_context_parses_documents_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = "<!-- PDF page 1 -->\n\n## Title\nbody\n"
    documents, _parsed, hierarchy = _hierarchy(tmp_path / "build", content)
    canonical = tmp_path / "canonical"
    target = canonical / "doc"
    target.mkdir(parents=True)
    (target / "document.md").write_text(content, encoding="utf-8")
    sections = tmp_path / "sections.jsonl"
    write_sections_jsonl(hierarchy, sections)
    rows = tmp_path / "a6_rows.jsonl"
    leaf = next(node for node in hierarchy.nodes if node.kind == "heading")
    rows.write_text(
        json.dumps(
            {
                "chunk_id": "c1",
                "document_id": "doc",
                "section_id": leaf.section_id,
                "chunk_index": 0,
                "page_start": leaf.page_start,
                "page_end": leaf.page_end,
                "content": "body",
                "dense_vector": [1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = {"parse": 0}
    real_parse = parse_markdown_document

    def counting(text: str, document_id: str):
        calls["parse"] += 1
        return real_parse(text, document_id=document_id)

    monkeypatch.setattr(
        "knowledge_base.section_store.parse_markdown_document", counting
    )
    monkeypatch.setattr(
        "retrieval.b6.context.load_generation_tokenizer",
        lambda name: CharTokenizer(),
    )
    from retrieval.b5.types import RerankingResult, RerankedCandidate
    from retrieval.b6.context import load_recovery_context, recover_request

    context = load_recovery_context(
        sections_jsonl=sections,
        rows_jsonl=rows,
        canonical_root=canonical,
        tokenizer_name_or_path="unused-local",
    )
    parsed_during_startup = calls["parse"]
    assert parsed_during_startup >= 1
    candidate = RerankedCandidate(
        chunk_id="c1",
        document_id="doc",
        section_id=leaf.section_id,
        chunk_index=0,
        page_start=leaf.page_start,
        page_end=leaf.page_end,
        content="body",
        rrf_score=0.0,
        fused_rank=1,
        rerank_score=0.0,
        rerank_rank=1,
    )
    first = recover_request(
        context, RerankingResult("req", "task", "query", (candidate,)), budget=1000
    )
    second = recover_request(
        context, RerankingResult("req", "task", "query", (candidate,)), budget=1000
    )
    assert calls["parse"] == parsed_during_startup
    assert first.evidence[0].content == second.evidence[0].content
    assert "<!-- PDF page" not in first.evidence[0].content
    assert documents[0].document_id == "doc"


class CharTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1] * len(text)
