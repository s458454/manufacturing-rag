"""Unit tests for the formal-index Markdown semantic projection.

These fakes intentionally model only the public Docling tree contract used by
``pdf_preprocess``.  They let the safety policy be tested without a Docling
runtime or GPU model installation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


class _FakeRef:
    def __init__(self, cref: str) -> None:
        self.cref = cref

    def resolve(self, doc: "_FakeDocument") -> Any:
        return doc.items[self.cref]


class _FakeProv:
    def __init__(self, page_no: int, bbox: Any = None) -> None:
        self.page_no = page_no
        self.bbox = bbox


class _FakeNode:
    def __init__(
        self,
        ref: str,
        *,
        page_no: int | None = None,
        children: list[str] | None = None,
        text: str | None = None,
        label: object | None = None,
    ) -> None:
        self.self_ref = ref
        self.children = [_FakeRef(child) for child in children or []]
        self.prov = [_FakeProv(page_no)] if page_no is not None else []
        self.text = text
        self.label = label


class _FakeGroup:
    """A structural node with no DocItem provenance, like Docling's body/group."""

    def __init__(self, ref: str, *, children: list[str] | None = None) -> None:
        self.self_ref = ref
        self.children = [_FakeRef(child) for child in children or []]


class _FakeText(_FakeNode):
    pass


class _FakeVisual(_FakeNode):
    def __init__(
        self,
        ref: str,
        *,
        page_no: int,
        captions: list[str],
        children: list[str],
        footnotes: list[str] | None = None,
    ) -> None:
        super().__init__(ref, page_no=page_no, children=children)
        self.captions = [_FakeRef(caption) for caption in captions]
        self.footnotes = [_FakeRef(footnote) for footnote in footnotes or []]


class _FakeTable(_FakeVisual):
    pass


class _FakePicture(_FakeVisual):
    pass


class _FakeTableCell:
    def __init__(self, ref: str) -> None:
        self.ref = _FakeRef(ref)


class _FakeTableData:
    def __init__(self, refs: list[str]) -> None:
        self.table_cells = [_FakeTableCell(ref) for ref in refs]


class _FakeDocument:
    def __init__(self, items: dict[str, _FakeNode]) -> None:
        self.items = items

    def iterate_items(self, **_kwargs: Any):
        for item in self.items.values():
            yield item, 0


def _load_module(monkeypatch: Any):
    """Install minimal import fakes, then import the policy module under test."""
    import importlib

    class FakeLabel:
        CAPTION = "caption"
        SECTION_HEADER = "section_header"

        def __iter__(self):
            return iter((self.CAPTION, self.SECTION_HEADER))

    fake_doc = types.ModuleType("docling_core.types.doc")
    fake_doc.DocItemLabel = FakeLabel()
    fake_doc.PictureItem = _FakePicture
    fake_doc.TableItem = _FakeTable
    fake_doc.TextItem = _FakeText
    fake_doc.DocItem = _FakeNode
    fake_common = types.ModuleType("docling_core.transforms.serializer.common")
    fake_common.create_ser_result = lambda **kwargs: types.SimpleNamespace(
        text=kwargs.get("text", ""), spans=[]
    )

    class FakeMarkdownParams:
        model_fields = {
            "pages": object(),
            "traverse_pictures": object(),
            "enable_chart_tables": object(),
        }

        def __init__(self, **kwargs: Any) -> None:
            self.labels = kwargs.get("labels", set())
            self.pages = kwargs.get("pages")
            self.caption_delim = " "

        def merge_with_patch(self, patch: dict[str, Any]):
            values = {"labels": self.labels, "pages": self.pages}
            values.update(patch)
            return FakeMarkdownParams(**values)

        def model_copy(self, update: dict[str, Any]):
            return self.merge_with_patch(update)

    class FakeMarkdownSerializer(BaseModel):
        """Match the Pydantic construction model used by Docling serializers."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        doc: Any
        params: Any

        def get_excluded_refs(self, **_kwargs: Any) -> set[str]:
            return set()

        def serialize(self, **_kwargs: Any) -> Any:
            return fake_common.create_ser_result()

        def post_process(self, *, text: str, **_kwargs: Any) -> str:
            return text

    fake_markdown = types.ModuleType("docling_core.transforms.serializer.markdown")
    fake_markdown.MarkdownDocSerializer = FakeMarkdownSerializer
    fake_markdown.MarkdownParams = FakeMarkdownParams

    fake_accelerator = types.ModuleType("docling.datamodel.accelerator_options")
    fake_accelerator.AcceleratorOptions = object
    fake_base = types.ModuleType("docling.datamodel.base_models")
    fake_base.ConversionStatus = object
    fake_base.InputFormat = object
    fake_layout = types.ModuleType("docling.datamodel.layout_model_specs")
    fake_layout.DOCLING_LAYOUT_HERON = object()
    fake_pipeline = types.ModuleType("docling.datamodel.pipeline_options")
    for name in (
        "HeadingHierarchyOptions",
        "LayoutOptions",
        "PdfPipelineOptions",
        "RapidOcrOptions",
        "TableFormerMode",
        "TableStructureOptions",
    ):
        setattr(fake_pipeline, name, object)
    fake_converter = types.ModuleType("docling.document_converter")
    fake_converter.DocumentConverter = object
    fake_converter.PdfFormatOption = object

    modules = {
        "docling": types.ModuleType("docling"),
        "docling.datamodel": types.ModuleType("docling.datamodel"),
        "docling.datamodel.accelerator_options": fake_accelerator,
        "docling.datamodel.base_models": fake_base,
        "docling.datamodel.layout_model_specs": fake_layout,
        "docling.datamodel.pipeline_options": fake_pipeline,
        "docling.document_converter": fake_converter,
        "docling_core": types.ModuleType("docling_core"),
        "docling_core.transforms": types.ModuleType("docling_core.transforms"),
        "docling_core.transforms.serializer": types.ModuleType("docling_core.transforms.serializer"),
        "docling_core.transforms.serializer.common": fake_common,
        "docling_core.transforms.serializer.markdown": fake_markdown,
        "docling_core.types": types.ModuleType("docling_core.types"),
        "docling_core.types.doc": fake_doc,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("pdf_preprocess", None)
    return importlib.import_module("pdf_preprocess")


def _pages(*eligible: bool) -> list[dict[str, object]]:
    return [
        {"page_no": index + 1, "eligible_for_indexing": decision}
        for index, decision in enumerate(eligible)
    ]


def test_semantic_projection_fails_closed_without_required_docling_serializer_params(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module.MarkdownParams, "model_fields", {"pages": object()})

    with pytest.raises(RuntimeError, match="docling-core>=2.88.0"):
        module.require_semantic_projection_serializer_api()


def test_missing_tableformer_assets_fail_fast(monkeypatch: Any, tmp_path: Path) -> None:
    module = _load_module(monkeypatch)
    monkeypatch.setattr(module, "TABLEFORMER_ACCURATE_DIR", tmp_path / "accurate")

    with pytest.raises(FileNotFoundError, match="TableFormer V1 accurate"):
        module.require_local_tableformer_assets()

    model_dir = tmp_path / "accurate"
    model_dir.mkdir()
    config_path = model_dir / "tm_config.json"
    config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="tableformer_accurate.safetensors"):
        module.require_local_tableformer_assets()

    model_path = model_dir / "tableformer_accurate.safetensors"
    model_path.write_bytes(b"not-the-approved-weights")
    with pytest.raises(RuntimeError, match="incorrect file size: tm_config.json"):
        module.require_local_tableformer_assets()

    # Both runtime assets are integrity-protected.  Use small test expectations
    # after proving that the real configuration file size is independently
    # checked, then exercise each SHA-256 path independently.
    monkeypatch.setattr(
        module,
        "TABLEFORMER_EXPECTED_ASSETS",
        {
            "tm_config.json": {
                "size_bytes": config_path.stat().st_size,
                "sha256": "f" * 64,
            },
            "tableformer_accurate.safetensors": {
                "size_bytes": model_path.stat().st_size,
                "sha256": module.sha256_file(model_path),
            },
        },
    )
    with pytest.raises(RuntimeError, match="incorrect SHA-256"):
        module.require_local_tableformer_assets()

    monkeypatch.setattr(
        module,
        "TABLEFORMER_EXPECTED_ASSETS",
        {
            "tm_config.json": {
                "size_bytes": config_path.stat().st_size,
                "sha256": module.sha256_file(config_path),
            },
            "tableformer_accurate.safetensors": {
                "size_bytes": model_path.stat().st_size,
                "sha256": "0" * 64,
            },
        },
    )
    with pytest.raises(RuntimeError, match="incorrect SHA-256"):
        module.require_local_tableformer_assets()


def test_table_section_hierarchy_prefers_docling_resolved_hierarchy(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)

    class Document:
        def get_heading_hierarchy(self, _item: Any) -> list[str]:
            return ["1 Scope", "1.2 Requirements"]

    hierarchy = module._section_hierarchy_before(
        object(), Document(), ["fallback heading"]
    )

    assert hierarchy == ["1 Scope", "1.2 Requirements"]


def test_section_hierarchy_stack_uses_assigned_heading_levels(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    h1 = types.SimpleNamespace(text="1 Scope", level=1)
    h2 = types.SimpleNamespace(text="1.1 Purpose", level=2)
    next_h1 = types.SimpleNamespace(text="2 Requirements", level=1)

    hierarchy = module._advance_section_hierarchy([], h1)
    hierarchy = module._advance_section_hierarchy(hierarchy, h2)
    assert hierarchy == ["1 Scope", "1.1 Purpose"]
    assert module._advance_section_hierarchy(hierarchy, next_h1) == ["2 Requirements"]


def test_projection_preserves_explicit_caption_and_excludes_other_visual_descendants(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0",
                page_no=1,
                captions=["#/texts/0"],
                children=["#/texts/0", "#/groups/0"],
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="TABLE 5.31 Alloy data"),
            "#/groups/0": _FakeNode("#/groups/0", children=["#/texts/1"]),
            "#/texts/1": _FakeText("#/texts/1", page_no=1, text="AM54028 00-A(d01)"),
            "#/texts/2": _FakeText("#/texts/2", page_no=1, text="Body paragraph"),
        }
    )

    projection = module.build_semantic_projection(document, _pages(True))

    assert "#/tables/0" in projection.excluded_refs
    assert "#/groups/0" in projection.excluded_refs
    assert "#/texts/1" in projection.excluded_refs
    assert "#/texts/0" not in projection.excluded_refs
    assert projection.accepted_caption_refs == {"#/texts/0"}
    assert projection.visual_caption_trust["#/tables/0"] == "accepted"


def test_projection_excludes_descendants_reachable_only_from_rich_table_cells(monkeypatch: Any) -> None:
    """A RichTableCell may be authoritative even if not duplicated in children."""
    module = _load_module(monkeypatch)
    table = _FakeTable(
        "#/tables/0",
        page_no=1,
        captions=["#/texts/0"],
        children=["#/texts/0"],
    )
    table.data = _FakeTableData(["#/groups/0"])
    document = _FakeDocument(
        {
            "#/tables/0": table,
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Trusted caption"),
            "#/groups/0": _FakeNode("#/groups/0", children=["#/texts/1"]),
            "#/texts/1": _FakeText("#/texts/1", page_no=1, text="831 00-A(d01)"),
        }
    )

    projection = module.build_semantic_projection(document, _pages(True))

    assert "#/texts/0" in projection.accepted_caption_refs
    assert "#/groups/0" in projection.excluded_refs
    assert "#/texts/1" in projection.excluded_refs


def test_projection_excludes_visual_caption_on_untrusted_page(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/pictures/0": _FakePicture(
                "#/pictures/0",
                page_no=2,
                captions=["#/texts/0"],
                children=["#/texts/0", "#/texts/1"],
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=2, text="Figure caption"),
            "#/texts/1": _FakeText("#/texts/1", page_no=2, text="OCR inside chart"),
            "#/texts/2": _FakeText("#/texts/2", page_no=1, text="Trusted page body"),
        }
    )

    projection = module.build_semantic_projection(document, _pages(True, False))

    assert projection.visual_caption_trust["#/pictures/0"] == "page_untrusted"
    assert "#/texts/0" in projection.excluded_refs
    assert "#/texts/1" in projection.excluded_refs
    assert "#/texts/2" not in projection.excluded_refs


def test_projection_keeps_page_less_structural_group_for_trusted_children(monkeypatch: Any) -> None:
    """Page filtering must not discard document.body/ListGroup containers."""
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/body": _FakeGroup(
                "#/body", children=["#/texts/0", "#/texts/1"]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Trusted body"),
            "#/texts/1": _FakeText("#/texts/1", page_no=2, text="Blocked body"),
        }
    )

    projection = module.build_semantic_projection(document, _pages(True, False))

    assert "#/body" not in projection.excluded_refs
    assert "#/texts/0" not in projection.excluded_refs
    assert "#/texts/1" in projection.excluded_refs


def test_projection_excludes_caption_with_untrusted_provenance_page(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0",
                page_no=1,
                captions=["#/texts/0"],
                children=["#/texts/0"],
            ),
            # The owner is on page 1, but the explicitly linked caption's
            # source is page 2, which is not eligible for formal indexing.
            "#/texts/0": _FakeText("#/texts/0", page_no=2, text="Cross-page caption"),
        }
    )

    projection = module.build_semantic_projection(document, _pages(True, False))

    assert projection.visual_caption_trust["#/tables/0"] == "page_untrusted"
    assert "#/texts/0" in projection.excluded_refs


def test_projection_does_not_allow_garbled_caption(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0",
                page_no=1,
                captions=["#/texts/0"],
                children=["#/texts/0"],
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="@@@ !!! ???"),
        }
    )

    projection = module.build_semantic_projection(document, _pages(True))

    assert projection.visual_caption_trust["#/tables/0"] == "garbled"
    assert "#/texts/0" in projection.excluded_refs


def test_short_visibly_corrupt_caption_is_not_treated_as_trusted(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    assert module.caption_is_obviously_garbled('*"F      e  t!') is True
    assert module.caption_is_obviously_garbled("Fig. 1") is False
    assert module.caption_is_obviously_garbled("A-1") is False


def test_export_keeps_original_page_marker_for_an_eligible_empty_projection(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument({})
    projection = module.build_semantic_projection(document, _pages(True))

    assert module.export_semantic_markdown(document, projection) == "<!-- PDF page 1 -->"


def test_regions_record_new_explicit_markdown_and_caption_trust_fields(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0", page_no=1, captions=["#/texts/0"], children=["#/texts/0"]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Trusted table caption"),
        }
    )
    projection = module.build_semantic_projection(document, _pages(True))
    regions = module.collect_regions(document, projection)

    assert regions[0]["visual_body_in_semantic_markdown"] is False
    assert regions[0]["caption_in_semantic_markdown"] is True
    assert regions[0]["caption_trust_decision"] == "accepted"
    assert regions[0]["region_id"] == "#/tables/0"


def test_accepted_native_table_is_injected_once_while_its_caption_is_retained(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0", page_no=1, captions=["#/texts/0"], children=["#/texts/0"]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Table 1. Requirements"),
        }
    )
    accepted = module.TableExtractionResult(
        table_id="doc-p0001-t001",
        document_id="doc",
        page_no=1,
        bbox=[0.0, 0.0, 10.0, 10.0],
        section_hierarchy=[],
        caption="Table 1. Requirements",
        docling_ref="#/tables/0",
        source_kind="native",
        extractor="native_pdf_table",
        decision="accepted",
        row_count=2,
        column_count=2,
        cells=[
            {"row_start": 0, "row_span": 1, "column_start": 0, "column_span": 1, "text": "Section", "source_cell_refs": ["p/1"]},
            {"row_start": 0, "row_span": 1, "column_start": 1, "column_span": 1, "text": "Requirement", "source_cell_refs": ["p/2"]},
            {"row_start": 1, "row_span": 1, "column_start": 0, "column_span": 1, "text": "A", "source_cell_refs": ["p/3"]},
            {"row_start": 1, "row_span": 1, "column_start": 1, "column_span": 1, "text": "Required", "source_cell_refs": ["p/4"]},
        ],
        validation={},
    )
    projection = module.build_semantic_projection(
        document, _pages(True), {accepted.table_id: accepted}
    )
    serializer = module.SemanticMarkdownSerializer(
        doc=document,
        params=module.MarkdownParams(labels={module.DocItemLabel.CAPTION}, pages={1}),
    )
    serializer.configure_projection(projection, [accepted])
    table_at_original_position = serializer.serialize(
        item=document.items["#/tables/0"], pages={1}
    ).text
    markdown = module.inject_accepted_tables_into_markdown(
        "<!-- PDF page 1 -->\n\n## Requirements\n\nBefore table\n\n"
        + table_at_original_position
        + "\n\nAfter table",
        [accepted],
    )

    assert accepted.table_id in projection.accepted_table_ids
    assert markdown.count("<!-- TABLE id=doc-p0001-t001") == 1
    assert markdown.count("| Section | Requirement |") == 1
    assert markdown.count("Table 1. Requirements") == 1
    assert "CANONICAL_TABLE_PLACEHOLDER" not in markdown
    assert markdown.index("Before table") < markdown.index("Table 1. Requirements")
    assert markdown.index("Table 1. Requirements") < markdown.index("<!-- TABLE id=")
    assert markdown.index("<!-- /TABLE -->") < markdown.index("After table")


def test_table_placeholder_is_emitted_only_for_its_owner_page(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0", page_no=1, captions=[], children=[]
            )
        }
    )
    accepted = module.TableExtractionResult(
        table_id="doc-p0001-t001",
        document_id="doc",
        page_no=1,
        bbox=None,
        section_hierarchy=[],
        caption=None,
        docling_ref="#/tables/0",
        source_kind="native",
        extractor="native_pdf_table",
        decision="accepted",
        row_count=1,
        column_count=1,
        cells=[
            {
                "row_start": 0,
                "row_span": 1,
                "column_start": 0,
                "column_span": 1,
                "column_header": False,
                "row_header": False,
                "text": "A",
                "source_cell_refs": ["p/1"],
            }
        ],
        validation={},
    )
    projection = module.build_semantic_projection(
        document, _pages(True, True), {accepted.table_id: accepted}
    )
    serializer = module.SemanticMarkdownSerializer(
        doc=document,
        params=module.MarkdownParams(labels=set(), pages=None),
    )
    serializer.configure_projection(projection, [accepted])

    assert "CANONICAL_TABLE_PLACEHOLDER" in serializer.serialize(
        item=document.items["#/tables/0"], pages={1}
    ).text
    assert serializer.serialize(
        item=document.items["#/tables/0"], pages={2}
    ).text == ""


def test_accepted_native_table_retains_linked_trusted_footnote_once(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    table = _FakeTable(
        "#/tables/0",
        page_no=1,
        captions=[],
        children=[],
        footnotes=["#/texts/0"],
    )
    document = _FakeDocument(
        {
            "#/tables/0": table,
            "#/texts/0": _FakeText(
                "#/texts/0", page_no=1, text="Note: GWR applicability is conditional."
            ),
        }
    )
    accepted = module.TableExtractionResult(
        table_id="doc-p0001-t001",
        document_id="doc",
        page_no=1,
        bbox=None,
        section_hierarchy=[],
        caption=None,
        docling_ref="#/tables/0",
        source_kind="native",
        extractor="native_pdf_table",
        decision="accepted",
        row_count=1,
        column_count=1,
        cells=[
            {
                "row_start": 0,
                "row_span": 1,
                "column_start": 0,
                "column_span": 1,
                "column_header": False,
                "row_header": False,
                "text": "A",
                "source_cell_refs": ["p/1"],
            }
        ],
        validation={},
        footnotes=[
            {
                "footnote_ref": "#/texts/0",
                "text": "Note: GWR applicability is conditional.",
                "page_nos": [1],
                "trust_decision": "accepted",
                "in_semantic_markdown": True,
            }
        ],
    )
    projection = module.build_semantic_projection(
        document, _pages(True), {accepted.table_id: accepted}
    )
    assert "#/texts/0" in projection.excluded_refs
    serializer = module.SemanticMarkdownSerializer(
        doc=document,
        params=module.MarkdownParams(labels=set(), pages=None),
    )
    serializer.configure_projection(projection, [accepted])
    serialized = serializer.serialize(item=table, pages={1}).text

    assert serialized.count("Note: GWR applicability is conditional.") == 1
    assert "CANONICAL_TABLE_PLACEHOLDER" in serialized


def test_accepted_table_without_original_placeholder_fails_closed(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    accepted = module.TableExtractionResult(
        table_id="doc-p0001-t001", document_id="doc", page_no=1, bbox=None,
        section_hierarchy=[], caption=None, docling_ref="#/tables/0", source_kind="native",
        extractor="native_pdf_table", decision="accepted", row_count=1, column_count=1,
        cells=[{"row_start": 0, "row_span": 1, "column_start": 0, "column_span": 1, "text": "A", "source_cell_refs": ["p/1"]}], validation={},
    )

    with pytest.raises(RuntimeError, match="placeholder must occur exactly once"):
        module.inject_accepted_tables_into_markdown("<!-- PDF page 1 -->", [accepted])


def test_accepted_table_on_untrusted_page_is_never_injected(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    accepted = module.TableExtractionResult(
        table_id="doc-p0001-t001", document_id="doc", page_no=1, bbox=None,
        section_hierarchy=[], caption=None, docling_ref="#/tables/0", source_kind="native",
        extractor="native_pdf_table", decision="accepted", row_count=1, column_count=1,
        cells=[{"row_start": 0, "row_span": 1, "column_start": 0, "column_span": 1, "text": "A", "source_cell_refs": ["p/1"]}], validation={},
    )

    markdown = module.inject_accepted_tables_into_markdown(
        "<!-- PDF page 1 -->", [accepted], set()
    )

    assert "<!-- TABLE" not in markdown


def test_deferred_table_emits_only_its_trusted_caption_at_original_position(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0", page_no=1, captions=["#/texts/0"], children=["#/texts/0"]
            ),
            "#/texts/0": _FakeText(
                "#/texts/0", page_no=1, text="Table 2. Deferred image table"
            ),
        }
    )
    deferred = module.TableExtractionResult(
        table_id="doc-p0001-t001", document_id="doc", page_no=1, bbox=None,
        section_hierarchy=[], caption="Table 2. Deferred image table",
        docling_ref="#/tables/0", source_kind="image_only", extractor="ocr_table",
        decision="deferred", row_count=0, column_count=0, cells=[], validation={},
    )
    projection = module.build_semantic_projection(document, _pages(True))
    serializer = module.SemanticMarkdownSerializer(
        doc=document,
        params=module.MarkdownParams(labels={module.DocItemLabel.CAPTION}, pages={1}),
    )
    serializer.configure_projection(projection, [deferred])

    serialized = serializer.serialize(item=document.items["#/tables/0"], pages={1}).text
    markdown = module.inject_accepted_tables_into_markdown(serialized, [deferred], {1})

    assert markdown == "Table 2. Deferred image table"
    assert "CANONICAL_TABLE_PLACEHOLDER" not in markdown
    assert "<!-- TABLE" not in markdown


def test_regions_record_table_route_and_audit_artifact(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {"#/tables/0": _FakeTable("#/tables/0", page_no=1, captions=[], children=[])}
    )
    projection = module.build_semantic_projection(document, _pages(True))
    deferred = module.TableExtractionResult(
        table_id="doc-p0001-t001", document_id="doc", page_no=1, bbox=None,
        section_hierarchy=[], caption=None, docling_ref="#/tables/0", source_kind="ocr",
        extractor="ocr_table", decision="deferred", row_count=0, column_count=0,
        cells=[], validation={},
    )
    regions = module.collect_regions(document, projection, {"#/tables/0": [deferred]})

    assert regions[0]["source_kind"] == "ocr"
    assert regions[0]["table_decision"] == "deferred"
    assert regions[0]["table_artifact"] == "tables/doc-p0001-t001.json"
    assert regions[0]["visual_body_in_semantic_markdown"] is False
    assert regions[0]["canonical_table_in_semantic_markdown"] is False


def test_multi_page_table_item_is_one_deferred_artifact_without_copied_cells(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    table = _FakeTable("#/tables/0", page_no=1, captions=[], children=[])
    table.prov = [_FakeProv(1), _FakeProv(2)]
    document = _FakeDocument({"#/tables/0": table})
    projection = module.build_semantic_projection(document, _pages(True, True))
    pages = [
        types.SimpleNamespace(
            page_no=page_no,
            size=types.SimpleNamespace(height=100.0),
            cells=[],
        )
        for page_no in (1, 2)
    ]

    results = module.extract_tables(document, pages, "doc", projection)

    assert len(results) == 1
    assert results[0].page_no == 1
    assert results[0].decision == "deferred"
    assert results[0].cells == []
    assert results[0].validation["provenance_page_numbers"] == [1, 2]
    assert results[0].validation["failure_reasons"] == [
        "multi_page_table_not_supported_v0_1"
    ]


def test_regions_distinguish_canonical_native_table_from_visual_body(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {"#/tables/0": _FakeTable("#/tables/0", page_no=1, captions=[], children=[])}
    )
    projection = module.build_semantic_projection(document, _pages(True))
    accepted = module.TableExtractionResult(
        table_id="doc-p0001-t001", document_id="doc", page_no=1, bbox=None,
        section_hierarchy=[], caption=None, docling_ref="#/tables/0", source_kind="native",
        extractor="native_pdf_table", decision="accepted", row_count=1, column_count=1,
        cells=[], validation={},
    )
    region = module.collect_regions(document, projection, {"#/tables/0": [accepted]})[0]

    assert region["visual_body_in_semantic_markdown"] is False
    assert region["canonical_table_in_semantic_markdown"] is True


def test_serializer_stops_excluded_visual_group_but_keeps_visual_root_for_caption(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0", page_no=1, captions=["#/texts/0"], children=["#/texts/0"]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Caption"),
            "#/groups/0": _FakeNode("#/groups/0", children=["#/texts/1"]),
            "#/texts/1": _FakeText("#/texts/1", page_no=1, text="Visual OCR"),
        }
    )
    projection = module.build_semantic_projection(document, _pages(True))
    serializer = module.SemanticMarkdownSerializer(
        doc=document,
        params=module.MarkdownParams(labels={"caption"}, pages={1}),
    )
    serializer.configure_projection(projection)

    assert serializer.serialize(item=document.items["#/groups/0"]).text == ""
    # Table/Picture roots deliberately reach their specialized serializer so it
    # can output an allowed explicit caption while suppressing visual body OCR.
    assert serializer.serialize(item=document.items["#/tables/0"]).text == "Caption"


def test_caption_serializer_emits_shared_explicit_caption_once(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable(
                "#/tables/0", page_no=1, captions=["#/texts/0"], children=["#/texts/0"]
            ),
            "#/pictures/0": _FakePicture(
                "#/pictures/0", page_no=1, captions=["#/texts/0"], children=["#/texts/0"]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="One shared caption"),
        }
    )
    projection = module.build_semantic_projection(document, _pages(True))
    serializer = module.SemanticMarkdownSerializer(
        doc=document,
        params=module.MarkdownParams(labels={"caption"}, pages={1}),
    )
    serializer.configure_projection(projection)

    assert serializer.serialize_captions(document.items["#/tables/0"]).text == "One shared caption"
    assert serializer.serialize_captions(document.items["#/pictures/0"]).text == ""


def test_semantic_text_sanity_uses_only_high_precision_corruption_rules(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    assert module.semantic_text_sanity("Aluminum 7075-T6 yield strength: 503 MPa.")[
        "has_hard_corruption"
    ] is False
    assert module.semantic_text_sanity("normal engineering text \ufffd")[
        "has_hard_corruption"
    ] is True
    assert module.semantic_text_sanity("A B C D E F G H I J K L M N O P Q R S T")[
        "has_hard_corruption"
    ] is True


def test_conversion_exit_code_rejects_partial_errors_and_no_eligible_pages(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)

    class Status:
        SUCCESS = "success"

    monkeypatch.setattr(module, "ConversionStatus", Status)
    assert module.conversion_exit_code("success", [], [{"eligible_for_indexing": True}]) == 0
    assert module.conversion_exit_code("partial_success", [], [{"eligible_for_indexing": True}]) == 1
    assert module.conversion_exit_code("success", [{"error": "failed"}], [{"eligible_for_indexing": True}]) == 1
    assert module.conversion_exit_code("success", [], [{"eligible_for_indexing": False}]) == 3
    assert module.conversion_exit_code("success", [], []) == 3


def test_output_transaction_preserves_previous_good_run_on_rejected_replacement(
    monkeypatch: Any, tmp_path: Path
) -> None:
    module = _load_module(monkeypatch)
    final = tmp_path / "outputs" / "document-id"
    final.mkdir(parents=True)
    (final / "document.md").write_text("previous-good", encoding="utf-8")

    transaction = module.OutputTransaction(final, overwrite=True)
    with transaction as staging:
        (staging / "document.md").write_text("rejected", encoding="utf-8")
        for name in (
            "document.json",
            "regions.json",
            "quality_report.json",
            "tables/index.json",
        ):
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            (staging / name).write_text("{}", encoding="utf-8")
        failure = transaction.retain_failure()

    assert (final / "document.md").read_text(encoding="utf-8") == "previous-good"
    assert (failure / "document.md").read_text(encoding="utf-8") == "rejected"
    assert not transaction.lock_path.exists()


def _region_ocr_page(rectangles: list[dict[str, float]]) -> dict[str, object]:
    return {
        "page_no": 1,
        "eligible_for_indexing": True,
        "route_observed": "region_ocr",
        "ocr_route_evidence": {"ocr_rectangles": rectangles},
    }


def test_region_ocr_rectangle_isolates_stray_visual_text_but_keeps_trusted_caption(
    monkeypatch: Any,
) -> None:
    """Reproduces the NASA-STD-5006A Figure 4 leak: OCR text in the gap
    between detected Picture boxes must not reach document.md, while the
    trusted caption is kept even though its own bbox also falls inside the
    same region-OCR rectangle (Task 1.5)."""

    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/pictures/0": _FakePicture(
                "#/pictures/0", page_no=1, captions=["#/texts/0"], children=["#/texts/0"]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Figure 4-Fillet Welds"),
            "#/texts/1": _FakeText(
                "#/texts/1", page_no=1, text="Notes: Root of Joint leaked OCR text"
            ),
            "#/texts/2": _FakeText("#/texts/2", page_no=1, text="Normal chapter body text"),
        }
    )
    document.items["#/pictures/0"].prov = [
        _FakeProv(1, bbox={"l": 0, "t": 0, "r": 100, "b": 80})
    ]
    # The caption's own bbox happens to sit inside the same rectangle as the
    # picture it describes; the trusted-caption exception must still apply.
    document.items["#/texts/0"].prov = [
        _FakeProv(1, bbox={"l": 10, "t": 65, "r": 90, "b": 78})
    ]
    document.items["#/texts/1"].prov = [
        _FakeProv(1, bbox={"l": 20, "t": 20, "r": 80, "b": 60})
    ]
    document.items["#/texts/2"].prov = [
        _FakeProv(1, bbox={"l": 20, "t": 150, "r": 80, "b": 170})
    ]
    pages = [_region_ocr_page([{"l": 0, "t": 0, "r": 100, "b": 80}])]

    projection = module.build_semantic_projection(
        document, pages, page_heights={1: 200.0}
    )

    assert 1 in projection.visual_ocr_rectangles_by_page
    assert "#/texts/1" in projection.visual_ocr_isolated_refs
    assert "#/texts/0" in projection.visual_ocr_isolated_refs
    assert "#/texts/1" in projection.excluded_refs
    assert "#/texts/0" not in projection.excluded_refs
    assert "#/texts/2" not in projection.excluded_refs


def test_full_page_ocr_route_never_applies_visual_ocr_isolation(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/pictures/0": _FakePicture(
                "#/pictures/0", page_no=1, captions=[], children=[]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Whole page OCR body text"),
        }
    )
    document.items["#/pictures/0"].prov = [
        _FakeProv(1, bbox={"l": 0, "t": 0, "r": 100, "b": 100})
    ]
    document.items["#/texts/0"].prov = [
        _FakeProv(1, bbox={"l": 10, "t": 10, "r": 90, "b": 90})
    ]
    pages = [
        {
            "page_no": 1,
            "eligible_for_indexing": True,
            "route_observed": "full_page_ocr",
            "ocr_route_evidence": {
                "ocr_rectangles": [{"l": 0, "t": 0, "r": 100, "b": 100}]
            },
        }
    ]

    projection = module.build_semantic_projection(
        document, pages, page_heights={1: 200.0}
    )

    assert projection.visual_ocr_rectangles_by_page == {}
    assert "#/texts/0" not in projection.excluded_refs


def test_native_only_route_never_applies_visual_ocr_isolation(monkeypatch: Any) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/pictures/0": _FakePicture(
                "#/pictures/0", page_no=1, captions=[], children=[]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Body text near a picture"),
        }
    )
    document.items["#/pictures/0"].prov = [
        _FakeProv(1, bbox={"l": 0, "t": 0, "r": 100, "b": 80})
    ]
    document.items["#/texts/0"].prov = [
        _FakeProv(1, bbox={"l": 20, "t": 20, "r": 80, "b": 60})
    ]
    # Contrived: a native_only page should never carry OCR rectangles in
    # practice, but the isolation must be gated on route, not just presence.
    pages = [
        {
            "page_no": 1,
            "eligible_for_indexing": True,
            "route_observed": "native_only",
            "ocr_route_evidence": {
                "ocr_rectangles": [{"l": 0, "t": 0, "r": 100, "b": 80}]
            },
        }
    ]

    projection = module.build_semantic_projection(
        document, pages, page_heights={1: 200.0}
    )

    assert projection.visual_ocr_rectangles_by_page == {}
    assert "#/texts/0" not in projection.excluded_refs


def test_native_table_does_not_contribute_to_visual_ocr_rectangle_marking(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable("#/tables/0", page_no=1, captions=[], children=[]),
            "#/texts/0": _FakeText(
                "#/texts/0", page_no=1, text="Body text near a native table"
            ),
        }
    )
    document.items["#/tables/0"].prov = [
        _FakeProv(1, bbox={"l": 0, "t": 0, "r": 100, "b": 80})
    ]
    document.items["#/texts/0"].prov = [
        _FakeProv(1, bbox={"l": 20, "t": 20, "r": 80, "b": 60})
    ]
    accepted_native = module.TableExtractionResult(
        table_id="doc-p0001-t001", document_id="doc", page_no=1, bbox=[0, 0, 100, 80],
        section_hierarchy=[], caption=None, docling_ref="#/tables/0", source_kind="native",
        extractor="native_pdf_table", decision="accepted", row_count=1, column_count=1,
        cells=[], validation={},
    )
    pages = [_region_ocr_page([{"l": 0, "t": 0, "r": 100, "b": 80}])]

    projection = module.build_semantic_projection(
        document,
        pages,
        {accepted_native.table_id: accepted_native},
        page_heights={1: 200.0},
    )

    assert projection.visual_ocr_rectangles_by_page == {}
    assert "#/texts/0" not in projection.excluded_refs


def test_non_native_table_contributes_to_visual_ocr_rectangle_marking(
    monkeypatch: Any,
) -> None:
    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/tables/0": _FakeTable("#/tables/0", page_no=1, captions=[], children=[]),
            "#/texts/0": _FakeText(
                "#/texts/0", page_no=1, text="Stray OCR text inside a scanned table"
            ),
        }
    )
    document.items["#/tables/0"].prov = [
        _FakeProv(1, bbox={"l": 0, "t": 0, "r": 100, "b": 80})
    ]
    document.items["#/texts/0"].prov = [
        _FakeProv(1, bbox={"l": 20, "t": 20, "r": 80, "b": 60})
    ]
    deferred = module.TableExtractionResult(
        table_id="doc-p0001-t001", document_id="doc", page_no=1, bbox=[0, 0, 100, 80],
        section_hierarchy=[], caption=None, docling_ref="#/tables/0", source_kind="ocr",
        extractor="ocr_table", decision="deferred", row_count=0, column_count=0,
        cells=[], validation={},
    )
    pages = [_region_ocr_page([{"l": 0, "t": 0, "r": 100, "b": 80}])]

    projection = module.build_semantic_projection(
        document,
        pages,
        {deferred.table_id: deferred},
        page_heights={1: 200.0},
    )

    assert 1 in projection.visual_ocr_rectangles_by_page
    assert "#/texts/0" in projection.excluded_refs


def test_visual_ocr_isolation_is_a_no_op_without_page_heights(monkeypatch: Any) -> None:
    """Callers that never pass ``page_heights`` (the pre-table-routing
    provisional projections) must see byte-identical behavior to before
    Task 1 existed."""

    module = _load_module(monkeypatch)
    document = _FakeDocument(
        {
            "#/pictures/0": _FakePicture(
                "#/pictures/0", page_no=1, captions=[], children=[]
            ),
            "#/texts/0": _FakeText("#/texts/0", page_no=1, text="Stray text"),
        }
    )
    document.items["#/pictures/0"].prov = [
        _FakeProv(1, bbox={"l": 0, "t": 0, "r": 100, "b": 80})
    ]
    document.items["#/texts/0"].prov = [
        _FakeProv(1, bbox={"l": 20, "t": 20, "r": 80, "b": 60})
    ]
    pages = [_region_ocr_page([{"l": 0, "t": 0, "r": 100, "b": 80}])]

    projection = module.build_semantic_projection(document, pages)

    assert projection.visual_ocr_rectangles_by_page == {}
    assert projection.visual_ocr_isolated_refs == set()
    assert "#/texts/0" not in projection.excluded_refs
