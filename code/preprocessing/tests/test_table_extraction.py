"""Regression coverage for canonical native-PDF table extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from table_extraction import (
    NativePdfTableExtractor,
    OcrTableExtractor,
    TableRegion,
    table_summary,
    table_to_markdown,
)


class _Box:
    def __init__(self, l: float, t: float, r: float, b: float) -> None:
        self.l, self.t, self.r, self.b = l, t, r, b
        self.coord_origin = "TOPLEFT"

    def to_bounding_box(self):
        return self


class _BottomLeftBox(_Box):
    def __init__(self, l: float, b: float, r: float, t: float) -> None:
        super().__init__(l, t, r, b)
        self.coord_origin = "BOTTOMLEFT"


def _page(*cells: object) -> object:
    return SimpleNamespace(cells=list(cells), size=SimpleNamespace(height=100.0))


def _source(text: str, box: _Box, *, from_ocr: bool = False, index: int = 0) -> object:
    return SimpleNamespace(text=text, rect=box, from_ocr=from_ocr, index=index)


def _table_cell(
    row: int,
    column: int,
    box: _Box,
    *,
    row_span: int = 1,
    col_span: int = 1,
    column_header: bool | None = None,
    row_header: bool = False,
) -> object:
    return SimpleNamespace(
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=column,
        end_col_offset_idx=column + col_span,
        row_span=row_span,
        col_span=col_span,
        column_header=(row == 0 if column_header is None else column_header),
        row_header=row_header,
        bbox=box,
    )


def _region(*cells: object, rows: int = 2, columns: int = 2) -> TableRegion:
    return TableRegion(
        table_id="doc-p0001-t001",
        document_id="doc",
        page_no=1,
        bbox=[0.0, 0.0, 100.0, 100.0],
        section_hierarchy=["Requirements"],
        caption="Table 1. Requirements",
        docling_ref="#/tables/0",
        item=SimpleNamespace(data=SimpleNamespace(num_rows=rows, num_cols=columns, table_cells=list(cells))),
    )


def test_native_table_is_accepted_with_complete_provenance() -> None:
    region = _region(
        _table_cell(0, 0, _Box(0, 0, 50, 50)),
        _table_cell(0, 1, _Box(50, 0, 100, 50)),
        _table_cell(1, 0, _Box(0, 50, 50, 100)),
        _table_cell(1, 1, _Box(50, 50, 100, 100)),
    )
    page = _page(
        _source("Section", _Box(1, 1, 40, 20), index=1),
        _source("Requirement", _Box(55, 1, 90, 20), index=2),
        _source("A", _Box(1, 55, 20, 75), index=3),
        _source("Required", _Box(55, 55, 95, 75), index=4),
    )

    result = NativePdfTableExtractor().extract(region, page, object())

    assert result.source_kind == "native"
    assert result.decision == "accepted"
    assert result.validation["text_conservation_ratio"] == 1.0
    assert result.cells[0]["source_cell_refs"] == ["page/1/cells/0"]
    assert result.cells[0]["column_header"] is True
    markdown = table_to_markdown(result)
    assert markdown.count("Section") == 1
    assert "| Section | Requirement |" in markdown


def test_ocr_mixed_and_image_only_tables_are_deferred_without_cells() -> None:
    region = _region(_table_cell(0, 0, _Box(0, 0, 100, 50)))
    for page, expected in (
        (_page(_source("OCR", _Box(1, 1, 20, 20), from_ocr=True)), "ocr"),
        (_page(_source("Native", _Box(1, 1, 20, 20)), _source("OCR", _Box(30, 1, 50, 20), from_ocr=True)), "mixed"),
        (_page(), "image_only"),
    ):
        result = OcrTableExtractor().extract(region, page, object())
        assert result.source_kind == expected
        assert result.decision == "deferred"
        assert result.cells == []


def test_native_table_rejects_ambiguous_source_cell_geometry() -> None:
    region = _region(
        _table_cell(0, 0, _Box(0, 0, 60, 60)),
        _table_cell(0, 1, _Box(40, 0, 100, 60)),
        rows=1,
    )
    page = _page(_source("Ambiguous", _Box(45, 10, 55, 20), index=1))

    result = NativePdfTableExtractor().extract(region, page, object())

    assert result.decision == "rejected"
    assert result.validation["failure_reasons"] == [
        "source_cell_geometry_is_unassigned_or_ambiguous"
    ]


def test_native_table_accepts_spans_and_projects_title_anchor_only() -> None:
    region = _region(
        _table_cell(0, 0, _Box(0, 0, 90, 30), col_span=3, column_header=False),
        _table_cell(1, 0, _Box(0, 30, 60, 50), col_span=2, column_header=True),
        _table_cell(1, 2, _Box(60, 30, 90, 50), column_header=True),
        _table_cell(2, 0, _Box(0, 50, 30, 70), column_header=True),
        _table_cell(2, 1, _Box(30, 50, 60, 70), column_header=True),
        _table_cell(2, 2, _Box(60, 50, 90, 70), column_header=True),
        _table_cell(3, 0, _Box(0, 70, 30, 100), column_header=False),
        _table_cell(3, 1, _Box(30, 70, 60, 100), column_header=False),
        _table_cell(3, 2, _Box(60, 70, 90, 100), column_header=False),
        rows=4,
        columns=3,
    )
    page = _page(
        _source("Requirements compliance matrix", _Box(1, 1, 89, 20), index=1),
        _source("Requirement detail", _Box(1, 31, 59, 45), index=2),
        _source("Status", _Box(61, 31, 89, 45), index=3),
        _source("Section", _Box(1, 51, 29, 65), index=4),
        _source("Description", _Box(31, 51, 59, 65), index=5),
        _source("Requirement", _Box(61, 51, 89, 65), index=6),
        _source("5.1", _Box(1, 71, 29, 85), index=7),
        _source("Control", _Box(31, 71, 59, 85), index=8),
        _source("Required", _Box(61, 71, 89, 85), index=9),
    )

    result = NativePdfTableExtractor().extract(region, page, object())

    assert result.decision == "accepted"
    assert result.cells[0]["column_span"] == 3
    assert result.validation["markdown_span_projection"] == "anchor_only"
    markdown = table_to_markdown(result)
    assert markdown.count("Requirements compliance matrix") == 1
    assert "| Section | Description | Requirement |" in markdown
    assert "| Requirements compliance matrix |" not in markdown
    assert markdown.index("Requirements compliance matrix") < markdown.index(
        "Requirement detail |  | Status"
    )
    assert markdown.index("Requirement detail |  | Status") < markdown.index(
        "| Section | Description | Requirement |"
    )


def test_native_table_conserves_interleaved_multiline_columns_by_source_ref() -> None:
    region = _region(
        _table_cell(0, 0, _Box(0, 0, 50, 100)),
        _table_cell(0, 1, _Box(50, 0, 100, 100)),
        rows=1,
        columns=2,
    )
    # Page Y/X order is Left 1, Right 1, Left 2, Right 2.  Grid order is
    # Left 1, Left 2, Right 1, Right 2; conservation must not reject this.
    page = _page(
        _source("Left 1", _Box(1, 10, 40, 20), index=1),
        _source("Right 1", _Box(55, 20, 95, 30), index=2),
        _source("Left 2", _Box(1, 40, 40, 50), index=3),
        _source("Right 2", _Box(55, 50, 95, 60), index=4),
    )

    result = NativePdfTableExtractor().extract(region, page, object())

    assert result.decision == "accepted"
    assert result.cells[0]["text"] == "Left 1 Left 2"
    assert result.cells[1]["text"] == "Right 1 Right 2"
    assert result.validation["text_conservation_ratio"] == 1.0
    assert result.validation["source_cell_refs_match"] is True


def test_markdown_preserves_middle_full_width_group_row_order() -> None:
    cells = [
        _table_cell(0, 0, _Box(0, 0, 30, 20)),
        _table_cell(0, 1, _Box(30, 0, 65, 20)),
        _table_cell(0, 2, _Box(65, 0, 100, 20)),
        _table_cell(1, 0, _Box(0, 20, 30, 45)),
        _table_cell(1, 1, _Box(30, 20, 65, 45)),
        _table_cell(1, 2, _Box(65, 20, 100, 45)),
        _table_cell(2, 0, _Box(0, 45, 100, 70), col_span=3),
        _table_cell(3, 0, _Box(0, 70, 30, 100)),
        _table_cell(3, 1, _Box(30, 70, 65, 100)),
        _table_cell(3, 2, _Box(65, 70, 100, 100)),
    ]
    region = _region(*cells, rows=4, columns=3)
    page = _page(
        _source("Section", _Box(1, 1, 25, 15)),
        _source("Description", _Box(31, 1, 60, 15)),
        _source("Requirement", _Box(66, 1, 95, 15)),
        _source("2.4.2", _Box(1, 25, 25, 40)),
        _source("Previous", _Box(31, 25, 60, 40)),
        _source("Required", _Box(66, 25, 95, 40)),
        _source("4. Requirements", _Box(1, 50, 95, 65)),
        _source("4.1", _Box(1, 75, 25, 90)),
        _source("Next", _Box(31, 75, 60, 90)),
        _source("Required", _Box(66, 75, 95, 90)),
    )

    result = NativePdfTableExtractor().extract(region, page, object())
    markdown = table_to_markdown(result)

    assert result.decision == "accepted"
    assert markdown.index("2.4.2") < markdown.index("4. Requirements")
    assert markdown.index("4. Requirements") < markdown.index("4.1")
    assert "| 4. Requirements |  |  |" in markdown


def test_native_table_rejects_incomplete_logical_grid() -> None:
    region = _region(
        _table_cell(0, 0, _Box(0, 0, 50, 100)), rows=1, columns=2
    )
    page = _page(_source("Left", _Box(1, 1, 40, 30), index=1))

    result = NativePdfTableExtractor().extract(region, page, object())

    assert result.decision == "rejected"
    assert result.validation["failure_reasons"] == [
        "logical_grid_has_uncovered_coordinates"
    ]


def test_headerless_table_uses_blank_markdown_header_without_promoting_data() -> None:
    region = _region(
        _table_cell(0, 0, _Box(0, 0, 50, 50), column_header=False),
        _table_cell(0, 1, _Box(50, 0, 100, 50), column_header=False),
        _table_cell(1, 0, _Box(0, 50, 50, 100), column_header=False),
        _table_cell(1, 1, _Box(50, 50, 100, 100), column_header=False),
    )
    page = _page(
        _source("A1", _Box(1, 1, 40, 20)),
        _source("B1", _Box(55, 1, 90, 20)),
        _source("A2", _Box(1, 55, 40, 75)),
        _source("B2", _Box(55, 55, 90, 75)),
    )

    result = NativePdfTableExtractor().extract(region, page, object())
    markdown = table_to_markdown(result)

    assert markdown.splitlines()[:2] == ["|  |  |", "| --- | --- |"]
    assert markdown.index("| A1 | B1 |") < markdown.index("| A2 | B2 |")


def test_late_header_flag_cannot_move_earlier_data_into_preamble() -> None:
    region = _region(
        _table_cell(0, 0, _Box(0, 0, 50, 50), column_header=False),
        _table_cell(0, 1, _Box(50, 0, 100, 50), column_header=False),
        _table_cell(1, 0, _Box(0, 50, 50, 100), column_header=True),
        _table_cell(1, 1, _Box(50, 50, 100, 100), column_header=True),
    )
    page = _page(
        _source("First data", _Box(1, 1, 40, 20)),
        _source("Value 1", _Box(55, 1, 90, 20)),
        _source("Misflagged", _Box(1, 55, 40, 75)),
        _source("Value 2", _Box(55, 55, 90, 75)),
    )

    markdown = table_to_markdown(
        NativePdfTableExtractor().extract(region, page, object())
    )

    assert markdown.splitlines()[:2] == ["|  |  |", "| --- | --- |"]
    assert markdown.index("First data") < markdown.index("Misflagged")


def test_native_table_accepts_docling_bottom_left_table_bboxes() -> None:
    region = _region(
        _table_cell(0, 0, _BottomLeftBox(0, 50, 50, 100)),
        _table_cell(0, 1, _BottomLeftBox(50, 50, 100, 100)),
        _table_cell(1, 0, _BottomLeftBox(0, 0, 50, 50)),
        _table_cell(1, 1, _BottomLeftBox(50, 0, 100, 50)),
    )
    page = _page(
        _source("Section", _Box(1, 1, 40, 20)),
        _source("Requirement", _Box(55, 1, 90, 20)),
        _source("A", _Box(1, 55, 20, 75)),
        _source("Required", _Box(55, 55, 95, 75)),
    )

    result = NativePdfTableExtractor().extract(region, page, object())

    assert result.decision == "accepted"
    assert result.cells[0]["text"] == "Section"


def test_table_summary_never_contains_cells() -> None:
    region = _region(_table_cell(0, 0, _Box(0, 0, 100, 100)), rows=1, columns=1)
    accepted = NativePdfTableExtractor().extract(
        region, _page(_source("A", _Box(1, 1, 20, 20), index=1)), object()
    )
    summary = table_summary([accepted])

    assert summary == {
        "detected": 1,
        "accepted_native": 1,
        "deferred_native": 0,
        "deferred_ocr": 0,
        "deferred_mixed": 0,
        "deferred_image_only": 0,
        "rejected_structure": 0,
    }
    assert "cells" not in summary
