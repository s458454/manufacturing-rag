"""Deterministic, auditable table extraction contracts for PDF preprocessing.

TableFormer provides only a proposed grid.  This module never accepts model
text: accepted cells are rebuilt exclusively from native PDF text cells and are
admitted only when every source cell has one unambiguous geometric owner.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


SourceKind = Literal["native", "ocr", "mixed", "image_only"]
TableDecision = Literal["accepted", "deferred", "rejected"]
TableExtractorName = Literal["native_pdf_table", "ocr_table"]


def normalize_table_text(value: Any) -> str:
    """Canonicalize text only for conservation checks and Markdown rendering."""

    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def bbox_to_topleft(value: Any, page_height: float | None) -> list[float] | None:
    """Convert Docling-like boxes to ``[left, top, right, bottom]``."""

    if value is None:
        return None
    converter = getattr(value, "to_bounding_box", None)
    if callable(converter):
        value = converter()
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    if isinstance(value, dict):
        left, top, right, bottom = (_number(value.get(key)) for key in ("l", "t", "r", "b"))
        origin = value.get("coord_origin", "")
    else:
        left, top, right, bottom = (
            _number(getattr(value, key, None)) for key in ("l", "t", "r", "b")
        )
        origin = getattr(value, "coord_origin", "")
    if None in (left, top, right, bottom):
        return None
    assert left is not None and top is not None and right is not None and bottom is not None
    if "bottomleft" in str(origin).lower().replace("_", "") and page_height is not None:
        top, bottom = page_height - top, page_height - bottom
    return [min(left, right), min(top, bottom), max(left, right), max(top, bottom)]


def _center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _contains(box: list[float], x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _int(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer == value or isinstance(value, int) else None


@dataclass(frozen=True)
class TableRegion:
    """One Docling table region, constrained to one original PDF page."""

    table_id: str
    document_id: str
    page_no: int
    bbox: list[float] | None
    section_hierarchy: list[str]
    caption: str | None
    docling_ref: str | None
    item: Any = field(repr=False, compare=False)
    source_cells: list[SourceTextCell] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class SourceTextCell:
    source_ref: str
    text: str
    bbox: list[float]
    from_ocr: bool


@dataclass
class TableExtractionResult:
    """Canonical schema shared by the native and future OCR extractors."""

    table_id: str
    document_id: str
    page_no: int
    bbox: list[float] | None
    section_hierarchy: list[str]
    caption: str | None
    docling_ref: str | None
    source_kind: SourceKind
    extractor: TableExtractorName
    decision: TableDecision
    row_count: int
    column_count: int
    cells: list[dict[str, Any]]
    validation: dict[str, Any]
    footnotes: list[dict[str, Any]] = field(default_factory=list)
    continuation_group_id: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "document_id": self.document_id,
            "page_no": self.page_no,
            "bbox": self.bbox,
            "section_hierarchy": self.section_hierarchy,
            "caption": self.caption,
            "docling_ref": self.docling_ref,
            "source_kind": self.source_kind,
            "extractor": self.extractor,
            "decision": self.decision,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "cells": self.cells,
            "validation": self.validation,
            "footnotes": self.footnotes,
            "continuation_group_id": self.continuation_group_id,
        }


class TableExtractor(Protocol):
    def extract(
        self, table_region: TableRegion, page: Any, document: Any
    ) -> TableExtractionResult: ...


def source_cells_in_region(table_region: TableRegion, page: Any) -> list[SourceTextCell]:
    """Collect real page text cells whose centres fall inside the table bbox."""

    if table_region.source_cells is not None:
        return list(table_region.source_cells)
    if table_region.bbox is None:
        return []
    page_height = _number(getattr(getattr(page, "size", None), "height", None))
    result: list[SourceTextCell] = []
    for ordinal, cell in enumerate(getattr(page, "cells", []) or []):
        text = normalize_table_text(getattr(cell, "text", None))
        rect = getattr(cell, "rect", None)
        bbox = bbox_to_topleft(rect, page_height)
        if not text or bbox is None:
            continue
        center_x, center_y = _center(bbox)
        if not _contains(table_region.bbox, center_x, center_y):
            continue
        source_ref = f"page/{table_region.page_no}/cells/{ordinal}"
        result.append(
            SourceTextCell(
                source_ref=source_ref,
                text=text,
                bbox=bbox,
                from_ocr=bool(getattr(cell, "from_ocr", False)),
            )
        )
    return result


def classify_source_kind(cells: list[SourceTextCell]) -> tuple[SourceKind, int, int]:
    native_chars = sum(len(cell.text) for cell in cells if not cell.from_ocr)
    ocr_chars = sum(len(cell.text) for cell in cells if cell.from_ocr)
    if native_chars and not ocr_chars:
        return "native", native_chars, ocr_chars
    if ocr_chars and not native_chars:
        return "ocr", native_chars, ocr_chars
    if native_chars and ocr_chars:
        return "mixed", native_chars, ocr_chars
    return "image_only", native_chars, ocr_chars


def _base_validation(
    source_cells: list[SourceTextCell], native_chars: int, ocr_chars: int
) -> dict[str, Any]:
    return {
        "native_character_count": native_chars,
        "ocr_character_count": ocr_chars,
        "source_text_cell_count": len(source_cells),
        "native_source_cell_count": sum(not cell.from_ocr for cell in source_cells),
        "ocr_source_cell_count": sum(cell.from_ocr for cell in source_cells),
        "source_cells": [
            {
                "source_cell_ref": cell.source_ref,
                "text": cell.text,
                "bbox": cell.bbox,
                "from_ocr": cell.from_ocr,
            }
            for cell in source_cells
        ],
        "text_normalization": "NFKC + collapsed whitespace",
        "failure_reasons": [],
    }


class OcrTableExtractor:
    """Reserved parallel extractor. V0.1 deliberately does not invent OCR tables."""

    def extract(
        self, table_region: TableRegion, page: Any, document: Any
    ) -> TableExtractionResult:
        source_cells = source_cells_in_region(table_region, page)
        source_kind, native_chars, ocr_chars = classify_source_kind(source_cells)
        validation = _base_validation(source_cells, native_chars, ocr_chars)
        validation["failure_reasons"] = [
            "ocr_table_extractor_not_implemented_in_v0_1"
        ]
        return TableExtractionResult(
            table_id=table_region.table_id,
            document_id=table_region.document_id,
            page_no=table_region.page_no,
            bbox=table_region.bbox,
            section_hierarchy=table_region.section_hierarchy,
            caption=table_region.caption,
            docling_ref=table_region.docling_ref,
            source_kind=source_kind,
            extractor="ocr_table",
            decision="deferred",
            row_count=0,
            column_count=0,
            cells=[],
            validation=validation,
        )


class NativePdfTableExtractor:
    """Admit only native tables with complete source-text conservation."""

    extractor_name: TableExtractorName = "native_pdf_table"

    def _result(
        self,
        region: TableRegion,
        source_kind: SourceKind,
        validation: dict[str, Any],
        *,
        decision: TableDecision,
        row_count: int = 0,
        column_count: int = 0,
        cells: list[dict[str, Any]] | None = None,
    ) -> TableExtractionResult:
        return TableExtractionResult(
            table_id=region.table_id,
            document_id=region.document_id,
            page_no=region.page_no,
            bbox=region.bbox,
            section_hierarchy=region.section_hierarchy,
            caption=region.caption,
            docling_ref=region.docling_ref,
            source_kind=source_kind,
            extractor=self.extractor_name,
            decision=decision,
            row_count=row_count,
            column_count=column_count,
            cells=cells or [],
            validation=validation,
        )

    def extract(
        self, table_region: TableRegion, page: Any, document: Any
    ) -> TableExtractionResult:
        source_cells = source_cells_in_region(table_region, page)
        source_kind, native_chars, ocr_chars = classify_source_kind(source_cells)
        validation = _base_validation(source_cells, native_chars, ocr_chars)
        if source_kind != "native":
            validation["failure_reasons"] = ["source_kind_is_not_native"]
            return self._result(
                table_region, source_kind, validation, decision="deferred"
            )

        data = getattr(table_region.item, "data", None)
        row_count = _int(getattr(data, "num_rows", None))
        column_count = _int(getattr(data, "num_cols", None))
        table_cells = list(getattr(data, "table_cells", []) or [])
        if row_count is None or column_count is None or row_count <= 0 or column_count <= 0:
            validation["failure_reasons"] = ["invalid_row_or_column_count"]
            return self._result(table_region, source_kind, validation, decision="rejected")
        if not table_cells:
            validation["failure_reasons"] = ["missing_tableformer_cells"]
            return self._result(
                table_region,
                source_kind,
                validation,
                decision="rejected",
                row_count=row_count,
                column_count=column_count,
            )

        page_height = _number(getattr(getattr(page, "size", None), "height", None))
        candidates: list[dict[str, Any]] = []
        occupied: set[tuple[int, int]] = set()
        reasons: list[str] = []
        for index, cell in enumerate(table_cells):
            row_start = _int(getattr(cell, "start_row_offset_idx", None))
            column_start = _int(getattr(cell, "start_col_offset_idx", None))
            row_end = _int(getattr(cell, "end_row_offset_idx", None))
            column_end = _int(getattr(cell, "end_col_offset_idx", None))
            row_span = _int(getattr(cell, "row_span", None))
            column_span = _int(getattr(cell, "col_span", None))
            if row_start is None or column_start is None:
                reasons.append(f"cell_{index}_missing_grid_start")
                continue
            if row_end is None:
                row_end = row_start + (row_span or 1)
            if column_end is None:
                column_end = column_start + (column_span or 1)
            row_span = row_end - row_start
            column_span = column_end - column_start
            bbox = bbox_to_topleft(getattr(cell, "bbox", None), page_height)
            if (
                row_span <= 0
                or column_span <= 0
                or row_start < 0
                or column_start < 0
                or row_end > row_count
                or column_end > column_count
            ):
                reasons.append(f"cell_{index}_invalid_grid_span")
                continue
            if bbox is None or bbox[0] == bbox[2] or bbox[1] == bbox[3]:
                reasons.append(f"cell_{index}_missing_or_empty_bbox")
                continue
            for row in range(row_start, row_end):
                for column in range(column_start, column_end):
                    coordinate = (row, column)
                    if coordinate in occupied:
                        reasons.append(f"cell_{index}_overlaps_logical_grid")
                    occupied.add(coordinate)
            candidates.append(
                {
                    "row_start": row_start,
                    "row_span": row_span,
                    "column_start": column_start,
                    "column_span": column_span,
                    # These are structural predictions, not inferred from row
                    # position.  Carry them into the canonical artifact so the
                    # Markdown projection never promotes a data row to a
                    # header merely because it happens to be first.
                    "column_header": bool(getattr(cell, "column_header", False)),
                    "row_header": bool(getattr(cell, "row_header", False)),
                    "bbox": bbox,
                    "sources": [],
                }
            )

        if reasons:
            validation["failure_reasons"] = sorted(set(reasons))
            validation["candidate_cell_count"] = len(candidates)
            return self._result(
                table_region,
                source_kind,
                validation,
                decision="rejected",
                row_count=row_count,
                column_count=column_count,
            )

        # A Markdown grid cannot safely imply cells which TableFormer did not
        # emit.  Empty *text* cells are fine, but every logical coordinate has
        # to be represented by exactly one proposed structural cell.
        expected_coordinates = {
            (row, column)
            for row in range(row_count)
            for column in range(column_count)
        }
        missing_coordinates = expected_coordinates - occupied
        if missing_coordinates:
            validation["failure_reasons"] = [
                "logical_grid_has_uncovered_coordinates"
            ]
            validation["uncovered_logical_coordinates"] = [
                {"row": row, "column": column}
                for row, column in sorted(missing_coordinates)
            ]
            validation["candidate_cell_count"] = len(candidates)
            return self._result(
                table_region,
                source_kind,
                validation,
                decision="rejected",
                row_count=row_count,
                column_count=column_count,
            )

        for source in source_cells:
            center_x, center_y = _center(source.bbox)
            owners = [candidate for candidate in candidates if _contains(candidate["bbox"], center_x, center_y)]
            if len(owners) != 1:
                validation["failure_reasons"] = [
                    "source_cell_geometry_is_unassigned_or_ambiguous"
                ]
                validation["geometry_problem_source_refs"] = [source.source_ref]
                return self._result(
                    table_region,
                    source_kind,
                    validation,
                    decision="rejected",
                    row_count=row_count,
                    column_count=column_count,
                )
            owners[0]["sources"].append(source)

        output_cells: list[dict[str, Any]] = []
        assigned_source_refs: list[str] = []
        for candidate in sorted(
            candidates, key=lambda cell: (cell["row_start"], cell["column_start"])
        ):
            sources = sorted(
                candidate["sources"],
                key=lambda source: (source.bbox[1], source.bbox[0], source.source_ref),
            )
            text = " ".join(source.text for source in sources)
            assigned_source_refs.extend(source.source_ref for source in sources)
            output_cells.append(
                {
                    "row_start": candidate["row_start"],
                    "row_span": candidate["row_span"],
                    "column_start": candidate["column_start"],
                    "column_span": candidate["column_span"],
                    "column_header": candidate["column_header"],
                    "row_header": candidate["row_header"],
                    "text": text,
                    "source_cell_refs": [source.source_ref for source in sources],
                    "bbox": candidate["bbox"],
                }
            )

        # Do not compare page reading order with grid order here.  In a table
        # containing two multi-line columns, page Y/X order interleaves lines
        # from the columns even when no text was lost.  Conservation is instead
        # defined by the source-cell provenance contract: every native source
        # reference occurs exactly once and every output cell is reconstructed
        # deterministically from its own references.
        source_by_ref = {source.source_ref: source for source in source_cells}
        source_ref_set = set(source_by_ref)
        assigned_ref_set = set(assigned_source_refs)
        one_to_one = (
            len(source_by_ref) == len(source_cells)
            and len(assigned_source_refs) == len(source_cells)
            and len(assigned_ref_set) == len(source_cells)
            and assigned_ref_set == source_ref_set
        )
        reconstructed_cells = True
        for cell in output_cells:
            refs = cell["source_cell_refs"]
            if any(ref not in source_by_ref for ref in refs):
                reconstructed_cells = False
                break
            expected_text = " ".join(source_by_ref[ref].text for ref in refs)
            if cell["text"] != expected_text:
                reconstructed_cells = False
                break
        uniquely_assigned_native_chars = sum(
            len(source_by_ref[ref].text)
            for ref in assigned_ref_set
            if ref in source_by_ref and not source_by_ref[ref].from_ocr
        )
        conservation_ratio = (
            uniquely_assigned_native_chars / native_chars if native_chars else 1.0
        )
        has_spans = any(
            cell["row_span"] != 1 or cell["column_span"] != 1
            for cell in output_cells
        )
        validation.update(
            {
                "candidate_cell_count": len(candidates),
                "assigned_native_source_cell_count": len(assigned_source_refs),
                "unique_assigned_native_source_cell_count": len(assigned_ref_set),
                "source_cell_refs_match": assigned_ref_set == source_ref_set,
                "output_cells_deterministically_rebuilt": reconstructed_cells,
                "text_conservation_ratio": conservation_ratio,
                "structure_model_text_used": False,
                "markdown_span_projection": (
                    "anchor_only" if has_spans else "not_required"
                ),
            }
        )
        if not one_to_one:
            validation["failure_reasons"] = ["source_cell_assignment_is_not_one_to_one"]
            return self._result(
                table_region,
                source_kind,
                validation,
                decision="rejected",
                row_count=row_count,
                column_count=column_count,
            )
        if not reconstructed_cells:
            validation["failure_reasons"] = [
                "output_cell_text_is_not_deterministically_rebuilt"
            ]
            return self._result(
                table_region,
                source_kind,
                validation,
                decision="rejected",
                row_count=row_count,
                column_count=column_count,
            )
        if conservation_ratio != 1.0:
            validation["failure_reasons"] = ["native_text_conservation_failed"]
            return self._result(
                table_region,
                source_kind,
                validation,
                decision="rejected",
                row_count=row_count,
                column_count=column_count,
            )

        validation["failure_reasons"] = []
        return self._result(
            table_region,
            source_kind,
            validation,
            decision="accepted",
            row_count=row_count,
            column_count=column_count,
            cells=output_cells,
        )


def table_to_markdown(result: TableExtractionResult) -> str:
    """Render an accepted table with a deterministic, span-safe pipe projection.

    Canonical JSON retains TableFormer's true spans and explicit header flags.
    Markdown has no span primitive, so text is emitted only at a span's
    top-left anchor and all covered logical coordinates remain empty. Rows
    preceding the final explicit column-header row are emitted as ordered
    preamble text. If no explicit complete column header exists, a blank
    synthetic Markdown header is used and every real row remains body data.
    """

    if result.decision != "accepted" or result.row_count < 1 or result.column_count < 1:
        return ""
    grid = [["" for _column in range(result.column_count)] for _row in range(result.row_count)]
    owners: list[list[dict[str, Any] | None]] = [
        [None for _column in range(result.column_count)]
        for _row in range(result.row_count)
    ]
    for cell in result.cells:
        row = cell.get("row_start")
        column = cell.get("column_start")
        if (
            not isinstance(row, int)
            or not isinstance(column, int)
            or row < 0
            or row >= result.row_count
            or column < 0
            or column >= result.column_count
        ):
            return ""
        text = normalize_table_text(cell.get("text"))
        row_span = cell.get("row_span", 1)
        column_span = cell.get("column_span", 1)
        if (
            not isinstance(row_span, int)
            or not isinstance(column_span, int)
            or row_span <= 0
            or column_span <= 0
            or row + row_span > result.row_count
            or column + column_span > result.column_count
        ):
            return ""
        for logical_row in range(row, row + row_span):
            for logical_column in range(column, column + column_span):
                if owners[logical_row][logical_column] is not None:
                    return ""
                owners[logical_row][logical_column] = cell
        # Anchor-only projection deliberately leaves the covered cells empty.
        grid[row][column] = text.replace("|", "\\|")

    if any(owner is None for row in owners for owner in row):
        return ""
    all_rows = list(range(result.row_count))

    def is_explicit_complete_header(row: int) -> bool:
        """Require TableFormer's explicit flag across the full logical row."""

        return all(
            owners[row][column] is not None
            and owners[row][column].get("column_header") is True
            for column in range(result.column_count)
        )

    def is_structural_preamble(row: int) -> bool:
        anchors = list({
            id(owner): owner
            for owner in owners[row]
            if owner is not None and owner.get("row_start") == row
        }.values())
        return bool(anchors) and any(cell.get("column_span", 1) > 1 for cell in anchors)

    header_start = next(
        (
            row
            for row in all_rows
            if is_explicit_complete_header(row)
            and all(is_structural_preamble(previous) for previous in all_rows[:row])
        ),
        None,
    )
    explicit_header_band: list[int] = []
    if header_start is not None:
        for row in all_rows[header_start:]:
            if not is_explicit_complete_header(row):
                break
            explicit_header_band.append(row)
    header_row = explicit_header_band[-1] if explicit_header_band else None

    # Markdown requires its header before its body.  Preserve every earlier
    # title/grouped-heading row in source order as a plain preamble rather than
    # moving it below the leaf header.  Keeping separators but omitting leading
    # and trailing pipes avoids accidentally starting another Markdown table.
    preamble_rows = all_rows[:header_row] if header_row is not None else []
    preamble_lines = [" | ".join(grid[row]).rstrip() for row in preamble_rows]
    preamble_lines = [line for line in preamble_lines if line.strip(" |")]

    markdown_header = (
        grid[header_row]
        if header_row is not None
        else ["" for _column in range(result.column_count)]
    )
    body_rows = (
        all_rows[header_row + 1 :]
        if header_row is not None
        else all_rows
    )
    lines = ["| " + " | ".join(markdown_header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(result.column_count)) + " |")
    lines.extend("| " + " | ".join(grid[row]) + " |" for row in body_rows)
    blocks = ["\n".join(lines)]
    if preamble_lines:
        blocks.insert(0, "\n\n".join(preamble_lines))
    return "\n\n".join(blocks)


def table_summary(results: list[TableExtractionResult]) -> dict[str, int]:
    return {
        "detected": len(results),
        "accepted_native": sum(
            result.decision == "accepted" and result.source_kind == "native"
            for result in results
        ),
        "deferred_native": sum(
            result.decision == "deferred" and result.source_kind == "native"
            for result in results
        ),
        "deferred_ocr": sum(
            result.decision == "deferred" and result.source_kind == "ocr"
            for result in results
        ),
        "deferred_mixed": sum(
            result.decision == "deferred" and result.source_kind == "mixed"
            for result in results
        ),
        "deferred_image_only": sum(
            result.decision == "deferred" and result.source_kind == "image_only"
            for result in results
        ),
        "rejected_structure": sum(result.decision == "rejected" for result in results),
    }
