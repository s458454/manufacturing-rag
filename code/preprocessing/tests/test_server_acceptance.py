"""Regression tests for the pinned NASA-STD-5006A Golden acceptance gate."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from verify_pdf_preprocess_server import (
    _validate_visual_body_isolation,
    validate_5006a_golden_artifacts,
)
from table_extraction import TableExtractionResult, table_to_markdown


def _summary(*, accepted: int, deferred_image_only: int = 0) -> dict[str, int]:
    return {
        "detected": accepted + deferred_image_only,
        "accepted_native": accepted,
        "deferred_native": 0,
        "deferred_ocr": 0,
        "deferred_mixed": 0,
        "deferred_image_only": deferred_image_only,
        "rejected_structure": 0,
    }


def _result_from_record(record: dict[str, object]) -> TableExtractionResult:
    return TableExtractionResult(
        table_id=record["table_id"],
        document_id=record["document_id"],
        page_no=record["page_no"],
        bbox=record.get("bbox"),
        section_hierarchy=record["section_hierarchy"],
        caption=record.get("caption"),
        docling_ref=record.get("docling_ref"),
        source_kind=record["source_kind"],
        extractor=record["extractor"],
        decision=record["decision"],
        row_count=record["row_count"],
        column_count=record["column_count"],
        cells=record["cells"],
        validation=record["validation"],
        footnotes=record.get("footnotes", []),
    )


def _accepted_record(table_id: str, page_no: int) -> dict[str, object]:
    rows = [
        ["Section", "Description", "Requirement in this Standard", "GWR"],
        ["2.4.2", "Previous", "Required", "Yes"],
        ["4. Requirements"],
        ["4.1", "Next", "Required", "Yes"],
    ] if page_no == 36 else [
        ["Section", "Description", "Requirement in this Standard", "GWR"],
        ["4.1", "Next", "Required", "Yes"],
    ]
    cells: list[dict[str, object]] = []
    source_cells: list[dict[str, object]] = []
    source_ordinal = 0
    for row_index, values in enumerate(rows):
        if len(values) == 1:
            placements = [(0, 4, values[0])]
        else:
            placements = [
                (column_index, 1, text)
                for column_index, text in enumerate(values)
            ]
        for column_index, column_span, text in placements:
            source_ref = f"page/{page_no}/cells/{source_ordinal}"
            source_ordinal += 1
            cells.append(
                {
                    "row_start": row_index,
                    "row_span": 1,
                    "column_start": column_index,
                    "column_span": column_span,
                    "column_header": row_index == 0,
                    "row_header": False,
                    "text": text,
                    "source_cell_refs": [source_ref],
                }
            )
            source_cells.append(
                {
                    "source_cell_ref": source_ref,
                    "text": text,
                    "bbox": [0, 0, 1, 1],
                    "from_ocr": False,
                }
            )
    return {
        "table_id": table_id,
        "document_id": "golden",
        "page_no": page_no,
        "bbox": [0, 0, 1, 1],
        "section_hierarchy": ["Requirements Compliance Matrix"],
        "caption": None,
        "docling_ref": f"#/tables/{page_no}",
        "source_kind": "native",
        "extractor": "native_pdf_table",
        "decision": "accepted",
        "row_count": len(rows),
        "column_count": 4,
        "cells": cells,
        "validation": {
            "text_conservation_ratio": 1.0,
            "output_cells_deterministically_rebuilt": True,
            "source_cells": source_cells,
            "markdown_span_projection": (
                "anchor_only" if page_no == 36 else "not_required"
            ),
        },
        "footnotes": [],
        "continuation_group_id": None,
    }


def _golden_fixture(tmp_path: Path) -> tuple[
    Path, dict[str, object], list[dict[str, object]], dict[str, object], str
]:
    artifact_dir = tmp_path / "artifact"
    table_dir = artifact_dir / "tables"
    table_dir.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    page_blocks: dict[int, list[str]] = {page_no: [] for page_no in range(1, 49)}
    for page_no in range(36, 49):
        table_id = f"golden-p{page_no:04d}-t001"
        record = _accepted_record(table_id, page_no)
        artifact = f"tables/{table_id}.json"
        (artifact_dir / artifact).write_text(
            json.dumps(record), encoding="utf-8"
        )
        entries.append(
            {
                "table_id": table_id,
                "page_no": page_no,
                "source_kind": "native",
                "extractor": "native_pdf_table",
                "decision": "accepted",
                "artifact": artifact,
            }
        )
        rows = table_to_markdown(_result_from_record(record))
        page_blocks[page_no].append(
            f"<!-- TABLE id={table_id} page={page_no} source=native -->\n\n"
            f"{rows}\n\n<!-- /TABLE -->"
        )
    markdown = "\n\n".join(
        f"<!-- PDF page {page_no} -->"
        + (
            "\n\n" + "\n\n".join(page_blocks[page_no])
            if page_blocks[page_no]
            else ""
        )
        for page_no in range(1, 49)
    )
    quality: dict[str, object] = {
        "pages": [
            {"page_no": page_no, "eligible_for_indexing": True}
            for page_no in range(1, 49)
        ]
    }
    regions: list[dict[str, object]] = [
        {
            "page_no": 17,
            "region_type": "table",
            "canonical_table_in_semantic_markdown": False,
        },
        *[
            {
                "page_no": page_no,
                "region_type": "picture",
                "visual_body_in_semantic_markdown": False,
            }
            for page_no in (20, 22, 25)
        ],
    ]
    table_index: dict[str, object] = {
        "tables": entries,
        "summary": _summary(accepted=13),
    }
    return artifact_dir, quality, regions, table_index, markdown


def test_5006a_golden_requires_all_pages_and_every_matrix_page(
    tmp_path: Path,
) -> None:
    fixture = _golden_fixture(tmp_path)

    evidence = validate_5006a_golden_artifacts(*fixture)

    assert evidence["all_pages_trusted"] is True
    assert evidence["matrix_pages_with_accepted_native_table"] == list(range(36, 49))
    assert evidence["page36_row_order"] == "2.4.2 < 4. Requirements < 4.1"


def test_5006a_golden_rejects_one_untrusted_page(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    quality["pages"][6]["eligible_for_indexing"] = False

    with pytest.raises(RuntimeError, match="every page to be trusted"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_missing_matrix_page(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    table_index["tables"] = [
        entry for entry in table_index["tables"] if entry["page_no"] != 48
    ]
    table_index["summary"] = _summary(accepted=12)
    markdown = re.sub(
        r"<!-- TABLE id=golden-p0048-t001 .*?<!-- /TABLE -->",
        "",
        markdown,
        flags=re.S,
    )

    with pytest.raises(RuntimeError, match="every matrix page"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_page36_row_reordering(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    entry = next(item for item in table_index["tables"] if item["page_no"] == 36)
    path = artifact_dir / entry["artifact"]
    record = json.loads(path.read_text(encoding="utf-8"))
    for cell in record["cells"]:
        if cell["row_start"] == 1:
            cell["row_start"] = 2
        elif cell["row_start"] == 2:
            cell["row_start"] = 1
    old_block = table_to_markdown(_result_from_record(_accepted_record(entry["table_id"], 36)))
    new_block = table_to_markdown(_result_from_record(record))
    path.write_text(json.dumps(record), encoding="utf-8")
    markdown = markdown.replace(old_block, new_block)

    with pytest.raises(RuntimeError, match="row order"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )
def test_5006a_golden_rejects_json_markdown_content_mismatch(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    markdown = markdown.replace(
        "| 4.1 | Next | Required | Yes |",
        "| 4.1 | Wrong | Required | Yes |",
        1,
    )

    with pytest.raises(RuntimeError, match="JSON and Markdown"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_table_under_wrong_page_marker(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    match = re.search(
        r"<!-- TABLE id=golden-p0036-t001 .*?<!-- /TABLE -->",
        markdown,
        flags=re.S,
    )
    assert match is not None
    block = match.group(0)
    markdown = markdown[: match.start()] + markdown[match.end() :]
    markdown = markdown.replace("<!-- PDF page 35 -->", f"<!-- PDF page 35 -->\n\n{block}")

    with pytest.raises(RuntimeError, match="declared PDF page"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_duplicate_page_marker(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    markdown = markdown.replace(
        "<!-- PDF page 10 -->",
        "<!-- PDF page 10 -->\n\n<!-- PDF page 10 -->",
    )

    with pytest.raises(RuntimeError, match="exactly once"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_page17_native_canonical_table(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    table_id = "golden-p0017-t001"
    record = _accepted_record(table_id, 17)
    artifact = f"tables/{table_id}.json"
    (artifact_dir / artifact).write_text(json.dumps(record), encoding="utf-8")
    table_index["tables"].append(
        {
            "table_id": table_id,
            "page_no": 17,
            "source_kind": "native",
            "extractor": "native_pdf_table",
            "decision": "accepted",
            "artifact": artifact,
        }
    )
    table_index["summary"] = _summary(accepted=14)
    block = table_to_markdown(_result_from_record(record))
    markdown = markdown.replace(
        "<!-- PDF page 17 -->",
        f"<!-- PDF page 17 -->\n\n"
        f"<!-- TABLE id={table_id} page=17 source=native -->\n\n"
        f"{block}\n\n<!-- /TABLE -->",
    )

    with pytest.raises(RuntimeError, match="page 17 image table"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_picture_body_leak_flag(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    picture = next(region for region in regions if region["page_no"] == 20)
    picture["visual_body_in_semantic_markdown"] = True

    with pytest.raises(RuntimeError, match="page 20 picture body isolation"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_markdown_picture_body(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    markdown = markdown.replace(
        "<!-- PDF page 20 -->",
        "<!-- PDF page 20 -->\n\n![visual body](page20.png)",
    )

    with pytest.raises(RuntimeError, match="Markdown picture body"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_source_ref_nonconservation(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    entry = next(item for item in table_index["tables"] if item["page_no"] == 36)
    path = artifact_dir / entry["artifact"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["cells"][0]["source_cell_refs"].append(
        record["cells"][0]["source_cell_refs"][0]
    )
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="one-to-one reconstruction"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_5006a_golden_rejects_deferred_table_body_leak(tmp_path: Path) -> None:
    artifact_dir, quality, regions, table_index, markdown = _golden_fixture(tmp_path)
    table_id = "golden-p0010-t001"
    record = {
        "table_id": table_id,
        "document_id": "golden",
        "page_no": 10,
        "bbox": None,
        "section_hierarchy": [],
        "caption": None,
        "docling_ref": "#/tables/deferred",
        "source_kind": "image_only",
        "extractor": "ocr_table",
        "decision": "deferred",
        "row_count": 0,
        "column_count": 0,
        "cells": [],
        "validation": {},
        "footnotes": [],
    }
    artifact = f"tables/{table_id}.json"
    (artifact_dir / artifact).write_text(json.dumps(record), encoding="utf-8")
    table_index["tables"].append(
        {
            "table_id": table_id,
            "page_no": 10,
            "source_kind": "image_only",
            "extractor": "ocr_table",
            "decision": "deferred",
            "artifact": artifact,
        }
    )
    table_index["summary"] = _summary(accepted=13, deferred_image_only=1)
    markdown += (
        f"\n\n<!-- TABLE id={table_id} page=10 source=native -->\n\n"
        "leaked\n\n<!-- /TABLE -->"
    )

    with pytest.raises(RuntimeError, match="Deferred/rejected"):
        validate_5006a_golden_artifacts(
            artifact_dir, quality, regions, table_index, markdown
        )


def test_visual_body_sentinel_rejects_picture_ocr_but_protects_caption(
    tmp_path: Path,
) -> None:
    del tmp_path
    document = {
        "pictures": [
            {
                "self_ref": "#/pictures/0",
                "children": [{"$ref": "#/texts/1"}],
                "captions": [{"$ref": "#/texts/0"}],
                "footnotes": [],
            }
        ],
        "texts": [
            {
                "self_ref": "#/texts/0",
                "text": "Figure 1. Trusted caption",
                "children": [],
            },
            {
                "self_ref": "#/texts/1",
                "text": "UNIQUE VISUAL OCR BODY 8739-X",
                "children": [],
            },
        ],
    }
    regions = [
        {
            "region_id": "#/pictures/0",
            "docling_ref": "#/pictures/0",
            "region_type": "picture",
        }
    ]
    caption_only = "<!-- PDF page 1 -->\n\nFigure 1. Trusted caption"
    assert _validate_visual_body_isolation(
        document, regions, {}, caption_only, {1}
    ) == 1

    with pytest.raises(RuntimeError, match="body text leaked"):
        _validate_visual_body_isolation(
            document,
            regions,
            {},
            caption_only + "\n\nUNIQUE VISUAL OCR BODY 8739-X",
            {1},
        )
