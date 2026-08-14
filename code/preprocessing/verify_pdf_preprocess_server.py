"""One-command server acceptance for the real preprocessing dependency path.

This script intentionally runs without pytest. It validates model integrity,
constructs Docling's real RapidOCR 3.9.2 wrapper with project models, proves
CUDA-preferred Det/Cls/Rec sessions, executes a real OCR inference on a PDF
page rendering, and can then launch the complete preprocessing CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DOCLING_SOURCE = PROJECT_ROOT / "docling"
for directory in (SCRIPT_DIR, DOCLING_SOURCE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from table_extraction import (  # noqa: E402 - local path is established above
    TableExtractionResult,
    normalize_table_text,
    table_summary,
    table_to_markdown,
)

GOLDEN_5006A_SHA256 = "fa229965758a0f0c630034084173341e2a0053a1ca25d35a15b6d14e9b8e5c20"
GOLDEN_5006A_PAGE_COUNT = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-model server acceptance without requiring pytest."
    )
    parser.add_argument("--input-pdf", type=Path, required=True)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument(
        "--document-timeout",
        type=float,
        default=7200.0,
        help="Maximum Docling processing time passed to the preprocessing CLI.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "server-acceptance",
    )
    parser.add_argument(
        "--skip-end-to-end",
        action="store_true",
        help="Only validate real OCR construction/inference; do not run the CLI.",
    )
    parser.add_argument(
        "--allow-quality-rejection",
        action="store_true",
        help=(
            "Diagnostic only: allow preprocessing exit code 3 after proving that "
            "the rejected artifacts were isolated under .failed. Partial-success "
            "exit code 1 is never accepted as deployment success."
        ),
    )
    parser.add_argument(
        "--golden-5006a",
        action="store_true",
        help=(
            "Run the complete pinned 48-page NASA-STD-5006A Golden PDF and "
            "enforce its page, table, picture, row-order, and provenance gates."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_asset_fingerprint(source: Path) -> dict[str, Any]:
    from pypdf import PdfReader

    reader = PdfReader(str(source))
    if reader.is_encrypted:
        raise RuntimeError("Encrypted PDFs are not supported by server acceptance")
    return {
        "source_sha256": sha256_file(source),
        "source_size_bytes": source.stat().st_size,
        "source_page_count": len(reader.pages),
    }


def normalized_markdown_evidence(value: str) -> str:
    """Normalize plain caption text and Markdown for containment checks.

    Docling may escape punctuation that has Markdown meaning.  Removing only
    those escape backslashes and folding Unicode/whitespace preserves the
    caption's actual text while avoiding a false failure caused by formatting.
    """
    normalized = unicodedata.normalize("NFKC", value)
    normalized = re.sub(r"\\([\\`*{}\[\]()#+\-.!_|>~])", r"\1", normalized)
    return " ".join(normalized.split()).casefold()


def render_page(source: Path, page_no: int) -> Any:
    import numpy as np
    import pypdfium2 as pdfium

    document: Any | None = None
    page: Any | None = None
    bitmap: Any | None = None
    try:
        document = pdfium.PdfDocument(str(source))
        if not 1 <= page_no <= len(document):
            raise ValueError(f"--page must be within [1, {len(document)}]")
        page = document.get_page(page_no - 1)
        bitmap = page.render(scale=216 / 72, rev_byteorder=True)
        return np.asarray(bitmap.to_pil().convert("RGB").copy())
    finally:
        for value in (bitmap, page, document):
            close = getattr(value, "close", None)
            if callable(close):
                close()


def create_real_ocr(device: str, num_threads: int) -> Any:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import RapidOcrOptions
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
    from pdf_preprocess import MODEL_PATHS

    options = RapidOcrOptions(
        lang=["english"],
        backend="onnxruntime",
        use_det=True,
        use_cls=True,
        use_rec=True,
        det_model_path=str(MODEL_PATHS["det"]),
        cls_model_path=str(MODEL_PATHS["cls"]),
        rec_model_path=str(MODEL_PATHS["rec"]),
        font_path=str(MODEL_PATHS["font"]),
    )
    return RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(
            device=device,
            num_threads=num_threads,
        ),
    )


def provider_payload(model: Any) -> dict[str, list[str]]:
    return {
        stage: list(providers)
        for stage, providers in model._provider_audit.items()
    }


def _table_block(markdown: str, table_id: str) -> str | None:
    match = re.search(
        rf"<!-- TABLE id={re.escape(table_id)} .*?-->\s*(.*?)\s*<!-- /TABLE -->",
        markdown,
        flags=re.S,
    )
    return match.group(1) if match is not None else None


def _page_markdown(markdown: str, page_no: int) -> str:
    match = re.search(
        rf"<!-- PDF page {page_no} -->\s*(.*?)(?=<!-- PDF page \d+ -->|\Z)",
        markdown,
        flags=re.S,
    )
    return match.group(1) if match is not None else ""


def _record_to_result(record: dict[str, Any]) -> TableExtractionResult:
    """Reconstruct the canonical object and fail on an incomplete artifact."""

    required = {
        "table_id",
        "document_id",
        "page_no",
        "section_hierarchy",
        "source_kind",
        "extractor",
        "decision",
        "row_count",
        "column_count",
        "cells",
        "validation",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError(
            f"Canonical table artifact is missing required fields {missing}: "
            f"{record.get('table_id')!r}"
        )
    if (
        not isinstance(record.get("table_id"), str)
        or not isinstance(record.get("document_id"), str)
        or not isinstance(record.get("page_no"), int)
        or not isinstance(record.get("row_count"), int)
        or not isinstance(record.get("column_count"), int)
        or not isinstance(record.get("section_hierarchy"), list)
        or not isinstance(record.get("cells"), list)
        or any(not isinstance(cell, dict) for cell in record["cells"])
        or not isinstance(record.get("validation"), dict)
        or not isinstance(record.get("footnotes", []), list)
    ):
        raise RuntimeError(
            f"Canonical table artifact has invalid field types: {record.get('table_id')!r}"
        )
    if (
        record["page_no"] < 0
        or record["row_count"] < 0
        or record["column_count"] < 0
        or record.get("source_kind") not in {"native", "ocr", "mixed", "image_only"}
        or record.get("extractor") not in {"native_pdf_table", "ocr_table"}
        or record.get("decision") not in {"accepted", "deferred", "rejected"}
    ):
        raise RuntimeError(
            f"Canonical table artifact has invalid enum/range values: {record.get('table_id')!r}"
        )
    try:
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
            continuation_group_id=record.get("continuation_group_id"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Malformed canonical table artifact: {record.get('table_id')!r}"
        ) from exc


def _canonical_markdown_block(value: str) -> str:
    """Normalize only Unicode/newlines/trailing whitespace for exact comparison."""

    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _ref_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("$ref") or value.get("cref")
        return candidate if isinstance(candidate, str) else None
    return None


def _validate_visual_body_isolation(
    document: dict[str, Any],
    regions: list[dict[str, Any]],
    table_records: dict[str, dict[str, Any]],
    markdown: str,
    eligible_page_nos: set[int],
) -> int:
    """Independently reject unique raw visual-body text leaking into Markdown."""

    nodes: dict[str, dict[str, Any]] = {}
    for collection in (
        "body",
        "furniture",
        "groups",
        "texts",
        "pictures",
        "tables",
        "key_value_items",
        "form_items",
    ):
        values = document.get(collection, [])
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            continue
        for node in values:
            if not isinstance(node, dict):
                continue
            node_ref = node.get("self_ref")
            if isinstance(node_ref, str):
                nodes[node_ref] = node

    normalized_text_by_ref = {
        node_ref: normalized_markdown_evidence(str(node.get("text")))
        for node_ref, node in nodes.items()
        if isinstance(node.get("text"), str) and str(node.get("text")).strip()
    }
    text_frequency = Counter(normalized_text_by_ref.values())
    accepted_table_refs = {
        record.get("docling_ref")
        for record in table_records.values()
        if record.get("decision") == "accepted"
        and record.get("source_kind") == "native"
        and record.get("page_no") in eligible_page_nos
        and isinstance(record.get("docling_ref"), str)
    }
    normalized_markdown = normalized_markdown_evidence(markdown)
    checked = 0
    for region in regions:
        root_ref = region.get("docling_ref") or region.get("region_id")
        if not isinstance(root_ref, str) or root_ref not in nodes:
            continue
        if region.get("region_type") == "table" and root_ref in accepted_table_refs:
            continue
        root = nodes[root_ref]
        protected_refs = {
            ref
            for value in (root.get("captions", []),)
            for item in (value if isinstance(value, list) else [])
            if (ref := _ref_value(item)) is not None
        }
        pending = list(root.get("children", []) or [])
        if region.get("region_type") == "table":
            data = root.get("data") or {}
            if isinstance(data, dict):
                for cell in data.get("table_cells", []) or []:
                    if isinstance(cell, dict) and cell.get("ref") is not None:
                        pending.append(cell["ref"])
        visited: set[str] = set()
        while pending:
            node_ref = _ref_value(pending.pop())
            if node_ref is None or node_ref in visited or node_ref in protected_refs:
                continue
            visited.add(node_ref)
            node = nodes.get(node_ref)
            if node is None:
                continue
            pending.extend(node.get("children", []) or [])
            text = normalized_text_by_ref.get(node_ref)
            # Unique, reasonably long strings are high-precision provenance
            # sentinels. Their occurrence in Markdown cannot be explained by a
            # duplicate non-visual text node elsewhere in the document.
            if text and len("".join(text.split())) >= 12 and text_frequency[text] == 1:
                checked += 1
                if text in normalized_markdown:
                    raise RuntimeError(
                        "Deferred table/picture body text leaked into document.md: "
                        f"region={root_ref} text_ref={node_ref}"
                    )
    if re.search(r"!\[[^\]]*\]\(|<img\b|data:image/", markdown, flags=re.I):
        raise RuntimeError("A picture body or embedded image leaked into document.md")
    return checked


def _load_table_records(
    artifact_dir: Path, table_index: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    entries = table_index.get("tables")
    if not isinstance(entries, list):
        raise RuntimeError("tables/index.json must contain a tables list")
    table_root = (artifact_dir / "tables").resolve()
    seen_artifacts: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Malformed table index entry: {entry!r}")
        table_id = entry.get("table_id")
        artifact = entry.get("artifact")
        if not isinstance(table_id, str) or not isinstance(artifact, str):
            raise RuntimeError(f"Malformed table index entry: {entry!r}")
        if table_id in records:
            raise RuntimeError(f"Duplicate table_id in table index: {table_id}")
        path = (artifact_dir / artifact).resolve()
        if not path.is_relative_to(table_root) or path == table_root / "index.json":
            raise RuntimeError(f"Table artifact escapes the tables directory: {artifact!r}")
        if path in seen_artifacts:
            raise RuntimeError(f"Duplicate table artifact path in index: {artifact!r}")
        seen_artifacts.add(path)
        if not path.is_file():
            raise RuntimeError(f"Table index points to a missing artifact: {entry!r}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise RuntimeError(f"Table artifact is not a JSON object: {table_id}")
        for field in ("table_id", "page_no", "source_kind", "extractor", "decision"):
            if record.get(field) != entry.get(field):
                raise RuntimeError(
                    f"Table index/artifact mismatch for {field}: {table_id}"
                )
        _record_to_result(record)
        records[table_id] = record
    recomputed_summary = table_summary(
        [_record_to_result(record) for record in records.values()]
    )
    if table_index.get("summary") != recomputed_summary:
        raise RuntimeError(
            "tables/index.json summary does not match its canonical artifacts: "
            f"stored={table_index.get('summary')!r} recomputed={recomputed_summary!r}"
        )
    return records


def _validate_accepted_table_provenance(record: dict[str, Any]) -> str:
    table_id = record.get("table_id")
    validation = record.get("validation", {})
    if record.get("source_kind") != "native":
        raise RuntimeError(f"Accepted table is not native: {table_id}")
    if not isinstance(record.get("page_no"), int) or record["page_no"] <= 0:
        raise RuntimeError(f"Accepted table has no valid owner page: {table_id}")
    if validation.get("text_conservation_ratio") != 1.0:
        raise RuntimeError(f"Accepted table lacks full text conservation: {table_id}")
    source_cells = validation.get("source_cells")
    if not isinstance(source_cells, list) or any(
        not isinstance(source, dict) for source in source_cells
    ):
        raise RuntimeError(f"Accepted table has malformed source cells: {table_id}")
    source_refs = [source.get("source_cell_ref") for source in source_cells]
    if (
        any(not isinstance(ref, str) for ref in source_refs)
        or len(source_refs) != len(set(source_refs))
        or any(source.get("from_ocr") is not False for source in source_cells)
        or any(not normalize_table_text(source.get("text")) for source in source_cells)
    ):
        raise RuntimeError(
            f"Accepted native table has duplicate, missing, or OCR source refs: {table_id}"
        )
    source_by_ref = {
        source["source_cell_ref"]: normalize_table_text(source.get("text"))
        for source in source_cells
    }
    output_source_refs = [
        source_ref
        for cell in record.get("cells", [])
        for source_ref in cell.get("source_cell_refs", [])
        if isinstance(source_ref, str)
    ]
    if (
        len(output_source_refs) != len(set(output_source_refs))
        or set(output_source_refs) != set(source_by_ref)
        or validation.get("output_cells_deterministically_rebuilt") is not True
    ):
        raise RuntimeError(
            "Accepted table source-cell provenance is not a deterministic "
            f"one-to-one reconstruction: {table_id}"
        )
    for cell in record.get("cells", []):
        refs = cell.get("source_cell_refs")
        if not isinstance(refs, list) or any(ref not in source_by_ref for ref in refs):
            raise RuntimeError(f"Accepted table cell has invalid source refs: {table_id}")
        expected_text = " ".join(source_by_ref[ref] for ref in refs)
        if normalize_table_text(cell.get("text")) != expected_text:
            raise RuntimeError(
                f"Accepted table cell text is not rebuilt from source refs: {table_id}"
            )
    has_spans = any(
        cell.get("row_span") != 1 or cell.get("column_span") != 1
        for cell in record.get("cells", [])
    )
    if has_spans and validation.get("markdown_span_projection") != "anchor_only":
        raise RuntimeError(
            "Accepted spanned table lacks the anchor-only Markdown projection "
            f"record: {table_id}"
        )
    expected_markdown = table_to_markdown(_record_to_result(record))
    if not expected_markdown:
        raise RuntimeError(
            f"Accepted canonical table cannot be projected to Markdown: {table_id}"
        )
    return expected_markdown


def _validate_table_markdown(
    markdown: str,
    record: dict[str, Any],
    *,
    eligible_page_nos: set[int],
) -> None:
    """Prove accepted table identity, content equality, and page ownership."""

    table_id = record["table_id"]
    page_no = record["page_no"]
    marker = f"<!-- TABLE id={table_id} "
    should_render = (
        record.get("decision") == "accepted"
        and record.get("source_kind") == "native"
        and page_no in eligible_page_nos
    )
    if not should_render:
        if marker in markdown:
            raise RuntimeError(
                f"Deferred/rejected/untrusted table leaked into Markdown: {table_id}"
            )
        return
    expected = _validate_accepted_table_provenance(record)
    if markdown.count(marker) != 1:
        raise RuntimeError(
            f"Accepted table must occur exactly once in document.md: {table_id}"
        )
    page_markdown = _page_markdown(markdown, page_no)
    expected_open = (
        f"<!-- TABLE id={table_id} page={page_no} source=native -->"
    )
    if page_markdown.count(expected_open) != 1:
        raise RuntimeError(
            f"Accepted table is not located under its declared PDF page: {table_id}"
        )
    actual = _table_block(page_markdown, table_id)
    if actual is None or _canonical_markdown_block(actual) != _canonical_markdown_block(expected):
        raise RuntimeError(
            f"Canonical JSON and Markdown table content differ: {table_id}"
        )
    for footnote in record.get("footnotes", []):
        if not isinstance(footnote, dict):
            raise RuntimeError(f"Malformed table footnote audit record: {table_id}")
        if footnote.get("trust_decision") != "accepted":
            continue
        text = footnote.get("text")
        if (
            not isinstance(text, str)
            or normalized_markdown_evidence(text)
            not in normalized_markdown_evidence(page_markdown)
        ):
            raise RuntimeError(
                f"Accepted table footnote is absent from its owner page: {table_id}"
            )


def validate_5006a_golden_artifacts(
    artifact_dir: Path,
    quality: dict[str, Any],
    regions: list[dict[str, Any]],
    table_index: dict[str, Any],
    markdown: str,
) -> dict[str, Any]:
    """Enforce the complete, pinned NASA-STD-5006A Golden acceptance contract."""

    expected_pages = list(range(1, GOLDEN_5006A_PAGE_COUNT + 1))
    quality_pages = quality.get("pages") or []
    actual_pages = [page.get("page_no") for page in quality_pages]
    if actual_pages != expected_pages:
        raise RuntimeError(
            f"5006A must audit all 48 pages in order; received {actual_pages}"
        )
    untrusted_pages = [
        page.get("page_no")
        for page in quality_pages
        if page.get("eligible_for_indexing") is not True
    ]
    if untrusted_pages:
        raise RuntimeError(
            f"5006A requires every page to be trusted and indexable: {untrusted_pages}"
        )
    marker_sequence = [
        int(value) for value in re.findall(r"<!-- PDF page (\d+) -->", markdown)
    ]
    if marker_sequence != expected_pages:
        raise RuntimeError(
            "5006A Markdown page markers must occur exactly once in source order: "
            f"{marker_sequence}"
        )

    records = _load_table_records(artifact_dir, table_index)
    entries = table_index.get("tables", [])
    for entry in entries:
        table_id = entry["table_id"]
        _validate_table_markdown(
            markdown,
            records[table_id],
            eligible_page_nos=set(expected_pages),
        )

    page17_native = [
        entry
        for entry in entries
        if entry.get("page_no") == 17
        and entry.get("decision") == "accepted"
        and entry.get("source_kind") == "native"
    ]
    if page17_native or any(
        region.get("page_no") == 17
        and region.get("region_type") == "table"
        and region.get("canonical_table_in_semantic_markdown") is True
        for region in regions
    ):
        raise RuntimeError("5006A page 17 image table became a native canonical table")

    for page_no in (20, 22, 25):
        pictures = [
            region
            for region in regions
            if region.get("page_no") == page_no
            and region.get("region_type") == "picture"
        ]
        if not pictures or any(
            picture.get("visual_body_in_semantic_markdown") is not False
            for picture in pictures
        ):
            raise RuntimeError(
                f"5006A page {page_no} picture body isolation was not proven"
            )
        if re.search(
            r"!\[[^\]]*\]\(|<img\b|data:image/",
            _page_markdown(markdown, page_no),
            flags=re.I,
        ):
            raise RuntimeError(
                f"5006A page {page_no} contains a Markdown picture body"
            )

    matrix_tokens = (
        "section",
        "description",
        "requirement in this standard",
        "gwr",
    )
    accepted_matrix_by_page: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        page_no = entry.get("page_no")
        if (
            isinstance(page_no, int)
            and 36 <= page_no <= 48
            and entry.get("decision") == "accepted"
            and entry.get("source_kind") == "native"
        ):
            block = _table_block(_page_markdown(markdown, page_no), entry["table_id"])
            normalized_block = normalized_markdown_evidence(block or "")
            if all(token in normalized_block for token in matrix_tokens):
                accepted_matrix_by_page.setdefault(page_no, []).append(entry)
    missing_matrix_pages = [
        page_no for page_no in range(36, 49) if page_no not in accepted_matrix_by_page
    ]
    if missing_matrix_pages:
        raise RuntimeError(
            "5006A requires its identified requirements matrix (Section, Description, "
            "Requirement in this Standard, GWR) on every matrix page: "
            f"{missing_matrix_pages}"
        )

    page36_markdown = "\n".join(
        _table_block(markdown, entry["table_id"]) or ""
        for entry in accepted_matrix_by_page[36]
    )
    normalized_page36 = normalized_markdown_evidence(page36_markdown)
    first_match = re.search(r"(?<![\d.])2\.4\.2(?!\d)", normalized_page36)
    group_match = re.compile(r"(?<![\d.])4\.\s+requirements\b").search(
        normalized_page36,
        first_match.end() if first_match is not None else 0,
    )
    following_match = re.compile(r"(?<![\d.])4\.1(?!\d)").search(
        normalized_page36,
        group_match.end() if group_match is not None else 0,
    )
    if first_match is None or group_match is None or following_match is None:
        raise RuntimeError(
            "5006A page 36 row order must satisfy "
            "2.4.2 < 4. Requirements < 4.1"
        )

    return {
        "all_pages_trusted": True,
        "matrix_pages_with_accepted_native_table": sorted(accepted_matrix_by_page),
        "page36_row_order": "2.4.2 < 4. Requirements < 4.1",
        "page17_image_table_isolated": True,
        "picture_pages_isolated": [20, 22, 25],
    }


def validate_end_to_end_artifacts(
    combined_output: str,
    returncode: int,
    expected_provider: str,
    source: Path,
    source_fingerprint: dict[str, Any],
    requested_page: int,
    golden_5006a: bool = False,
) -> dict[str, Any]:
    published = re.findall(r"(?m)^published_output=(.+)$", combined_output)
    if len(published) != 1:
        raise RuntimeError(
            "End-to-end output must contain exactly one published_output marker"
        )
    artifact_dir = Path(published[0].strip()).resolve()
    required = [
        artifact_dir / "document.md",
        artifact_dir / "document.json",
        artifact_dir / "regions.json",
        artifact_dir / "quality_report.json",
        artifact_dir / "tables" / "index.json",
        artifact_dir / "orientation_report.json",
        artifact_dir / "normalized" / "oriented.pdf",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("End-to-end artifacts are incomplete: " + ", ".join(missing))

    quality = json.loads((artifact_dir / "quality_report.json").read_text(encoding="utf-8"))
    document = json.loads((artifact_dir / "document.json").read_text(encoding="utf-8"))
    orientation = json.loads(
        (artifact_dir / "orientation_report.json").read_text(encoding="utf-8")
    )
    regions = json.loads((artifact_dir / "regions.json").read_text(encoding="utf-8"))
    table_index = json.loads(
        (artifact_dir / "tables" / "index.json").read_text(encoding="utf-8")
    )
    markdown = (artifact_dir / "document.md").read_text(encoding="utf-8")
    if sha256_file(source) != source_fingerprint["source_sha256"]:
        raise RuntimeError("Source PDF SHA-256 changed during preprocessing")
    if source.stat().st_size != source_fingerprint["source_size_bytes"]:
        raise RuntimeError("Source PDF size changed during preprocessing")
    from pypdf import PdfReader

    normalized_page_count = len(
        PdfReader(str(artifact_dir / "normalized" / "oriented.pdf")).pages
    )
    if normalized_page_count != source_fingerprint["source_page_count"]:
        raise RuntimeError(
            "Normalized PDF page count does not match the source PDF: "
            f"{normalized_page_count} != {source_fingerprint['source_page_count']}"
        )
    if quality.get("source_sha256") != source_fingerprint["source_sha256"]:
        raise RuntimeError("quality_report source SHA-256 mismatch")
    if orientation.get("source_sha256") != source_fingerprint["source_sha256"]:
        raise RuntimeError("orientation_report source SHA-256 mismatch")
    expected_processed_pages = (
        list(range(1, GOLDEN_5006A_PAGE_COUNT + 1))
        if golden_5006a
        else [requested_page]
    )
    if orientation.get("processed_page_numbers") != expected_processed_pages:
        raise RuntimeError(
            "orientation_report processed page mismatch: "
            f"{orientation.get('processed_page_numbers')}"
        )
    quality_pages = quality.get("pages") or []
    if [page.get("page_no") for page in quality_pages] != expected_processed_pages:
        raise RuntimeError(
            f"quality_report page mismatch: {[page.get('page_no') for page in quality_pages]}"
        )
    runtime = quality.get("ocr_runtime") or {}
    if set(runtime) != {"det", "cls", "rec"}:
        raise RuntimeError(f"quality_report OCR provider evidence is incomplete: {runtime}")
    for stage, active in runtime.items():
        if not active or active[0] != expected_provider:
            raise RuntimeError(f"quality_report {stage} provider mismatch: {active}")
    if any(
        region.get("visual_body_in_semantic_markdown") is not False
        for region in regions
        if region.get("region_type") == "picture"
    ):
        raise RuntimeError("A picture region was not explicitly isolated from Markdown")
    summary = table_index.get("summary")
    if not isinstance(summary, dict) or "cells" in summary:
        raise RuntimeError("tables/index.json must contain a cell-free table summary")
    if quality.get("table_summary") != summary:
        raise RuntimeError("quality_report table_summary differs from tables/index.json")
    eligible_pages = {
        int(page["page_no"])
        for page in quality.get("pages", [])
        if page.get("eligible_for_indexing") is True
    }
    markdown_page_sequence = [
        int(value) for value in re.findall(r"<!-- PDF page (\d+) -->", markdown)
    ]
    expected_markdown_pages = sorted(eligible_pages)
    if markdown_page_sequence != expected_markdown_pages:
        raise RuntimeError(
            "Markdown page markers must occur exactly once in ascending source order: "
            f"actual={markdown_page_sequence} expected={expected_markdown_pages}"
        )
    table_records = _load_table_records(artifact_dir, table_index)
    for table in table_index.get("tables", []):
        table_id = table.get("table_id")
        _validate_table_markdown(
            markdown,
            table_records[table_id],
            eligible_page_nos=eligible_pages,
        )
    visual_body_sentinels_checked = _validate_visual_body_isolation(
        document,
        regions,
        table_records,
        markdown,
        eligible_pages,
    )

    accepted_caption_region_count = 0
    accepted_caption_refs: set[str] = set()
    caption_owner_pages_by_ref: dict[str, set[int]] = {}
    for region in regions:
        owner_pages = {
            occurrence.get("page_no")
            for occurrence in region.get("page_occurrences", [])
            if isinstance(occurrence, dict)
            and isinstance(occurrence.get("page_no"), int)
        }
        if isinstance(region.get("page_no"), int):
            owner_pages.add(region["page_no"])
        for caption_ref in region.get("caption_refs", []):
            if isinstance(caption_ref, str):
                caption_owner_pages_by_ref.setdefault(caption_ref, set()).update(
                    owner_pages
                )
    for region in regions:
        decision = region.get("caption_trust_decision")
        marked_present = region.get("caption_in_semantic_markdown")
        caption = region.get("caption")
        if decision == "accepted":
            accepted_caption_region_count += 1
            accepted_caption_refs.update(
                ref
                for ref in region.get("caption_refs", [])
                if isinstance(ref, str)
            )
            if marked_present is not True:
                raise RuntimeError(
                    f"Accepted caption was not marked as retained: {region.get('region_id')}"
                )
            if not isinstance(caption, str) or not caption.strip():
                raise RuntimeError(
                    f"Accepted caption has no text: {region.get('region_id')}"
                )
            allowed_pages: set[int] = set()
            for caption_ref in region.get("caption_refs", []):
                if isinstance(caption_ref, str):
                    allowed_pages.update(
                        caption_owner_pages_by_ref.get(caption_ref, set())
                    )
            if not allowed_pages and isinstance(region.get("page_no"), int):
                allowed_pages.add(region["page_no"])
            owner_markdown = "\n".join(
                _page_markdown(markdown, page_no)
                for page_no in sorted(allowed_pages & eligible_pages)
            )
            if normalized_markdown_evidence(caption) not in normalized_markdown_evidence(
                owner_markdown
            ):
                raise RuntimeError(
                    "Accepted caption text is absent from its owner page(s): "
                    f"{region.get('region_id')}"
                )
        elif marked_present is not False:
            raise RuntimeError(
                "A non-accepted visual caption was marked as retained: "
                f"{region.get('region_id')} decision={decision!r}"
            )

    if "CANONICAL_TABLE_PLACEHOLDER" in markdown:
        raise RuntimeError("An unresolved canonical table placeholder reached document.md")
    golden_evidence = None
    if golden_5006a:
        if returncode != 0:
            raise RuntimeError("5006A Golden preprocessing must finish with return code 0")
        golden_evidence = validate_5006a_golden_artifacts(
            artifact_dir,
            quality,
            regions,
            table_index,
            markdown,
        )
    if returncode == 0 and not eligible_pages:
        raise RuntimeError("A successful formal publication contains no eligible page")
    if returncode == 0 and artifact_dir.parent.name in {".failed", ".staging"}:
        raise RuntimeError("A successful run was not published to its stable directory")
    if returncode in {1, 3} and ".failed" not in artifact_dir.parts:
        raise RuntimeError("A rejected/partial run was not retained under .failed")
    if returncode == 3 and eligible_pages:
        raise RuntimeError("Quality-gate return code 3 contains an eligible page")
    if returncode == 1 and quality.get("status") != "partial_success":
        raise RuntimeError("Return code 1 does not carry partial_success audit status")
    return {
        "artifact_dir": str(artifact_dir),
        "quality_status": quality.get("status"),
        "eligible_pages": sorted(eligible_pages),
        "region_count": len(regions),
        "table_summary": summary,
        "accepted_caption_region_count": accepted_caption_region_count,
        "accepted_caption_ref_count": len(accepted_caption_refs),
        "visual_body_sentinels_checked": visual_body_sentinels_checked,
        "caption_retention_check": (
            "verified_in_markdown"
            if accepted_caption_region_count
            else "not_applicable_no_accepted_caption"
        ),
        "formal_markdown_page_markers": markdown_page_sequence,
        "source_sha256_preserved": True,
        "source_page_count": source_fingerprint["source_page_count"],
        "normalized_page_count": normalized_page_count,
        "golden_5006a": golden_evidence,
    }


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.document_timeout) or args.document_timeout <= 0:
        raise ValueError("--document-timeout must be a finite positive number")
    source = args.input_pdf.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_fingerprint = source_asset_fingerprint(source)
    if not 1 <= args.page <= source_fingerprint["source_page_count"]:
        raise ValueError(
            f"--page must be within [1, {source_fingerprint['source_page_count']}]"
        )
    if args.golden_5006a:
        if args.allow_quality_rejection:
            raise ValueError(
                "--golden-5006a cannot be combined with --allow-quality-rejection"
            )
        if args.skip_end_to_end:
            raise ValueError(
                "--golden-5006a cannot be combined with --skip-end-to-end"
            )
        if (
            source_fingerprint["source_page_count"] != GOLDEN_5006A_PAGE_COUNT
            or source_fingerprint["source_sha256"] != GOLDEN_5006A_SHA256
        ):
            raise RuntimeError(
                "--golden-5006a requires the pinned 48-page NASA-STD-5006A PDF "
                f"with SHA-256 {GOLDEN_5006A_SHA256}"
            )
    if version("rapidocr") != "3.9.2":
        raise RuntimeError(
            f"RapidOCR 3.9.2 is required; installed: {version('rapidocr')}"
        )
    from page_orientation import load_orientation_model_metadata
    from pdf_preprocess import verify_preprocessing_model_assets

    verify_preprocessing_model_assets(include_orientation=True)
    orientation_metadata = load_orientation_model_metadata(
        PROJECT_ROOT / "models" / "PageOrientation" / "PP-LCNet_x1_0_doc_ori"
    )

    model = create_real_ocr(args.device, args.num_threads)
    providers = provider_payload(model)
    expected = (
        "CUDAExecutionProvider"
        if args.device.lower().startswith("cuda")
        else "CPUExecutionProvider"
    )
    for stage, active in providers.items():
        if not active or active[0] != expected:
            raise RuntimeError(f"{stage} primary provider mismatch: {active}")

    image = render_page(source, args.page)
    result = model.reader(
        image,
        use_det=True,
        use_cls=True,
        use_rec=True,
    )
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    evidence: dict[str, Any] = {
        "status": "real_ocr_passed",
        "rapidocr_version": version("rapidocr"),
        "providers": providers,
        "orientation_model_sha256": orientation_metadata.model_sha256,
        "input_pdf": str(source),
        **source_fingerprint,
        "page": args.page,
        "document_timeout_seconds": args.document_timeout,
        "detected_box_count": len(boxes) if boxes is not None else 0,
        "recognized_text_count": len(texts) if texts is not None else 0,
        "mean_recognition_confidence": (
            sum(float(score) for score in scores) / len(scores)
            if scores is not None and len(scores)
            else None
        ),
    }

    if not args.skip_end_to_end:
        subprocess_env = os.environ.copy()
        python_paths = [str(DOCLING_SOURCE), str(SCRIPT_DIR)]
        if subprocess_env.get("PYTHONPATH"):
            python_paths.append(subprocess_env["PYTHONPATH"])
        subprocess_env["PYTHONPATH"] = os.pathsep.join(python_paths)
        command = [
            sys.executable,
            str(SCRIPT_DIR / "pdf_preprocess.py"),
            str(source),
            "--device",
            args.device,
            "--num-threads",
            str(args.num_threads),
            "--document-timeout",
            str(args.document_timeout),
        ]
        if not args.golden_5006a:
            command.extend(["--page-range", str(args.page), str(args.page)])
        command.extend(
            [
                "--output-root",
                str(args.output_root.expanduser().resolve()),
                "--overwrite",
            ]
        )
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=subprocess_env,
            bufsize=1,
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)
        returncode = process.wait()
        combined_output = "".join(output_lines)
        evidence["end_to_end"] = {
            "returncode": returncode,
            "combined_output": combined_output,
        }
        # Validate any deliberately retained quality rejection before deciding
        # whether this invocation's acceptance policy permits it. A Docling
        # partial_success is never a deployment pass.
        if returncode not in {0, 1, 3}:
            raise RuntimeError(
                "End-to-end preprocessing failed:\n"
                + combined_output
            )
        evidence["end_to_end"]["artifact_validation"] = validate_end_to_end_artifacts(
            combined_output,
            returncode,
            expected,
            source,
            source_fingerprint,
            args.page,
            golden_5006a=args.golden_5006a,
        )
        if returncode == 1:
            raise RuntimeError(
                "Deployment acceptance failed: Docling returned partial_success. "
                "The auditable artifact was retained under .failed, but it is not "
                "formal index input."
            )
        if returncode == 3 and not args.allow_quality_rejection:
            raise RuntimeError(
                "Deployment acceptance failed: the selected page was rejected by "
                "the deterministic quality gate. Use --allow-quality-rejection only "
                "when deliberately testing the safe-rejection path."
            )

    evidence_dir = args.output_root.expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=evidence_dir,
        prefix=".server-acceptance.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, evidence_dir / "server_acceptance.json")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"server_acceptance_error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
