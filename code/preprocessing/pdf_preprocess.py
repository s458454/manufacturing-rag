"""V0.1 PDF preprocessing entry point.

This module deliberately stops before chunking, embedding, or indexing.  Before
Docling Layout and RapidOCR run, it normalizes trustworthy whole-page PDF
orientations with an independent PP-LCNet model.  It then runs Docling with the
project-local Heron layout model and PP-OCRv6 medium weights and writes
auditable intermediate artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    LayoutOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    PictureItem,
    TableItem,
    TextItem,
)

try:
    from .page_orientation import (
        OrientationConfig,
        PageOrientationNormalizer,
        NormalizedDocument,
        document_id_for_source,
        orientation_summary_from_records,
        sha256_file,
    )
    from .table_extraction import (
        NativePdfTableExtractor,
        OcrTableExtractor,
        TableExtractionResult,
        TableRegion,
        bbox_to_topleft,
        classify_source_kind,
        normalize_table_text,
        source_cells_in_region,
        table_summary,
        table_to_markdown,
    )
except ImportError:  # Support direct execution: python code/preprocessing/pdf_preprocess.py
    from page_orientation import (
        OrientationConfig,
        PageOrientationNormalizer,
        NormalizedDocument,
        document_id_for_source,
        orientation_summary_from_records,
        sha256_file,
    )
    from table_extraction import (
        NativePdfTableExtractor,
        OcrTableExtractor,
        TableExtractionResult,
        TableRegion,
        bbox_to_topleft,
        classify_source_kind,
        normalize_table_text,
        source_cells_in_region,
        table_summary,
        table_to_markdown,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"
MODEL_MANIFEST_PATH = MODEL_ROOT / "preprocessing-models.manifest.json"
RAPIDOCR_ROOT = MODEL_ROOT / "RapidOcr"
HERON_ROOT = MODEL_ROOT / "docling-project--docling-layout-heron"
DEFAULT_ORIENTATION_MODEL_DIR = (
    MODEL_ROOT / "PageOrientation" / "PP-LCNet_x1_0_doc_ori"
)
MIN_SEMANTIC_PROJECTION_DOCLING_CORE = "2.88.0"
TABLEFORMER_ACCURATE_DIR = (
    MODEL_ROOT
    / "docling-project--docling-models"
    / "model_artifacts"
    / "tableformer"
    / "accurate"
)
TABLEFORMER_EXPECTED_ASSETS = {
    "tm_config.json": {
        "size_bytes": 7060,
        "sha256": "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd",
    },
    "tableformer_accurate.safetensors": {
        "size_bytes": 212758388,
        "sha256": "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9",
    },
}
TABLEFORMER_REQUIRED_FILES = tuple(TABLEFORMER_EXPECTED_ASSETS)
REQUIRED_SEMANTIC_MARKDOWN_PARAMS = frozenset(
    {"pages", "traverse_pictures", "enable_chart_tables"}
)

MODEL_PATHS = {
    "det": RAPIDOCR_ROOT
    / "onnx"
    / "PP-OCRv6"
    / "det"
    / "PP-OCRv6_det_medium.onnx",
    "rec": RAPIDOCR_ROOT
    / "onnx"
    / "PP-OCRv6"
    / "rec"
    / "PP-OCRv6_rec_medium.onnx",
    "cls": RAPIDOCR_ROOT
    / "onnx"
    / "PP-OCRv4"
    / "cls"
    / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "font": RAPIDOCR_ROOT / "resources" / "fonts" / "FZYTK.TTF",
    "heron_config": HERON_ROOT / "config.json",
    "heron_weights": HERON_ROOT / "model.safetensors",
    "heron_preprocessor": HERON_ROOT / "preprocessor_config.json",
}


def import_onnxruntime_for_preprocessing() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime-gpu is required for CUDA preprocessing; install the "
            "pinned server requirements before running."
        ) from exc
    return ort


def require_cuda_runtime(device: str) -> None:
    """Reject CUDA requests before layout/OCR if the shared runtime is unusable."""

    if not device.lower().startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA preprocessing requires a CUDA-enabled PyTorch build") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is False"
        )
    ort = import_onnxruntime_for_preprocessing()
    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        preload_dlls(directory="")
    providers = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "--device cuda was requested, but ONNX Runtime does not expose "
            f"CUDAExecutionProvider (available: {providers})"
        )
    print(
        f"onnxruntime_cuda_preflight=ok version={ort.__version__} providers={providers}",
        flush=True,
    )

# Formal indexing gates use confidence values only where their semantics are
# known.  Docling defines 0.5 as the lower boundary between POOR and FAIR parse
# quality. RapidOCR first discards recognitions below 0.5; 0.75 is the midpoint
# of the remaining accepted interval and 0.90 is reserved for short OCR text,
# where there is too little redundancy to expose a bad line.
MIN_NATIVE_PARSE_SCORE = 0.50
MIN_OCR_MEAN_CONFIDENCE = 0.75
MIN_SHORT_OCR_MEAN_CONFIDENCE = 0.90
SHORT_OCR_ALNUM_LIMIT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse one PDF with Docling Heron and PP-OCRv6 medium. "
            "Pictures and OCR-table bodies are excluded from Markdown semantics; "
            "audited native-PDF tables are projected from canonical JSON."
        )
    )
    parser.add_argument("input_pdf", type=Path, help="PDF file to preprocess")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprocessing",
        help="Root directory for generated artifacts",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "Shared page-orientation, Docling layout, and OCR device: cpu, "
            "cuda, or cuda:N. Page orientation requires CUDA as the first active "
            "provider in CUDA mode; all ONNX sessions use CUDA-preferred mixed "
            "execution with CPU node-level fallback."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="CPU inference threads used by Docling/RapidOCR",
    )
    parser.add_argument(
        "--document-timeout",
        type=float,
        default=3600.0,
        help=(
            "Maximum Docling processing time in seconds for one PDF "
            "(default: 3600). A timeout produces partial_success, excludes "
            "affected formal-index pages, and returns a non-zero exit code."
        ),
    )
    parser.add_argument(
        "--page-range",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        help=(
            "Optional 1-based inclusive source-PDF range, e.g. --page-range 1 3. "
            "It constrains both page-orientation classification and Docling while "
            "preserving original PDF page numbers."
        ),
    )
    parser.add_argument(
        "--enable-direction-classifier",
        action="store_true",
        help=(
            "Enable RapidOCR's independent 0/180-degree text-line classifier "
            "after detection; this is not the page-orientation model."
        ),
    )
    parser.add_argument(
        "--orientation-model-dir",
        type=Path,
        default=DEFAULT_ORIENTATION_MODEL_DIR,
        help=(
            "Directory containing local PP-LCNet_x1_0_doc_ori model.onnx, "
            "inference.yml, labels.json, and manifest.json"
        ),
    )
    parser.add_argument(
        "--orientation-render-dpi",
        type=int,
        default=150,
        help="DPI for full-page orientation classification renderings, 120-150 (default: 150)",
    )
    parser.add_argument(
        "--orientation-min-score",
        type=float,
        default=0.90,
        help="Minimum orientation top-1 score before a page can be accepted",
    )
    parser.add_argument(
        "--orientation-min-margin",
        type=float,
        default=0.15,
        help="Minimum top-1 versus top-2 orientation score margin",
    )
    parser.add_argument(
        "--orientation-postcheck-min-zero",
        type=float,
        default=0.90,
        help="Minimum 0-degree score required after a rotated-page post-check",
    )
    parser.add_argument(
        "--disable-page-orientation",
        action="store_true",
        help=(
            "Skip independent page-orientation normalization. Use only for "
            "controlled diagnostics; no orientation report or indexing gate is produced."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing this document's existing output files",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, str]:
    source = args.input_pdf.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF input is supported in V0.1: {source}")
    if args.num_threads < 1:
        raise ValueError("--num-threads must be at least 1")
    if not math.isfinite(args.document_timeout) or args.document_timeout <= 0:
        raise ValueError("--document-timeout must be a finite positive number")
    device = args.device.strip().lower()
    if device == "auto":
        raise ValueError(
            "--device auto is not allowed: choose explicit cuda/cuda:N or cpu so "
            "page-orientation inference cannot silently fall back to CPU."
        )
    if device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise ValueError("--device must be cpu, cuda, or cuda:N with a non-negative integer N")
    if args.page_range is not None:
        first, last = args.page_range
        if first < 1 or last < first:
            raise ValueError("--page-range must satisfy 1 <= FIRST <= LAST")
    if not 120 <= args.orientation_render_dpi <= 150:
        raise ValueError("--orientation-render-dpi must be within the supported 120-150 range")
    require_semantic_projection_serializer_api()

    verify_preprocessing_model_assets(
        include_orientation=not args.disable_page_orientation
    )

    # Validate the upper page bound before creating output directories.  The
    # orientation stage repeats this validation against the same source PDF,
    # but doing it here makes --page-range fail consistently even in controlled
    # orientation-disabled diagnostics.
    if args.page_range is not None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf>=4 is required to validate --page-range") from exc
        reader = PdfReader(str(source))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported in V0.1")
        if args.page_range[1] > len(reader.pages):
            raise ValueError(
                "--page-range LAST exceeds source PDF page count "
                f"({len(reader.pages)})"
            )

    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    source_sha256 = sha256_file(source)
    document_id = document_id_for_source(source, source_sha256)
    output_dir = output_root.resolve() / document_id
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists; pass --overwrite to replace it: {output_dir}"
        )
    return source, output_dir, document_id


def verify_preprocessing_model_assets(*, include_orientation: bool = True) -> None:
    """Fail before GPU initialization if a local model is missing or corrupted."""

    if not MODEL_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Required model integrity manifest is missing: {MODEL_MANIFEST_PATH}"
        )
    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read model integrity manifest: {MODEL_MANIFEST_PATH}"
        ) from exc
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("assets"), list):
        raise ValueError("preprocessing-models.manifest.json has an unsupported schema")

    failures: list[str] = []
    for asset in manifest["assets"]:
        if not isinstance(asset, dict):
            failures.append("manifest contains a non-object asset record")
            continue
        relative = asset.get("path")
        if not isinstance(relative, str) or not relative:
            failures.append("manifest asset has no path")
            continue
        if not include_orientation and relative.startswith("PageOrientation/"):
            continue
        path = (MODEL_ROOT / relative).resolve()
        try:
            path.relative_to(MODEL_ROOT.resolve())
        except ValueError:
            failures.append(f"asset path escapes models/: {relative}")
            continue
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        expected_size = asset.get("size_bytes")
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            failures.append(
                f"size mismatch: {path} (expected {expected_size}, got {path.stat().st_size})"
            )
            continue
        expected_sha = str(asset.get("sha256", "")).lower()
        actual_sha = sha256_file(path).lower()
        if actual_sha != expected_sha:
            failures.append(
                f"SHA-256 mismatch: {path} (expected {expected_sha}, got {actual_sha})"
            )
    if failures:
        raise RuntimeError("Project-local model integrity check failed:\n- " + "\n- ".join(failures))

    # TableFormer has its own direct verification as a second, named guard.
    # The manifest above verifies the same fixed v2.3.0 assets in the project
    # wide inventory; this guard gives an exact no-runtime-download failure.
    require_local_tableformer_assets()


def _remove_generated_tree(root: Path, target: Path) -> None:
    """Remove only a known transaction path beneath its resolved output root."""

    resolved_root = root.resolve()
    resolved_target = target.resolve()
    try:
        relative = resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to remove path outside output root: {target}") from exc
    if not relative.parts or resolved_target == resolved_root:
        raise RuntimeError(f"Refusing to remove output root itself: {target}")
    if target.is_symlink():
        raise RuntimeError(f"Refusing to remove symlinked output directory: {target}")
    if target.exists():
        shutil.rmtree(target)


class OutputTransaction:
    """Build a complete artifact set off-path and publish it as one run."""

    def __init__(self, final_dir: Path, *, overwrite: bool) -> None:
        self.final_dir = final_dir.resolve()
        self.root = self.final_dir.parent
        self.overwrite = overwrite
        self.token = uuid.uuid4().hex
        self.staging_dir = self.root / ".staging" / f"{self.final_dir.name}.{self.token}"
        self.lock_path = self.root / ".locks" / f"{self.final_dir.name}.lock"
        self.committed = False
        self.published_dir: Path | None = None

    def __enter__(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                "Another preprocessing run holds this document lock. "
                f"If no process is running, inspect and remove the stale lock: {self.lock_path}"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "transaction": self.token}, stream)
            stream.write("\n")
        try:
            self.staging_dir.mkdir(parents=True, exist_ok=False)
        except Exception:
            self.lock_path.unlink(missing_ok=True)
            raise
        return self.staging_dir

    def commit(self) -> None:
        self._rewrite_embedded_paths(self.final_dir)
        required = {
            "document.md",
            "document.json",
            "regions.json",
            "quality_report.json",
            "tables/index.json",
        }
        missing = sorted(
            name for name in required if not (self.staging_dir / name).is_file()
        )
        if missing:
            raise RuntimeError(
                "Refusing to publish incomplete preprocessing output; missing: "
                + ", ".join(missing)
            )
        if self.final_dir.exists() and not self.overwrite:
            raise FileExistsError(f"Output already exists: {self.final_dir}")

        backup = self.root / ".staging" / f"{self.final_dir.name}.backup.{self.token}"
        had_previous = self.final_dir.exists()
        if had_previous:
            if self.final_dir.is_symlink():
                raise RuntimeError(
                    f"Refusing to replace symlinked output directory: {self.final_dir}"
                )
            os.replace(self.final_dir, backup)
        try:
            os.replace(self.staging_dir, self.final_dir)
        except Exception:
            if had_previous and backup.exists() and not self.final_dir.exists():
                os.replace(backup, self.final_dir)
            raise
        if backup.exists():
            _remove_generated_tree(self.root, backup)
        self.committed = True
        self.published_dir = self.final_dir

    def retain_failure(self) -> Path:
        """Keep a complete rejected run without replacing the last good run."""

        failure_dir = self.root / ".failed" / f"{self.final_dir.name}.{self.token}"
        failure_dir.parent.mkdir(parents=True, exist_ok=True)
        self._rewrite_embedded_paths(failure_dir)
        os.replace(self.staging_dir, failure_dir)
        self.committed = True
        self.published_dir = failure_dir
        return failure_dir

    def _rewrite_embedded_paths(self, destination: Path) -> None:
        """Replace transaction-only absolute paths in audit JSON before moving it."""

        old_prefix = str(self.staging_dir.resolve())
        new_prefix = str(destination.resolve())

        def relocate(value: Any) -> Any:
            if isinstance(value, str):
                return new_prefix + value[len(old_prefix) :] if value.startswith(old_prefix) else value
            if isinstance(value, list):
                return [relocate(item) for item in value]
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            return value

        for name in ("orientation_report.json", "quality_report.json"):
            path = self.staging_dir / name
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            write_json(path, relocate(payload))

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.staging_dir.exists():
                _remove_generated_tree(self.root, self.staging_dir)
        finally:
            self.lock_path.unlink(missing_ok=True)


def orientation_config_from_args(args: argparse.Namespace) -> OrientationConfig:
    """Create one centralized configuration for the independent pre-Docling stage."""
    model_dir = args.orientation_model_dir.expanduser()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    return OrientationConfig(
        model_dir=model_dir,
        device=args.device,
        render_dpi=args.orientation_render_dpi,
        min_top1_score=args.orientation_min_score,
        min_top1_margin=args.orientation_min_margin,
        postcheck_min_zero=args.orientation_postcheck_min_zero,
        reject_uncertain=True,
    )


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def require_semantic_projection_serializer_api() -> None:
    """Fail before conversion if Docling lacks the required safe export API.

    ``enable_chart_tables`` is required to suppress chart-derived Markdown
    tables from excluded pictures. It was introduced in docling-core 2.88.0;
    accepting an older serializer would make visual isolation unverifiable.
    """
    fields = getattr(MarkdownParams, "model_fields", {})
    missing = sorted(REQUIRED_SEMANTIC_MARKDOWN_PARAMS - set(fields))
    if missing:
        raise RuntimeError(
            "The installed docling-core Markdown serializer lacks required "
            f"parameters {missing}. Install docling-core>={MIN_SEMANTIC_PROJECTION_DOCLING_CORE},<3.0.0 "
            "before running formal PDF preprocessing."
        )


def require_local_tableformer_assets() -> None:
    """Fail before conversion instead of letting Docling download table models."""

    missing = [
        str(TABLEFORMER_ACCURATE_DIR / filename)
        for filename in TABLEFORMER_REQUIRED_FILES
        if not (TABLEFORMER_ACCURATE_DIR / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "TableFormer V1 accurate model assets are required for formal PDF table "
            "processing and must be present locally; refusing runtime download. Missing:\n- "
            + "\n- ".join(missing)
        )
    size_mismatches = [
        filename
        for filename, expected in TABLEFORMER_EXPECTED_ASSETS.items()
        if (TABLEFORMER_ACCURATE_DIR / filename).stat().st_size
        != expected["size_bytes"]
    ]
    if size_mismatches:
        raise RuntimeError(
            "TableFormer V1 accurate model integrity check failed; incorrect file size: "
            + ", ".join(size_mismatches)
        )
    sha_mismatches = [
        filename
        for filename, expected in TABLEFORMER_EXPECTED_ASSETS.items()
        if sha256_file(TABLEFORMER_ACCURATE_DIR / filename).lower()
        != expected["sha256"]
    ]
    if sha_mismatches:
        raise RuntimeError(
            "TableFormer V1 accurate model integrity check failed; incorrect SHA-256: "
            + ", ".join(sha_mismatches)
        )


def tableformer_asset_requirement_payload() -> dict[str, Any]:
    return {
        "model": "TableFormer V1",
        "mode": "accurate",
        "local_directory": str(TABLEFORMER_ACCURATE_DIR),
        "required_files": list(TABLEFORMER_REQUIRED_FILES),
        "expected_assets": TABLEFORMER_EXPECTED_ASSETS,
        "runtime_download": "forbidden",
        "provisioning_status": (
            "present"
            if all((TABLEFORMER_ACCURATE_DIR / name).is_file() for name in TABLEFORMER_REQUIRED_FILES)
            else "missing"
        ),
    }


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def bbox_payload(prov: Any) -> dict[str, Any] | None:
    bbox = getattr(prov, "bbox", None)
    if bbox is None:
        return None
    if hasattr(bbox, "model_dump"):
        return bbox.model_dump(mode="json")
    return {
        name: finite_number(getattr(bbox, name, None))
        for name in ("l", "t", "r", "b", "coord_origin")
    }


def _ref_cref(ref_or_item: Any) -> str | None:
    """Return a Docling JSON-pointer reference without using private APIs."""
    cref = getattr(ref_or_item, "cref", None)
    if isinstance(cref, str) and cref:
        return cref
    self_ref = getattr(ref_or_item, "self_ref", None)
    return self_ref if isinstance(self_ref, str) and self_ref else None


def _resolve_ref(ref: Any, document: Any) -> Any | None:
    resolver = getattr(ref, "resolve", None)
    if not callable(resolver):
        return None
    try:
        return resolver(doc=document)
    except TypeError:
        try:
            return resolver(document)
        except Exception:
            return None
    except Exception:
        return None


def _item_page_numbers(item: Any) -> set[int]:
    return {
        page_no
        for prov in getattr(item, "prov", []) or []
        if isinstance((page_no := getattr(prov, "page_no", None)), int) and page_no > 0
    }


def _is_document_item(item: Any) -> bool:
    """Return whether an object bears page-level content provenance.

    ``GroupItem`` has a ``label`` but no page provenance.  Only ``DocItem``
    instances must be excluded wholesale for an ineligible page; structural
    groups are still traversed so trusted descendants are not lost.
    """
    return isinstance(item, DocItem)


def _table_cell_refs(item: Any) -> list[Any]:
    """Include RichTableCell tree references when the Docling version exposes them."""
    data = getattr(item, "data", None)
    return [
        ref
        for cell in getattr(data, "table_cells", []) or []
        if _ref_cref(ref := getattr(cell, "ref", None)) is not None
    ]


def _descendant_refs_from_document_tree(item: Any, document: Any) -> set[str]:
    """Return every child reachable in the canonical Docling document tree."""
    refs: set[str] = set()
    pending: list[Any] = [item]
    while pending:
        current = pending.pop()
        current_ref = _ref_cref(current)
        if current_ref is None or current_ref in refs:
            continue
        refs.add(current_ref)
        for child_ref in getattr(current, "children", []) or []:
            child = _resolve_ref(child_ref, document)
            if child is not None:
                pending.append(child)
    return refs


def _collect_subtree_refs(item: Any, document: Any) -> set[str]:
    """Collect visual descendants, including cell and linked-note tree roots."""
    refs = _descendant_refs_from_document_tree(item, document)
    if isinstance(item, TableItem):
        for cell_ref in _table_cell_refs(item):
            cell = _resolve_ref(cell_ref, document)
            if cell is not None:
                refs.update(_descendant_refs_from_document_tree(cell, document))
    # ReadingOrderModel associates footnotes through an explicit ref list.
    # Keep those nodes out of ordinary traversal; trusted native-table notes
    # are emitted exactly once by the specialized table serializer below.
    for footnote_ref in getattr(item, "footnotes", []) or []:
        footnote = _resolve_ref(footnote_ref, document)
        if footnote is not None:
            refs.update(_descendant_refs_from_document_tree(footnote, document))
    return refs


def _caption_plain_text(item: Any, document: Any) -> str:
    """Read only the referenced caption subtree, never unrelated visual OCR."""
    parts: list[str] = []
    pending: list[Any] = [item]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        current_ref = _ref_cref(current)
        if current_ref is None or current_ref in visited:
            continue
        visited.add(current_ref)
        text = getattr(current, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
            # A TextItem's inline child tree represents the same text.
            continue
        for child_ref in reversed(list(getattr(current, "children", []) or [])):
            child = _resolve_ref(child_ref, document)
            if child is not None:
                pending.append(child)
    return " ".join(parts).strip()


def caption_text(item: Any, document: Any) -> str | None:
    parts: list[str] = []
    for caption_ref in getattr(item, "captions", []) or []:
        caption_item = _resolve_ref(caption_ref, document)
        if caption_item is not None and (text := _caption_plain_text(caption_item, document)):
            parts.append(text)
    caption = " ".join(parts).strip()
    return caption or None


def caption_is_obviously_garbled(text: str) -> bool:
    """High-precision only; borderline captions remain available for audit."""
    compact = "".join(char for char in text if not char.isspace())
    if not compact:
        return False
    if "\ufffd" in compact or any(ord(char) < 32 for char in compact):
        return True
    alphanumeric = sum(char.isalnum() for char in compact)
    if alphanumeric == 0:
        return True
    allowed_punctuation = set(".,;:()[]{}'\"-–—?/&%+#=_\\")
    unusual = sum(
        not char.isalnum() and char not in allowed_punctuation for char in compact
    )
    # Short OCR fragments such as ``*\"F e t!`` are visibly corrupt even
    # though a long-text ratio threshold has too little evidence to trigger.
    # Keep this deliberately narrow so normal engineering labels like ``Fig. 1``
    # and ``A-1`` remain accepted.
    if len(compact) <= 6 and alphanumeric <= 3 and unusual >= 2:
        return True
    return len(compact) >= 8 and unusual / len(compact) > 0.35


def table_footnote_payload(
    item: Any,
    document: Any,
    eligible_page_nos: set[int],
    *,
    table_accepted: bool,
) -> list[dict[str, Any]]:
    """Audit explicitly linked table footnotes and retain only trusted ones."""

    payload: list[dict[str, Any]] = []
    owner_pages = _item_page_numbers(item)
    for reference in getattr(item, "footnotes", []) or []:
        footnote_ref = _ref_cref(reference)
        footnote = _resolve_ref(reference, document)
        if footnote_ref is None or footnote is None:
            continue
        text = _caption_plain_text(footnote, document)
        provenance_pages = _item_page_numbers(footnote)
        effective_pages = provenance_pages or owner_pages
        if not table_accepted:
            decision = "table_not_accepted"
        elif provenance_pages and not provenance_pages.issubset(owner_pages):
            decision = "owner_page_mismatch"
        elif not effective_pages or not effective_pages.issubset(eligible_page_nos):
            decision = "page_untrusted"
        elif not text:
            decision = "missing"
        elif caption_is_obviously_garbled(text):
            decision = "garbled"
        else:
            decision = "accepted"
        payload.append(
            {
                "footnote_ref": footnote_ref,
                "text": text or None,
                "page_nos": sorted(effective_pages),
                "trust_decision": decision,
                "in_semantic_markdown": decision == "accepted",
            }
        )
    return payload


@dataclass
class SemanticProjection:
    """The formal-index projection of a full, untouched Docling document."""

    excluded_refs: set[str]
    accepted_caption_refs: set[str]
    caption_owner_by_ref: dict[str, str]
    caption_owner_pages: dict[str, set[int]]
    caption_trust_by_ref: dict[str, str]
    visual_caption_refs: dict[str, list[str]]
    visual_caption_trust: dict[str, str]
    accepted_table_ids: set[str]
    eligible_page_nos: set[int]


def _caption_decision(
    caption_refs: list[str],
    caption_values: dict[str, str],
    caption_pages: dict[str, set[int]],
    visual_pages: set[int],
    eligible_page_nos: set[int],
) -> str:
    if not caption_refs:
        return "missing"
    if not visual_pages or not visual_pages.issubset(eligible_page_nos):
        return "page_untrusted"
    # A caption can be explicitly linked to a visual region while carrying its
    # own provenance on a different page.  Do not leak such a caption through
    # the trusted visual page when the caption's source page is untrusted.
    for caption_ref in caption_refs:
        provenance_pages = caption_pages.get(caption_ref, set())
        if provenance_pages and not provenance_pages.issubset(eligible_page_nos):
            return "page_untrusted"
    if any(not caption_values.get(ref, "").strip() for ref in caption_refs):
        return "missing"
    if any(caption_is_obviously_garbled(caption_values[ref]) for ref in caption_refs):
        return "garbled"
    return "accepted"


def build_semantic_projection(
    document: Any,
    pages: list[dict[str, Any]],
    table_results: dict[str, TableExtractionResult] | None = None,
) -> SemanticProjection:
    """Create reference-level visual isolation and final page eligibility gate."""
    eligible_page_nos = {
        int(page["page_no"])
        for page in pages
        if page.get("eligible_for_indexing") is True and isinstance(page.get("page_no"), int)
    }
    visual_infos: list[dict[str, Any]] = []
    seen_visual_refs: set[str] = set()
    all_doc_items: list[Any] = []
    seen_item_refs: set[str] = set()

    for item, _level in document.iterate_items(with_groups=True, traverse_pictures=True):
        item_ref = _ref_cref(item)
        if item_ref is not None and item_ref not in seen_item_refs:
            seen_item_refs.add(item_ref)
            all_doc_items.append(item)
        if not isinstance(item, (TableItem, PictureItem)) or item_ref is None:
            continue
        if item_ref in seen_visual_refs:
            continue
        seen_visual_refs.add(item_ref)
        caption_refs: list[str] = []
        caption_values: dict[str, str] = {}
        caption_pages: dict[str, set[int]] = {}
        for ref in getattr(item, "captions", []) or []:
            caption_ref = _ref_cref(ref)
            caption_item = _resolve_ref(ref, document)
            if caption_ref is None or caption_item is None:
                continue
            if caption_ref not in caption_refs:
                caption_refs.append(caption_ref)
                caption_values[caption_ref] = _caption_plain_text(caption_item, document)
                caption_pages[caption_ref] = _item_page_numbers(caption_item)
        visual_infos.append(
            {
                "root_ref": item_ref,
                "pages": _item_page_numbers(item),
                "subtree_refs": _collect_subtree_refs(item, document),
                "caption_refs": caption_refs,
                "caption_values": caption_values,
                "caption_pages": caption_pages,
            }
        )

    caption_owner_by_ref: dict[str, str] = {}
    caption_owner_pages: dict[str, set[int]] = {}
    caption_trust_by_ref: dict[str, str] = {}
    visual_caption_refs: dict[str, list[str]] = {}
    visual_caption_trust: dict[str, str] = {}
    all_visual_descendants: set[str] = set()
    caption_decisions: dict[str, list[str]] = {}
    for info in visual_infos:
        root_ref = info["root_ref"]
        if not isinstance(root_ref, str):
            continue
        refs = [ref for ref in info["caption_refs"] if isinstance(ref, str)]
        decision = _caption_decision(
            refs,
            info["caption_values"],
            info["caption_pages"],
            info["pages"],
            eligible_page_nos,
        )
        visual_caption_refs[root_ref] = refs
        visual_caption_trust[root_ref] = decision
        all_visual_descendants.update(info["subtree_refs"])
        for caption_ref in refs:
            caption_owner_by_ref.setdefault(caption_ref, root_ref)
            caption_owner_pages.setdefault(caption_ref, set()).update(info["pages"])
            caption_decisions.setdefault(caption_ref, []).append(decision)

    decision_priority = {"accepted": 0, "missing": 1, "garbled": 2, "page_untrusted": 3}
    for caption_ref, decisions in caption_decisions.items():
        caption_trust_by_ref[caption_ref] = max(
            decisions, key=lambda decision: decision_priority[decision]
        )
    accepted_caption_refs = {
        ref for ref, decision in caption_trust_by_ref.items() if decision == "accepted"
    }

    accepted_table_ids = {
        table_id
        for table_id, result in (table_results or {}).items()
        if result.decision == "accepted"
        and result.source_kind == "native"
        and result.page_no in eligible_page_nos
    }

    # Required formula for a visual region remains: root and descendants minus
    # explicit trusted captions. Accepted tables are injected from their
    # canonical audited representation, so their Docling body stays excluded.
    excluded_refs = all_visual_descendants - accepted_caption_refs
    for item in all_doc_items:
        item_ref = _ref_cref(item)
        if item_ref is None or not _is_document_item(item):
            continue
        item_pages = _item_page_numbers(item)
        trusted_caption_without_prov = (
            item_ref in accepted_caption_refs
            and caption_owner_pages.get(item_ref, set()).issubset(eligible_page_nos)
        )
        if (not item_pages and not trusted_caption_without_prov) or (
            item_pages and not item_pages.issubset(eligible_page_nos)
        ):
            excluded_refs.add(item_ref)

    return SemanticProjection(
        excluded_refs=excluded_refs,
        accepted_caption_refs=accepted_caption_refs,
        caption_owner_by_ref=caption_owner_by_ref,
        caption_owner_pages=caption_owner_pages,
        caption_trust_by_ref=caption_trust_by_ref,
        visual_caption_refs=visual_caption_refs,
        visual_caption_trust=visual_caption_trust,
        accepted_table_ids=accepted_table_ids,
        eligible_page_nos=eligible_page_nos,
    )


class SemanticMarkdownSerializer(MarkdownDocSerializer):
    """Docling Markdown serializer constrained to a formal-index projection."""

    _semantic_excluded_refs: set[str] = PrivateAttr(default_factory=set)
    _accepted_caption_refs: set[str] = PrivateAttr(default_factory=set)
    _caption_owner_pages: dict[str, set[int]] = PrivateAttr(default_factory=dict)
    _emitted_caption_refs: set[str] = PrivateAttr(default_factory=set)
    _table_placeholder_by_ref: dict[str, tuple[str, int]] = PrivateAttr(
        default_factory=dict
    )
    _table_footnotes_by_ref: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def configure_projection(
        self,
        projection: SemanticProjection,
        table_results: list[TableExtractionResult] | None = None,
    ) -> None:
        self._semantic_excluded_refs = set(projection.excluded_refs)
        self._accepted_caption_refs = set(projection.accepted_caption_refs)
        self._caption_owner_pages = {
            ref: set(page_nos) for ref, page_nos in projection.caption_owner_pages.items()
        }
        self._table_placeholder_by_ref = {}
        self._table_footnotes_by_ref = {}
        for result in table_results or []:
            if (
                result.decision != "accepted"
                or result.source_kind != "native"
                or result.page_no not in projection.eligible_page_nos
                or not isinstance(result.docling_ref, str)
            ):
                continue
            if result.docling_ref in self._table_placeholder_by_ref:
                raise RuntimeError(
                    "A Docling TableItem resolved to multiple accepted canonical tables: "
                    f"{result.docling_ref}"
                )
            self._table_placeholder_by_ref[result.docling_ref] = (
                canonical_table_placeholder(result),
                result.page_no,
            )
            self._table_footnotes_by_ref[result.docling_ref] = [
                str(note["text"])
                for note in result.footnotes
                if note.get("trust_decision") == "accepted"
                and isinstance(note.get("text"), str)
                and str(note["text"]).strip()
            ]

    def get_excluded_refs(self, **kwargs: Any) -> set[str]:
        excluded = set(super().get_excluded_refs(**kwargs))
        excluded.update(self._semantic_excluded_refs)
        params = self.params.merge_with_patch(patch=kwargs)
        if params.pages is not None:
            for ref in self._accepted_caption_refs:
                if self._caption_owner_pages.get(ref, set()) & set(params.pages):
                    excluded.discard(ref)
        return excluded

    def serialize(
        self,
        *,
        item: Any = None,
        **kwargs: Any,
    ) -> Any:
        # The base serializer treats GroupItem as a structural node and does not
        # consult ``get_excluded_refs`` before traversing it.  Stop every
        # excluded non-visual subtree here.  Table/Picture roots remain allowed
        # through so their serializers can emit their explicitly protected
        # captions before suppressing the visual body.
        item_ref = _ref_cref(item)
        if isinstance(item, TableItem):
            params = self.params.merge_with_patch(patch=kwargs)
            parts = []
            caption = self.serialize_captions(item=item, **kwargs)
            if caption.text:
                parts.append(caption)
            placeholder_info = self._table_placeholder_by_ref.get(item_ref or "")
            if placeholder_info:
                placeholder, owner_page = placeholder_info
                if params.pages is None or owner_page in set(params.pages):
                    parts.append(create_ser_result(text=placeholder, span_source=item))
                    for footnote in self._table_footnotes_by_ref.get(item_ref or "", []):
                        parts.append(create_ser_result(text=footnote, span_source=item))
            return create_ser_result(
                text="\n\n".join(part.text for part in parts if part.text),
                span_source=parts,
            )
        if (
            item_ref is not None
            and item_ref in self.get_excluded_refs(**kwargs)
            and not isinstance(item, (TableItem, PictureItem))
        ):
            return create_ser_result()
        return super().serialize(item=item, **kwargs)

    def serialize_captions(self, item: Any, **kwargs: Any) -> Any:
        """Emit trusted explicit captions once although their visual root is excluded."""
        params = self.params.merge_with_patch(patch=kwargs)
        if DocItemLabel.CAPTION not in params.labels:
            return create_ser_result()
        results = []
        for caption_ref in getattr(item, "captions", []) or []:
            cref = _ref_cref(caption_ref)
            if cref is None or cref not in self._accepted_caption_refs:
                continue
            if cref in self._emitted_caption_refs:
                continue
            if params.pages is not None and not (
                self._caption_owner_pages.get(cref, set()) & set(params.pages)
            ):
                continue
            caption_item = _resolve_ref(caption_ref, self.doc)
            if caption_item is None:
                continue
            text = _caption_plain_text(caption_item, self.doc)
            if not text:
                continue
            self._emitted_caption_refs.add(cref)
            results.append(
                create_ser_result(
                    text=text,
                    span_source=caption_item if isinstance(caption_item, TextItem) else [],
                )
            )
        caption = params.caption_delim.join(result.text for result in results)
        return create_ser_result(
            text=self.post_process(text=caption) if caption else "", span_source=results
        )


def export_semantic_markdown(
    document: Any,
    projection: SemanticProjection,
    table_results: list[TableExtractionResult] | None = None,
) -> str:
    """Render only eligible formal-index content with original PDF page markers."""
    if not projection.eligible_page_nos:
        return ""
    serializer = SemanticMarkdownSerializer(
        doc=document,
        params=MarkdownParams(
            labels=set(DocItemLabel),
            image_placeholder="",
            page_break_placeholder=None,
            traverse_pictures=True,
            # Chart annotations may otherwise be emitted as a Markdown table
            # even if the PictureItem itself is excluded from this view.
            enable_chart_tables=False,
        ),
    )
    serializer.configure_projection(projection, table_results)
    page_parts: list[str] = []
    for page_no in sorted(projection.eligible_page_nos):
        # Keep the base params immutable across pages.  In particular, a
        # `pages={38}` export must not accidentally carry that filter into the
        # later `pages={39}` iteration through serializer state.
        page_markdown = serializer.serialize(pages={page_no}).text.strip()
        marker = f"<!-- PDF page {page_no} -->"
        page_parts.append(f"{marker}\n\n{page_markdown}" if page_markdown else marker)
    return "\n\n".join(page_parts)


def _table_id(document_id: str, page_no: int, ordinal: int) -> str:
    return f"{document_id}-p{page_no:04d}-t{ordinal:03d}"


def _section_hierarchy_before(
    item: Any, document: Any, fallback: list[str]
) -> list[str]:
    """Use Docling's resolved heading hierarchy when available.

    The standard PDF pipeline enables ``HeadingHierarchyModel``.  Its API may
    differ across pinned Docling versions, so a last-seen section fallback is
    retained for audit continuity rather than guessing a hierarchy.
    """

    for attribute in (
        "get_heading_hierarchy",
        "heading_hierarchy",
        "get_section_hierarchy",
    ):
        resolver = getattr(document, attribute, None)
        if not callable(resolver):
            continue
        try:
            value = resolver(item)
        except (AttributeError, TypeError, ValueError):
            continue
        if isinstance(value, str):
            value = [value]
        if isinstance(value, (list, tuple)):
            headings = [normalize_table_text(part) for part in value]
            headings = [heading for heading in headings if heading]
            if headings:
                return headings
    return list(fallback)


def _advance_section_hierarchy(current: list[str], heading: Any) -> list[str]:
    """Update an H1/H2/H3-style stack from ``SectionHeaderItem.level``."""

    text = normalize_table_text(getattr(heading, "text", ""))
    if not text:
        return list(current)
    try:
        level = max(1, int(getattr(heading, "level", 1)))
    except (TypeError, ValueError):
        level = 1
    # Preserve only real ancestors. A direct H1-to-H3 jump must not fabricate
    # a missing H2 entry.
    ancestors = list(current[: min(level - 1, len(current))])
    return [*ancestors, text]


def extract_tables(
    document: Any,
    result_pages: list[Any],
    document_id: str,
    projection: SemanticProjection,
) -> list[TableExtractionResult]:
    """Route every table region independently and never infer a whole-PDF type."""

    pages_by_no = {
        int(getattr(page, "page_no")): page
        for page in result_pages
        if isinstance(getattr(page, "page_no", None), int)
    }
    current_section_hierarchy: list[str] = []
    table_ordinals: dict[int, int] = {}
    native_extractor = NativePdfTableExtractor()
    ocr_extractor = OcrTableExtractor()
    results: list[TableExtractionResult] = []

    for item, _level in document.iterate_items(with_groups=False, traverse_pictures=True):
        if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
            current_section_hierarchy = _advance_section_hierarchy(
                current_section_hierarchy, item
            )
            continue
        if not isinstance(item, TableItem):
            continue
        provenance = list(getattr(item, "prov", []) or [])
        page_nos = [
            int(getattr(entry, "page_no"))
            for entry in provenance
            if isinstance(getattr(entry, "page_no", None), int)
            and int(getattr(entry, "page_no")) > 0
        ]
        if not page_nos:
            page_nos = [0]
        unique_page_nos = list(dict.fromkeys(page_nos))
        if len(unique_page_nos) > 1:
            # A single TableItem with multiple provenance pages is one logical
            # multi-page object. V0.1 neither merges it nor copies its shared
            # TableFormer cells into page-local artifacts: emit exactly one
            # deferred audit record anchored to the first provenance page.
            primary_page_no = unique_page_nos[0]
            table_ordinals[primary_page_no] = table_ordinals.get(primary_page_no, 0) + 1
            table_id = _table_id(
                document_id,
                primary_page_no,
                table_ordinals[primary_page_no],
            )
            section_hierarchy = _section_hierarchy_before(
                item, document, current_section_hierarchy
            )
            caption = caption_text(item, document)
            multi_page_sources = []
            primary_bbox = None
            for page_no in unique_page_nos:
                page = pages_by_no.get(page_no)
                prov = next(
                    (
                        entry
                        for entry in provenance
                        if getattr(entry, "page_no", None) == page_no
                    ),
                    None,
                )
                page_height = finite_number(
                    getattr(getattr(page, "size", None), "height", None)
                )
                bbox = bbox_to_topleft(getattr(prov, "bbox", None), page_height)
                if page_no == primary_page_no:
                    primary_bbox = bbox
                if page is None or bbox is None:
                    continue
                occurrence = TableRegion(
                    table_id=table_id,
                    document_id=document_id,
                    page_no=page_no,
                    bbox=bbox,
                    section_hierarchy=section_hierarchy,
                    caption=caption,
                    docling_ref=_ref_cref(item),
                    item=item,
                )
                multi_page_sources.extend(source_cells_in_region(occurrence, page))
            source_kind, native_characters, ocr_characters = classify_source_kind(
                multi_page_sources
            )
            deferred_result = TableExtractionResult(
                    table_id=table_id,
                    document_id=document_id,
                    page_no=primary_page_no,
                    bbox=primary_bbox,
                    section_hierarchy=section_hierarchy,
                    caption=caption,
                    docling_ref=_ref_cref(item),
                    source_kind=source_kind,
                    extractor=(
                        "native_pdf_table" if source_kind == "native" else "ocr_table"
                    ),
                    decision="deferred",
                    row_count=0,
                    column_count=0,
                    cells=[],
                    validation={
                        "native_character_count": native_characters,
                        "ocr_character_count": ocr_characters,
                        "source_text_cell_count": len(multi_page_sources),
                        "source_cells": [
                            {
                                "source_cell_ref": source.source_ref,
                                "text": source.text,
                                "bbox": source.bbox,
                                "from_ocr": source.from_ocr,
                            }
                            for source in multi_page_sources
                        ],
                        "provenance_page_numbers": unique_page_nos,
                        "failure_reasons": [
                            "multi_page_table_not_supported_v0_1"
                        ],
                        "text_conservation_ratio": None,
                        "markdown_span_projection": "not_applicable_deferred",
                    },
                )
            deferred_result.footnotes = table_footnote_payload(
                item,
                document,
                projection.eligible_page_nos,
                table_accepted=False,
            )
            results.append(deferred_result)
            continue

        for page_no in unique_page_nos:
            table_ordinals[page_no] = table_ordinals.get(page_no, 0) + 1
            table_id = _table_id(document_id, page_no, table_ordinals[page_no])
            page = pages_by_no.get(page_no)
            prov = next(
                (entry for entry in provenance if getattr(entry, "page_no", None) == page_no),
                None,
            )
            page_height = finite_number(getattr(getattr(page, "size", None), "height", None))
            table_region = TableRegion(
                table_id=table_id,
                document_id=document_id,
                page_no=page_no,
                bbox=bbox_to_topleft(getattr(prov, "bbox", None), page_height),
                section_hierarchy=_section_hierarchy_before(
                    item, document, current_section_hierarchy
                ),
                caption=caption_text(item, document),
                docling_ref=_ref_cref(item),
                item=item,
            )
            if page is None or table_region.bbox is None:
                rejected = NativePdfTableExtractor()._result(
                    table_region,
                    "image_only",
                    {
                        "native_character_count": 0,
                        "ocr_character_count": 0,
                        "source_text_cell_count": 0,
                        "failure_reasons": ["missing_page_or_table_bbox"],
                    },
                    decision="rejected",
                )
                rejected.footnotes = table_footnote_payload(
                    item,
                    document,
                    projection.eligible_page_nos,
                    table_accepted=False,
                )
                results.append(rejected)
                continue
            sources = source_cells_in_region(table_region, page)
            source_kind, _native_characters, _ocr_characters = classify_source_kind(sources)
            extractor = native_extractor if source_kind == "native" else ocr_extractor
            extracted = extractor.extract(table_region, page, document)
            extracted.footnotes = table_footnote_payload(
                item,
                document,
                projection.eligible_page_nos,
                table_accepted=(
                    extracted.decision == "accepted"
                    and extracted.source_kind == "native"
                    and extracted.page_no in projection.eligible_page_nos
                ),
            )
            results.append(extracted)
    return results


def table_results_by_docling_ref(
    table_results: list[TableExtractionResult],
) -> dict[str, list[TableExtractionResult]]:
    """Keep all per-page table results for the corresponding visual region."""

    chosen: dict[str, list[TableExtractionResult]] = {}
    for result in table_results:
        if not isinstance(result.docling_ref, str):
            continue
        chosen.setdefault(result.docling_ref, []).append(result)
    for results in chosen.values():
        results.sort(key=lambda result: (result.page_no, result.table_id))
    return chosen


def canonical_table_placeholder(result: TableExtractionResult) -> str:
    """Return the exact temporary marker emitted at a TableItem's tree position."""

    return f"<!-- CANONICAL_TABLE_PLACEHOLDER id={result.table_id} -->"


def inject_accepted_tables_into_markdown(
    markdown: str,
    table_results: list[TableExtractionResult],
    eligible_page_nos: set[int] | None = None,
) -> str:
    """Replace each accepted TableItem placeholder at its original tree position."""

    accepted: list[TableExtractionResult] = []
    for result in table_results:
        if (
            result.decision == "accepted"
            and result.source_kind == "native"
            and (eligible_page_nos is None or result.page_no in eligible_page_nos)
        ):
            accepted.append(result)

    rendered = markdown
    for result in accepted:
        placeholder = canonical_table_placeholder(result)
        occurrence_count = rendered.count(placeholder)
        if occurrence_count != 1:
            raise RuntimeError(
                "Accepted canonical table placeholder must occur exactly once at its "
                f"Docling position: {result.table_id} occurrences={occurrence_count}"
            )
        table_markdown = table_to_markdown(result)
        if not table_markdown:
            raise RuntimeError(
                f"Accepted canonical table could not be projected to Markdown: {result.table_id}"
            )
        block = (
            f"<!-- TABLE id={result.table_id} page={result.page_no} source=native -->\n\n"
            f"{table_markdown}\n\n<!-- /TABLE -->"
        )
        rendered = rendered.replace(placeholder, block, 1)

    if "<!-- CANONICAL_TABLE_PLACEHOLDER " in rendered:
        raise RuntimeError(
            "A canonical table placeholder remained after semantic Markdown projection"
        )
    return rendered.strip()


def write_table_artifacts(
    output_dir: Path, table_results: list[TableExtractionResult]
) -> dict[str, Any]:
    """Persist one canonical audit document per table plus a compact index."""

    records = [result.to_dict() for result in table_results]
    index_records: list[dict[str, Any]] = []
    for record in records:
        path = output_dir / "tables" / f"{record['table_id']}.json"
        write_json(path, record)
        index_records.append(
            {
                "table_id": record["table_id"],
                "page_no": record["page_no"],
                "source_kind": record["source_kind"],
                "extractor": record["extractor"],
                "decision": record["decision"],
                "artifact": f"tables/{record['table_id']}.json",
            }
        )
    index = {"tables": index_records, "summary": table_summary(table_results)}
    write_json(output_dir / "tables" / "index.json", index)
    return index


def collect_regions(
    document: Any,
    projection: SemanticProjection,
    table_results_by_ref: dict[str, list[TableExtractionResult]] | None = None,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    counters: dict[tuple[int, str], int] = {}
    current_section_hierarchy: list[str] = []

    for item, _level in document.iterate_items(with_groups=False, traverse_pictures=True):
        if getattr(item, "label", None) == DocItemLabel.SECTION_HEADER:
            current_section_hierarchy = _advance_section_hierarchy(
                current_section_hierarchy, item
            )
            continue
        if not isinstance(item, (TableItem, PictureItem)):
            continue

        root_ref = _ref_cref(item)
        if root_ref is None:
            continue
        region_type = "table" if isinstance(item, TableItem) else "picture"
        provenance = list(getattr(item, "prov", []))
        prov = provenance[0] if provenance else None
        page_occurrences = [
            {
                "page_no": int(getattr(entry, "page_no", 0)) or None,
                "pdf_page_index": (
                    int(getattr(entry, "page_no", 0)) - 1
                    if int(getattr(entry, "page_no", 0)) > 0
                    else None
                ),
                "bbox": bbox_payload(entry),
            }
            for entry in provenance
        ]
        page_no = int(getattr(prov, "page_no", 0)) if prov is not None else 0
        key = (page_no, region_type)
        counters[key] = counters.get(key, 0) + 1
        region_index = counters[key]
        caption = caption_text(item, document)
        caption_refs = projection.visual_caption_refs.get(root_ref, [])
        caption_trust = projection.visual_caption_trust.get(root_ref, "missing")
        caption_in_markdown = any(
            ref in projection.accepted_caption_refs
            for ref in caption_refs
        )
        table_results = (
            (table_results_by_ref or {}).get(root_ref, [])
            if region_type == "table"
            else []
        )
        table_result = table_results[0] if table_results else None
        page_label = page_no if page_no > 0 else "unknown"
        visual_name = "表格" if region_type == "table" else "图片"
        display_label = (
            caption
            if caption_trust == "accepted" and caption
            else f"PDF 第 {page_label} 页未命名{visual_name} {region_index}"
        )
        regions.append(
            {
                "region_id": root_ref,
                "docling_ref": root_ref,
                "region_type": region_type,
                "page_no": page_no or None,
                "pdf_page_index": page_no - 1 if page_no > 0 else None,
                "page_occurrences": page_occurrences,
                "region_index_on_page": region_index,
                "caption": caption,
                "caption_refs": caption_refs,
                "display_label": display_label,
                "section": (
                    current_section_hierarchy[-1]
                    if current_section_hierarchy
                    else None
                ),
                "section_hierarchy": list(current_section_hierarchy),
                "bbox": bbox_payload(prov) if prov is not None else None,
                # The raw Docling visual body never enters the semantic view.
                # Accepted native tables replace their original-position
                # placeholders from the canonical audited schema below.
                "visual_body_in_semantic_markdown": False,
                "canonical_table_in_semantic_markdown": bool(
                    table_result is not None
                    and table_result.decision == "accepted"
                    and table_result.source_kind == "native"
                    and table_result.page_no in projection.eligible_page_nos
                ),
                "caption_in_semantic_markdown": caption_in_markdown,
                "caption_trust_decision": caption_trust,
                **(
                    {
                        "source_kind": table_result.source_kind,
                        "table_decision": table_result.decision,
                        "table_artifact": f"tables/{table_result.table_id}.json",
                        "table_id": table_result.table_id,
                        "continuation_group_id": table_result.continuation_group_id,
                        "footnotes": table_result.footnotes,
                        "table_occurrences": [
                            {
                                "table_id": result.table_id,
                                "page_no": result.page_no,
                                "source_kind": result.source_kind,
                                "table_decision": result.decision,
                                "table_artifact": f"tables/{result.table_id}.json",
                                "continuation_group_id": result.continuation_group_id,
                            }
                            for result in table_results
                        ],
                    }
                    if table_result is not None
                    else {}
                ),
            }
        )
    return regions


def page_quality_payload(result: Any) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    confidence_pages = result.confidence.pages
    for page in result.pages:
        native_cells = 0
        ocr_cells = 0
        ocr_cell_audit: list[dict[str, Any]] = []
        for cell in page.cells:
            if bool(getattr(cell, "from_ocr", False)):
                ocr_cells += 1
                rect = getattr(cell, "rect", None)
                bbox = rect.to_bounding_box() if rect is not None else None
                ocr_cell_audit.append(
                    {
                        "confidence": finite_number(getattr(cell, "confidence", None)),
                        "bbox": (
                            bbox.model_dump(mode="json")
                            if bbox is not None and hasattr(bbox, "model_dump")
                            else None
                        ),
                    }
                )
            else:
                native_cells += 1

        ocr_audit = dict(getattr(page, "ocr_audit", {}) or {})
        route = str(ocr_audit.get("route_requested") or "unknown")

        scores = confidence_pages.get(page.page_no)
        payload.append(
            {
                "page_no": page.page_no,
                "pdf_page_index": page.page_no - 1,
                "route_observed": route,
                "native_text_cells": native_cells,
                "ocr_text_cells": ocr_cells,
                "ocr_route_evidence": {
                    key: value
                    for key, value in ocr_audit.items()
                    if key != "onnx_providers"
                },
                "ocr_providers": ocr_audit.get("onnx_providers"),
                "_ocr_cell_audit": ocr_cell_audit,
                "_page_height": finite_number(getattr(page.size, "height", None)),
                "parse_score": finite_number(getattr(scores, "parse_score", None)),
                "layout_score": finite_number(getattr(scores, "layout_score", None)),
                "ocr_score": finite_number(getattr(scores, "ocr_score", None)),
                "trust_decision": "pending_orientation_and_parse_gate",
            }
        )
    return payload


def _bbox_topleft(payload: Any, page_height: float) -> tuple[float, float, float, float] | None:
    if not isinstance(payload, dict):
        return None
    values = [finite_number(payload.get(name)) for name in ("l", "t", "r", "b")]
    if any(value is None for value in values):
        return None
    left, top, right, bottom = (float(value) for value in values if value is not None)
    origin = str(payload.get("coord_origin", "")).lower().replace("_", "-")
    if "bottom" in origin:
        top, bottom = page_height - top, page_height - bottom
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)


def add_nonvisual_ocr_evidence(
    pages: list[dict[str, Any]], regions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Exclude OCR lines centred inside table/picture boxes from text-quality gates."""

    visual_boxes: dict[int, list[dict[str, Any]]] = {}
    for region in regions:
        if region.get("region_type") not in {"table", "picture"}:
            continue
        occurrences = region.get("page_occurrences") or []
        if not occurrences and isinstance(region.get("page_no"), int):
            occurrences = [{"page_no": region["page_no"], "bbox": region.get("bbox")}]
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue
            occurrence_page_no = occurrence.get("page_no")
            bbox = occurrence.get("bbox")
            if isinstance(occurrence_page_no, int) and isinstance(bbox, dict):
                visual_boxes.setdefault(occurrence_page_no, []).append(bbox)

    for page in pages:
        page_no = int(page["page_no"])
        page_height = finite_number(page.pop("_page_height", None))
        cell_audit = page.pop("_ocr_cell_audit", [])
        outside_scores: list[float] = []
        inside_count = 0
        boxes = visual_boxes.get(page_no, [])
        for cell in cell_audit:
            score = finite_number(cell.get("confidence")) if isinstance(cell, dict) else None
            if score is None or page_height is None:
                continue
            cell_box = _bbox_topleft(cell.get("bbox"), page_height)
            if cell_box is None:
                continue
            center_x = (cell_box[0] + cell_box[2]) / 2.0
            center_y = (cell_box[1] + cell_box[3]) / 2.0
            inside_visual = False
            for visual_payload in boxes:
                visual_box = _bbox_topleft(visual_payload, page_height)
                if visual_box is not None and (
                    visual_box[0] <= center_x <= visual_box[2]
                    and visual_box[1] <= center_y <= visual_box[3]
                ):
                    inside_visual = True
                    break
            if inside_visual:
                inside_count += 1
            else:
                outside_scores.append(score)
        page["semantic_ocr_evidence"] = {
            "nonvisual_ocr_text_cells": len(outside_scores),
            "visual_region_ocr_text_cells_excluded_from_gate": inside_count,
            "nonvisual_ocr_mean_confidence": (
                sum(outside_scores) / len(outside_scores) if outside_scores else None
            ),
            "visual_membership_rule": "OCR cell centre lies inside table/picture bbox",
        }
    return pages


def json_error(error: Any) -> dict[str, Any]:
    if hasattr(error, "model_dump"):
        return error.model_dump(mode="json")
    return {"message": str(error)}


def page_eligibility_summary(pages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total_pages": len(pages),
        "accepted_pages": sum(page.get("eligible_for_indexing") is True for page in pages),
        "excluded_pages": sum(page.get("eligible_for_indexing") is not True for page in pages),
    }


def conversion_exit_code(
    status: ConversionStatus,
    errors: list[dict[str, Any]],
    pages: list[dict[str, Any]],
) -> int:
    """Return zero only for a complete, error-free conversion with usable output."""

    if status != ConversionStatus.SUCCESS or errors:
        return 1
    if not pages or not any(page.get("eligible_for_indexing") is True for page in pages):
        return 3
    return 0


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run_page_orientation_stage(
    source: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> NormalizedDocument | None:
    """Normalize pages before Docling; never put the whole-page model in RapidOCR."""
    if args.disable_page_orientation:
        print("orientation_provider=disabled", flush=True)
        return None

    normalizer = PageOrientationNormalizer(orientation_config_from_args(args))
    result = normalizer.normalize(
        source=source,
        normalized_pdf=output_dir / "normalized" / "oriented.pdf",
        report_path=output_dir / "orientation_report.json",
        page_range=(tuple(args.page_range) if args.page_range is not None else None),
    )
    summary = result.report["orientation_summary"]
    print(
        "orientation_summary="
        + json.dumps(summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return result


def orientation_summary_from_pages(pages: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize decisions defensively for reports made with orientation disabled."""
    return orientation_summary_from_records(
        [{"decision": page.get("orientation_decision")} for page in pages]
    )


def merge_orientation_with_page_quality(
    pages: list[dict[str, Any]],
    orientation_result: NormalizedDocument | None,
) -> list[dict[str, Any]]:
    """Attach pre-Docling eligibility to each Docling page report.

    ``eligible_for_indexing`` is provisional here. The deterministic
    post-Docling gate below finalizes it before Markdown is serialized.
    """
    if orientation_result is None:
        for quality_page in pages:
            quality_page["orientation_decision"] = "disabled"
            quality_page["orientation_eligible_for_indexing"] = False
            quality_page["eligible_for_indexing"] = False
            quality_page["trust_decision"] = "orientation_disabled"
        return pages

    orientation_pages = {
        int(page["page_no"]): page for page in orientation_result.report["pages"]
    }
    docling_pages: dict[int, dict[str, Any]] = {
        int(page["page_no"]): page for page in pages
    }
    merged: list[dict[str, Any]] = []

    # The orientation report is the authoritative list of source pages that
    # were requested for this run.  Outer-join against it rather than relying
    # only on ``result.pages``: Docling can fail before producing a Page object,
    # and that failure must remain visible in quality_report.json.
    for page_no, orientation_page in sorted(orientation_pages.items()):
        existing_page = docling_pages.pop(page_no) if page_no in docling_pages else None
        if existing_page is None:
            page: dict[str, Any] = {
                "page_no": page_no,
                "pdf_page_index": page_no - 1,
                "route_observed": "missing_from_docling",
                "native_text_cells": 0,
                "ocr_text_cells": 0,
                "parse_score": None,
                "layout_score": None,
                "ocr_score": None,
                "trust_decision": "docling_page_missing",
            }
        else:
            page = existing_page
        decision = orientation_page["decision"]
        eligible = bool(orientation_page["eligible_for_indexing"])
        page["orientation_decision"] = decision
        page["orientation_eligible_for_indexing"] = eligible
        page["orientation_report_page"] = int(orientation_page["page_no"])
        page["eligible_for_indexing"] = eligible
        page["trust_decision"] = (
            "orientation_accepted_pending_deterministic_quality_gate"
            if eligible
            else decision
        )
        merged.append(page)

    # A Docling page outside the orientation scope is never eligible. This is
    # defensive: with --page-range both stages should have the same pages.
    for page_no, page in sorted(docling_pages.items()):
        page["orientation_decision"] = "review_required"
        page["orientation_eligible_for_indexing"] = False
        page["eligible_for_indexing"] = False
        page["trust_decision"] = "orientation_missing_for_docling_page"
        merged.append(page)
    return sorted(merged, key=lambda page: int(page["page_no"]))


def apply_post_docling_orientation_gate(
    pages: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    orientation_result: NormalizedDocument | None,
    document: Any,
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply deterministic formal-index gates without discarding raw audit data."""
    if orientation_result is None:
        return pages

    region_counts: dict[int, dict[str, int]] = {}
    for region in regions:
        region_type = region.get("region_type")
        if region_type not in {"table", "picture"}:
            continue
        occurrence_pages: set[int] = {
            int(occurrence["page_no"])
            for occurrence in region.get("page_occurrences", [])
            if isinstance(occurrence, dict)
            and isinstance(occurrence.get("page_no"), int)
        }
        if not occurrence_pages and isinstance(region.get("page_no"), int):
            occurrence_pages.add(int(region["page_no"]))
        for page_no in occurrence_pages:
            counts = region_counts.setdefault(page_no, {"tables": 0, "pictures": 0})
            counts[f"{region_type}s"] += 1

    error_pages: set[int] = set()
    document_scope_error = False
    for error in errors:
        error_page_no = error.get("page_no")
        if isinstance(error_page_no, int):
            error_pages.add(error_page_no)
        else:
            document_scope_error = True

    # Build one page-specific formal view while all orientation-accepted pages
    # are still provisionally eligible. Visual bodies are already removed by
    # the semantic serializer, so their noisy OCR cannot poison text checks.
    provisional_projection = build_semantic_projection(document, pages)
    serializer = SemanticMarkdownSerializer(
        doc=document,
        params=MarkdownParams(
            labels=set(DocItemLabel),
            image_placeholder="",
            page_break_placeholder=None,
            traverse_pictures=True,
            enable_chart_tables=False,
        ),
    )
    serializer.configure_projection(provisional_projection)

    orientation_pages = {
        int(page["page_no"]): page for page in orientation_result.report["pages"]
    }
    parse_uncertain_pages = 0
    for page in pages:
        page_no = int(page["page_no"])
        orientation_page = orientation_pages.get(page_no)
        if orientation_page is None:
            continue
        text_cells = int(page["native_text_cells"]) + int(page["ocr_text_cells"])
        page_text = serializer.serialize(pages={page_no}).text.strip()
        text_sanity = semantic_text_sanity(page_text)
        detected_regions = region_counts.get(page_no, {"tables": 0, "pictures": 0})
        rejection_reasons: list[str] = []
        if page_no in error_pages or document_scope_error:
            rejection_reasons.append("docling_reported_conversion_error")
        if text_sanity["has_hard_corruption"]:
            rejection_reasons.extend(text_sanity["reasons"])
        has_semantic_text = bool(text_sanity["compact_char_count"])
        parse_score = page["parse_score"]
        semantic_ocr_evidence = page.get("semantic_ocr_evidence") or {}
        ocr_score = finite_number(
            semantic_ocr_evidence.get("nonvisual_ocr_mean_confidence")
        )
        if ocr_score is None:
            ocr_score = page["ocr_score"]
        route = page["route_observed"]
        if has_semantic_text and int(page["native_text_cells"]) > 0:
            if parse_score is None or parse_score < MIN_NATIVE_PARSE_SCORE:
                rejection_reasons.append("native_parse_score_below_0.50_or_missing")
        if has_semantic_text and int(page["native_text_cells"]) == 0:
            required_ocr_score = (
                MIN_SHORT_OCR_MEAN_CONFIDENCE
                if text_sanity["alphanumeric_count"] < SHORT_OCR_ALNUM_LIMIT
                else MIN_OCR_MEAN_CONFIDENCE
            )
            if ocr_score is None or ocr_score < required_ocr_score:
                rejection_reasons.append(
                    f"ocr_mean_confidence_below_{required_ocr_score:.2f}_or_missing"
                )
        if not has_semantic_text and text_cells == 0 and not any(detected_regions.values()):
            rejection_reasons.append("no_semantic_text_or_locatable_visual_region")

        parse_evidence = {
            "total_text_cells": text_cells,
            "ocr_mean_confidence": page["ocr_score"],
            "nonvisual_ocr_mean_confidence": finite_number(
                semantic_ocr_evidence.get("nonvisual_ocr_mean_confidence")
            ),
            "layout_confidence": page["layout_score"],
            "parse_confidence": page["parse_score"],
            "route_observed": route,
            "table_regions": detected_regions["tables"],
            "picture_regions": detected_regions["pictures"],
            "semantic_text_sanity": text_sanity,
            "rejection_reasons": rejection_reasons,
            "gate_decision": "accepted",
        }
        if (
            orientation_page["decision"]
            in {"accepted_upright", "accepted_rotated"}
            and rejection_reasons
        ):
            orientation_page["decision"] = "orientation_or_parse_uncertain"
            orientation_page["eligible_for_indexing"] = False
            orientation_page["review_reasons"].extend(rejection_reasons)
            parse_evidence["gate_decision"] = "orientation_or_parse_uncertain"
            page["orientation_decision"] = "orientation_or_parse_uncertain"
            page["orientation_eligible_for_indexing"] = False
            page["eligible_for_indexing"] = False
            page["trust_decision"] = "orientation_or_parse_uncertain"
            parse_uncertain_pages += 1
        elif not page["eligible_for_indexing"]:
            parse_evidence["gate_decision"] = page["orientation_decision"]
        else:
            page["trust_decision"] = "accepted"
        orientation_page["post_docling_parse"] = parse_evidence

    orientation_result.report["post_docling_parse_summary"] = {
        "orientation_or_parse_uncertain_pages": parse_uncertain_pages,
        "accepted_pages": sum(page["eligible_for_indexing"] for page in pages),
    }
    orientation_result.report["orientation_summary"] = orientation_summary_from_records(
        orientation_result.report["pages"]
    )
    orientation_result.report["manual_review_required"] = bool(
        orientation_result.report["orientation_summary"]["review_required_pages"]
    )
    orientation_result.write_report()
    return pages


def semantic_text_sanity(text: str) -> dict[str, Any]:
    """High-precision corruption checks for the already visual-isolated page text."""

    compact = "".join(character for character in text if not character.isspace())
    alphanumeric = sum(character.isalnum() for character in compact)
    controls = sum(ord(character) < 32 and character not in "\t\n\r" for character in text)
    replacement = text.count("\ufffd")
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    singleton_ratio = (
        sum(len(token) == 1 for token in tokens) / len(tokens) if tokens else 0.0
    )
    alphanumeric_ratio = alphanumeric / len(compact) if compact else 0.0
    reasons: list[str] = []
    if controls or replacement:
        reasons.append("replacement_or_control_character_present")
    if len(compact) >= 20 and alphanumeric_ratio < 0.35:
        reasons.append("alphanumeric_ratio_below_0.35")
    if len(tokens) >= 10 and singleton_ratio > 0.50:
        reasons.append("single_character_token_ratio_above_0.50")
    return {
        "compact_char_count": len(compact),
        "alphanumeric_count": alphanumeric,
        "alphanumeric_ratio": alphanumeric_ratio,
        "token_count": len(tokens),
        "single_character_token_ratio": singleton_ratio,
        "replacement_character_count": replacement,
        "control_character_count": controls,
        "has_hard_corruption": bool(reasons),
        "reasons": reasons,
    }


def run_preprocessing(
    args: argparse.Namespace,
    source: Path,
    output_dir: Path,
    document_id: str,
) -> int:
    require_cuda_runtime(args.device)
    print("preprocess_stage=page_orientation", flush=True)
    orientation_result = run_page_orientation_stage(source, output_dir, args)
    docling_source = (
        orientation_result.normalized_pdf if orientation_result is not None else source
    )

    accelerator = AcceleratorOptions(
        device=args.device,
        num_threads=args.num_threads,
    )
    ocr_options = RapidOcrOptions(
        lang=["english"],
        backend="onnxruntime",
        use_det=True,
        use_cls=args.enable_direction_classifier,
        use_rec=True,
        det_model_path=str(MODEL_PATHS["det"]),
        cls_model_path=str(MODEL_PATHS["cls"]),
        rec_model_path=str(MODEL_PATHS["rec"]),
        font_path=str(MODEL_PATHS["font"]),
    )
    pipeline_options = PdfPipelineOptions(
        artifacts_path=MODEL_ROOT,
        accelerator_options=accelerator,
        do_ocr=True,
        ocr_options=ocr_options,
        do_table_structure=True,
        table_structure_options=TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        ),
        do_picture_classification=False,
        do_picture_description=False,
        do_chart_extraction=False,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        generate_parsed_pages=True,
        layout_options=LayoutOptions(
            model_spec=DOCLING_LAYOUT_HERON,
            create_orphan_clusters=True,
        ),
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
        layout_batch_size=1,
        ocr_batch_size=1,
        document_timeout=args.document_timeout,
    )
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    conversion_kwargs: dict[str, Any] = {"raises_on_error": False}
    if args.page_range is not None:
        conversion_kwargs["page_range"] = tuple(args.page_range)
    print("preprocess_stage=docling_conversion", flush=True)
    result = converter.convert(docling_source, **conversion_kwargs)

    print("preprocess_stage=quality_and_semantic_projection", flush=True)
    converted_errors = [json_error(error) for error in result.errors]
    pages = merge_orientation_with_page_quality(page_quality_payload(result), orientation_result)
    # The post-Docling gate needs region counts. Build a provisional projection
    # only for those audit fields, then rebuild it after the gate so document.md
    # sees the final eligibility decisions.
    provisional_projection = build_semantic_projection(result.document, pages)
    provisional_regions = collect_regions(result.document, provisional_projection)
    pages = add_nonvisual_ocr_evidence(pages, provisional_regions)
    pages = apply_post_docling_orientation_gate(
        pages,
        provisional_regions,
        orientation_result,
        result.document,
        converted_errors,
    )
    # Rebuild after the definitive quality gate. Table artifacts (including
    # linked-footnote trust decisions) must use the same final page eligibility
    # that later governs document.md, never the provisional pre-gate view.
    post_gate_projection = build_semantic_projection(result.document, pages)
    table_results = extract_tables(
        result.document,
        result.pages,
        document_id,
        post_gate_projection,
    )
    table_results_by_ref = table_results_by_docling_ref(table_results)
    projection = build_semantic_projection(
        result.document,
        pages,
        {result.table_id: result for result in table_results},
    )
    regions = collect_regions(result.document, projection, table_results_by_ref)
    markdown = inject_accepted_tables_into_markdown(
        export_semantic_markdown(result.document, projection, table_results),
        table_results,
        projection.eligible_page_nos,
    )
    table_summary_payload = table_summary(table_results)
    report = {
        "source": str(source),
        "normalized_source": (
            str(orientation_result.normalized_pdf)
            if orientation_result is not None
            else None
        ),
        "document_id": document_id,
        "source_sha256": (
            orientation_result.report["source_sha256"]
            if orientation_result is not None
            else sha256_file(source)
        ),
        "status": result.status.value,
        "errors": converted_errors,
        "versions": {
            "docling-slim": package_version("docling-slim"),
            "docling-core": package_version("docling-core"),
            "docling-ibm-models": package_version("docling-ibm-models"),
            "docling-parse": package_version("docling-parse"),
            "rapidocr": package_version("rapidocr"),
            "onnxruntime": package_version("onnxruntime-gpu")
            or package_version("onnxruntime"),
        },
        "configuration": {
            "layout_model": "docling-project/docling-layout-heron",
            "layout_revision": (HERON_ROOT / "REVISION").read_text(
                encoding="ascii"
            ).strip(),
            "ocr_engine": (
                "RapidOCR/ONNXRuntime CUDA-preferred mixed execution"
                if args.device.lower().startswith("cuda")
                else "RapidOCR/ONNXRuntime CPU"
            ),
            "ocr_cuda_provider_policy": (
                "CUDAExecutionProvider primary; CPUExecutionProvider permits node-level fallback"
                if args.device.lower().startswith("cuda")
                else "CPUExecutionProvider"
            ),
            "ocr_session_creation": "RapidOCR internal ONNXRuntime session creation",
            "ocr_detection_model": MODEL_PATHS["det"].name,
            "ocr_recognition_model": MODEL_PATHS["rec"].name,
            "model_integrity_manifest": str(MODEL_MANIFEST_PATH),
            "direction_classifier_enabled": args.enable_direction_classifier,
            "page_orientation_enabled": not args.disable_page_orientation,
            "page_orientation": (
                orientation_result.report["configuration"]
                if orientation_result is not None
                else {"enabled": False}
            ),
            "orientation_provider": (
                orientation_result.report["orientation_provider"]
                if orientation_result is not None
                else "disabled"
            ),
            "table_structure_enabled": True,
            "table_structure_engine": "TableFormer V1",
            "table_structure_mode": "accurate",
            "table_structure_cell_matching": True,
            "table_structure_model_dir": str(TABLEFORMER_ACCURATE_DIR),
            "table_structure_model_assets": tableformer_asset_requirement_payload(),
            "picture_semantics_enabled": False,
            "device": args.device,
            "num_threads": args.num_threads,
            "document_timeout_seconds": args.document_timeout,
            "page_range": args.page_range,
        },
        "document_confidence": {
            "parse_score": finite_number(result.confidence.parse_score),
            "layout_score": finite_number(result.confidence.layout_score),
            "ocr_score": finite_number(result.confidence.ocr_score),
        },
        "pages": pages,
        "page_eligibility_summary": page_eligibility_summary(pages),
        "ocr_runtime": next(
            (
                page["ocr_providers"]
                for page in pages
                if isinstance(page.get("ocr_providers"), dict)
            ),
            None,
        ),
        "orientation_summary": (
            orientation_result.report["orientation_summary"]
            if orientation_result is not None
            else orientation_summary_from_pages(pages)
        ),
        "orientation_report": (
            str(orientation_result.report_path)
            if orientation_result is not None
            else None
        ),
        "region_counts": {
            "tables": sum(r["region_type"] == "table" for r in regions),
            "pictures": sum(r["region_type"] == "picture" for r in regions),
        },
        "table_summary": table_summary_payload,
        "semantic_projection": {
            "eligible_pdf_pages": sorted(projection.eligible_page_nos),
            "excluded_docling_ref_count": len(projection.excluded_refs),
            "accepted_caption_ref_count": len(projection.accepted_caption_refs),
        },
        "trust_policy": {
            "formal_markdown": (
                "Only eligible pages; pictures and deferred/rejected table bodies are "
                "excluded. Accepted native canonical tables and accepted explicit "
                "table/picture captions are retained."
            ),
            "page_eligibility": (
                "Orientation gate, Docling error propagation, native parse-score floor, "
                "OCR confidence floor, and deterministic corruption checks."
            ),
            "layout_score_role": (
                "Diagnostic only; an unsupervised layout confidence is not treated as "
                "ground-truth box accuracy."
            ),
            "thresholds": {
                "native_parse_score_minimum": MIN_NATIVE_PARSE_SCORE,
                "ocr_mean_confidence_minimum": MIN_OCR_MEAN_CONFIDENCE,
                "short_ocr_mean_confidence_minimum": MIN_SHORT_OCR_MEAN_CONFIDENCE,
                "short_ocr_alphanumeric_limit": SHORT_OCR_ALNUM_LIMIT,
                "semantic_alphanumeric_ratio_minimum_for_20_plus_chars": 0.35,
                "single_character_token_ratio_maximum_for_10_plus_tokens": 0.50,
            },
            "threshold_rationale": (
                "0.50 is Docling's POOR/FAIR parse boundary and RapidOCR's own first-stage "
                "text filter. 0.75 is the midpoint of the retained OCR-confidence interval; "
                "short text requires 0.90 because it has little redundancy. Text sanity "
                "rules reject only replacement/control characters or strongly punctuation/"
                "singleton-dominated output."
            ),
        },
    }

    print("preprocess_stage=write_audit_artifacts", flush=True)
    table_index = write_table_artifacts(output_dir, table_results)
    if table_index["summary"] != table_summary_payload:
        raise RuntimeError("table audit index summary diverged from quality-report summary")
    write_text(output_dir / "document.md", markdown.rstrip() + "\n")
    write_json(output_dir / "document.json", result.document.export_to_dict())
    write_json(output_dir / "regions.json", regions)
    write_json(output_dir / "quality_report.json", report)

    print(f"status={result.status.value}")
    print(f"output={output_dir}")
    print(f"pages={len(pages)} tables={report['region_counts']['tables']} pictures={report['region_counts']['pictures']}")
    return conversion_exit_code(result.status, converted_errors, pages)


def main() -> int:
    args = parse_args()
    source, final_output_dir, document_id = validate_inputs(args)
    transaction = OutputTransaction(final_output_dir, overwrite=args.overwrite)
    with transaction as staging_output_dir:
        exit_code = run_preprocessing(
            args,
            source,
            staging_output_dir,
            document_id,
        )
        if exit_code == 0:
            transaction.commit()
        else:
            transaction.retain_failure()
    # ``run_preprocessing`` reports the staging path while work is in progress;
    # this final stable marker is intentionally last so shell acceptance can
    # select the published artifact directory.
    print(f"published_output={transaction.published_dir}")
    if exit_code != 0:
        print(
            "formal_index_output=not_published; previous successful output was preserved",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
