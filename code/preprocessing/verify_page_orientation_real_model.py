"""Server-only acceptance harness for the real PP-LCNet orientation model.

This deliberately does not use the fake classifier from the offline unit
tests.  It extracts one selectable-text engineering-document page, makes four
metadata-only visual rotations (0/90/180/270), and proves that the deployed
model and PDF correction mapping return each variant to an upright page.

Example:
    python code/preprocessing/verify_page_orientation_real_model.py \
        --input-pdf /data/engineering.pdf --page 12 --device cuda \
        --work-dir outputs/real-orientation-acceptance
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from page_orientation import (  # noqa: E402
    EXPECTED_ORIENTATIONS,
    OrientationConfig,
    OrientationError,
    PageOrientationClassifier,
    PageOrientationNormalizer,
    orientation_model_input_evidence,
    render_pdf_page,
    sha256_file,
    write_json_atomic,
    write_rotated_pdf,
)


EXPECTED_MODEL_SHA256 = "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92"
BASELINE_ONNXRUNTIME_VERSION = "1.23.2"
HISTORICAL_CPU_NODE_OPS = frozenset({"Slice", "Concat"})
HISTORICAL_MAJOR_COMPUTE_OPS = frozenset(
    {"Conv", "MatMul", "GlobalAveragePool", "Softmax"}
)
HISTORICAL_CPU_NODE_DURATION_RATIO = 0.00022


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the real PP-LCNet four-orientation acceptance check on one "
            "selectable-text engineering-document PDF page."
        )
    )
    parser.add_argument("--input-pdf", required=True, type=Path)
    parser.add_argument(
        "--page",
        required=True,
        type=int,
        help="One-based page number in --input-pdf to use as the upright source page.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/PageOrientation/PP-LCNet_x1_0_doc_ori"),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="cpu, cuda, or cuda:N. CUDA is the acceptance default.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        type=Path,
        help="Directory for variants, normalized PDFs, reports, and final evidence.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this harness's known artifact paths in --work-dir.",
    )
    parser.add_argument("--render-dpi", type=int, default=150)
    parser.add_argument("--min-top1-score", type=float, default=0.90)
    parser.add_argument("--min-top1-margin", type=float, default=0.15)
    parser.add_argument("--postcheck-min-zero", type=float, default=0.90)
    return parser.parse_args()


def _read_pdf_page_count_and_text(source: Path, page_no: int) -> tuple[int, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise OrientationError("pypdf>=4 is required for real-model acceptance") from exc

    reader = PdfReader(str(source))
    if reader.is_encrypted:
        raise OrientationError("Encrypted PDFs are not supported for real-model acceptance")
    page_count = len(reader.pages)
    if not 1 <= page_no <= page_count:
        raise OrientationError(
            f"--page must be within [1, {page_count}], received {page_no}"
        )
    try:
        text = reader.pages[page_no - 1].extract_text() or ""
    except Exception as exc:
        raise OrientationError(
            f"Could not extract selectable text from source PDF page {page_no}"
        ) from exc
    if not text.strip():
        raise OrientationError(
            f"Source PDF page {page_no} has no selectable text. Choose a digitally "
            "generated engineering-document page for this acceptance check."
        )
    return page_count, text


def _write_single_page_source(source: Path, page_no: int, destination: Path) -> None:
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise OrientationError("pypdf>=4 is required for real-model acceptance") from exc

    reader = PdfReader(str(source))
    if reader.is_encrypted:
        raise OrientationError("Encrypted PDFs are not supported for real-model acceptance")
    writer = PdfWriter()
    writer.add_page(reader.pages[page_no - 1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        writer.write(stream)


def _text_witness(text: str) -> str:
    """Choose a stable extractable token without assuming an English-only PDF."""

    tokens = re.findall(r"\w{3,}", text, flags=re.UNICODE)
    if tokens:
        return max(tokens, key=len)
    compact = "".join(text.split())
    if not compact:
        raise OrientationError("Source page text became empty while creating acceptance evidence")
    return compact[: min(12, len(compact))]


def _assert_selectable_text(path: Path, witness: str) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise OrientationError("pypdf>=4 is required for real-model acceptance") from exc

    reader = PdfReader(str(path))
    if len(reader.pages) != 1:
        raise OrientationError(
            f"Expected one page in acceptance artifact {path}, found {len(reader.pages)}"
        )
    try:
        output_text = reader.pages[0].extract_text() or ""
    except Exception as exc:
        raise OrientationError(f"Could not extract text from normalized PDF {path}") from exc
    if not output_text.strip() or witness not in output_text:
        raise OrientationError(
            f"Normalized PDF no longer preserves selectable source text witness {witness!r}: {path}"
        )


def _classify_one_page(
    classifier: PageOrientationClassifier,
    pdf_path: Path,
    dpi: int,
) -> dict[str, Any]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise OrientationError("pypdfium2 is required for real-model acceptance") from exc

    document: Any | None = None
    image: Any | None = None
    try:
        document = pdfium.PdfDocument(str(pdf_path))
        if len(document) != 1:
            raise OrientationError(f"Expected one page in classifier input {pdf_path}")
        image = render_pdf_page(document, 0, dpi)
        prediction = classifier.classify(image)
        input_evidence = orientation_model_input_evidence(
            image,
            classifier.metadata.preprocess,
            classifier.config,
        )
        if not input_evidence["has_sufficient_ink"]:
            raise OrientationError(
                f"Selected acceptance page has insufficient ink in the model's "
                f"actual centre crop: {pdf_path}"
            )
        return {
            "predicted_orientation": prediction.predicted_orientation,
            "top1_score": float(prediction.top1_score),
            "top1_margin": float(prediction.top1_margin),
            "scores": {label: float(score) for label, score in prediction.scores.items()},
            "model_output_shape": list(classifier.last_output_shape or ()),
            "model_input_evidence": input_evidence,
        }
    finally:
        close = getattr(image, "close", None)
        if callable(close):
            close()
        close = getattr(document, "close", None)
        if callable(close):
            close()


def _assert_expected_prediction(
    evidence: dict[str, Any],
    expected_orientation: int,
    config: OrientationConfig,
    artifact: Path,
) -> None:
    if evidence["model_output_shape"] != [1, 4]:
        raise OrientationError(
            f"Real model output shape was not (1, 4) for {artifact.name}: "
            f"{evidence['model_output_shape']}"
        )
    if evidence["predicted_orientation"] != expected_orientation:
        raise OrientationError(
            "Real model label mapping failed: "
            f"{artifact.name} was expected to be {expected_orientation} degrees, but model "
            f"predicted {evidence['predicted_orientation']} degrees (scores={evidence['scores']})"
        )
    if evidence["top1_score"] < config.min_top1_score:
        raise OrientationError(
            f"Real model confidence is below min_top1_score for {artifact.name}: "
            f"{evidence['top1_score']:.6f} < {config.min_top1_score:.6f}"
        )
    if evidence["top1_margin"] < config.min_top1_margin:
        raise OrientationError(
            f"Real model confidence margin is below min_top1_margin for {artifact.name}: "
            f"{evidence['top1_margin']:.6f} < {config.min_top1_margin:.6f}"
        )


def _artifact_paths(work_dir: Path) -> list[Path]:
    paths = [
        work_dir / "selected-source-page.pdf",
        work_dir / "real_orientation_acceptance.json",
        work_dir / "onnxruntime_profile.json",
    ]
    for orientation in sorted(EXPECTED_ORIENTATIONS):
        paths.extend(
            [
                work_dir / "variants" / f"page-{orientation}.pdf",
                work_dir / "normalized" / f"page-{orientation}-normalized.pdf",
                work_dir / "reports" / f"page-{orientation}-orientation-report.json",
            ]
        )
    return paths


def _node_profile_summary(profile_path: Path) -> dict[str, Any]:
    """Summarize ONNX Runtime per-node placement for acceptance evidence."""

    try:
        decoded = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrientationError(f"Could not read ONNX Runtime profile: {profile_path}") from exc

    if isinstance(decoded, list):
        events = decoded
    elif isinstance(decoded, dict) and isinstance(decoded.get("traceEvents"), list):
        events = decoded["traceEvents"]
    else:
        raise OrientationError("ONNX Runtime profile did not contain a trace-event list")

    node_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("cat") != "Node":
            continue
        arguments = event.get("args")
        # ONNX Runtime may emit fence events in the Node category in addition
        # to actual ``*_kernel_time`` executions.  Only events carrying both
        # the operator and assigned provider represent measurable placement.
        if not isinstance(arguments, dict):
            continue
        op_name = arguments.get("op_name")
        provider = arguments.get("provider") or arguments.get("execution_provider")
        duration = event.get("dur")
        if not isinstance(op_name, str) or not isinstance(provider, str):
            continue
        if not isinstance(duration, (int, float)) or duration < 0:
            raise OrientationError(
                f"Invalid duration in ONNX Runtime node profile event {event.get('name')!r}"
            )
        node_events.append(
            {"op_name": op_name, "provider": provider, "duration_us": float(duration)}
        )

    if not node_events:
        raise OrientationError("ONNX Runtime profile contained no node execution events")

    total_duration_us = sum(event["duration_us"] for event in node_events)
    if total_duration_us <= 0:
        raise OrientationError("ONNX Runtime profile had zero total node duration")
    cpu_events = [event for event in node_events if event["provider"] == "CPUExecutionProvider"]
    cpu_node_ops = sorted({event["op_name"] for event in cpu_events})
    cpu_duration_us = sum(event["duration_us"] for event in cpu_events)
    cpu_duration_ratio = cpu_duration_us / total_duration_us

    provider_by_op: dict[str, set[str]] = {}
    for event in node_events:
        provider_by_op.setdefault(event["op_name"], set()).add(event["provider"])
    return {
        "profile_path": str(profile_path),
        "node_event_count": len(node_events),
        "historical_cpu_node_ops": sorted(HISTORICAL_CPU_NODE_OPS),
        "observed_cpu_node_ops": cpu_node_ops,
        "cpu_nodes_match_historical_baseline": set(cpu_node_ops) <= HISTORICAL_CPU_NODE_OPS,
        "cpu_node_duration_us": cpu_duration_us,
        "total_node_duration_us": total_duration_us,
        "cpu_node_duration_ratio": cpu_duration_ratio,
        "historical_cpu_node_duration_ratio": HISTORICAL_CPU_NODE_DURATION_RATIO,
        "historical_major_compute_provider": "CUDAExecutionProvider",
        "observed_major_compute_ops": {
            op_name: sorted(provider_by_op.get(op_name, set()))
            for op_name in sorted(HISTORICAL_MAJOR_COMPUTE_OPS)
        },
    }


def _move_profile_to_work_dir(profile_source: Path, work_dir: Path) -> Path:
    destination = work_dir / "onnxruntime_profile.json"
    if profile_source.resolve() == destination.resolve():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        profile_source.replace(destination)
    except OSError:
        shutil.copy2(profile_source, destination)
        profile_source.unlink()
    return destination


def _ensure_safe_destinations(work_dir: Path, overwrite: bool) -> None:
    existing = [path for path in _artifact_paths(work_dir) if path.exists()]
    if existing and not overwrite:
        sample = ", ".join(str(path) for path in existing[:3])
        raise OrientationError(
            "Acceptance artifact path already exists. Use a new --work-dir or "
            f"pass --overwrite after reviewing the files: {sample}"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input_pdf.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise OrientationError(f"--input-pdf must be a PDF: {source}")
    _ensure_safe_destinations(work_dir, args.overwrite)

    source_sha256_before = sha256_file(source)
    source_page_count_before, source_page_text = _read_pdf_page_count_and_text(source, args.page)
    witness = _text_witness(source_page_text)
    selected_source = work_dir / "selected-source-page.pdf"
    _write_single_page_source(source, args.page, selected_source)
    _assert_selectable_text(selected_source, witness)

    config = OrientationConfig(
        model_dir=model_dir,
        device=args.device,
        render_dpi=args.render_dpi,
        min_top1_score=args.min_top1_score,
        min_top1_margin=args.min_top1_margin,
        postcheck_min_zero=args.postcheck_min_zero,
        enable_profiling=True,
    )
    classifier = PageOrientationClassifier(config)
    if classifier.metadata.model_sha256.lower() != EXPECTED_MODEL_SHA256:
        raise OrientationError(
            "Unexpected page-orientation model SHA-256: "
            f"{classifier.metadata.model_sha256}; expected {EXPECTED_MODEL_SHA256}"
        )
    normalizer = PageOrientationNormalizer(config, classifier=classifier)
    cases: list[dict[str, Any]] = []

    for expected_orientation in sorted(EXPECTED_ORIENTATIONS):
        variant = work_dir / "variants" / f"page-{expected_orientation}.pdf"
        normalized = work_dir / "normalized" / f"page-{expected_orientation}-normalized.pdf"
        orientation_report = work_dir / "reports" / f"page-{expected_orientation}-orientation-report.json"
        write_rotated_pdf(selected_source, variant, {0: expected_orientation})
        _assert_selectable_text(variant, witness)

        before = _classify_one_page(classifier, variant, config.render_dpi)
        _assert_expected_prediction(before, expected_orientation, config, variant)
        result = normalizer.normalize(
            source=variant,
            normalized_pdf=normalized,
            report_path=orientation_report,
        )
        record = result.report["pages"][0]
        if record["predicted_orientation"] != expected_orientation:
            raise OrientationError(
                f"Normalization report disagrees with direct real-model inference for {variant}"
            )
        if not record["eligible_for_indexing"]:
            raise OrientationError(
                f"Normalization rejected real-model acceptance variant {variant}: "
                f"{record['review_reasons']}"
            )

        after = _classify_one_page(classifier, normalized, config.render_dpi)
        _assert_expected_prediction(after, 0, config, normalized)
        _assert_selectable_text(normalized, witness)
        if expected_orientation != 0:
            if record["post_rotation_predicted_orientation"] != 0:
                raise OrientationError(
                    f"Normalizer's built-in postcheck was not 0 degrees for {variant}: "
                    f"{record['post_rotation_predicted_orientation']}"
                )
            if record["post_rotation_zero_score"] < config.postcheck_min_zero:
                raise OrientationError(
                    f"Normalizer's built-in postcheck confidence was below threshold for {variant}"
                )

        cases.append(
            {
                "input_visual_rotation": expected_orientation,
                "variant": str(variant),
                "normalized_pdf": str(normalized),
                "orientation_report": str(orientation_report),
                "variant_page_count": 1,
                "normalized_page_count": 1,
                "direct_before": before,
                "normalizer_page_record": record,
                "direct_after": after,
            }
        )

    source_sha256_after = sha256_file(source)
    source_page_count_after, _ = _read_pdf_page_count_and_text(source, args.page)
    if source_sha256_after != source_sha256_before:
        raise OrientationError("Original input PDF SHA-256 changed during acceptance run")
    if source_page_count_after != source_page_count_before:
        raise OrientationError("Original input PDF page count changed during acceptance run")

    profile_path = _move_profile_to_work_dir(classifier.end_profiling(), work_dir)
    profile_summary = _node_profile_summary(profile_path)

    return {
        "status": "passed",
        "input_pdf": str(source),
        "source_page": args.page,
        "source_sha256_before": source_sha256_before,
        "source_sha256_after": source_sha256_after,
        "source_page_count_before": source_page_count_before,
        "source_page_count_after": source_page_count_after,
        "selectable_text_witness": witness,
        "model": classifier.metadata.report_payload(),
        "orientation_provider": classifier.provider,
        "configuration": config.report_payload(),
        "real_model_profile_baseline": {
            "model_sha256": EXPECTED_MODEL_SHA256,
            "onnxruntime": BASELINE_ONNXRUNTIME_VERSION,
            "historical_cpu_nodes": sorted(HISTORICAL_CPU_NODE_OPS),
            "historical_major_compute_provider": "CUDAExecutionProvider",
            "historical_cpu_node_duration_ratio": HISTORICAL_CPU_NODE_DURATION_RATIO,
        },
        "onnxruntime_version": str(getattr(classifier._ort, "__version__", "unknown")),
        "onnxruntime_profile": profile_summary,
        "cases": cases,
    }


def main() -> int:
    args = _parse_args()
    work_dir = args.work_dir.expanduser().resolve()
    evidence_path = work_dir / "real_orientation_acceptance.json"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        evidence = _run(args)
    except Exception as exc:
        failure = {
            "status": "failed",
            "input_pdf": str(args.input_pdf.expanduser()),
            "source_page": args.page,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        # Do not turn a failed safety preflight into an implicit overwrite of
        # prior evidence when the caller intentionally omitted --overwrite.
        if args.overwrite or not evidence_path.exists():
            write_json_atomic(evidence_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    write_json_atomic(evidence_path, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
