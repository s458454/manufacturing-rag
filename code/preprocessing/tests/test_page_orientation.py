"""Focused offline tests for the independent page-orientation stage.

They intentionally use a fake classifier: offline unit tests never execute or
download the PP-LCNet ONNX asset.  The separate server-only harness validates
the real model and its CUDA profile.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from page_orientation import (  # noqa: E402
    DEFAULT_PDF_ROTATION_CORRECTIONS,
    OrientationError,
    InferencePreprocess,
    OrientationConfig,
    OrientationModelMetadata,
    OrientationPrediction,
    _pdf_page_info,
    PageOrientationClassifier,
    PageOrientationNormalizer,
    document_id_for_source,
    load_orientation_model_metadata,
    orientation_model_input_evidence,
    probabilities_from_model_output,
    selected_pdf_page_indexes,
    sha256_file,
)
from verify_page_orientation_real_model import _node_profile_summary  # noqa: E402


class FakeClassifier:
    """Predict from the page's current /Rotate as a deterministic test oracle."""

    provider = "CPUExecutionProvider"

    def __init__(self, model_dir: Path) -> None:
        self.metadata = OrientationModelMetadata(
            model_name="PP-LCNet_x1_0_doc_ori",
            model_version="test",
            source="test-fixture",
            model_path=model_dir / "model.onnx",
            model_sha256="test-model-sha256",
            inference_yml_path=model_dir / "inference.yml",
            inference_yml_sha256="test-yaml-sha256",
            labels_path=model_dir / "labels.json",
            labels=("0", "90", "180", "270"),
            manifest_path=model_dir / "manifest.json",
            runtime="onnxruntime",
            preprocess=InferencePreprocess(
                resize_short=256,
                crop_size=(224, 224),
                scale=1.0 / 255.0,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

    def classify(self, image: Any) -> OrientationPrediction:
        # Tests attach this marker to a rendered image through a monkeypatch.
        orientation = int(getattr(image, "test_orientation", 0))
        scores = {str(angle): 0.0 for angle in (0, 90, 180, 270)}
        scores[str(orientation)] = 0.99
        scores["0" if orientation != 0 else "90"] = 0.01
        return OrientationPrediction(
            scores=scores,
            predicted_label=str(orientation),
            top1_score=0.99,
            top1_margin=0.98,
        )


def _make_pdf(path: Path, rotations: list[int]) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for rotation in rotations:
        page = writer.add_blank_page(width=612, height=792)
        if rotation:
            page.rotate(rotation)
    with path.open("wb") as stream:
        writer.write(stream)


def _patch_render_with_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    def render(pdf_document: Any, page_index: int, dpi: int) -> Any:
        page = pdf_document.get_page(page_index)
        image = Image.new("RGB", (64, 64), "white")
        # Keep enough dark pixels so the blank-page gate does not intercept the
        # fake model (the production threshold is deliberately conservative).
        for x in range(8):
            for y in range(8):
                image.putpixel((x, y), (0, 0, 0))
        image.test_orientation = page.get_rotation()  # type: ignore[attr-defined]
        page.close()
        return image

    monkeypatch.setattr("page_orientation.render_pdf_page", render)


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    return directory


def test_all_four_pdf_rotations_normalize_to_upright(
    tmp_path: Path, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_render_with_rotation(monkeypatch)
    source = tmp_path / "source.pdf"
    _make_pdf(source, [0, 90, 180, 270])
    original_sha256 = sha256_file(source)

    normalizer = PageOrientationNormalizer(
        OrientationConfig(model_dir=model_dir, device="cpu"),
        classifier=FakeClassifier(model_dir),
    )
    result = normalizer.normalize(
        source=source,
        normalized_pdf=tmp_path / "normalized" / "oriented.pdf",
        report_path=tmp_path / "orientation_report.json",
    )

    from pypdf import PdfReader

    output = PdfReader(str(result.normalized_pdf))
    assert len(output.pages) == 4
    assert [page.rotation % 360 for page in output.pages] == [0, 0, 0, 0]
    assert sha256_file(source) == original_sha256
    assert result.report["document_id"] == document_id_for_source(source, original_sha256)
    assert result.report["orientation_summary"] == {
        "upright_pages": 1,
        "rotated_pages": 3,
        "not_applicable_pages": 0,
        "review_required_pages": 0,
    }
    assert [page["applied_rotation"] for page in result.report["pages"]] == [
        0,
        270,
        180,
        90,
    ]
    assert result.report_path.is_file()


def test_page_range_limits_orientation_work_and_preserves_original_page_numbers(
    tmp_path: Path, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_render_with_rotation(monkeypatch)
    source = tmp_path / "source.pdf"
    _make_pdf(source, [90, 180, 270])
    normalizer = PageOrientationNormalizer(
        OrientationConfig(model_dir=model_dir, device="cpu"),
        classifier=FakeClassifier(model_dir),
    )

    result = normalizer.normalize(
        source=source,
        normalized_pdf=tmp_path / "normalized" / "oriented.pdf",
        report_path=tmp_path / "orientation_report.json",
        page_range=(2, 3),
    )

    from pypdf import PdfReader

    output = PdfReader(str(result.normalized_pdf))
    # Page 1 is intentionally outside the smoke-test range and its source
    # rotation is retained.  The physically complete PDF preserves original
    # page numbers, so Docling's page 2/3 provenance stays stable.
    assert [page.rotation % 360 for page in output.pages] == [90, 0, 0]
    assert [page["page_no"] for page in result.report["pages"]] == [2, 3]
    assert [page["pdf_page_index"] for page in result.report["pages"]] == [1, 2]
    assert result.report["page_count"] == 3
    assert result.report["processed_page_count"] == 2
    assert result.report["processed_page_range"] == [2, 3]
    assert result.report["processed_page_numbers"] == [2, 3]


def test_selected_pdf_page_indexes_rejects_out_of_bounds_ranges() -> None:
    assert list(selected_pdf_page_indexes(3, None)) == [0, 1, 2]
    assert list(selected_pdf_page_indexes(3, (2, 3))) == [1, 2]
    with pytest.raises(OrientationError, match="LAST <= source page count"):
        selected_pdf_page_indexes(3, (2, 4))


def test_orientation_report_cannot_overwrite_pdf_artifacts(
    tmp_path: Path, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_render_with_rotation(monkeypatch)
    source = tmp_path / "source.pdf"
    _make_pdf(source, [0])
    normalizer = PageOrientationNormalizer(
        OrientationConfig(model_dir=model_dir, device="cpu"),
        classifier=FakeClassifier(model_dir),
    )
    normalized = tmp_path / "normalized" / "oriented.pdf"

    with pytest.raises(OrientationError, match="report destination"):
        normalizer.normalize(
            source=source,
            normalized_pdf=normalized,
            report_path=normalized,
        )


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.enable_profiling = False


class _FakeOrtSession:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class _FakeCudaOrt:
    SessionOptions = _FakeSessionOptions

    def __init__(
        self,
        active_providers: list[str] | None = None,
        available_providers: list[str] | None = None,
    ) -> None:
        self.active_providers = active_providers or ["CUDAExecutionProvider"]
        self.available_providers = available_providers or [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        self.calls: list[dict[str, Any]] = []

    def get_available_providers(self) -> list[str]:
        return self.available_providers

    def InferenceSession(self, model_path: str, **kwargs: Any) -> _FakeOrtSession:
        self.calls.append({"model_path": model_path, **kwargs})
        return _FakeOrtSession(self.active_providers)


def test_cuda_orientation_session_prefers_cuda_and_registers_cpu_for_audited_nodes(
    model_dir: Path,
) -> None:
    classifier = object.__new__(PageOrientationClassifier)
    classifier.config = OrientationConfig(model_dir=model_dir, device="cuda")
    classifier.metadata = FakeClassifier(model_dir).metadata
    fake_ort = _FakeCudaOrt()
    classifier._ort = fake_ort

    session = classifier._create_session()

    assert session.get_providers() == ["CUDAExecutionProvider"]
    assert fake_ort.calls[0]["providers"] == [
        ("CUDAExecutionProvider", {}),
        "CPUExecutionProvider",
    ]
    assert "sess_options" not in fake_ort.calls[0]


def test_cuda_orientation_session_forwards_explicit_visible_device_id(
    model_dir: Path,
) -> None:
    classifier = object.__new__(PageOrientationClassifier)
    classifier.config = OrientationConfig(model_dir=model_dir, device="cuda:2")
    classifier.metadata = FakeClassifier(model_dir).metadata
    fake_ort = _FakeCudaOrt()
    classifier._ort = fake_ort

    classifier._create_session()

    assert fake_ort.calls[0]["providers"] == [
        ("CUDAExecutionProvider", {"device_id": "2"}),
        "CPUExecutionProvider",
    ]


def test_cuda_orientation_profiling_uses_session_options_without_disabling_cpu_fallback(
    model_dir: Path,
) -> None:
    classifier = object.__new__(PageOrientationClassifier)
    classifier.config = OrientationConfig(
        model_dir=model_dir,
        device="cuda:2",
        enable_profiling=True,
    )
    classifier.metadata = FakeClassifier(model_dir).metadata
    fake_ort = _FakeCudaOrt()
    classifier._ort = fake_ort

    classifier._create_session()

    assert fake_ort.calls[0]["providers"] == [
        ("CUDAExecutionProvider", {"device_id": "2"}),
        "CPUExecutionProvider",
    ]
    options = fake_ort.calls[0]["sess_options"]
    assert options.enable_profiling is True


def test_cuda_orientation_session_rejects_non_cuda_active_provider(model_dir: Path) -> None:
    classifier = object.__new__(PageOrientationClassifier)
    classifier.config = OrientationConfig(model_dir=model_dir, device="cuda")
    classifier.metadata = FakeClassifier(model_dir).metadata
    classifier._ort = _FakeCudaOrt(active_providers=["CPUExecutionProvider"])

    with pytest.raises(OrientationError, match="CUDA was requested, but the model did not activate"):
        classifier._create_session()


def test_cuda_orientation_session_rejects_when_cuda_is_not_available(model_dir: Path) -> None:
    classifier = object.__new__(PageOrientationClassifier)
    classifier.config = OrientationConfig(model_dir=model_dir, device="cuda")
    classifier.metadata = FakeClassifier(model_dir).metadata
    classifier._ort = _FakeCudaOrt(available_providers=["CPUExecutionProvider"])

    with pytest.raises(OrientationError, match="does not expose CUDAExecutionProvider"):
        classifier._create_session()


def test_cpu_orientation_session_registers_only_cpu_provider(model_dir: Path) -> None:
    classifier = object.__new__(PageOrientationClassifier)
    classifier.config = OrientationConfig(model_dir=model_dir, device="cpu")
    classifier.metadata = FakeClassifier(model_dir).metadata
    fake_ort = _FakeCudaOrt(active_providers=["CPUExecutionProvider"])
    classifier._ort = fake_ort

    classifier._create_session()

    assert fake_ort.calls[0]["providers"] == ["CPUExecutionProvider"]
    assert "sess_options" not in fake_ort.calls[0]


def test_vector_text_page_copy_preserves_text_and_page_mapping(tmp_path: Path, model_dir: Path) -> None:
    from reportlab.pdfgen.canvas import Canvas
    from pypdf import PdfReader
    from page_orientation import write_rotated_pdf

    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source))
    canvas.drawString(72, 720, "Selectable source text")
    canvas.showPage()
    canvas.save()
    original_sha256 = sha256_file(source)
    output = tmp_path / "normalized" / "oriented.pdf"

    write_rotated_pdf(source, output, {0: 90})

    original_reader = PdfReader(str(source))
    output_reader = PdfReader(str(output))
    assert len(output_reader.pages) == len(original_reader.pages) == 1
    assert "Selectable source text" in output_reader.pages[0].extract_text()
    assert output_reader.pages[0].rotation == 90
    assert sha256_file(source) == original_sha256


def test_document_id_uses_original_name_and_hash_prefix(tmp_path: Path) -> None:
    source = tmp_path / "same-name.pdf"
    source.write_bytes(b"original PDF identity")
    source_sha256 = sha256_file(source)

    assert document_id_for_source(source, source_sha256) == (
        f"same-name-{source_sha256[:16]}"
    )


def test_document_id_normalizes_path_unsafe_source_stems(tmp_path: Path) -> None:
    source = tmp_path / "A source (draft).pdf"
    source.write_bytes(b"source identity")

    assert document_id_for_source(source).startswith("A_source__draft-")


def test_orientation_dpi_is_limited_to_the_configured_rendering_range(
    model_dir: Path,
) -> None:
    with pytest.raises(ValueError, match="120-150 DPI"):
        OrientationConfig(model_dir=model_dir, device="cpu", render_dpi=119)
    with pytest.raises(ValueError, match="120-150 DPI"):
        OrientationConfig(model_dir=model_dir, device="cpu", render_dpi=151)


def test_orientation_model_input_evidence_checks_the_actual_center_crop(
    model_dir: Path,
) -> None:
    from PIL import Image

    config = OrientationConfig(model_dir=model_dir, device="cpu")
    preprocess = FakeClassifier(model_dir).metadata.preprocess
    image = Image.new("RGB", (600, 900), "white")
    # Ink only at the page edge proves that full-page ink alone is not enough.
    for x in range(0, 40):
        for y in range(0, 40):
            image.putpixel((x, y), (0, 0, 0))
    evidence = orientation_model_input_evidence(image, preprocess, config)

    assert evidence["has_sufficient_ink"] is False
    assert evidence["ink_ratio"] == 0.0


@pytest.mark.parametrize("device", ["cuda:", "cuda:-1", "cuda:１", "cuda:0:1", "cudaish"])
def test_orientation_config_rejects_ambiguous_cuda_device_selectors(
    model_dir: Path, device: str
) -> None:
    with pytest.raises(ValueError, match="cuda"):
        OrientationConfig(model_dir=model_dir, device=device)


def test_untrusted_model_metadata_is_rejected_without_download(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No model will be downloaded"):
        load_orientation_model_metadata(model_dir)


def test_placeholder_manifest_is_rejected_before_model_initialization(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"not-a-real-model")
    (model_dir / "inference.yml").write_text(
        """PreProcess:
  transform_ops:
    - ResizeImage: {resize_short: 256}
    - CropImage: {size: 224}
    - NormalizeImage:
        channel_num: 3
        mean: [0.485, 0.456, 0.406]
        order: ''
        scale: 0.00392156862745098
        std: [0.229, 0.224, 0.225]
    - ToCHWImage: null
PostProcess:
  Topk: {label_list: ['0', '90', '180', '270']}
""",
        encoding="utf-8",
    )
    (model_dir / "labels.json").write_text(
        '["0", "90", "180", "270"]', encoding="utf-8"
    )
    (model_dir / "manifest.json").write_text(
        """{
  "model_name": "PP-LCNet_x1_0_doc_ori",
  "model_version": "REPLACE_WITH_PROVISIONED_MODEL_VERSION",
  "source": "REPLACE_WITH_OFFICIAL_ONNX_SOURCE_URL",
  "sha256": "REPLACE_WITH_MODEL_ONNX_SHA256",
  "labels": ["0", "90", "180", "270"],
  "runtime": "onnxruntime"
}
""",
        encoding="utf-8",
    )

    with pytest.raises(OrientationError, match="model_version, sha256, source"):
        load_orientation_model_metadata(model_dir)


def test_softmax_keeps_four_class_probability_order() -> None:
    probabilities = probabilities_from_model_output([1.0, 2.0, 3.0, 4.0])
    assert probabilities.shape == (4,)
    assert float(probabilities.sum()) == pytest.approx(1.0)
    assert int(probabilities.argmax()) == 3


class _OutputShapeSession:
    def __init__(self, output: Any) -> None:
        self.output = output

    def run(self, output_names: Any, inputs: Any) -> list[Any]:
        return [self.output]


def test_orientation_classifier_requires_exact_one_by_four_model_output(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    classifier = object.__new__(PageOrientationClassifier)
    classifier.config = OrientationConfig(model_dir=model_dir, device="cpu")
    classifier.metadata = FakeClassifier(model_dir).metadata
    classifier._input_name = "image"
    classifier._session = _OutputShapeSession(np.zeros((4,), dtype=np.float32))
    classifier.last_output_shape = None
    monkeypatch.setattr(classifier, "_preprocess", lambda image: image)

    with pytest.raises(OrientationError, match=r"output shape \(1, 4\)"):
        classifier.classify(object())
    assert classifier.last_output_shape == (4,)


def test_real_model_profile_records_historical_cpu_and_compute_evidence(tmp_path: Path) -> None:
    profile_path = tmp_path / "onnxruntime_profile.json"
    profile_path.write_text(
        json.dumps(
            [
                {
                    "cat": "Node",
                    "dur": 1000,
                    "args": {"op_name": "Conv", "provider": "CUDAExecutionProvider"},
                },
                {
                    "cat": "Node",
                    "dur": 1000,
                    "args": {"op_name": "MatMul", "provider": "CUDAExecutionProvider"},
                },
                {
                    "cat": "Node",
                    "dur": 1000,
                    "args": {
                        "op_name": "GlobalAveragePool",
                        "provider": "CUDAExecutionProvider",
                    },
                },
                {
                    "cat": "Node",
                    "dur": 1000,
                    "args": {"op_name": "Softmax", "provider": "CUDAExecutionProvider"},
                },
                {
                    "cat": "Node",
                    "dur": 10,
                    "args": {"op_name": "Slice", "provider": "CPUExecutionProvider"},
                },
                {
                    "cat": "Node",
                    "dur": 10,
                    "args": {"op_name": "Concat", "provider": "CPUExecutionProvider"},
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = _node_profile_summary(profile_path)

    assert summary["observed_cpu_node_ops"] == ["Concat", "Slice"]
    assert summary["cpu_nodes_match_historical_baseline"] is True
    assert summary["cpu_node_duration_ratio"] == pytest.approx(20 / 4020)
    assert summary["historical_major_compute_provider"] == "CUDAExecutionProvider"
    assert summary["observed_major_compute_ops"]["Conv"] == ["CUDAExecutionProvider"]


def test_pdf_rotation_conversion_table_is_explicit_and_complete() -> None:
    assert DEFAULT_PDF_ROTATION_CORRECTIONS == {
        0: 0,
        90: 270,
        180: 180,
        270: 90,
    }


def test_non_right_angle_pdf_rotation_is_rejected(tmp_path: Path) -> None:
    from pypdf import PdfWriter
    from pypdf.generic import NameObject, NumberObject

    source = tmp_path / "unsupported-rotation.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Rotate")] = NumberObject(45)
    with source.open("wb") as stream:
        writer.write(stream)

    with pytest.raises(OrientationError, match="unsupported /Rotate"):
        _pdf_page_info(source)


def test_postcheck_temporary_pdf_is_removed_after_rotation_write_failure(
    tmp_path: Path, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import page_orientation

    _patch_render_with_rotation(monkeypatch)
    source = tmp_path / "source.pdf"
    _make_pdf(source, [90])
    normalized_pdf = tmp_path / "normalized" / "oriented.pdf"

    def write_then_fail(source_path: Path, destination: Path, rotations: dict[int, int]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"partially-written-provisional-output")
        raise OrientationError("simulated PDF write failure")

    monkeypatch.setattr(page_orientation, "write_rotated_pdf", write_then_fail)
    normalizer = PageOrientationNormalizer(
        OrientationConfig(model_dir=model_dir, device="cpu"),
        classifier=FakeClassifier(model_dir),
    )

    with pytest.raises(OrientationError, match="simulated PDF write failure"):
        normalizer.normalize(
            source=source,
            normalized_pdf=normalized_pdf,
            report_path=tmp_path / "orientation_report.json",
        )

    assert not list((tmp_path / "normalized").glob(".oriented.postcheck.*.pdf"))
