"""Auditable, page-level PDF orientation normalization before Docling.

The PP-LCNet document-orientation model is intentionally kept outside
RapidOCR.  It receives a low-resolution rendering of the PDF's current visual
appearance, writes a vector/text-preserving PDF copy with verified page
rotations, and records every decision for downstream indexing gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


MODEL_NAME = "PP-LCNet_x1_0_doc_ori"
EXPECTED_ORIENTATIONS = frozenset({0, 90, 180, 270})
DOCUMENT_ID_PREFIX_LENGTH = 16

# pypdf's PageObject.rotate() is clockwise.  This table is deliberately kept
# separate from model-label loading: labels come from the audited model assets,
# while the PDF correction mapping is covered by the four-direction test suite.
DEFAULT_PDF_ROTATION_CORRECTIONS: dict[int, int] = {
    0: 0,
    90: 270,
    180: 180,
    270: 90,
}


class OrientationError(RuntimeError):
    """Raised when orientation normalization cannot safely continue."""


@dataclass(frozen=True)
class OrientationConfig:
    """All tunable page-orientation settings in one auditable object."""

    model_dir: Path
    device: str
    render_dpi: int = 150
    min_top1_score: float = 0.90
    min_top1_margin: float = 0.15
    postcheck_min_zero: float = 0.90
    reject_uncertain: bool = True
    blank_ink_ratio_threshold: float = 0.0005
    min_model_input_ink_ratio: float = 0.0005
    blank_gray_threshold: int = 245
    enable_profiling: bool = False
    pdf_rotation_corrections: Mapping[int, int] = field(
        default_factory=lambda: dict(DEFAULT_PDF_ROTATION_CORRECTIONS)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_dir", Path(self.model_dir).expanduser().resolve())
        device = self.device.strip().lower()
        object.__setattr__(self, "device", device)
        if not 120 <= self.render_dpi <= 150:
            raise ValueError("orientation render_dpi must be within the supported 120-150 DPI range")
        for name in (
            "min_top1_score",
            "min_top1_margin",
            "postcheck_min_zero",
            "blank_ink_ratio_threshold",
            "min_model_input_ink_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"orientation {name} must be within [0, 1]")
        if not 0 <= self.blank_gray_threshold <= 255:
            raise ValueError("orientation blank_gray_threshold must be within [0, 255]")
        if device != "cpu" and re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
            raise ValueError(
                "Page orientation requires --device cpu, cuda, or cuda:N with a "
                "non-negative ASCII integer N; auto is not allowed because CPU "
                "fallback must be explicit."
            )

        corrections = {int(key): int(value) for key, value in self.pdf_rotation_corrections.items()}
        if set(corrections) != EXPECTED_ORIENTATIONS:
            raise ValueError(
                "pdf_rotation_corrections must define exactly 0, 90, 180, and 270"
            )
        if any(value not in EXPECTED_ORIENTATIONS for value in corrections.values()):
            raise ValueError("PDF correction angles must be 0, 90, 180, or 270")
        object.__setattr__(self, "pdf_rotation_corrections", corrections)

    def report_payload(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "model_dir": str(self.model_dir),
            "device": self.device,
            "onnx_execution_policy": (
                "cuda_preferred_mixed_execution"
                if self.device.startswith("cuda")
                else "cpu_only_execution"
            ),
            "cuda_provider_order": (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device.startswith("cuda")
                else ["CPUExecutionProvider"]
            ),
            "cuda_primary_provider_required": self.device.startswith("cuda"),
            "enable_profiling": self.enable_profiling,
            "render_dpi": self.render_dpi,
            "min_top1_score": self.min_top1_score,
            "min_top1_margin": self.min_top1_margin,
            "postcheck_min_zero": self.postcheck_min_zero,
            "reject_uncertain": self.reject_uncertain,
            "blank_ink_ratio_threshold": self.blank_ink_ratio_threshold,
            "min_model_input_ink_ratio": self.min_model_input_ink_ratio,
            "blank_gray_threshold": self.blank_gray_threshold,
            "pdf_rotation_corrections": {
                str(key): value
                for key, value in sorted(self.pdf_rotation_corrections.items())
            },
        }


@dataclass(frozen=True)
class InferencePreprocess:
    """Preprocessing values parsed from the official ``inference.yml``."""

    resize_short: int
    crop_size: tuple[int, int]
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


@dataclass(frozen=True)
class OrientationModelMetadata:
    """Verified local model metadata used in every audit record."""

    model_name: str
    model_version: str
    source: str
    model_path: Path
    model_sha256: str
    inference_yml_path: Path
    inference_yml_sha256: str
    labels_path: Path
    labels: tuple[str, ...]
    manifest_path: Path
    runtime: str
    preprocess: InferencePreprocess

    def report_payload(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source": self.source,
            "runtime": self.runtime,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "inference_yml_path": str(self.inference_yml_path),
            "inference_yml_sha256": self.inference_yml_sha256,
            "labels_path": str(self.labels_path),
            "labels": list(self.labels),
            "manifest_path": str(self.manifest_path),
            "preprocess": {
                "resize_short": self.preprocess.resize_short,
                "crop_size": list(self.preprocess.crop_size),
                "scale": self.preprocess.scale,
                "mean": list(self.preprocess.mean),
                "std": list(self.preprocess.std),
                "color_space": "RGB",
                "layout": "NCHW",
            },
        }


@dataclass(frozen=True)
class OrientationPrediction:
    """The complete four-class output for one rendered PDF page."""

    scores: Mapping[str, float]
    predicted_label: str
    top1_score: float
    top1_margin: float

    @property
    def predicted_orientation(self) -> int:
        return parse_orientation_label(self.predicted_label)


@dataclass
class NormalizedDocument:
    """Artifacts and audit state returned by one normalization run."""

    source: Path
    normalized_pdf: Path
    report_path: Path
    report: dict[str, Any]

    def write_report(self) -> None:
        write_json_atomic(self.report_path, self.report)


class OrientationClassifierProtocol(Protocol):
    metadata: OrientationModelMetadata
    provider: str

    def classify(self, image: Any) -> OrientationPrediction:
        """Return all four orientation probabilities for an RGB page image."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_id_for_source(source: Path, source_sha256: str | None = None) -> str:
    """Create a stable, readable ID from the original file only.

    The file stem keeps output paths usable to people, while the SHA-256 prefix
    prevents collisions between identically named source PDFs.  The normalized
    copy is never part of this identifier.
    """

    source = Path(source)
    digest = source_sha256 or sha256_file(source)
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in source.stem
    ).strip("._") or "document"
    return f"{safe_stem}-{digest[:DOCUMENT_ID_PREFIX_LENGTH]}"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def parse_orientation_label(label: str) -> int:
    try:
        value = int(str(label).strip())
    except (TypeError, ValueError) as exc:
        raise OrientationError(f"Invalid orientation label: {label!r}") from exc
    if value not in EXPECTED_ORIENTATIONS:
        raise OrientationError(
            f"Unsupported orientation label {label!r}; expected 0, 90, 180, or 270"
        )
    return value


def _required_path(model_dir: Path, filename: str) -> Path:
    path = model_dir / filename
    if not path.is_file():
        raise FileNotFoundError(
            "Required local page-orientation model asset is missing: "
            f"{path}. No model will be downloaded at runtime."
        )
    return path


def _is_manifest_placeholder(value: Any) -> bool:
    """Reject template values before they can be mistaken for audited provenance."""

    return not isinstance(value, str) or not value.strip() or "REPLACE_WITH_" in value


def _load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrientationError(f"Cannot read {description}: {path}") from exc


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OrientationError(
            "PyYAML is required to parse the official page-orientation "
            f"inference.yml ({path}); install PyYAML before running preprocessing."
        ) from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OrientationError(f"Cannot parse official inference.yml: {path}") from exc
    if not isinstance(payload, Mapping):
        raise OrientationError("Official inference.yml must contain a mapping at its root")
    return payload


def _labels_from_payload(payload: Any, path: Path) -> tuple[str, ...]:
    if isinstance(payload, list):
        raw_labels = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("labels"), list):
        raw_labels = payload["labels"]
    else:
        raise OrientationError(
            f"{path} must be a JSON label array or an object with a 'labels' array"
        )
    labels = tuple(str(label) for label in raw_labels)
    parsed = [parse_orientation_label(label) for label in labels]
    if len(labels) != 4 or len(set(labels)) != 4 or set(parsed) != EXPECTED_ORIENTATIONS:
        raise OrientationError(
            "labels.json must contain each of the four labels exactly once: "
            "0, 90, 180, 270"
        )
    return labels


def _number_list(value: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise OrientationError(f"Official inference.yml {field_name} must contain three values")
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise OrientationError(
            f"Official inference.yml {field_name} contains a non-numeric value"
        ) from exc
    if not all(math.isfinite(item) for item in numbers):
        raise OrientationError(f"Official inference.yml {field_name} contains a non-finite value")
    return numbers  # type: ignore[return-value]


def _parse_crop_size(value: Any) -> tuple[int, int]:
    if isinstance(value, bool):
        raise OrientationError("Official inference.yml CropImage.size must be an integer or two integers")
    if isinstance(value, int):
        size = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        try:
            size = (int(value[0]), int(value[1]))
        except (TypeError, ValueError) as exc:
            raise OrientationError(
                "Official inference.yml CropImage.size must contain integer values"
            ) from exc
    else:
        raise OrientationError("Official inference.yml CropImage.size is missing or invalid")
    if min(size) <= 0:
        raise OrientationError("Official inference.yml CropImage.size must be positive")
    return size


def _extract_inference_preprocess(payload: Mapping[str, Any]) -> tuple[InferencePreprocess, tuple[str, ...]]:
    try:
        transforms = payload["PreProcess"]["transform_ops"]
    except (KeyError, TypeError) as exc:
        raise OrientationError("Official inference.yml lacks PreProcess.transform_ops") from exc
    if not isinstance(transforms, list):
        raise OrientationError("Official inference.yml PreProcess.transform_ops must be a list")

    transform_map: dict[str, Any] = {}
    transform_order: list[str] = []
    for operation in transforms:
        if not isinstance(operation, Mapping) or len(operation) != 1:
            raise OrientationError("Every official preprocessing operation must be a one-key mapping")
        name, value = next(iter(operation.items()))
        if not isinstance(name, str):
            raise OrientationError("Official preprocessing operation names must be strings")
        transform_order.append(name)
        transform_map[name] = value

    expected_operations = ["ResizeImage", "CropImage", "NormalizeImage", "ToCHWImage"]
    if transform_order != expected_operations:
        raise OrientationError(
            "Unsupported page-orientation preprocessing pipeline in inference.yml; "
            f"expected {expected_operations}, received {transform_order}"
        )

    resize = transform_map["ResizeImage"]
    crop = transform_map["CropImage"]
    normalize = transform_map["NormalizeImage"]
    if not isinstance(resize, Mapping) or not isinstance(crop, Mapping) or not isinstance(normalize, Mapping):
        raise OrientationError("Official inference.yml preprocessing operation parameters are invalid")

    try:
        resize_short = int(resize["resize_short"])
        crop_size = _parse_crop_size(crop["size"])
        scale = float(normalize["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OrientationError("Official inference.yml is missing required preprocessing values") from exc
    if resize_short <= 0 or scale <= 0 or not math.isfinite(scale):
        raise OrientationError("Official inference.yml resize_short and scale must be finite positive values")
    if normalize.get("channel_num") != 3 or normalize.get("order") != "":
        raise OrientationError(
            "This implementation supports the official RGB HWC, three-channel "
            "PP-LCNet preprocessing configuration only"
        )

    try:
        labels = tuple(str(label) for label in payload["PostProcess"]["Topk"]["label_list"])
    except (KeyError, TypeError) as exc:
        raise OrientationError("Official inference.yml lacks PostProcess.Topk.label_list") from exc
    parsed_labels = [parse_orientation_label(label) for label in labels]
    if len(labels) != 4 or len(set(labels)) != 4 or set(parsed_labels) != EXPECTED_ORIENTATIONS:
        raise OrientationError(
            "Official inference.yml must declare each orientation label exactly once"
        )

    return (
        InferencePreprocess(
            resize_short=resize_short,
            crop_size=crop_size,
            scale=scale,
            mean=_number_list(normalize.get("mean"), "NormalizeImage.mean"),
            std=_number_list(normalize.get("std"), "NormalizeImage.std"),
        ),
        labels,
    )


def load_orientation_model_metadata(model_dir: Path) -> OrientationModelMetadata:
    """Validate all required local assets before creating an ONNX session."""

    model_dir = Path(model_dir).expanduser().resolve()
    model_path = _required_path(model_dir, "model.onnx")
    inference_yml_path = _required_path(model_dir, "inference.yml")
    labels_path = _required_path(model_dir, "labels.json")
    manifest_path = _required_path(model_dir, "manifest.json")

    manifest = _load_json(manifest_path, "orientation manifest.json")
    if not isinstance(manifest, Mapping):
        raise OrientationError("orientation manifest.json must contain a JSON object")
    required_manifest_fields = {
        "model_name",
        "model_version",
        "source",
        "sha256",
        "labels",
        "runtime",
    }
    missing_fields = sorted(
        field for field in required_manifest_fields if not manifest.get(field)
    )
    placeholder_fields = sorted(
        field
        for field in ("model_version", "source", "sha256")
        if _is_manifest_placeholder(manifest.get(field))
    )
    missing_fields = sorted(set(missing_fields) | set(placeholder_fields))
    if missing_fields:
        raise OrientationError(
            "orientation manifest.json is missing required values: " + ", ".join(missing_fields)
        )
    if manifest["model_name"] != MODEL_NAME:
        raise OrientationError(
            f"Expected orientation model {MODEL_NAME}, received {manifest['model_name']!r}"
        )
    if manifest["runtime"] != "onnxruntime":
        raise OrientationError("orientation manifest runtime must be 'onnxruntime'")

    labels = _labels_from_payload(_load_json(labels_path, "orientation labels.json"), labels_path)
    manifest_labels = _labels_from_payload({"labels": manifest["labels"]}, manifest_path)
    inference_payload = _load_yaml(inference_yml_path)
    preprocess, inference_labels = _extract_inference_preprocess(inference_payload)
    if labels != manifest_labels or labels != inference_labels:
        raise OrientationError(
            "labels.json, manifest.json, and inference.yml must contain the same "
            "ordered orientation labels"
        )

    model_sha256 = sha256_file(model_path)
    expected_sha256 = str(manifest["sha256"]).lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise OrientationError("orientation manifest sha256 must be a 64-character hexadecimal SHA-256")
    if model_sha256.lower() != expected_sha256:
        raise OrientationError(
            "orientation model SHA-256 does not match manifest.json: "
            f"expected {expected_sha256}, received {model_sha256}"
        )

    return OrientationModelMetadata(
        model_name=MODEL_NAME,
        model_version=str(manifest["model_version"]),
        source=str(manifest["source"]),
        model_path=model_path,
        model_sha256=model_sha256,
        inference_yml_path=inference_yml_path,
        inference_yml_sha256=sha256_file(inference_yml_path),
        labels_path=labels_path,
        labels=labels,
        manifest_path=manifest_path,
        runtime="onnxruntime",
        preprocess=preprocess,
    )


def _cuda_provider_options(device: str) -> dict[str, str]:
    if ":" not in device:
        return {}
    _, raw_device_id = device.split(":", maxsplit=1)
    try:
        device_id = int(raw_device_id)
    except ValueError as exc:
        raise OrientationError(f"Invalid CUDA device selector: {device!r}") from exc
    if device_id < 0:
        raise OrientationError(f"Invalid CUDA device selector: {device!r}")
    return {"device_id": str(device_id)}


class PageOrientationClassifier:
    """One ONNX Runtime session reused for every page in one process."""

    def __init__(self, config: OrientationConfig) -> None:
        self.config = config
        self.metadata = load_orientation_model_metadata(config.model_dir)
        self._ort = self._load_onnxruntime()
        self._session = self._create_session()
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise OrientationError(
                f"Expected one orientation-model input, received {len(inputs)}"
            )
        self._input_name = inputs[0].name
        self.last_output_shape: tuple[int, ...] | None = None

    def _load_onnxruntime(self) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OrientationError(
                "onnxruntime-gpu (for CUDA) or onnxruntime (for explicit CPU) is "
                "required for page orientation classification"
            ) from exc
        return ort

    def _create_session(self) -> Any:
        session_options: Any | None = None
        if self.config.device.startswith("cuda"):
            preload_dlls = getattr(self._ort, "preload_dlls", None)
            if callable(preload_dlls):
                preload_dlls(directory="")
            available = self._ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise OrientationError(
                    "--device cuda was requested, but ONNX Runtime does not expose "
                    "CUDAExecutionProvider for the page-orientation model "
                    f"(available: {available})."
                )
            # CUDA is intentionally first, with CPU available only for ONNX
            # nodes that CUDA EP cannot execute.  The post-creation primary
            # provider check below prevents a damaged CUDA setup from silently
            # turning this whole session into CPU execution.
            providers: list[Any] = [
                ("CUDAExecutionProvider", _cuda_provider_options(self.config.device)),
                "CPUExecutionProvider",
            ]
            if self.config.enable_profiling:
                session_options = self._ort.SessionOptions()
                session_options.enable_profiling = True
        else:
            providers = ["CPUExecutionProvider"]
            if self.config.enable_profiling:
                session_options = self._ort.SessionOptions()
                session_options.enable_profiling = True

        try:
            session_kwargs: dict[str, Any] = {"providers": providers}
            if session_options is not None:
                session_kwargs["sess_options"] = session_options
            session = self._ort.InferenceSession(str(self.metadata.model_path), **session_kwargs)
        except Exception as exc:
            raise OrientationError(
                "Unable to initialize the local PP-LCNet page-orientation ONNX session"
            ) from exc

        active = session.get_providers()
        if not active:
            raise OrientationError("ONNX Runtime created no page-orientation execution provider")
        if self.config.device.startswith("cuda"):
            if active[0] != "CUDAExecutionProvider":
                raise OrientationError(
                    "CUDA was requested, but the model did not activate "
                    "CUDAExecutionProvider as its primary provider. "
                    f"Active providers: {active}"
                )
        elif active[0] != "CPUExecutionProvider":
            raise OrientationError(
                "Explicit CPU page-orientation inference did not activate "
                f"CPUExecutionProvider (active: {active})"
            )

        self.provider = active[0]
        print(f"orientation_provider={self.provider}", flush=True)
        print(f"orientation_model={self.metadata.model_name}", flush=True)
        return session

    def end_profiling(self) -> Path:
        """Finish an explicitly enabled ONNX Runtime profile and return its JSON path."""

        if not self.config.enable_profiling:
            raise OrientationError("ONNX Runtime profiling was not enabled for this session")
        end_profiling = getattr(self._session, "end_profiling", None)
        if not callable(end_profiling):
            raise OrientationError("ONNX Runtime session does not support end_profiling()")
        try:
            profile_path = Path(str(end_profiling())).expanduser().resolve()
        except Exception as exc:
            raise OrientationError("Could not finalize ONNX Runtime orientation profile") from exc
        if not profile_path.is_file():
            raise OrientationError(f"ONNX Runtime profile was not written: {profile_path}")
        return profile_path

    def classify(self, image: Any) -> OrientationPrediction:
        tensor = self._preprocess(image)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:
            raise OrientationError("Page-orientation ONNX inference failed") from exc
        if not outputs:
            raise OrientationError("Page-orientation ONNX inference returned no outputs")

        try:
            import numpy as np
        except ImportError as exc:
            raise OrientationError("numpy is required for page-orientation inference") from exc

        raw_output = np.asarray(outputs[0], dtype=np.float64)
        self.last_output_shape = tuple(int(dimension) for dimension in raw_output.shape)
        if self.last_output_shape != (1, 4):
            raise OrientationError(
                "Expected page-orientation ONNX output shape (1, 4), received "
                f"{self.last_output_shape}"
            )
        raw_scores = raw_output[0]
        if raw_scores.size != len(self.metadata.labels):
            raise OrientationError(
                "Page-orientation ONNX output does not match audited label count: "
                f"expected {len(self.metadata.labels)}, received {raw_scores.size}"
            )
        if not np.all(np.isfinite(raw_scores)):
            raise OrientationError("Page-orientation ONNX output contains non-finite scores")

        probabilities = probabilities_from_model_output(raw_scores)
        ranked_indexes = np.argsort(probabilities)[::-1]
        top_index = int(ranked_indexes[0])
        second_index = int(ranked_indexes[1])
        scores = {
            self.metadata.labels[index]: float(probabilities[index])
            for index in range(len(self.metadata.labels))
        }
        return OrientationPrediction(
            scores=scores,
            predicted_label=self.metadata.labels[top_index],
            top1_score=float(probabilities[top_index]),
            top1_margin=float(probabilities[top_index] - probabilities[second_index]),
        )

    def _preprocess(self, image: Any) -> Any:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise OrientationError(
                "numpy and Pillow are required for page-orientation preprocessing"
            ) from exc

        if not isinstance(image, Image.Image):
            raise OrientationError("Page-orientation classifier requires a PIL image")
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise OrientationError("Cannot classify an empty rendered PDF page")

        scale = self.metadata.preprocess.resize_short / min(width, height)
        resized_size = (round(width * scale), round(height * scale))
        image = image.resize(resized_size, Image.Resampling.BILINEAR)
        crop_width, crop_height = self.metadata.preprocess.crop_size
        if image.width < crop_width or image.height < crop_height:
            raise OrientationError(
                "Official orientation preprocessing crop is larger than the resized page"
            )
        left = (image.width - crop_width) // 2
        top = (image.height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))

        pixels = np.asarray(image, dtype=np.float32)
        mean = np.asarray(self.metadata.preprocess.mean, dtype=np.float32)
        std = np.asarray(self.metadata.preprocess.std, dtype=np.float32)
        normalized = (pixels * self.metadata.preprocess.scale - mean) / std
        return np.ascontiguousarray(normalized.transpose((2, 0, 1))[None, ...])


def probabilities_from_model_output(raw_scores: Any) -> Any:
    """Keep probability outputs intact; apply softmax only when logits are returned."""

    try:
        import numpy as np
    except ImportError as exc:
        raise OrientationError("numpy is required for page-orientation inference") from exc

    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    if scores.size != 4 or not np.all(np.isfinite(scores)):
        raise OrientationError("Expected four finite page-orientation scores")
    score_sum = float(scores.sum())
    if np.all(scores >= 0.0) and math.isclose(score_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        return scores / score_sum

    shifted = scores - float(scores.max())
    exponentials = np.exp(shifted)
    denominator = float(exponentials.sum())
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise OrientationError("Unable to normalize page-orientation model scores")
    return exponentials / denominator


def render_pdf_page(pdf_document: Any, page_index: int, dpi: int) -> Any:
    """Render the current PDF display appearance, including its existing /Rotate."""

    try:
        page = pdf_document.get_page(page_index)
        bitmap = page.render(scale=dpi / 72.0, rev_byteorder=True)
        try:
            # Copy disconnects Pillow storage from the PDFium bitmap before closing it.
            return bitmap.to_pil().convert("RGB").copy()
        finally:
            close = getattr(bitmap, "close", None)
            if callable(close):
                close()
    finally:
        close = getattr(locals().get("page"), "close", None)
        if callable(close):
            close()


def rendered_page_evidence(image: Any, config: OrientationConfig) -> dict[str, Any]:
    """Identify blank / effectively blank pages before a classifier can overfit them."""

    try:
        import numpy as np
    except ImportError as exc:
        raise OrientationError("numpy is required for page-orientation evidence checks") from exc

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if pixels.size == 0:
        return {
            "width": 0,
            "height": 0,
            "ink_ratio": 0.0,
            "gray_mean": None,
            "gray_std": None,
            "is_blank": True,
        }
    gray = (
        pixels[..., 0].astype(np.float32) * 0.299
        + pixels[..., 1].astype(np.float32) * 0.587
        + pixels[..., 2].astype(np.float32) * 0.114
    )
    ink_ratio = float((gray < config.blank_gray_threshold).mean())
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    is_blank = ink_ratio <= config.blank_ink_ratio_threshold
    return {
        "width": int(image.width),
        "height": int(image.height),
        "ink_ratio": ink_ratio,
        "gray_mean": gray_mean,
        "gray_std": gray_std,
        "is_blank": is_blank,
    }


def orientation_model_input_evidence(
    image: Any,
    preprocess: InferencePreprocess,
    config: OrientationConfig,
) -> dict[str, Any]:
    """Measure ink in the exact resized centre crop consumed by PP-LCNet."""

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise OrientationError(
            "numpy and Pillow are required for orientation input evidence"
        ) from exc
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        return {"ink_ratio": 0.0, "has_sufficient_ink": False}
    scale = preprocess.resize_short / min(width, height)
    resized = rgb.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.BILINEAR,
    )
    crop_width, crop_height = preprocess.crop_size
    if resized.width < crop_width or resized.height < crop_height:
        raise OrientationError(
            "Official orientation preprocessing crop is larger than the resized page"
        )
    left = (resized.width - crop_width) // 2
    top = (resized.height - crop_height) // 2
    crop = resized.crop((left, top, left + crop_width, top + crop_height))
    pixels = np.asarray(crop, dtype=np.uint8)
    gray = (
        pixels[..., 0].astype(np.float32) * 0.299
        + pixels[..., 1].astype(np.float32) * 0.587
        + pixels[..., 2].astype(np.float32) * 0.114
    )
    ink_ratio = float((gray < config.blank_gray_threshold).mean())
    return {
        "resized_size": [int(resized.width), int(resized.height)],
        "crop_box": [left, top, left + crop_width, top + crop_height],
        "crop_size": [crop_width, crop_height],
        "ink_ratio": ink_ratio,
        "minimum_ink_ratio": config.min_model_input_ink_ratio,
        "has_sufficient_ink": ink_ratio > config.min_model_input_ink_ratio,
    }


def _prediction_payload(prediction: OrientationPrediction) -> dict[str, Any]:
    return {
        "scores": {label: float(score) for label, score in prediction.scores.items()},
        "predicted_orientation": prediction.predicted_orientation,
        "top1_score": float(prediction.top1_score),
        "top1_margin": float(prediction.top1_margin),
    }


def is_confident_prediction(prediction: OrientationPrediction, config: OrientationConfig) -> bool:
    return (
        prediction.top1_score >= config.min_top1_score
        and prediction.top1_margin >= config.min_top1_margin
    )


def orientation_summary_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return the four stable orientation counters used by downstream reports.

    A later parse-quality gate may replace an initially accepted orientation
    decision with ``orientation_or_parse_uncertain``.  That is still a
    review-required outcome for the stable four-category summary; the detailed
    post-Docling counter records the more specific reason separately.
    """

    decisions = [str(record.get("decision", "review_required")) for record in records]
    return {
        "upright_pages": sum(decision == "accepted_upright" for decision in decisions),
        "rotated_pages": sum(decision == "accepted_rotated" for decision in decisions),
        "not_applicable_pages": sum(decision == "not_applicable" for decision in decisions),
        "review_required_pages": sum(
            decision in {"review_required", "orientation_or_parse_uncertain"}
            for decision in decisions
        ),
    }


def write_rotated_pdf(source: Path, destination: Path, rotations: Mapping[int, int]) -> None:
    """Create a new PDF by changing page rotation metadata only, never rasterizing."""

    try:
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import NameObject, NumberObject
    except ImportError as exc:
        raise OrientationError(
            "pypdf>=4 is required to write the text-preserving normalized PDF copy"
        ) from exc

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise OrientationError("Refusing to overwrite the original PDF during orientation normalization")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        reader = PdfReader(str(source))
        if reader.is_encrypted:
            raise OrientationError("Encrypted PDFs are not supported for page-orientation normalization")
        writer = PdfWriter(clone_from=reader)
        for page_index, clockwise_rotation in rotations.items():
            if not 0 <= page_index < len(writer.pages):
                raise OrientationError(
                    f"Page index {page_index} is outside the PDF page range"
                )
            if clockwise_rotation not in EXPECTED_ORIENTATIONS:
                raise OrientationError(
                    f"Page {page_index + 1} has unsupported rotation {clockwise_rotation}"
                )
            if clockwise_rotation:
                page = writer.pages[page_index]
                # Set the absolute visual rotation modulo 360 rather than using
                # PageObject.rotate(), which can serialise e.g. 90 + 270 as
                # /Rotate 360.  This preserves every page content stream while
                # keeping normalized page metadata canonical and auditable.
                page[NameObject("/Rotate")] = NumberObject(
                    (int(page.rotation) + clockwise_rotation) % 360
                )
        with temporary.open("wb") as stream:
            writer.write(stream)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _pdf_page_info(path: Path) -> tuple[int, list[int]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise OrientationError("pypdf>=4 is required for PDF orientation normalization") from exc
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise OrientationError("Encrypted PDFs are not supported for page-orientation normalization")
    rotations: list[int] = []
    for page_index, page in enumerate(reader.pages):
        rotation = int(page.rotation) % 360
        if rotation not in EXPECTED_ORIENTATIONS:
            raise OrientationError(
                f"PDF page {page_index + 1} has unsupported /Rotate value "
                f"{page.rotation!r}; expected a multiple of 90 degrees"
            )
        rotations.append(rotation)
    return len(reader.pages), rotations


def selected_pdf_page_indexes(
    page_count: int,
    page_range: tuple[int, int] | Sequence[int] | None,
) -> range:
    """Return zero-based source indexes for an optional inclusive page range.

    The CLI and Docling both use one-based, inclusive PDF page numbers.  The
    orientation worker must use that same convention before rendering, while
    the normalized PDF keeps all original pages in their original positions.
    """

    if page_count < 1:
        raise OrientationError("PDF has no pages to normalize")
    if page_range is None:
        return range(page_count)
    if (
        isinstance(page_range, (str, bytes))
        or not isinstance(page_range, Sequence)
        or len(page_range) != 2
    ):
        raise OrientationError(
            "page_range must contain exactly two one-based inclusive page numbers"
        )
    first, last = page_range
    if isinstance(first, bool) or isinstance(last, bool):
        raise OrientationError("page_range values must be integer page numbers")
    try:
        first = int(first)
        last = int(last)
    except (TypeError, ValueError) as exc:
        raise OrientationError("page_range values must be integer page numbers") from exc
    if first < 1 or last < first or last > page_count:
        raise OrientationError(
            "page_range must satisfy 1 <= FIRST <= LAST <= "
            f"source page count ({page_count}); received ({first}, {last})"
        )
    return range(first - 1, last)


class PageOrientationNormalizer:
    """Run the standalone pre-Docling orientation phase for one source PDF."""

    def __init__(
        self,
        config: OrientationConfig,
        classifier: OrientationClassifierProtocol | None = None,
    ) -> None:
        self.config = config
        self.classifier = classifier or PageOrientationClassifier(config)

    def normalize(
        self,
        source: Path,
        normalized_pdf: Path,
        report_path: Path,
        page_range: tuple[int, int] | Sequence[int] | None = None,
    ) -> NormalizedDocument:
        """Normalize trustworthy pages and write report/PDF artifacts atomically.

        ``page_range`` is a one-based inclusive source-PDF range.  It limits
        model rendering/classification and rotation changes, not the physical
        page sequence of the normalized PDF; this keeps page provenance stable
        for Docling and downstream audit records.
        """

        source = Path(source).expanduser().resolve()
        normalized_pdf = Path(normalized_pdf).expanduser().resolve()
        report_path = Path(report_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input PDF not found: {source}")
        if source.suffix.lower() != ".pdf":
            raise OrientationError(f"Page orientation supports PDF input only: {source}")

        source_sha256_before = sha256_file(source)
        document_id = document_id_for_source(source, source_sha256_before)
        page_count, existing_rotations = _pdf_page_info(source)
        page_indexes = selected_pdf_page_indexes(page_count, page_range)
        processed_page_numbers = [page_index + 1 for page_index in page_indexes]
        if normalized_pdf == source:
            raise OrientationError(
                "Normalized PDF destination must differ from the original source PDF"
            )
        if report_path in {source, normalized_pdf}:
            raise OrientationError(
                "Orientation report destination must differ from both the original "
                "and normalized PDF destinations"
            )
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise OrientationError(
                "pypdfium2 is required to render PDF pages for orientation classification"
            ) from exc

        pages: list[dict[str, Any]] = []
        page_records: dict[int, dict[str, Any]] = {}
        candidate_rotations: dict[int, int] = {}
        document = pdfium.PdfDocument(str(source))
        try:
            for page_index in page_indexes:
                image = render_pdf_page(document, page_index, self.config.render_dpi)
                evidence = rendered_page_evidence(image, self.config)
                model_input_evidence = orientation_model_input_evidence(
                    image,
                    self.classifier.metadata.preprocess,
                    self.config,
                )
                record: dict[str, Any] = {
                    "page_no": page_index + 1,
                    "pdf_page_index": page_index,
                    "existing_pdf_rotation": existing_rotations[page_index],
                    "scores": None,
                    "predicted_orientation": None,
                    "top1_score": None,
                    "top1_margin": None,
                    "attempted_rotation": 0,
                    "applied_rotation": 0,
                    "post_rotation_scores": None,
                    "post_rotation_predicted_orientation": None,
                    "post_rotation_zero_score": None,
                    "decision": "review_required",
                    "eligible_for_indexing": False,
                    "model_name": self.classifier.metadata.model_name,
                    "model_sha256": self.classifier.metadata.model_sha256,
                    "render_evidence": evidence,
                    "model_input_evidence": model_input_evidence,
                    "review_reasons": [],
                }
                page_records[page_index] = record
                if evidence["is_blank"]:
                    record["decision"] = "not_applicable"
                    record["review_reasons"].append("blank_or_insufficient_visual_evidence")
                    pages.append(record)
                    continue
                if not model_input_evidence["has_sufficient_ink"]:
                    record["decision"] = "review_required"
                    record["review_reasons"].append(
                        "orientation_model_center_crop_has_insufficient_ink"
                    )
                    pages.append(record)
                    continue

                prediction = self.classifier.classify(image)
                record.update(_prediction_payload(prediction))
                if not is_confident_prediction(prediction, self.config):
                    record["decision"] = "review_required"
                    record["review_reasons"].append(
                        "orientation_score_or_margin_below_configured_threshold"
                    )
                    pages.append(record)
                    continue

                predicted_orientation = prediction.predicted_orientation
                if predicted_orientation == 0:
                    record["decision"] = "accepted_upright"
                    record["eligible_for_indexing"] = True
                    pages.append(record)
                    continue

                correction = self.config.pdf_rotation_corrections[predicted_orientation]
                record["attempted_rotation"] = correction
                candidate_rotations[page_index] = correction
                pages.append(record)
        finally:
            close = getattr(document, "close", None)
            if callable(close):
                close()

        verified_rotations: dict[int, int] = {}
        provisional_pdf: Path | None = None
        if candidate_rotations:
            provisional_pdf = normalized_pdf.with_name(
                f".{normalized_pdf.stem}.postcheck.{uuid.uuid4().hex}.pdf"
            )
            provisional_document: Any | None = None
            try:
                write_rotated_pdf(source, provisional_pdf, candidate_rotations)
                provisional_document = pdfium.PdfDocument(str(provisional_pdf))
                for page_index, candidate_rotation in candidate_rotations.items():
                    post_image = render_pdf_page(
                        provisional_document,
                        page_index,
                        self.config.render_dpi,
                    )
                    post_prediction = self.classifier.classify(post_image)
                    record = page_records[page_index]
                    record["post_rotation_scores"] = {
                        label: float(score)
                        for label, score in post_prediction.scores.items()
                    }
                    record["post_rotation_predicted_orientation"] = (
                        post_prediction.predicted_orientation
                    )
                    zero_score = float(post_prediction.scores.get("0", 0.0))
                    record["post_rotation_zero_score"] = zero_score
                    if (
                        post_prediction.predicted_orientation == 0
                        and zero_score >= self.config.postcheck_min_zero
                    ):
                        verified_rotations[page_index] = candidate_rotation
                        record["applied_rotation"] = candidate_rotation
                        record["decision"] = "accepted_rotated"
                        record["eligible_for_indexing"] = True
                    else:
                        record["decision"] = "review_required"
                        record["review_reasons"].append(
                            "post_rotation_verification_failed"
                        )
            finally:
                close = getattr(provisional_document, "close", None)
                if callable(close):
                    close()
                if provisional_pdf.exists():
                    provisional_pdf.unlink()

        # Rebuild from the original so a failed post-check cannot leave an
        # unverified rotation in the PDF consumed by Docling. If no trustworthy
        # page needed rotation, retain a byte-for-byte copied standardization
        # artifact rather than needlessly rewriting a digital PDF.
        if verified_rotations:
            write_rotated_pdf(source, normalized_pdf, verified_rotations)
        else:
            normalized_pdf.parent.mkdir(parents=True, exist_ok=True)
            temporary = normalized_pdf.with_name(
                f".{normalized_pdf.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
                    while block := input_stream.read(1024 * 1024):
                        output_stream.write(block)
                os.replace(temporary, normalized_pdf)
            finally:
                if temporary.exists():
                    temporary.unlink(missing_ok=True)
        output_page_count, _output_rotations = _pdf_page_info(normalized_pdf)
        if output_page_count != page_count:
            raise OrientationError(
                "Normalized PDF page count differs from the original PDF; refusing output"
            )
        source_sha256_after = sha256_file(source)
        if source_sha256_after != source_sha256_before:
            raise OrientationError("Original PDF SHA-256 changed during normalization")

        summary = orientation_summary_from_records(pages)
        report = {
            "source": str(source),
            "source_sha256": source_sha256_before,
            "normalized_source": str(normalized_pdf),
            "normalized_source_sha256": sha256_file(normalized_pdf),
            "document_id": document_id,
            "page_count": page_count,
            "processed_page_count": len(pages),
            "processed_page_range": (
                [processed_page_numbers[0], processed_page_numbers[-1]]
                if processed_page_numbers
                else None
            ),
            "processed_page_numbers": processed_page_numbers,
            "page_mapping": "identity: normalized PDF page N corresponds to original PDF page N",
            "configuration": self.config.report_payload(),
            "model": self.classifier.metadata.report_payload(),
            "orientation_provider": self.classifier.provider,
            "orientation_summary": summary,
            "manual_review_required": bool(summary["review_required_pages"]),
            "pages": pages,
        }
        result = NormalizedDocument(
            source=source,
            normalized_pdf=normalized_pdf,
            report_path=report_path,
            report=report,
        )
        result.write_report()
        return result
