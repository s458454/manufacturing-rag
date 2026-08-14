"""Regression tests for Docling's serializable RapidOCR configuration path."""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPROCESSING_DIR = PROJECT_ROOT / "code" / "preprocessing"
DOCLING_SOURCE = PROJECT_ROOT / "docling"
for directory in (PREPROCESSING_DIR, DOCLING_SOURCE):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def _load_pdf_preprocess(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the CLI with its broad Docling type surface replaced by fakes."""

    class _FakeLabel:
        CAPTION = "caption"
        SECTION_HEADER = "section_header"

        def __iter__(self) -> Any:
            return iter((self.CAPTION, self.SECTION_HEADER))

    fake_doc_types = types.ModuleType("docling_core.types.doc")
    fake_doc_types.DocItemLabel = _FakeLabel()
    fake_doc_types.PictureItem = object
    fake_doc_types.TableItem = object
    fake_doc_types.TextItem = object
    fake_doc_types.DocItem = object

    fake_common = types.ModuleType("docling_core.transforms.serializer.common")
    fake_common.create_ser_result = lambda **kwargs: types.SimpleNamespace(
        text=kwargs.get("text", ""), spans=[]
    )

    class _FakeMarkdownParams:
        pass

    fake_markdown = types.ModuleType("docling_core.transforms.serializer.markdown")
    fake_markdown.MarkdownDocSerializer = object
    fake_markdown.MarkdownParams = _FakeMarkdownParams

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
        "docling_core.types.doc": fake_doc_types,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("pdf_preprocess", None)
    return importlib.import_module("pdf_preprocess")


def test_pdf_preprocess_passes_no_custom_rapidocr_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI must not inject ONNX Python objects into OmegaConf parameters."""

    module = _load_pdf_preprocess(monkeypatch)
    captured_ocr_options: dict[str, Any] = {}

    class _StopAfterOcrOptions(RuntimeError):
        pass

    args = types.SimpleNamespace(
        device="cuda:2",
        enable_direction_classifier=False,
        num_threads=4,
        document_timeout=3600.0,
        page_range=None,
        disable_page_orientation=True,
    )
    monkeypatch.setattr(module, "parse_args", lambda: args)
    monkeypatch.setattr(
        module,
        "validate_inputs",
        lambda _args: (tmp_path / "source.pdf", tmp_path / "output", "doc-id"),
    )
    monkeypatch.setattr(module, "run_page_orientation_stage", lambda *_args: None)
    monkeypatch.setattr(module, "require_cuda_runtime", lambda *_args: None)
    monkeypatch.setattr(module, "AcceleratorOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "LayoutOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "HeadingHierarchyOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "TableStructureOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        module, "TableFormerMode", types.SimpleNamespace(ACCURATE="accurate")
    )

    def capture_rapidocr_options(**kwargs: Any) -> object:
        captured_ocr_options.update(kwargs)
        return object()

    monkeypatch.setattr(module, "RapidOcrOptions", capture_rapidocr_options)
    monkeypatch.setattr(
        module,
        "PdfPipelineOptions",
        lambda **_kwargs: (_ for _ in ()).throw(_StopAfterOcrOptions()),
    )

    with pytest.raises(_StopAfterOcrOptions):
        module.run_preprocessing(
            args,
            tmp_path / "source.pdf",
            tmp_path / "output",
            "doc-id",
        )

    assert "rapidocr_params" not in captured_ocr_options
    assert all(
        isinstance(value, (str, int, float, bool, type(None), list, dict))
        for value in captured_ocr_options.values()
    )
    json.dumps(captured_ocr_options)


def test_pdf_preprocess_configures_tableformer_v1_accurate_with_matching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load_pdf_preprocess(monkeypatch)
    captured_pipeline_options: dict[str, Any] = {}

    class _StopAfterPipelineOptions(RuntimeError):
        pass

    args = types.SimpleNamespace(
        device="cuda:0",
        enable_direction_classifier=False,
        num_threads=4,
        document_timeout=3600.0,
        page_range=None,
        disable_page_orientation=True,
    )
    monkeypatch.setattr(module, "require_cuda_runtime", lambda *_args: None)
    monkeypatch.setattr(module, "run_page_orientation_stage", lambda *_args: None)
    monkeypatch.setattr(module, "AcceleratorOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "RapidOcrOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "LayoutOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "HeadingHierarchyOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "TableStructureOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        module, "TableFormerMode", types.SimpleNamespace(ACCURATE="accurate")
    )

    def capture_pipeline_options(**kwargs: Any) -> object:
        captured_pipeline_options.update(kwargs)
        raise _StopAfterPipelineOptions()

    monkeypatch.setattr(module, "PdfPipelineOptions", capture_pipeline_options)
    with pytest.raises(_StopAfterPipelineOptions):
        module.run_preprocessing(args, tmp_path / "source.pdf", tmp_path / "output", "doc-id")

    assert captured_pipeline_options["do_table_structure"] is True
    assert captured_pipeline_options["table_structure_options"] == {
        "mode": "accurate",
        "do_cell_matching": True,
    }


class _FakeAcceleratorDevice(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"


class _FakeBaseOcrModel:
    def __init__(self, enabled: bool, artifacts_path: Path | None, options: Any, accelerator_options: Any):
        self.enabled = enabled
        self.artifacts_path = artifacts_path
        self.options = options
        self.accelerator_options = accelerator_options


class _FakeEngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    OPENVINO = "openvino"
    PADDLE = "paddle"
    TORCH = "torch"


class _FakeOnnxSession:
    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers


class _FakeRapidOCR:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    last_params: dict[str, Any] | None = None

    def __init__(self, *, params: dict[str, Any]) -> None:
        type(self).last_params = params
        session = _FakeOnnxSession(type(self).providers)
        self.text_det = types.SimpleNamespace(session=types.SimpleNamespace(session=session))
        self.text_cls = types.SimpleNamespace(session=types.SimpleNamespace(session=session))
        self.text_rec = types.SimpleNamespace(session=types.SimpleNamespace(session=session))


def _load_rapid_ocr_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: str,
    providers: list[str],
) -> tuple[Any, type[_FakeRapidOCR]]:
    """Load the real Docling RapidOcrModel source with only external pieces faked."""

    for name in list(sys.modules):
        if name == "docling" or name.startswith("docling."):
            sys.modules.pop(name, None)
    fake_doc_types = types.ModuleType("docling_core.types.doc")
    fake_doc_types.BoundingBox = object
    fake_doc_types.CoordOrigin = types.SimpleNamespace(TOPLEFT="top-left")
    fake_page_types = types.ModuleType("docling_core.types.doc.page")
    fake_page_types.BoundingRectangle = object
    fake_page_types.TextCell = object
    monkeypatch.setitem(sys.modules, "docling_core", types.ModuleType("docling_core"))
    monkeypatch.setitem(sys.modules, "docling_core.types", types.ModuleType("docling_core.types"))
    monkeypatch.setitem(sys.modules, "docling_core.types.doc", fake_doc_types)
    monkeypatch.setitem(sys.modules, "docling_core.types.doc.page", fake_page_types)

    fake_accelerator = types.ModuleType("docling.datamodel.accelerator_options")
    fake_accelerator.AcceleratorDevice = _FakeAcceleratorDevice
    fake_accelerator.AcceleratorOptions = object
    fake_base_models = types.ModuleType("docling.datamodel.base_models")
    fake_base_models.Page = object
    fake_document = types.ModuleType("docling.datamodel.document")
    fake_document.ConversionResult = object
    fake_pipeline = types.ModuleType("docling.datamodel.pipeline_options")
    fake_pipeline.OcrOptions = object
    fake_pipeline.RapidOcrOptions = object
    fake_settings = types.ModuleType("docling.datamodel.settings")
    fake_settings.settings = types.SimpleNamespace(cache_dir=Path(".cache"))
    fake_base_ocr = types.ModuleType("docling.models.base_ocr_model")
    fake_base_ocr.BaseOcrModel = _FakeBaseOcrModel
    fake_accelerator_utils = types.ModuleType("docling.utils.accelerator_utils")
    fake_accelerator_utils.decide_device = lambda _device: device
    fake_profiling = types.ModuleType("docling.utils.profiling")
    fake_profiling.TimeRecorder = object
    fake_utils = types.ModuleType("docling.utils.utils")
    fake_utils.download_url_with_progress = lambda *args, **kwargs: None
    modules = {
        "docling.datamodel.accelerator_options": fake_accelerator,
        "docling.datamodel.base_models": fake_base_models,
        "docling.datamodel.document": fake_document,
        "docling.datamodel.pipeline_options": fake_pipeline,
        "docling.datamodel.settings": fake_settings,
        "docling.models.base_ocr_model": fake_base_ocr,
        "docling.utils.accelerator_utils": fake_accelerator_utils,
        "docling.utils.profiling": fake_profiling,
        "docling.utils.utils": fake_utils,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    rapidocr_module = types.ModuleType("rapidocr")
    rapidocr_module.EngineType = _FakeEngineType
    fake_reader = type("ConfiguredFakeRapidOCR", (_FakeRapidOCR,), {"providers": providers})
    rapidocr_module.RapidOCR = fake_reader
    monkeypatch.setitem(sys.modules, "rapidocr", rapidocr_module)
    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.preload_dlls = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    module = importlib.import_module("docling.models.stages.ocr.rapid_ocr_model")
    return module, fake_reader


def _rapid_options() -> Any:
    return types.SimpleNamespace(
        backend="onnxruntime",
        lang=["english"],
        det_model_path="det.onnx",
        cls_model_path="cls.onnx",
        rec_model_path="rec.onnx",
        rec_keys_path=None,
        font_path="font.ttf",
        text_score=0.5,
        use_det=True,
        use_cls=True,
        use_rec=True,
        rec_font_path=None,
        rapidocr_params={},
    )


def test_docling_rapidocr_sets_serializable_cuda_configuration_and_checks_sessions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    module, fake_reader = _load_rapid_ocr_model(
        monkeypatch,
        device="cuda:2",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    caplog.set_level(logging.INFO)

    model = module.RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=_rapid_options(),
        accelerator_options=types.SimpleNamespace(device="cuda:2", num_threads=4),
    )

    assert model.reader is not None
    assert fake_reader.last_params is not None
    params = fake_reader.last_params
    assert params["EngineConfig.onnxruntime.use_cuda"] is True
    assert params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] == 2
    assert params["Global.use_det"] is True
    assert params["Global.use_cls"] is True
    assert params["Global.use_rec"] is True
    assert not any(key.endswith(".session") for key in params)
    assert not any(key in params for key in ("Det.use_cuda", "Cls.use_cuda", "Rec.use_cuda"))
    assert all(
        isinstance(value, (str, int, float, bool, type(None), list, dict, Enum))
        for value in params.values()
    )
    json.dumps(
        {
            key: value.value if isinstance(value, Enum) else value
            for key, value in params.items()
        }
    )
    assert "rapidocr_det_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']" in caplog.text
    assert "rapidocr_cls_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']" in caplog.text
    assert "rapidocr_rec_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']" in caplog.text


def test_docling_rapidocr_rejects_cpu_primary_when_cuda_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_rapid_ocr_model(
        monkeypatch,
        device="cuda:0",
        providers=["CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="RapidOCR det session did not activate CUDAExecutionProvider"):
        module.RapidOcrModel(
            enabled=True,
            artifacts_path=None,
            options=_rapid_options(),
            accelerator_options=types.SimpleNamespace(device="cuda", num_threads=4),
        )


def test_docling_rapidocr_rejects_non_cuda_provider_when_cuda_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_rapid_ocr_model(
        monkeypatch,
        device="cuda:0",
        providers=["TensorrtExecutionProvider", "CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="RapidOCR det session did not activate CUDAExecutionProvider"):
        module.RapidOcrModel(
            enabled=True,
            artifacts_path=None,
            options=_rapid_options(),
            accelerator_options=types.SimpleNamespace(device="cuda", num_threads=4),
        )


def test_docling_rapidocr_allows_explicit_cpu_primary_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, fake_reader = _load_rapid_ocr_model(
        monkeypatch,
        device="cpu",
        providers=["CPUExecutionProvider"],
    )

    module.RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=_rapid_options(),
        accelerator_options=types.SimpleNamespace(device="cpu", num_threads=4),
    )

    assert fake_reader.last_params is not None
    assert fake_reader.last_params["EngineConfig.onnxruntime.use_cuda"] is False
    assert fake_reader.last_params["EngineConfig.onnxruntime.cuda_ep_cfg.device_id"] == 0


def test_docling_rapidocr_rejects_nonserializable_user_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = _load_rapid_ocr_model(
        monkeypatch,
        device="cpu",
        providers=["CPUExecutionProvider"],
    )
    options = _rapid_options()
    options.rapidocr_params = {"EngineConfig.onnxruntime.session": object()}

    with pytest.raises(TypeError, match="OmegaConf-compatible"):
        module.RapidOcrModel(
            enabled=True,
            artifacts_path=None,
            options=options,
            accelerator_options=types.SimpleNamespace(device="cpu", num_threads=4),
        )


def test_docling_rapidocr_forwards_explicit_visible_device_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, fake_reader = _load_rapid_ocr_model(
        monkeypatch,
        device="cuda:3",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    module.RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=_rapid_options(),
        accelerator_options=types.SimpleNamespace(device="cuda:3", num_threads=4),
    )

    assert fake_reader.last_params is not None
    assert fake_reader.last_params[
        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id"
    ] == 3


def test_rapidocr_params_accept_omegaconf_primitive_values() -> None:
    """The only user-supplied pass-through values are accepted by OmegaConf."""

    omegaconf = pytest.importorskip("omegaconf")
    params = {
        "EngineConfig.onnxruntime.use_cuda": True,
        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 2,
        "Det.limit_side_len": 960,
        "Rec.extra_options": {"keys": ["a", "b"], "enabled": False},
    }
    config = omegaconf.OmegaConf.create({})
    for key, value in params.items():
        omegaconf.OmegaConf.update(config, key, value, merge=True)
    assert omegaconf.OmegaConf.to_container(config, resolve=True) == {
        "EngineConfig": {
            "onnxruntime": {
                "use_cuda": True,
                "cuda_ep_cfg": {"device_id": 2},
            }
        },
        "Det": {"limit_side_len": 960},
        "Rec": {"extra_options": {"keys": ["a", "b"], "enabled": False}},
    }


def test_real_rapidocr_392_cpu_construction_smoke() -> None:
    """Exercise RapidOCR 3.9.2's real OmegaConf path with local project models."""

    # Earlier unit tests load the same source with lightweight stand-ins.  Drop
    # those modules so this smoke test cannot accidentally reuse a fake session.
    for name in list(sys.modules):
        if (
            name == "rapidocr"
            or name == "docling"
            or name.startswith("docling.")
            or name == "docling_core"
            or name.startswith("docling_core.")
        ):
            sys.modules.pop(name, None)

    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("omegaconf")
    pytest.importorskip("pydantic_settings")

    from importlib.metadata import version

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import RapidOcrOptions
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

    assert version("rapidocr") == "3.9.2"
    # Linux is case-sensitive; the deployed directory is exactly RapidOcr.
    model_root = PROJECT_ROOT / "models" / "RapidOcr"
    options = RapidOcrOptions(
        lang=["english"],
        backend="onnxruntime",
        use_det=True,
        use_cls=True,
        use_rec=True,
        det_model_path=str(model_root / "onnx" / "PP-OCRv6" / "det" / "PP-OCRv6_det_medium.onnx"),
        cls_model_path=str(model_root / "onnx" / "PP-OCRv4" / "cls" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
        rec_model_path=str(model_root / "onnx" / "PP-OCRv6" / "rec" / "PP-OCRv6_rec_medium.onnx"),
        font_path=str(model_root / "resources" / "fonts" / "FZYTK.TTF"),
        rapidocr_params={},
    )
    model = RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(device="cpu", num_threads=1),
    )

    assert model.reader.text_det.session.session.get_providers()[0] == "CPUExecutionProvider"
    assert model.reader.text_cls.session.session.get_providers()[0] == "CPUExecutionProvider"
    assert model.reader.text_rec.session.session.get_providers()[0] == "CPUExecutionProvider"


def test_real_rapidocr_392_cuda_construction_and_inference() -> None:
    """Server-only: real models, real OmegaConf, real CUDA sessions and inference."""

    if not bool(int(__import__("os").environ.get("RUN_REAL_CUDA_TESTS", "0"))):
        pytest.skip("set RUN_REAL_CUDA_TESTS=1 on a CUDA server")
    for name in list(sys.modules):
        if (
            name == "rapidocr"
            or name == "docling"
            or name.startswith("docling.")
            or name == "docling_core"
            or name.startswith("docling_core.")
        ):
            sys.modules.pop(name, None)

    numpy = pytest.importorskip("numpy")
    pytest.importorskip("rapidocr")
    onnxruntime = pytest.importorskip("onnxruntime")
    assert "CUDAExecutionProvider" in onnxruntime.get_available_providers()

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import RapidOcrOptions
    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

    model_root = PROJECT_ROOT / "models" / "RapidOcr"
    options = RapidOcrOptions(
        lang=["english"],
        backend="onnxruntime",
        use_det=True,
        use_cls=True,
        use_rec=True,
        det_model_path=str(model_root / "onnx" / "PP-OCRv6" / "det" / "PP-OCRv6_det_medium.onnx"),
        cls_model_path=str(model_root / "onnx" / "PP-OCRv4" / "cls" / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
        rec_model_path=str(model_root / "onnx" / "PP-OCRv6" / "rec" / "PP-OCRv6_rec_medium.onnx"),
        font_path=str(model_root / "resources" / "fonts" / "FZYTK.TTF"),
    )
    model = RapidOcrModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(device="cuda", num_threads=1),
    )
    assert all(
        providers[0] == "CUDAExecutionProvider"
        for providers in model._provider_audit.values()
    )
    # Blank input may legitimately return no text; the assertion is that all
    # three real models initialize and RapidOCR completes one real call.
    result = model.reader(
        numpy.full((256, 512, 3), 255, dtype=numpy.uint8),
        use_det=True,
        use_cls=True,
        use_rec=True,
    )
    assert result is not None
