import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Type, TypedDict

import numpy
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    OcrOptions,
    RapidOcrOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder
from docling.utils.utils import download_url_with_progress

_log = logging.getLogger(__name__)

_RAPIDOCR_ONNX_SESSION_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("det", ("text_det", "session", "session")),
    ("cls", ("text_cls", "session", "session")),
    ("rec", ("text_rec", "session", "session")),
)


def _rapidocr_config_path(path: str | Path | None) -> str | None:
    """Keep values passed through RapidOCR/OmegaConf configuration primitive-only."""

    return None if path is None else str(path)


def _validate_rapidocr_config_value(value: Any, location: str) -> None:
    """Reject Python runtime objects before RapidOCR passes parameters to OmegaConf."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_rapidocr_config_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "RapidOCR configuration dictionary keys must be strings; "
                    f"received {type(key)!r} at {location}"
                )
            _validate_rapidocr_config_value(item, f"{location}.{key}")
        return
    raise TypeError(
        "RapidOcrOptions.rapidocr_params must contain only OmegaConf-compatible "
        "primitive values, lists, and dictionaries; "
        f"received {type(value)!r} at {location}"
    )


def _validate_rapidocr_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        if not isinstance(key, str):
            raise TypeError(
                "RapidOcrOptions.rapidocr_params keys must be strings; "
                f"received {type(key)!r}"
            )
        _validate_rapidocr_config_value(value, key)

_ModelPathEngines = Literal["onnxruntime", "torch"]
_ModelPathTypes = Literal[
    "det_model_path", "cls_model_path", "rec_model_path", "rec_keys_path", "font_path"
]
_RAPIDOCR_BACKENDS: tuple[_ModelPathEngines, ...] = ("onnxruntime", "torch")


class _ModelPathDetail(TypedDict):
    url: str | None
    path: str | None


_RAPIDOCR_MODELSCOPE_RELEASE = "v3.9.0"
_RAPIDOCR_MODELSCOPE_BASE_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve"
)
_RAPIDOCR_DEFAULT_LANGUAGE = "chinese"
_RAPIDOCR_PPOCRV6_ONNX_MODEL_PATHS: dict[_ModelPathTypes, str | None] = {
    "det_model_path": "onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
    "cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "rec_model_path": "onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
    "rec_keys_path": None,
    "font_path": "resources/fonts/FZYTK.TTF",
}
_RAPIDOCR_CHINESE_MODEL_PATHS: dict[
    _ModelPathEngines, dict[_ModelPathTypes, str | None]
] = {
    "onnxruntime": {
        **_RAPIDOCR_PPOCRV6_ONNX_MODEL_PATHS,
    },
    "torch": {
        "det_model_path": "torch/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.pth",
        "cls_model_path": "torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth",
        "rec_model_path": "torch/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.pth",
        "rec_keys_path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
}
_RAPIDOCR_ENGLISH_MODEL_PATHS: dict[
    _ModelPathEngines, dict[_ModelPathTypes, str | None]
] = {
    "onnxruntime": {
        **_RAPIDOCR_PPOCRV6_ONNX_MODEL_PATHS,
    },
    "torch": {
        "det_model_path": "torch/PP-OCRv4/det/en_PP-OCRv3_det_mobile.pth",
        "cls_model_path": "torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth",
        "rec_model_path": "torch/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.pth",
        "rec_keys_path": "paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
}


_RAPIDOCR_LATIN_MODEL_PATHS: dict[
    _ModelPathEngines, dict[_ModelPathTypes, str | None]
] = {
    "onnxruntime": {
        **_RAPIDOCR_PPOCRV6_ONNX_MODEL_PATHS,
    },
    # The Torch backend does not have PP-OCRv6 assets yet. Keep the previous
    # Latin rec model + dict; detector/classifier mirror the English set.
    "torch": {
        "det_model_path": "torch/PP-OCRv4/det/en_PP-OCRv3_det_mobile.pth",
        "cls_model_path": "torch/PP-OCRv4/cls/ch_ptocr_mobile_v2.0_cls_mobile.pth",
        "rec_model_path": "torch/PP-OCRv4/rec/latin_PP-OCRv3_rec_mobile.pth",
        "rec_keys_path": "paddle/PP-OCRv4/rec/latin_PP-OCRv3_rec_mobile/latin_dict.txt",
        "font_path": "resources/fonts/FZYTK.TTF",
    },
}


def _build_model_detail(path: str | None) -> _ModelPathDetail:
    if path is None:
        return {
            "url": None,
            "path": None,
        }
    return {
        "url": f"{_RAPIDOCR_MODELSCOPE_BASE_URL}/{_RAPIDOCR_MODELSCOPE_RELEASE}/{path}",
        "path": path,
    }


# Maps user-facing language names (ISO 639-1/639-2 codes and English names,
# tesseract-style values included) onto the bundled RapidOCR model sets.
_RAPIDOCR_LANGUAGE_GROUPS: dict[str, str] = {
    # english model set
    "en": "english",
    "eng": "english",
    "english": "english",
    # chinese model set
    "ch": "chinese",
    "chi": "chinese",
    "zh": "chinese",
    "zho": "chinese",
    "chinese": "chinese",
    # latin model set (latin_dict covers most Latin-script European languages)
    "latin": "latin",
    "de": "latin",
    "deu": "latin",
    "ger": "latin",
    "german": "latin",
    "fr": "latin",
    "fra": "latin",
    "fre": "latin",
    "french": "latin",
    "es": "latin",
    "spa": "latin",
    "spanish": "latin",
    "it": "latin",
    "ita": "latin",
    "italian": "latin",
    "pt": "latin",
    "por": "latin",
    "portuguese": "latin",
    "nl": "latin",
    "nld": "latin",
    "dut": "latin",
    "dutch": "latin",
    "fi": "latin",
    "fin": "latin",
    "finnish": "latin",
    "sv": "latin",
    "swe": "latin",
    "swedish": "latin",
    "da": "latin",
    "dan": "latin",
    "danish": "latin",
    "no": "latin",
    "nor": "latin",
    "norwegian": "latin",
    "pl": "latin",
    "pol": "latin",
    "polish": "latin",
    "cs": "latin",
    "ces": "latin",
    "cze": "latin",
    "czech": "latin",
    "ro": "latin",
    "ron": "latin",
    "rum": "latin",
    "romanian": "latin",
    "hu": "latin",
    "hun": "latin",
    "hungarian": "latin",
    "tr": "latin",
    "tur": "latin",
    "turkish": "latin",
    "hr": "latin",
    "hrv": "latin",
    "croatian": "latin",
    "sk": "latin",
    "slk": "latin",
    "slovak": "latin",
    "sl": "latin",
    "slv": "latin",
    "slovenian": "latin",
    "ca": "latin",
    "cat": "latin",
    "catalan": "latin",
    "id": "latin",
    "ind": "latin",
    "indonesian": "latin",
}


def _resolve_rapidocr_language(languages: list[str] | None) -> str:
    """Map requested languages onto a bundled RapidOCR model set.

    Falls back to the default set *loudly*: silently running the Chinese
    recognition model on Latin-script documents drops inter-word spaces
    (see docling issues #2887, #1635, #2927).
    """
    if not languages:
        return _RAPIDOCR_DEFAULT_LANGUAGE

    groups: list[str] = []
    unknown: list[str] = []
    for language in languages:
        # "en-US" / "en_US" -> "en"
        normalized = language.strip().lower().replace("_", "-").split("-")[0]
        group = _RAPIDOCR_LANGUAGE_GROUPS.get(normalized)
        if group is None:
            unknown.append(language)
        else:
            groups.append(group)

    if unknown:
        _log.warning(
            "RapidOCR has no bundled model set for language(s) %s; known values "
            "map onto the 'english', 'latin' or 'chinese' model sets.",
            unknown,
        )
    if not groups:
        _log.warning(
            "Falling back to the '%s' RapidOCR model set; note the Chinese "
            "recognition model drops inter-word spaces in Latin-script text.",
            _RAPIDOCR_DEFAULT_LANGUAGE,
        )
        return _RAPIDOCR_DEFAULT_LANGUAGE

    distinct = set(groups)
    if distinct == {"english"}:
        return "english"
    if distinct <= {"english", "latin"}:
        # the latin set covers English plus other Latin-script languages
        return "latin"
    if len(distinct) == 1:
        return groups[0]
    _log.warning(
        "Requested languages %s span multiple RapidOCR model sets %s; using "
        "'%s' (first requested). Run separate conversions for the others.",
        languages,
        sorted(distinct),
        groups[0],
    )
    return groups[0]


def _rapidocr_lang_type_params(ocr_lang: str) -> dict[str, object]:
    """Language params for the no-pinned-paths flow (no artifacts_path).

    Without explicit model paths RapidOCR resolves models itself — and its
    defaults are the Chinese set regardless of what was requested here, so the
    resolved language must be passed through. Older rapidocr versions without
    the typings module keep their defaults.
    """
    try:
        from rapidocr.utils.typings import LangDet, LangRec  # type: ignore
    except ImportError:
        return {}
    mapping: dict[str, dict[str, object]] = {
        "english": {"Det.lang_type": LangDet.EN, "Rec.lang_type": LangRec.EN},
        "latin": {"Det.lang_type": LangDet.EN, "Rec.lang_type": LangRec.LATIN},
    }
    return mapping.get(ocr_lang, {})


def _rapidocr_torch_ppocrv4_params() -> dict[str, object]:
    try:
        from rapidocr.utils.typings import ModelType, OCRVersion  # type: ignore
    except ImportError:
        return {}
    return {
        "Det.ocr_version": OCRVersion.PPOCRV4,
        "Det.model_type": ModelType.MOBILE,
        "Cls.ocr_version": OCRVersion.PPOCRV4,
        "Cls.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV4,
        "Rec.model_type": ModelType.MOBILE,
    }


class RapidOcrModel(BaseOcrModel):
    _model_repo_folder = "RapidOcr"
    # from https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/default_models.yaml
    # matching the default config in https://github.com/RapidAI/RapidOCR/blob/main/python/rapidocr/config.yaml
    # and naming f"{file_info.engine_type.value}.{file_info.ocr_version.value}.{file_info.task_type.value}"
    _models_by_language: dict[
        str, dict[_ModelPathEngines, dict[_ModelPathTypes, _ModelPathDetail]]
    ] = {
        "chinese": {
            backend: {
                key: _build_model_detail(path)
                for key, path in _RAPIDOCR_CHINESE_MODEL_PATHS[backend].items()
            }
            for backend in _RAPIDOCR_BACKENDS
        },
        "english": {
            backend: {
                key: _build_model_detail(path)
                for key, path in _RAPIDOCR_ENGLISH_MODEL_PATHS[backend].items()
            }
            for backend in _RAPIDOCR_BACKENDS
        },
        "latin": {
            backend: {
                key: _build_model_detail(path)
                for key, path in _RAPIDOCR_LATIN_MODEL_PATHS[backend].items()
            }
            for backend in _RAPIDOCR_BACKENDS
        },
    }
    _default_models: dict[
        _ModelPathEngines, dict[_ModelPathTypes, _ModelPathDetail]
    ] = _models_by_language[_RAPIDOCR_DEFAULT_LANGUAGE]

    @staticmethod
    def _onnx_session_providers(reader: Any, path: tuple[str, ...]) -> list[str]:
        target = reader
        try:
            for attribute in path:
                target = getattr(target, attribute)
            providers = target.get_providers()
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "RapidOCR did not expose the expected ONNX Runtime session at "
                f"{'.'.join(path)}"
            ) from exc
        if not isinstance(providers, (list, tuple)):
            raise RuntimeError(
                "RapidOCR ONNX Runtime get_providers() returned an invalid value at "
                f"{'.'.join(path)}: {providers!r}"
            )
        return [str(provider) for provider in providers]

    def _verify_rapidocr_onnx_providers(self, use_cuda: bool) -> dict[str, list[str]]:
        """Verify the Provider actually selected by RapidOCR's own sessions."""

        expected_primary = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        provider_audit: dict[str, list[str]] = {}
        for stage, path in _RAPIDOCR_ONNX_SESSION_PATHS:
            providers = self._onnx_session_providers(self.reader, path)
            provider_audit[stage] = providers
            message = f"rapidocr_{stage}_providers={providers}"
            _log.info(message)
            print(message, flush=True)
            if not providers or providers[0] != expected_primary:
                if use_cuda:
                    raise RuntimeError(
                        "CUDA was requested, but the RapidOCR "
                        f"{stage} session did not activate CUDAExecutionProvider "
                        f"as its primary provider. Active providers: {providers}"
                    )
                raise RuntimeError(
                    "Explicit CPU execution was requested, but the RapidOCR "
                    f"{stage} session did not activate CPUExecutionProvider "
                    f"as its primary provider. Active providers: {providers}"
                )
        return provider_audit

    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: RapidOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: RapidOcrOptions

        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        if self.enabled:
            try:
                from rapidocr import EngineType, RapidOCR  # type: ignore
            except ImportError:
                raise ImportError(
                    "RapidOCR is not installed. Please install it via `pip install rapidocr onnxruntime` to use this OCR engine. "
                    "Alternatively, Docling has support for other OCR engines. See the documentation."
                )

            # Decide the accelerator devices
            device = decide_device(accelerator_options.device)
            use_cuda = str(AcceleratorDevice.CUDA.value).lower() in device
            intra_op_num_threads = accelerator_options.num_threads
            gpu_id = 0
            if use_cuda and ":" in device:
                gpu_id = int(device.split(":")[1])
            _ALIASES = {
                "onnxruntime": EngineType.ONNXRUNTIME,
                "openvino": EngineType.OPENVINO,
                "paddle": EngineType.PADDLE,
                "torch": EngineType.TORCH,
            }
            backend_enum = _ALIASES.get(self.options.backend, EngineType.ONNXRUNTIME)
            backend_key: _ModelPathEngines = "onnxruntime"
            if backend_enum == EngineType.TORCH:
                backend_key = "torch"
            if use_cuda and backend_enum == EngineType.ONNXRUNTIME:
                # ORT 1.23+ can preload the CUDA/cuDNN libraries bundled with
                # the already-imported PyTorch wheel.  This keeps an
                # orientation-disabled diagnostic from depending on the page
                # model to have performed that process-wide setup first.
                try:
                    import onnxruntime as ort
                except ImportError as exc:
                    raise RuntimeError(
                        "onnxruntime-gpu is required for RapidOCR CUDA execution"
                    ) from exc
                preload_dlls = getattr(ort, "preload_dlls", None)
                if callable(preload_dlls):
                    preload_dlls(directory="")

            ocr_lang = _resolve_rapidocr_language(self.options.lang)
            model_set = self._models_by_language[ocr_lang][backend_key]

            det_model_path = self.options.det_model_path
            cls_model_path = self.options.cls_model_path
            rec_model_path = self.options.rec_model_path
            rec_keys_path = self.options.rec_keys_path
            font_path = self.options.font_path

            if artifacts_path is not None:

                def resolve_artifact_path(
                    model_type: _ModelPathTypes, configured_path: str | None
                ) -> str | Path | None:
                    if configured_path is not None:
                        return configured_path
                    path = model_set[model_type]["path"]
                    if path is None:
                        return None
                    return artifacts_path / self._model_repo_folder / path

                det_model_path = resolve_artifact_path("det_model_path", det_model_path)
                cls_model_path = resolve_artifact_path("cls_model_path", cls_model_path)
                rec_model_path = resolve_artifact_path("rec_model_path", rec_model_path)
                rec_keys_path = resolve_artifact_path("rec_keys_path", rec_keys_path)
                font_path = resolve_artifact_path("font_path", font_path)

            for model_path in (
                det_model_path,
                rec_keys_path,
                cls_model_path,
                rec_model_path,
                font_path,
            ):
                if model_path is None:
                    continue
                if not Path(model_path).exists():
                    _log.warning(f"The provided model path {model_path} is not found.")

            params = {
                # Global settings (these are still correct)
                "Global.text_score": self.options.text_score,
                "Global.font_path": _rapidocr_config_path(font_path),
                # Engine-level ONNXRuntime settings
                "EngineConfig.onnxruntime.intra_op_num_threads": intra_op_num_threads,
                "EngineConfig.onnxruntime.use_cuda": use_cuda,
                "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": gpu_id,
                "EngineConfig.onnxruntime.use_dml": False,
                # Engine-level OpenVINO settings
                "EngineConfig.openvino.inference_num_threads": intra_op_num_threads,
                # "Global.verbose": self.options.print_verbose,
                # Detection model settings
                "Det.model_path": _rapidocr_config_path(det_model_path),
                # Classification model settings
                "Cls.model_path": _rapidocr_config_path(cls_model_path),
                # Recognition model settings
                "Rec.model_path": _rapidocr_config_path(rec_model_path),
                "Rec.font_path": _rapidocr_config_path(font_path),
                "Rec.rec_keys_path": _rapidocr_config_path(rec_keys_path),
                "Det.engine_type": backend_enum,
                "Cls.engine_type": backend_enum,
                "Rec.engine_type": backend_enum,
            }
            for option_name in ("use_det", "use_cls", "use_rec"):
                option_value = getattr(self.options, option_name)
                if option_value is not None:
                    params[f"Global.{option_name}"] = bool(option_value)

            if self.options.rec_font_path is not None:
                _log.warning(
                    "The 'rec_font_path' option for RapidOCR is deprecated. Please use 'font_path' instead."
                )
            if det_model_path is None and rec_model_path is None:
                params.update(_rapidocr_lang_type_params(ocr_lang))
                if backend_enum == EngineType.TORCH:
                    params.update(_rapidocr_torch_ppocrv4_params())

            user_params = self.options.rapidocr_params
            if user_params:
                _validate_rapidocr_params(user_params)
                _log.debug("Overwriting RapidOCR params with user-provided values.")
                params.update(user_params)

            self.reader = RapidOCR(params=params)
            if backend_enum != EngineType.ONNXRUNTIME:
                return
            self._provider_audit = self._verify_rapidocr_onnx_providers(use_cuda)

    @classmethod
    def download_models(
        cls,
        backend: _ModelPathEngines,
        local_dir: Path | None = None,
        force: bool = False,
        progress: bool = False,
        lang: str = "chinese",
    ) -> Path:
        if local_dir is None:
            local_dir = settings.cache_dir / "models" / RapidOcrModel._model_repo_folder

        local_dir.mkdir(parents=True, exist_ok=True)

        # Download models
        resolved_lang = _resolve_rapidocr_language([lang])
        model_set = cls._models_by_language[resolved_lang][backend]
        for model_type, model_details in model_set.items():
            if model_details["path"] is None or model_details["url"] is None:
                continue
            output_path = local_dir / model_details["path"]
            if output_path.exists() and not force:
                continue
            output_path.parent.mkdir(exist_ok=True, parents=True)
            buf = download_url_with_progress(model_details["url"], progress=progress)
            with output_path.open("wb") as fw:
                fw.write(buf.read())

        return local_dir

    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            if hasattr(self, "_provider_audit"):
                page.ocr_audit["onnx_providers"] = {
                    stage: list(providers)
                    for stage, providers in self._provider_audit.items()
                }
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
            else:
                with TimeRecorder(conv_res, "ocr"):
                    ocr_rects = self.get_ocr_rects(page)

                    all_ocr_cells = []
                    for ocr_rect in ocr_rects:
                        # Skip zero area boxes
                        if ocr_rect.area() == 0:
                            continue
                        high_res_image = page._backend.get_page_image(
                            scale=self.scale, cropbox=ocr_rect
                        )
                        im = numpy.array(high_res_image)
                        result = self.reader(
                            im,
                            use_det=self.options.use_det,
                            use_cls=self.options.use_cls,
                            use_rec=self.options.use_rec,
                        )
                        if result is None or result.boxes is None:
                            _log.warning("RapidOCR returned empty result!")
                            continue
                        result = list(
                            zip(result.boxes.tolist(), result.txts, result.scores)
                        )

                        del high_res_image
                        del im

                        if result is not None:
                            cells = [
                                TextCell(
                                    index=ix,
                                    text=line[1],
                                    orig=line[1],
                                    confidence=line[2],
                                    from_ocr=True,
                                    rect=BoundingRectangle.from_bounding_box(
                                        BoundingBox.from_tuple(
                                            coord=(
                                                (line[0][0][0] / self.scale)
                                                + ocr_rect.l,
                                                (line[0][0][1] / self.scale)
                                                + ocr_rect.t,
                                                (line[0][2][0] / self.scale)
                                                + ocr_rect.l,
                                                (line[0][2][1] / self.scale)
                                                + ocr_rect.t,
                                            ),
                                            origin=CoordOrigin.TOPLEFT,
                                        )
                                    ),
                                )
                                for ix, line in enumerate(result)
                            ]
                            all_ocr_cells.extend(cells)

                    page.ocr_audit["raw_ocr_text_cell_count"] = len(all_ocr_cells)
                    page.ocr_audit["raw_ocr_mean_confidence"] = (
                        float(numpy.mean([cell.confidence for cell in all_ocr_cells]))
                        if all_ocr_cells
                        else None
                    )

                    # Post-process the cells
                    self.post_process_cells(all_ocr_cells, page)

                # DEBUG code:
                if settings.debug.visualize_ocr:
                    self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return RapidOcrOptions
