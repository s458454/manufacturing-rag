# PDF 预处理服务器验收

本清单验证正式入库视图，而不是仅验证 PDF 是否发生旋转。通过前，不得把
`document.md` 交给 Chunk、Embedding 或索引模块。

> 2026-08-13 更新：正式验收不再依赖 pytest 才能发现部署问题。先执行第 1 节的
> `verify_pdf_preprocess_server.py`，它会走真实 RapidOCR 3.9.2、真实 Det/Cls/Rec
> ONNX Session、真实页面推理和端到端 CLI。pytest 仍保留为代码回归，但不是服务器
> 唯一验收依据。正式产物只在 exit code 0 时发布；`partial_success` 或质量门禁拒绝的
> 完整审计结果保存在输出根目录的 `.failed/`，不会覆盖上一次成功产物。

## 0. 部署前提

同步代码、项目内的 Docling 快照和模型目录后，在仓库根目录执行。必须提供真实
`models/PageOrientation/PP-LCNet_x1_0_doc_ori/model.onnx`，并确认
`manifest.json` 中的来源、版本和 SHA-256 与其一致；预处理与真实模型验收均会拒绝占位值或错误模型。

以下命令以 Linux 和 `bash` 为例，`DEVICE` 请按实际 GPU 调整。不要用 NASA
第 38 页做“可选文本保真”测试：该页是扫描页；四向测试要选择一页有真实文字层的
工程 PDF。

```bash
set -euo pipefail
cd /path/to/all-in-rag-main

# 在独立虚拟环境中执行，避免替换共享服务环境中的 ONNX Runtime。
python -m venv .venv-preprocess-acceptance
source .venv-preprocess-acceptance/bin/activate
python -m pip install --upgrade pip

# 先按服务器 CUDA/驱动版本安装 GPU 版 PyTorch（以下仅为 CUDA 12.6 示例）。
# 若服务器已有经验证的 CUDA PyTorch 环境，可跳过这一条，但不能接受 CPU-only torch。
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126

export PYTHONPATH="$PWD/code/preprocessing${PYTHONPATH:+:$PYTHONPATH}"
export DEVICE="cuda:0"
export NASA_PDF="$PWD/data/engineering_docs/raw/material/01_nasa_materials_aluminum_2014.pdf"
export ORIENTATION_MODEL="$PWD/models/PageOrientation/PP-LCNet_x1_0_doc_ori"
export ACCEPTANCE_ROOT="$PWD/outputs/acceptance-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$ACCEPTANCE_ROOT"

# 安装项目的 Docling 依赖，再安装本链路经过验收的固定运行面。
python -m pip install -e "./docling[standard]"
python -m pip uninstall -y onnxruntime || true
python -m pip install --upgrade --force-reinstall \
  -r code/preprocessing/requirements-server.txt

nvidia-smi
python - <<'PY'
import torch
import onnxruntime as ort
from importlib.metadata import version
from packaging.version import Version
from docling_core.transforms.serializer.markdown import MarkdownParams

print("torch=", torch.__version__, "torch_cuda=", torch.version.cuda)
print("torch.cuda.is_available=", torch.cuda.is_available())
print("onnxruntime=", ort.__version__, "providers=", ort.get_available_providers())
print("rapidocr=", version("rapidocr"))
print("docling-core=", version("docling-core"))
assert torch.cuda.is_available()
assert "CUDAExecutionProvider" in ort.get_available_providers()
assert version("rapidocr") == "3.9.2"
assert Version(version("docling-core")) >= Version("2.88.0")
assert {"pages", "traverse_pictures", "enable_chart_tables"} <= set(MarkdownParams.model_fields)
PY

python - "$ORIENTATION_MODEL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
for filename in ("model.onnx", "inference.yml", "labels.json", "manifest.json"):
    assert (model_dir / filename).is_file(), f"missing: {model_dir / filename}"
manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
for key in ("model_version", "source", "sha256"):
    assert "REPLACE_WITH_" not in str(manifest.get(key, "")), f"placeholder: {key}"
actual_sha = hashlib.sha256((model_dir / "model.onnx").read_bytes()).hexdigest()
assert actual_sha == manifest["sha256"].lower(), (actual_sha, manifest["sha256"])
assert actual_sha == "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92", actual_sha
assert manifest["labels"] == ["0", "90", "180", "270"]
print("verified_orientation_model_sha256=", actual_sha)
PY
```

## 1. 真实运行面一键验收（优先执行）

这条命令不要求 pytest。它先校验 OCR、Heron、页面方向模型的大小和 SHA-256，
然后用项目内 Docling 包装层构造 RapidOCR 3.9.2，检查三个真实 Session 的 provider，
在指定 PDF 页面上做一次真实 Det+Rec，最后运行同一页完整预处理：

```bash
CUDA_VISIBLE_DEVICES=0 python code/preprocessing/verify_pdf_preprocess_server.py \
  --input-pdf "$NASA_PDF" \
  --page 38 \
  --device cuda \
  --num-threads 8 \
  --output-root "$ACCEPTANCE_ROOT/one-command"
```

成功时必须生成 `server_acceptance.json`，且 `providers.det/cls/rec` 的第一项都是
`CUDAExecutionProvider`。默认部署验收只接受端到端返回码 0，即成功并发布正式产物；
1=Docling `partial_success`，始终判定验收失败；3=转换完成但所选页面被固定质量门禁排除，
默认也判定验收失败。只有专门验证安全拒绝路径时才传
`--allow-quality-rejection` 接受返回码 3，其完整审计结果仍只留在 `.failed/`；
2=环境、模型、参数或运行时错误。

## 2. 单元回归（有 pytest 时执行）

```bash
python -m pytest code/preprocessing/tests -q

# This test must run (not be skipped) in the deployed environment.  It invokes
# the actual RapidOcrOptions + RapidOcrModel + RapidOCR 3.9.2 construction path,
# including OmegaConf configuration merge and the real local OCR models.
python -m pytest code/preprocessing/tests/test_rapidocr_cuda_sessions.py \
  -q -k real_rapidocr_392_cpu_construction_smoke -rs

RUN_REAL_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 \
  python -m pytest code/preprocessing/tests/test_rapidocr_cuda_sessions.py \
  -q -k real_rapidocr_392_cuda_construction_and_inference -rs

python - <<'PY'
import os
from pathlib import Path
from page_orientation import OrientationConfig, PageOrientationClassifier

device = os.environ["DEVICE"]
model_dir = Path(os.environ["ORIENTATION_MODEL"])
orientation = PageOrientationClassifier(OrientationConfig(model_dir=model_dir, device=device))
print("orientation_active_providers=", orientation._session.get_providers())
assert orientation._session.get_providers()[0] == "CUDAExecutionProvider"

PY
```

页面方向模型按 `[('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']`
创建。RapidOCR Det/Cls/Rec 不能把 Python 会话对象放进 `rapidocr_params`：RapidOCR 会先经
OmegaConf 合并参数。Docling 改为传递可序列化的
`EngineConfig.onnxruntime.use_cuda`、`EngineConfig.onnxruntime.cuda_ep_cfg.device_id`，让 RapidOCR
自行建立会话。它完成后必须输出并检查：

```text
rapidocr_det_providers=[...]
rapidocr_cls_providers=[...]
rapidocr_rec_providers=[...]
```

CUDA 模式三个列表的首项均必须为 `CUDAExecutionProvider`；显式 `--device cpu` 时必须为
`CPUExecutionProvider`。CPU Provider 可为 CUDA 不支持的单个节点补充执行，但 CUDA 请求不能让
整个 OCR 会话静默退化为 CPU-only。不会使用强制 CPU EP 回退锁，也不提供隐式 auto 模式。

TableFormer V1 accurate 也必须在启动预处理前由运维同步至：

```text
models/docling-project--docling-models/model_artifacts/tableformer/accurate/
├── tm_config.json
└── tableformer_accurate.safetensors
```

缺任一必需文件，或 `tm_config.json`、`tableformer_accurate.safetensors` 任一文件的批准大小或 SHA-256 不匹配，均应在 Docling 初始化前失败；
它们均不得由运行时联网补齐。部署时先运行下列只读检查，确认两个文件存在后再继续完整预处理：

```bash
test -f models/docling-project--docling-models/model_artifacts/tableformer/accurate/tm_config.json
test -f models/docling-project--docling-models/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors
```

## 3. 真实模型四向验收

先选一个“工程 PDF 的原始第 N 页”，它必须有可选择的数字文字层，并且视觉上为
正向。脚本会生成 0/90/180/270 四个版本，要求真实模型预测全部正确；随后将每个
版本标准化并复验为 0，同时验证源 PDF 哈希、页数和文字层不变。

```bash
export ENGINEERING_PDF="/absolute/path/to/engineering-document-with-text-layer.pdf"
export ENGINEERING_PAGE="12"   # 改成已人工确认正向、含可选择文本的原始页码

python code/preprocessing/verify_page_orientation_real_model.py \
  --input-pdf "$ENGINEERING_PDF" \
  --page "$ENGINEERING_PAGE" \
  --model-dir "$ORIENTATION_MODEL" \
  --device "$DEVICE" \
  --work-dir "$ACCEPTANCE_ROOT/real-orientation" \
  --overwrite
```

检查 `$ACCEPTANCE_ROOT/real-orientation/real_orientation_acceptance.json`：
`status` 必须为 `passed`，四个 `cases` 的 `direct_before.predicted_orientation`
必须依次覆盖 0、90、180、270，全部 `direct_after.predicted_orientation` 为 0，且每一次
`model_output_shape` 都为 `[1, 4]`。任一失败都表示模型标签语义或 PDF 修正方向尚未验收，
不能用 FakeClassifier 代替。

本次真实模型 profile 工程基线如下；脚本会把 profile 原件保存到
`$ACCEPTANCE_ROOT/real-orientation/onnxruntime_profile.json`，并把汇总与断言结果写进上述
`real_orientation_acceptance.json`。

| 项目 | 验收基线 |
|---|---|
| `model_sha256` | `af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92` |
| ONNX Runtime 基线 | `1.23.2` |
| 历史 CPU 节点 | `Slice`、`Concat` |
| 历史主要计算 Provider | `CUDAExecutionProvider` |
| 历史 CPU 节点累计时长占比 | `0.022%` |

该 profile 是部署验收、性能排查和模型或 ONNX Runtime 升级后的复验证据；它不作为每次
正式预处理的 CPU 节点集合或时长门禁。运行后执行以下独立检查，连同 JSON 和 profile 一并回传：

```bash
python - "$ACCEPTANCE_ROOT/real-orientation/real_orientation_acceptance.json" <<'PY'
import json
import sys
from pathlib import Path

evidence = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert evidence["status"] == "passed"
assert evidence["model"]["model_sha256"] == (
    "af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92"
)
assert [case["direct_before"]["predicted_orientation"] for case in evidence["cases"]] == [0, 90, 180, 270]
assert all(case["direct_after"]["predicted_orientation"] == 0 for case in evidence["cases"])
assert all(
    part["model_output_shape"] == [1, 4]
    for case in evidence["cases"]
    for part in (case["direct_before"], case["direct_after"])
)
profile = evidence["onnxruntime_profile"]
assert profile["historical_major_compute_provider"] == "CUDAExecutionProvider"
print("real_orientation_profile_evidence=collected", profile)
PY
```

## 4. NASA 第 38 页：范围、题注和视觉 OCR 隔离

```bash
NASA_RUN="$ACCEPTANCE_ROOT/nasa-p38"
SOURCE_SHA_BEFORE="$(sha256sum "$NASA_PDF" | awk '{print $1}')"

python code/preprocessing/pdf_preprocess.py "$NASA_PDF" \
  --device "$DEVICE" \
  --page-range 38 38 \
  --output-root "$NASA_RUN" \
  --overwrite 2>&1 | tee "$NASA_RUN.log"

ARTIFACT_DIR="$(sed -n 's/^published_output=//p' "$NASA_RUN.log" | tail -n 1)"
test -n "$ARTIFACT_DIR"

python - "$NASA_PDF" "$ARTIFACT_DIR" "$SOURCE_SHA_BEFORE" <<'PY'
import json
import re
import sys
from pathlib import Path
from pypdf import PdfReader

source = Path(sys.argv[1])
artifact = Path(sys.argv[2])
source_sha = sys.argv[3]
orientation = json.loads((artifact / "orientation_report.json").read_text(encoding="utf-8"))
quality = json.loads((artifact / "quality_report.json").read_text(encoding="utf-8"))
regions = json.loads((artifact / "regions.json").read_text(encoding="utf-8"))
raw_document = json.loads((artifact / "document.json").read_text(encoding="utf-8"))
markdown = (artifact / "document.md").read_text(encoding="utf-8")

assert orientation["processed_page_count"] == 1
assert orientation["processed_page_range"] == [38, 38]
assert orientation["processed_page_numbers"] == [38]
assert [p["page_no"] for p in orientation["pages"]] == [38]
assert [p["page_no"] for p in quality["pages"]] == [38]
assert orientation["source_sha256"] == source_sha == quality["source_sha256"]
assert len(PdfReader(str(artifact / "normalized" / "oriented.pdf")).pages) == len(PdfReader(str(source)).pages)

tables = [r for r in regions if r["page_no"] == 38 and r["region_type"] == "table"]
assert tables, "page 38 table missing from regions.json"
assert any("TABLE 5.31" in (r.get("caption") or "").upper() for r in tables)
assert all(r["visual_body_in_semantic_markdown"] is False for r in tables)
assert any((t.get("prov") or [{}])[0].get("page_no") == 38 for t in raw_document.get("tables", []))

# 方向阶段本身必须完成该页的复验；若其拒绝，后面的 review 分支会继续验证
# 审计保留和 Markdown 门禁，但部署方仍需人工判定该真实模型质量是否可接受。
orientation_page = orientation["pages"][0]
assert orientation_page["decision"] in {
    "accepted_upright", "accepted_rotated", "review_required", "orientation_or_parse_uncertain", "not_applicable"
}
if orientation_page["decision"] == "accepted_rotated":
    assert orientation_page["post_rotation_predicted_orientation"] == 0
    assert orientation_page["post_rotation_zero_score"] >= orientation["configuration"]["postcheck_min_zero"]

# The core isolation properties must always hold, even when the real page is
# correctly routed to manual review.  Formal-MD assertions apply only after the
# actual model and post-Docling gate accept the page.  For the named NASA
# acceptance sample, an accepted page must expose the expected table caption.
page = quality["pages"][0]
if page["eligible_for_indexing"]:
    assert any("TABLE 5.31" in (r.get("caption") or "").upper() for r in tables)
    assert markdown.count("<!-- PDF page 38 -->") == 1
    assert len(re.findall(r"(?i)\bTABLE\s+5\.31\b", markdown)) == 1
    assert any(r["caption_in_semantic_markdown"] is True for r in tables)
    assert any(r["caption_trust_decision"] == "accepted" for r in tables)
else:
    assert "<!-- PDF page 38 -->" not in markdown
    assert all(r["caption_in_semantic_markdown"] is False for r in tables)
    assert all(r["caption_trust_decision"] in {"page_untrusted", "garbled", "missing"} for r in tables)
    print("NASA p38 correctly routed to review; formal Markdown assertions skipped")

# 由完整 document.json 的实际父子引用推导第 38 页表格后代，证明表体 OCR
# 仍在审计 JSON 中、但不会泄露至 document.md。兼容 $ref 与 cref 两种 JSON 形式。
nodes = {}
def walk(value):
    if isinstance(value, dict):
        if isinstance(value.get("self_ref"), str):
            nodes[value["self_ref"]] = value
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
walk(raw_document)

def ref(value):
    if not isinstance(value, dict):
        return None
    return value.get("$ref") or value.get("cref")

def descendants(root):
    seen, pending = set(), [root]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current, {})
        pending.extend(
            child_ref for child_ref in (ref(child) for child in node.get("children", []))
            if child_ref
        )
    return seen

raw_tables = [
    table for table in raw_document.get("tables", [])
    if (table.get("prov") or [{}])[0].get("page_no") == 38
]
caption_refs = {
    ref(caption) for table in raw_tables for caption in table.get("captions", [])
    if ref(caption)
}
body_refs = set().union(*(descendants(table["self_ref"]) for table in raw_tables)) - caption_refs
body_texts = [
    nodes[body_ref].get("text", "").strip() for body_ref in body_refs
    if isinstance(nodes.get(body_ref, {}).get("text"), str) and nodes[body_ref]["text"].strip()
]
# Some Docling tables store plain-cell OCR directly in ``data.table_cells``
# rather than as a separate text-node descendant.  It must remain in the raw
# JSON too, and must be absent from the formal Markdown just like rich cells.
body_texts.extend(
    cell.get("text", "").strip()
    for table in raw_tables
    for cell in (table.get("data", {}).get("table_cells", []) or [])
    if isinstance(cell, dict) and isinstance(cell.get("text"), str)
    and cell["text"].strip()
)
caption_text = " ".join(nodes[caption_ref].get("text", "") for caption_ref in caption_refs)
normalise = lambda value: re.sub(r"\s+", " ", value).strip().lower()
sentinels = [
    text for text in body_texts
    if len(normalise(text)) >= 4 and any(char.isdigit() for char in text)
    and normalise(text) not in normalise(caption_text)
]
assert sentinels, "document.json did not retain a table-body OCR sentinel"
leaked = [text for text in sentinels if normalise(text) in normalise(markdown)]
assert not leaked, f"table body leaked into document.md: {leaked[:5]}"
print("NASA p38 semantic acceptance passed")
PY
```

若该页的真实方向、题注或后置解析门禁没有达到阈值，脚本会验证它被完整保留在审计
JSON 中、且不进入 Markdown；这是待人工复核的真实质量结果，不是放宽题注保护或页面
门禁的理由。若该页通过门禁，则额外要求 `TABLE 5.31` 及其可信题注在 Markdown 恰好
出现一次。

## 5. 强制 `review_required` 门禁

该用例制造一张可选择文本的页面，并将两个方向阈值设为 1.0，使页面确定进入
`review_required`。它必须留在 `document.json`、方向/质量报告中，但绝不能泄露到
`document.md`。

```bash
REVIEW_ROOT="$ACCEPTANCE_ROOT/review-gate"
mkdir -p "$REVIEW_ROOT"

python - "$REVIEW_ROOT/review-gate.pdf" <<'PY'
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas
import sys

canvas = Canvas(sys.argv[1], pagesize=letter)
canvas.setFont("Helvetica-Bold", 18)
for row in range(28):
    canvas.drawString(36, 760 - row * 25, f"REVIEW_GATE_SENTINEL_314159 row {row:02d}")
canvas.save()
PY

python code/preprocessing/pdf_preprocess.py "$REVIEW_ROOT/review-gate.pdf" \
  --device "$DEVICE" \
  --orientation-min-score 1.0 \
  --orientation-min-margin 1.0 \
  --output-root "$REVIEW_ROOT/output" \
  --overwrite 2>&1 | tee "$REVIEW_ROOT/run.log"

REVIEW_ARTIFACT="$(sed -n 's/^published_output=//p' "$REVIEW_ROOT/run.log" | tail -n 1)"
python - "$REVIEW_ARTIFACT" <<'PY'
import json
import sys
from pathlib import Path

artifact = Path(sys.argv[1])
orientation = json.loads((artifact / "orientation_report.json").read_text(encoding="utf-8"))
quality = json.loads((artifact / "quality_report.json").read_text(encoding="utf-8"))
raw_document = (artifact / "document.json").read_text(encoding="utf-8")
markdown = (artifact / "document.md").read_text(encoding="utf-8")
marker = "REVIEW_GATE_SENTINEL_314159"

assert orientation["pages"][0]["decision"] == "review_required"
assert orientation["pages"][0]["eligible_for_indexing"] is False
assert quality["pages"][0]["eligible_for_indexing"] is False
assert marker in raw_document, "audit document.json lost the page"
assert marker not in markdown, "untrusted content leaked into document.md"
print("review-gate acceptance passed")
PY
```

## 6. NASA-STD-5006A：数字表格端到端验收

在代码、单元测试和 TableFormer 本地资产清单完成最终静态审阅后，再在服务器运行此项；
最终代码与回归测试同步完成后再覆盖服务器版本。Golden 文件固定为 48 页且 SHA-256 必须为
`fa229965758a0f0c630034084173341e2a0053a1ca25d35a15b6d14e9b8e5c20`。验收覆盖图片表格、
普通图片、第 36--48 页 Requirements Compliance Matrix、原位插入及真实表格行顺序。

```bash
export STD5006A_PDF="$PWD/data/engineering_docs/raw/joining/13_nasa_std_5006a_welding_requirements.pdf"
STD5006A_RUN="$ACCEPTANCE_ROOT/nasa-std-5006a"

python code/preprocessing/verify_pdf_preprocess_server.py \
  --input-pdf "$STD5006A_PDF" \
  --page 36 \
  --device "$DEVICE" \
  --num-threads 8 \
  --document-timeout 7200 \
  --output-root "$STD5006A_RUN" \
  --golden-5006a 2>&1 | tee "$STD5006A_RUN.log"

STD5006A_ARTIFACT="$(sed -n 's/^published_output=//p' "$STD5006A_RUN.log" | tail -n 1)"
test -n "$STD5006A_ARTIFACT"

# 上面的 --golden-5006a 已完成 canonical JSON/Markdown 逐表重建比对、
# 页归属、索引一致性及视觉正文泄漏检查；下面只打印便于人工阅读的二次摘要。
python - "$STD5006A_ARTIFACT" <<'PY'
import json
import re
import sys
from pathlib import Path

artifact = Path(sys.argv[1])
quality = json.loads((artifact / "quality_report.json").read_text(encoding="utf-8"))
regions = json.loads((artifact / "regions.json").read_text(encoding="utf-8"))
table_index = json.loads((artifact / "tables" / "index.json").read_text(encoding="utf-8"))
markdown = (artifact / "document.md").read_text(encoding="utf-8")

assert len(quality["pages"]) == 48, "all 48 source pages must be audited"
assert [page["page_no"] for page in quality["pages"]] == list(range(1, 49))
assert all(page["eligible_for_indexing"] is True for page in quality["pages"])
assert quality["table_summary"] == table_index["summary"]
assert "cells" not in quality["table_summary"]

def page_regions(page_no, kind=None):
    return [
        region for region in regions
        if region.get("page_no") == page_no
        and (kind is None or region.get("region_type") == kind)
    ]

# The WPS is an image-based example, never a canonical native table. Its
# explicit figure caption remains eligible exactly once if the page is trusted.
page17 = page_regions(17)
assert not any(
    region.get("region_type") == "table"
    and region.get("canonical_table_in_semantic_markdown")
    for region in page17
)
caption_pattern = r"Figure\s+1\s*[-—–]\s*Welding Procedure Specification Example"
if any(page["page_no"] == 17 and page["eligible_for_indexing"] for page in quality["pages"]):
    assert len(re.findall(caption_pattern, markdown, flags=re.I)) == 1

# Figures on pp. 20, 22 and 25 stay outside the semantic body; trustworthy
# Docling-associated captions are retained once by the existing projection.
for page_no in (20, 22, 25):
    pictures = page_regions(page_no, "picture")
    assert pictures, f"expected picture region on page {page_no}"
    assert all(not item["visual_body_in_semantic_markdown"] for item in pictures)
    for picture in pictures:
        if picture["caption_trust_decision"] == "accepted":
            assert picture["caption_in_semantic_markdown"] is True
            assert markdown.count(picture["caption"]) == 1

matrix_records = [
    entry for entry in table_index.get("tables", [])
    if 36 <= entry.get("page_no", 0) <= 48
]
assert matrix_records, "Requirements Compliance Matrix tables were not detected"
visible_matrix_blocks_by_page = {}
accepted_matrix_pages = set()
for entry in matrix_records:
    record = json.loads((artifact / entry["artifact"]).read_text(encoding="utf-8"))
    if record["decision"] != "accepted":
        assert record["decision"] in {"deferred", "rejected"}
        assert f"<!-- TABLE id={record['table_id']}" not in markdown
        continue
    assert record["source_kind"] == "native"
    accepted_matrix_pages.add(record["page_no"])
    assert record["validation"]["text_conservation_ratio"] == 1.0
    source_refs = [
        ref
        for cell in record["cells"]
        for ref in cell["source_cell_refs"]
    ]
    expected_refs = {
        item["source_cell_ref"]
        for item in record["validation"]["source_cells"]
        if item["from_ocr"] is False
    }
    assert len(source_refs) == len(set(source_refs))
    assert set(source_refs) == expected_refs
    assert record["validation"]["output_cells_deterministically_rebuilt"] is True
    if any(cell["row_span"] != 1 or cell["column_span"] != 1 for cell in record["cells"]):
        assert record["validation"]["markdown_span_projection"] == "anchor_only"
    marker = f"<!-- TABLE id={record['table_id']}"
    if any(page["page_no"] == record["page_no"] and page["eligible_for_indexing"] for page in quality["pages"]):
        assert markdown.count(marker) == 1
        block = re.search(
            rf"<!-- TABLE id={re.escape(record['table_id'])} .*?-->\s*(.*?)\s*<!-- /TABLE -->",
            markdown,
            flags=re.S,
        )
        assert block, f"missing complete Markdown block for {record['table_id']}"
        visible_matrix_blocks_by_page.setdefault(record["page_no"], []).append(
            block.group(1)
        )
    else:
        assert marker not in markdown
assert accepted_matrix_pages == set(range(36, 49)), (
    "every page from 36 through 48 must contain an accepted native table"
)
# Every matrix page must pass; the combined view must reveal the expected columns.
matrix_markdown = "\n".join(
    block
    for page_no in sorted(visible_matrix_blocks_by_page)
    for block in visible_matrix_blocks_by_page[page_no]
)
for label in ("section", "description", "requirement"):
    assert label in matrix_markdown.lower(), f"missing Matrix column: {label}"
page36_markdown = "\n".join(visible_matrix_blocks_by_page[36])
pos_242 = page36_markdown.find("2.4.2")
pos_group = page36_markdown.find("4. Requirements", pos_242 + 1)
pos_41 = page36_markdown.find("4.1", pos_group + 1)
assert 0 <= pos_242 < pos_group < pos_41, (
    "page 36 must preserve 2.4.2 < 4. Requirements < 4.1"
)
print("NASA-STD-5006A table acceptance passed", table_index["summary"])
PY
```

请回传以下证据以继续判定：`nvidia-smi` 和依赖版本输出、真实模型验收 JSON、NASA
运行日志及其四个 JSON/Markdown 产物、强制 review 用例的同名产物，以及 5006A 的
`quality_report.json`、`regions.json`、`tables/` 和 `document.md`。
