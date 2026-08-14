# 制造业文档 RAG V0.1：PDF 预处理实现与交接说明

> 交接日期：2026-08-14
>
> 当前范围：PDF 文档端预处理；不包含 Chunk、Embedding、Milvus、检索和生成
>
> 结论口径：主流程已经能够稳定执行并生成完整审计产物，但 **V0.1 最终验收尚未通过，不能标记为 100% 完成**。当前至少还有两个已经定位的 P0 问题：Figure 复合区域隔离，以及真实数字表格的结构准入。

## 1. 目标与已经冻结的产品口径

本模块的目标是把公开制造业技术 PDF 转换为可供后续 RAG 使用的正式文本视图，同时保留可追溯的原始解析和质量证据。

V0.1 已冻结的边界如下：

1. 当前只完成 PDF 主链路。TXT、Markdown、DOCX 可由后续输入 Adapter 接入，但不在本模块的现有实现内。
2. 扫描 PDF 与数字 PDF 都要支持；扫描正文允许通过 OCR 进入正式 Markdown。
3. V0.1 不进行图片、图表和 OCR 表格的语义理解。
4. 图片本体、图内文字、OCR 表格本体不应进入 `document.md`；可信题注应保留，以便后续检索返回来源页、章节、题注和区域坐标。
5. OCR 在 Docling 转换阶段可能对图片或表格区域产生计算，这部分额外开销可以接受；隔离发生在正式语义视图生成阶段，而不是强行阻止 OCR 执行。
6. 数字 PDF 的原生表格允许进入 Markdown，但必须通过结构合法性、来源文字一一对应和文字守恒校验。
7. OCR、mixed、image-only 表格走与数字表格平行的扩展接口；V0.1 只生成 `deferred` 审计记录，不生成表格正文。
8. 不引入视觉模型。图片/VLM、OCR 表格解析、CAD 等能力只保留可扩展接口。
9. 页面质量不可信时，不删除原始解析结果；完整内容保留在审计产物中，但该页不得进入正式 `document.md`。
10. `document.md` 是唯一允许交给后续 Chunk/Embedding 的正式文本；`document.json`、`regions.json` 和 `quality_report.json` 是审计与未来重处理依据，不应被下游当作正文直接入库。

## 2. 当前实际处理链路

```text
原始 PDF（只读）
    ↓
输入、依赖和本地模型完整性检查
    ↓
整页方向分类：PP-LCNet_x1_0_doc_ori（0/90/180/270）
    ↓
旋转候选复核，通过后只修改副本的 PDF /Rotate
    ↓
Docling Layout Heron
    ↓
RapidOCR：PP-OCRv6 medium Det + PP-OCRv6 medium Rec
    ↓
TableFormer V1 accurate：数字表格结构候选
    ↓
逐页质量门禁
    ↓
视觉区域隔离 + 可信题注投影
    ↓
数字表格 Canonical JSON 校验与 Markdown 投影
    ↓
document.md + document.json + regions.json + quality_report.json
    ↓
事务式发布到稳定输出目录
```

主入口为：

```text
code/preprocessing/pdf_preprocess.py
```

## 3. 实现方法

### 3.1 输入校验、稳定 ID 与事务发布

当前入口只接受 `.pdf`。执行前会检查：

- 输入文件存在且不是加密 PDF；
- `--page-range FIRST LAST` 是 1-based 闭区间且没有超过原 PDF 页数；
- `--device` 只能是 `cpu`、`cuda` 或 `cuda:N`，禁止隐式 `auto`；
- 所需模型文件存在，大小和 SHA-256 与项目清单一致；
- Docling Markdown Serializer 具有本项目隔离图片所需的 API；
- TableFormer 固定资产存在，禁止运行时下载。

`document_id` 由原文件名和原 PDF SHA-256 前缀稳定生成。输出先写到：

```text
<output-root>/.staging/<document_id>.<uuid>/
```

只有完整转换成功、没有转换错误且至少存在一个可入库页面时，才原子发布到：

```text
<output-root>/<document_id>/
```

失败或部分成功的产物进入 `.failed/`，不会覆盖上一次成功输出。`--overwrite` 只允许替换同一 `document_id` 的已知目标目录。

### 3.2 整页方向检测

整页方向检测在 Docling 和 RapidOCR 之前独立执行，模型是：

```text
models/PageOrientation/PP-LCNet_x1_0_doc_ori/
```

处理逻辑：

1. 按 PDF 当前 `/Rotate` 的显示效果，以默认 150 DPI 渲染页面。
2. 页面有效墨迹比例不高于 `0.0005` 时记为 `not_applicable`，不对空白页强行分类。
3. 模型同时输出 0、90、180、270 四类分数。
4. 只有 top-1 分数不低于 `0.90`，且 top-1 与 top-2 分差不低于 `0.15`，才形成方向候选。
5. 若候选不是 0 度，则生成旋转副本并再次渲染；复核后的 0 度分数必须不低于 `0.90`。
6. 复核通过才修改规范化 PDF 副本的 `/Rotate`；不栅格化页面，不修改原始 PDF。
7. 低置信度、分差不足或旋转后复核失败的页面标记为 `review_required`，不得进入正式 Markdown。

这里的 PP-LCNet 是整页四方向模型。RapidOCR 使用的 PP-OCRv4 `cls` 是 0/180 度文字行方向分类器，只在显式传入 `--enable-direction-classifier` 时启用，不能替代整页方向检测；当前正式命令默认没有启用该文字行分类器。

方向阶段输出 `orientation_report.json`，逐页记录原始旋转、四类分数、top-1、margin、应用旋转、复核分数、决定和 Provider。

### 3.3 OCR 路由与“栅格页”的量化标准

Docling 的 OCR 路由根据 PDF 页面中 bitmap 区域占页面面积的比例决定，当前沿用并审计以下固定值：

| 条件 | 路由 |
| --- | --- |
| bitmap coverage `> 0.75` | `full_page_ocr` |
| `0.05 < bitmap coverage <= 0.75` | `region_ocr` |
| bitmap coverage `<= 0.05` | `native_only` |

因此“页面主要由栅格图组成”在当前实现中不是程度词，而是 bitmap 覆盖率严格大于 75%。`force_full_page_ocr` 默认关闭；`bitmap_area_threshold` 默认 5%。每页实际覆盖率、阈值、请求路由和 OCR 矩形数量写入质量报告。

OCR 始终优先保留 PDF 原生文字。OCR Cell 与已有原生文字相交时，Docling 会做冲突过滤，避免无条件用 OCR 覆盖数字版文本。

当前 OCR 语言配置为 `english`，与 NASA/NIST 英文语料一致。如果后续加入中文或中英混排文档，必须单独评估识别模型、字典和 `lang` 配置，不能直接宣称现有配置已支持。

### 3.4 CUDA 执行策略

页面方向、Layout 和 OCR 共用显式 `--device`。CUDA 模式采用：

```text
CUDAExecutionProvider 优先 + CPUExecutionProvider 节点级补充
```

这不是强制所有 ONNX 节点只能在 GPU 上执行。部分 `Concat`、`Slice` 等 CUDA EP 不支持或不适合的辅助节点可以运行在 CPU；但真实会话的第一个 Provider 必须是 CUDA。若用户选择 CUDA 而 RapidOCR Det/Cls/Rec 的会话整体退化成 CPU-first，程序立即失败，不允许静默使用 CPU 跑完整 OCR。

项目内的 Docling 快照为此做了修改：

- 通过 RapidOCR 3.9.2 的可序列化配置入口设置 `EngineConfig.onnxruntime.use_cuda` 和 `device_id`；
- 不向 OmegaConf 塞入 Python `InferenceSession`、`SessionOptions` 或 `Path` 对象；
- RapidOCR 自己创建 Det/Cls/Rec 会话后，再读取真实 `get_providers()`；
- 每页记录实际 Provider 和原始 OCR 数量、均值置信度；
- ORT 1.23+ 可从已安装的 PyTorch CUDA 轮子预载 CUDA/cuDNN 动态库。

### 3.5 Docling 文档解析配置

当前配置为：

- Layout：`docling-project/docling-layout-heron`，项目内固定 revision；
- OCR：RapidOCR 3.9.2，ONNX Runtime；
- Det：PP-OCRv6 medium；
- Rec：PP-OCRv6 medium；
- 文字行 Cls：PP-OCRv4 mobile 0/180，可选，默认关闭；
- 表格结构：TableFormer V1 `accurate`，`do_cell_matching=True`；
- 标题层级：启用 Heading Hierarchy；
- 图片分类、图片描述、Chart Extraction、代码增强、公式增强：全部关闭；
- 不生成页面、图片或表格渲染图；保留 parsed pages 供质量和表格来源追溯；
- Layout batch 和 OCR batch 均为 1，优先兼容 2080 Ti 11 GB 等可用 GPU；
- 默认单文档超时 3600 秒，正式服务器命令使用 7200 秒。

### 3.6 页面质量门禁

质量门禁逐页执行，拒绝只影响正式 Markdown，不删除 `document.json` 中的原始结果。

固定阈值：

| 指标 | 阈值 | 当前解释 |
| --- | ---: | --- |
| 原生正文 parse score | `>= 0.50` | Docling 的 POOR/FAIR 边界 |
| 一般 OCR 正文均值置信度 | `>= 0.75` | RapidOCR 已过滤区间中点，作为无人持续调参时的固定准入线 |
| 少于 20 个字母数字字符的短 OCR | `>= 0.90` | 短文本冗余少，错误更难由上下文暴露 |
| 长度至少 20 的正文，字母数字比例 | `>= 0.35` | 低于该值视为明显由符号噪声主导 |
| 至少 10 个 token 时，单字符 token 比例 | `<= 0.50` | 超过该值视为明显字符碎片化 |

另外，出现 Unicode replacement character、非法控制字符、Docling 页级/文档级错误时会拒绝页面。

OCR 质量只统计表格/图片 bbox 之外的 OCR Cell。判断一个 OCR Cell 是否属于视觉区域的规则是其中心点是否落入 table/picture bbox。这样图片或表格内部的低置信 OCR 不会导致同页正常正文被整体拒绝。

Layout score 只记录为诊断证据，不作为“框一定画对”的无监督正确率，也不直接决定页面准入。

### 3.7 图片、图表和题注的语义投影

当前代码会遍历每个 `TableItem` 和 `PictureItem`：

1. 收集其显式 `captions` 引用；
2. 递归收集视觉对象自身和后代，包括 RichTableCell 指向的节点；
3. 从排除集合中减去可信、显式关联的题注引用；
4. 只序列化 `eligible_for_indexing=true` 的页面；
5. 每页在 Markdown 中保留 `<!-- PDF page N -->`；
6. 同一题注最多输出一次；
7. 关闭 `enable_chart_tables`，避免 Picture 派生表格绕过隔离。

题注的状态为 `accepted`、`missing`、`garbled` 或 `page_untrusted`。只把 `accepted` 题注写入 Markdown；其余原值仍留在审计产物。

这一机制对“题注和图内文字都是视觉对象子节点”的常规 Docling 树有效，但对“一个 Figure 被切成多个 PictureItem，部分说明文字和最终 Caption 被放在 body 顶层”的情况尚未闭环，详见第 7.1 节。

### 3.8 数字 PDF 表格

每个 Docling `TableItem` 独立路由，不按整份文档粗分“数字版/扫描版”。表格 bbox 内的来源文字根据 `from_ocr` 字符数分成：

- `native`：只有 PDF 原生文字；
- `ocr`：只有 OCR 文字；
- `mixed`：两者都有；
- `image_only`：没有可用文字。

V0.1 只尝试接纳 `native`。其余类型进入 `OcrTableExtractor` 预留路径，产出 `deferred`、`cells=[]` 的 Canonical 记录。

`NativePdfTableExtractor` 使用 TableFormer 的行列与 bbox，但不信任模型生成的单元格文字。每个输出 Cell 的文字必须由真实 PDF source cell 重新组合。接纳条件包括：

- 行列数和 span 合法；
- TableFormer 结构不存在逻辑格重叠；
- 每个 source cell 的中心点恰好属于一个结构 Cell；
- source ref 一一对应，不丢失、不重复；
- 输出 Cell 可以仅由 `source_cell_refs` 确定性重建；
- 原生文字守恒率严格等于 `1.0`；
- 当前实现还要求逻辑网格每个坐标都被某个 TableFormer Cell 覆盖。

通过后写入 `tables/<table_id>.json`，并在原 `TableItem` 位置用唯一占位符注入 Markdown pipe table。Canonical JSON 保留 rowspan/colspan；Markdown 采用 anchor-only 投影：span 文字只写左上角，覆盖格置空。没有可信显式表头时使用空 Markdown 表头，不把首行数据臆造为表头。

单个 `TableItem` 若跨多个 provenance 页，V0.1 只生成一个 `deferred` 审计记录，禁止复制 Cells，也不自动合并跨页表。

### 3.9 未来多模态和 OCR 表格接口

当前已经具备以下接入点：

- `OcrTableExtractor` 与 `NativePdfTableExtractor` 输出相同的 `TableExtractionResult`；
- `regions.json` 保留图片/表格的原始页号、bbox、章节、题注和 Docling ref；
- `document.json` 保留完整视觉区域 OCR 和 Docling 树；
- 原始/规范化 PDF 可按页和 bbox 重新渲染；
- 未来 VLM、OCR 表格模型只需生成标准化文本、Canonical Cells、来源页和 bbox，再通过独立质量门禁进入相同下游链路。

目前这些接口存在，但视觉语义解析和 OCR 表格解析均没有实现。

## 4. 产物职责

| 产物 | 当前职责 |
| --- | --- |
| `document.md` | 唯一正式入库文本；含通过门禁的正文、标题、列表、可信题注和通过校验的原生数字表格 |
| `document.json` | 完整、未裁剪的 Docling 文档树，供审计和未来重处理；不得直接送入 Chunk |
| `regions.json` | 表格/图片定位，包括页码、bbox、章节、题注、信任决定、Canonical 表格关联 |
| `quality_report.json` | 文档配置、版本、总体置信度、逐页准入、OCR 路由、Provider、区域计数和表格汇总 |
| `orientation_report.json` | 逐页方向模型输入证据、四类分数、旋转与复核结果 |
| `normalized/oriented.pdf` | 保持原页数和矢量文字的方向规范化副本 |
| `tables/index.json` | 紧凑表格索引和汇总，不包含所有 Cells |
| `tables/<table_id>.json` | 单表 Canonical Schema、来源 Cells、结构、决定和失败理由 |

`quality_report.json` 会保存 `pages` 数组，因此大小随页数线性增长；每页只放聚合指标、计数、阈值结果和 Provider 证据，不复制页面图像、OCR 全文或每个 OCR 框。Chunk metadata 不在该文件中，也尚未由本模块生成。

## 5. 代码与修改范围

### 5.1 新增的预处理模块

| 文件 | 职责 |
| --- | --- |
| `code/preprocessing/pdf_preprocess.py` | 主入口、Docling 配置、页面门禁、语义投影、表格注入、产物事务发布 |
| `code/preprocessing/page_orientation.py` | PP-LCNet 元数据校验、四方向推理、旋转复核、PDF `/Rotate` 重写和报告 |
| `code/preprocessing/table_extraction.py` | 表格路由、Canonical Schema、原生文字守恒校验、Markdown 投影、OCR 扩展桩 |
| `code/preprocessing/verify_page_orientation_real_model.py` | 真实方向模型验收 |
| `code/preprocessing/verify_pdf_preprocess_server.py` | 真实模型、OCR、产物、Golden 文档端到端验收 |
| `code/preprocessing/requirements-server.txt` | 服务器端预处理依赖范围 |
| `code/preprocessing/tests/` | 方向、CUDA 会话、语义投影、表格和服务器验收单元测试 |

### 5.2 项目内 Docling 快照的必要修改

这些改动不能在同步时遗漏：

| 文件 | 修改原因 |
| --- | --- |
| `docling/docling/datamodel/base_models.py` | 给 Page 增加运行期 `ocr_audit` |
| `docling/docling/datamodel/pipeline_options.py` | 明确 RapidOCR 参数必须可被 OmegaConf 序列化 |
| `docling/docling/models/base_ocr_model.py` | 记录 bitmap coverage、OCR 路由和矩形数 |
| `docling/docling/models/stages/ocr/rapid_ocr_model.py` | 正确配置 CUDA、验证 Det/Cls/Rec 真实 Provider、记录 OCR 证据 |
| `docling/pyproject.toml`、`docling/PKG-INFO` | 将 `docling-core` 下限提高到 2.88.0 |

### 5.3 当前代码文件 SHA-256

以下值用于确认接替者看到的是本次交接时的工作树，不代表长期发布版本：

```text
db438afc2fd83e6a3fe60a5d57eb768b74de32cbeab152d600d6625ba602d05e  code/preprocessing/page_orientation.py
dc459b81d0b91e5c1c4724d152025a77d1975d9e4e1b611c6e11a76cab6a4b5c  code/preprocessing/pdf_preprocess.py
e28b69f6264d6bb73665d1eb217c08af4a7800c5752324ca2ceab8899cffaf26  code/preprocessing/table_extraction.py
e4528955db2cf5e94268f6c915c6c7adea48992a0079097d9608de1e71ed3cd3  code/preprocessing/verify_page_orientation_real_model.py
aefe2e0f0b9edd33c79577ecc20eecc99e2b2789bd7ee032a83e37ca8b807b69  code/preprocessing/verify_pdf_preprocess_server.py
89015de2b2d77d51b7da12134eafcf4749f88d0507ce28c572a0d44f159dab1a  code/preprocessing/requirements-server.txt
c11f8498ef7bc1df15d024f5b7b0c32b6675d100ef7079906b9ae12a506c1f9e  docling/docling/datamodel/base_models.py
108f7b78241db7120248ce16b3d465437fcd1a96f2e63ade0ba3825e98242b9c  docling/docling/datamodel/pipeline_options.py
c15c449a95fb83224e09d6bdcebfcaf1e09c1e941f72455d16855325a47c1017  docling/docling/models/base_ocr_model.py
fb8143aba925dcd99f00b580b707fd5615d3bfddd548f04171a343c4256f77c4  docling/docling/models/stages/ocr/rapid_ocr_model.py
2a99624574ab9ef051d93c14529aa2a133ac0c67b0b92a24fb7432e7d9003edd  models/preprocessing-models.manifest.json
```

当前工作树不是干净提交：`code/preprocessing/` 整体仍是未跟踪目录，Docling 快照、README、依赖和规范文件有未提交修改；大型模型权重多数被 `.gitignore` 排除。接替者不能只依赖 Git commit 恢复现状，必须以本地工作区和模型清单为准。

## 6. 环境和模型资产

服务器实际验收环境曾确认：

```text
Python 3.11
docling-slim 2.115.0
rapidocr 3.9.2
onnxruntime-gpu 1.23.2
PyTorch 2.6.0+cu126（后一次真实 CUDA 验收环境）
CUDAExecutionProvider 可用
```

PyTorch CUDA Wheel 与系统 `nvidia-smi` 显示的最高 CUDA 版本不需要完全相同；关键是驱动向后兼容、ORT CUDA Provider 能创建真实会话，并在 Det/Cls/Rec 中处于首位。

本地必须存在的模型资产：

```text
models/RapidOcr/onnx/PP-OCRv6/det/PP-OCRv6_det_medium.onnx
models/RapidOcr/onnx/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx
models/RapidOcr/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx
models/RapidOcr/resources/fonts/FZYTK.TTF

models/docling-project--docling-layout-heron/model.safetensors
models/docling-project--docling-layout-heron/config.json
models/docling-project--docling-layout-heron/preprocessor_config.json
models/docling-project--docling-layout-heron/REVISION

models/docling-project--docling-models/model_artifacts/tableformer/accurate/tm_config.json
models/docling-project--docling-models/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors

models/PageOrientation/PP-LCNet_x1_0_doc_ori/model.onnx
models/PageOrientation/PP-LCNet_x1_0_doc_ori/inference.yml
models/PageOrientation/PP-LCNet_x1_0_doc_ori/labels.json
models/PageOrientation/PP-LCNet_x1_0_doc_ori/manifest.json
```

完整大小、来源和 SHA-256 位于：

```text
models/preprocessing-models.manifest.json
```

## 7. 当前实现状态与真实验收结果

### 7.1 NASA-STD-5006A：转换成功，但最终验收失败

本地回传产物目录：

```text
outputs/final-acceptance-5006a/
  13_nasa_std_5006a_welding_requirements-fa229965758a0f0c/
```

主程序真实结果：

```text
status=success
pages=48
tables=27
pictures=7
published_output=.../13_nasa_std_5006a_welding_requirements-fa229965758a0f0c
```

质量概况：

| 项目 | 结果 |
| --- | ---: |
| 总页数 | 48 |
| 可入库页 | 48 |
| 排除页 | 0 |
| 方向 upright | 48 |
| 方向 rotated / review / N/A | 0 / 0 / 0 |
| 文档 parse score | 1.0 |
| 文档 layout score | 0.8829949435 |
| 文档 OCR score | 0.9826746811 |
| 表格数 | 27 |
| 图片数 | 7 |
| accepted native 表格 | 7 |
| deferred OCR 表格 | 3 |
| rejected structure 表格 | 17 |
| accepted 图片题注 | 2 |
| missing 图片题注 | 5 |

最终命令返回：

```text
server_acceptance_error: Deferred table/picture body text leaked into document.md:
region=#/pictures/5 text_ref=#/texts/361
final_acceptance_exit_code=1
```

这个报错包含一个验收器假阳性和一个真实投影缺口，必须分开理解。

#### 假阳性

`#/texts/361` 的真实内容是：

```text
Equal leg fillet weld
```

它是 `#/pictures/5` 的子节点，实际上没有进入 `document.md`。当前验收器把整个 Markdown 规范化后做任意子串判断：

```python
if text in normalized_markdown:
```

Markdown 中存在另一个顶层文本：

```text
Unequal leg fillet weld
```

因此 `Equal leg fillet weld` 被匹配为 `Unequal leg fillet weld` 的子串，错误地把 `#/texts/361` 报成泄露。验收器应改为精确规范化 block/段落匹配或结构引用验证，不能做无边界子串搜索。

#### 真实缺口

PDF 第 25 页的 Figure 4 被 Docling 切成四个顶层 `PictureItem`：

```text
#/pictures/3
#/pictures/4
#/pictures/5
#/pictures/6
```

其中框内子节点已被排除，但以下 Figure 说明被 Docling 放在 `document.body` 顶层，而不是某个 PictureItem 的 children：

```text
#/texts/349  Notes: Root of Joint ...
#/texts/350  Root of Weld ...
#/texts/351  The root of the weld shall penetrate ...
#/texts/368  Unequal leg fillet weld
#/texts/369  For unequal-leg fillet welds ...
```

最终题注也是 body 顶层节点：

```text
#/texts/370  label=caption  Figure 4-Fillet Welds
```

当前语义投影只隔离 PictureItem 的显式后代，只信任 `PictureItem.captions` 的显式引用，所以五段 Figure 正文进入了 `document.md`，而 `#/texts/370` 虽然自然出现在 Markdown 中，却没有与四个 PictureItem 建立可审计关联。按照当前“图片本体和图内说明不入库，只保留题注”的冻结口径，这是真实 P0 问题。

建议的高精度修复边界：只在同一 PDF 页的 body 顺序中，识别“一个或多个 PictureItem → 中间普通文本/PictureItem → 明确 `label=caption` 的终止节点”，且中间没有 Table、Section Header、页切换或其他结构边界时，才建立 inferred figure group；排除中间 Figure body refs，保留终止 Caption，并在 `regions.json` 中记录推断关联。没有终止 Caption 时不得向后扩大排除范围。

### 7.2 NASA-STD-5006A：数字表格仍没有通过 Golden 基线

即使修复第 7.1 节，当前 Golden 验收还会继续失败。验收器要求第 36–48 页的 requirements matrix 每页都有 accepted native table；真实产物中这些 13 页全部是 `rejected`，失败理由均为：

```text
logical_grid_has_uncovered_coordinates
```

真实 accepted native 表格仅出现在：

```text
PDF page 4, 7, 8, 32, 33, 34, 35
```

第 17 页的三个表格被正确识别为 OCR 表格并 `deferred`，符合 V0.1 边界。

因此当前数字表格实现不能称为完成。`89 passed` 的单元测试使用了构造的 Canonical 表格和模拟 Docling 结构，没有证明真实 TableFormer 对 5006A 第 36–48 页输出的逻辑网格满足现有严格覆盖条件。接替者需要检查真实 `TableItem.data.table_cells` 与 `num_rows/num_cols`：

1. 未覆盖坐标究竟是 TableFormer 漏框、合法的空逻辑格、span 表示差异，还是坐标/边界转换错误；
2. 不能直接删除网格检查后宣布通过；任何放宽都必须继续保证 source cell 一一归属、文字守恒率 1.0、输出可由 source refs 确定性重建；
3. 应增加使用回传 `document.json`/真实 TableFormer 结构的离线回归测试，而不是只用手工 Fake；
4. 修复后必须重新检查第 36 页顺序 `2.4.2 < 4. Requirements < 4.1` 以及第 36–48 页矩阵表头是否都存在。

### 7.3 扫描件阶段性结果

早期扫描件：

```text
data/engineering_docs/raw/material/01_nasa_materials_aluminum_2014.pdf
```

一次完整产物显示：

| 项目 | 结果 |
| --- | ---: |
| 总页数 | 136 |
| 可入库页 | 113 |
| 排除页 | 23 |
| upright | 112 |
| rotated | 1 |
| not_applicable | 6 |
| review_required | 17 |
| tables | 42 |
| pictures | 73 |

这证明方向、OCR、质量隔离和审计主链可以运行，但该原始 PDF 自身质量较差，输出观感一般，不能用它单独证明工业可用性；也不能用 5006A 数字版的结果替代扫描正文准确率评估。扫描 OCR 仍需要一个小规模、有原文对照或人工核对的独立验收集。

### 7.4 单元测试状态

服务器最近一次结果：

```text
89 passed, 1 skipped, 1 warning in 24.11s
```

Skip 是需要真实服务器/模型条件的测试分支；Warning 是 Docling `rec_font_path` deprecation。单元测试通过只能证明确定性代码分支符合测试，不等于 Golden 真实模型验收通过。

## 8. 接替者的处理优先级

### P0-1：修复 Figure 复合区域隔离和验收器假阳性

应一次完成以下内容：

1. 用同页、同 body 顺序、明确终止 Caption 和结构边界建立 inferred figure group；
2. 把 group 中间的普通文本加入正式 Markdown 排除集合；
3. 保留 Caption，且在 `regions.json` 中记录 `caption_ref`、关联类型和 group 成员；
4. 不重复输出一个共享 Caption；
5. 验收器从任意子串匹配改为精确 block 或引用级验证；
6. 增加 `Equal leg...` 不得误匹配 `Unequal leg...` 的回归测试；
7. 增加“无终止 Caption 时不扩展排除”的负例，防止误删正常正文；
8. 直接使用现有 5006A 回传产物验证第 25 页只留下 `Figure 4-Fillet Welds`，不留下 349–351、368–369。

### P0-2：修复真实数字表格准入

1. 用第 36 页真实 `document.json` 检查 `num_rows/num_cols`、TableFormer Cells、span、bbox 和 uncovered coordinates；
2. 明确哪些空坐标是结构上合法的，哪些是模型漏检；
3. 在不降低来源文字守恒合同的前提下修复 Canonical 网格构造；
4. 用第 36–48 页真实矩阵做离线回归；
5. 重新运行 Golden，必须得到 13 个矩阵页全部 accepted；
6. 验证表格 Markdown 顺序、表头、span anchor-only 投影和 table markers。

### P1：建立最小扫描 OCR 质量验收

主链路 Golden 通过后，再从扫描文档选少量有代表性的正文页，至少覆盖正常扫描、旋转页、低质量页和空白页。固定核对正文遗漏、字符错误、页码、标题层级和是否误收图表内部文本。该项尚未完成，不应在修复两个 P0 前扩展到视觉模型。

## 9. 服务器验收命令

### 9.1 环境与单元测试

```bash
cd ~/MyMethod/all-in-rag-main
conda activate mfg-rag-preprocess

export PYTHONPATH="$PWD/docling:$PWD/code/preprocessing${PYTHONPATH:+:$PYTHONPATH}"

python -m pip check
python -m pytest code/preprocessing/tests -q
```

### 9.2 5006A 最终 Golden

```bash
mkdir -p logs/preprocessing
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python \
  code/preprocessing/verify_pdf_preprocess_server.py \
  --input-pdf data/engineering_docs/raw/joining/13_nasa_std_5006a_welding_requirements.pdf \
  --page 36 \
  --device cuda \
  --num-threads 8 \
  --document-timeout 7200 \
  --output-root outputs/final-acceptance-5006a \
  --golden-5006a \
  2>&1 | tee logs/preprocessing/final-5006a.log

rc=${PIPESTATUS[0]}
echo "final_acceptance_exit_code=$rc"
test "$rc" -eq 0
```

`--golden-5006a` 会处理并检查完整 48 页；`--page 36` 仍用于通用真实页面 OCR 检查，但不是只处理第 36 页。

最终合格必须同时满足：

- `final_acceptance_exit_code=0`；
- 48 页全部有且只有一个顺序正确的 PDF page marker；
- CUDA 为方向和 RapidOCR 三阶段的首 Provider；
- 第 17 页 OCR 表格保持 deferred；
- 第 20、22、25 页图片本体没有进入 Markdown，可信题注仍在；
- 第 36–48 页 requirements matrix 每页都有 accepted native table；
- 第 36 页顺序满足 `2.4.2 < 4. Requirements < 4.1`；
- 没有未替换 Table placeholder、重复表块、错误页题注或视觉正文泄露。

## 10. 交接时不要误判的事项

1. `status=success` 只说明 Docling 主转换成功并发布了产物，不说明 Golden 所有业务断言通过。
2. `regions.json` 中 `visual_body_in_semantic_markdown=false` 是程序声明；仍需独立检查实际 Markdown，当前 Figure 4 已证明仅靠该字段不足。
3. `#/texts/361` 本身没有泄露；它是 `Equal`/`Unequal` 子串假阳性。不要围绕错误 ref 修补正文。
4. Figure 4 确实有五段顶层关联文字被保留；不能因为 361 是假阳性就忽略真实隔离缺口。
5. 修好 Figure 后 Golden 仍会在第 36–48 页表格失败；这个问题已经从现有产物确定，不要等待下一次服务器返工才发现。
6. 不能把 `logical_grid_has_uncovered_coordinates` 简单从拒绝条件中删除；应先用真实结构证明安全的空格规则。
7. 当前模型文件是项目本地部署资产，不要求放进 Python 包或 Docling 源码目录；代码按 `models/` 下固定路径读取。
8. 不要重新引入“CUDA-only 每个节点都必须在 GPU”限制；正确口径是 CUDA-first、允许节点级 CPU fallback、禁止整体 CPU-first 静默降级。
9. 不要让后续 Chunk 读取 `document.json` 或 `quality_report.json`；正式输入只能是最终验收通过的 `document.md` 加来源/区域 metadata。

## 11. 当前交接结论

已完成且有真实运行证据的部分：

- 本地模型完整性检查；
- 四方向页面标准化与旋转复核；
- CUDA-first mixed ONNX 执行；
- RapidOCR Det/Cls/Rec Provider 核验；
- Docling Heron 解析；
- OCR 路由审计；
- 逐页质量门禁和不可信页面隔离；
- 事务式产物发布；
- 常规 Picture/Table 子树隔离和显式题注保留；
- 数字表格 Canonical Schema、来源文字守恒框架及 OCR 表格并行接口；
- 完整审计产物和服务器验收脚本。

尚未完成且阻止 V0.1 关闭的部分：

- 多 PictureItem + 顶层说明 + 顶层 Caption 的 Figure 复合区域隔离；
- 视觉正文验收器的子串假阳性；
- 5006A 第 36–48 页真实数字矩阵表格的准入；
- 有代表性的扫描正文 OCR 准确率小样本验收。

本次交接没有继续修改上述生产代码，只记录当前实现、证据和已经确认的缺口。接替对话应从现有回传产物直接做离线诊断，先闭环两个 P0，再发起下一次服务器 Golden 验收。
