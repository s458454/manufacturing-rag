# MinerU A0 迁移实现需求 v1

状态：`[APPROVED FOR IMPLEMENTATION]`

日期：2026-09-01

适用范围：A0 MinerU 主链、A0 正式产物、A3/A4 最小兼容扩展。A1、A2、A5、A6 的正文消费合同原则上保持不变。

上游固定版本：

```text
repository: https://github.com/opendatalab/MinerU
tag: mineru-3.4.5-released
git ref: fbb1257a555a3fde78ae5aaaa931e3b3f8fb2883
vendored path: mineru/
```

注意：该标签的 `mineru/version.py` 仍报告 `3.4.4`。实现和报告必须同时记录 Git ref 与运行时版本，不能只比较版本字符串。

---

## 1. 目标

将 A0 的 PDF 解析主后端从 Docling/RapidOCR/TableFormer 迁移为 MinerU，并满足以下行为：

1. 数字 PDF 和扫描 PDF 全部走 OCR；
2. 正文、标题、列表、公式和所有可识别文字进入正式切片链；
3. 表格与图片/图表 OCR 文字分别形成独立原子块，不与相邻正文混装；
4. 页码作为结构化 provenance 保留，但页码标记不得进入 Leaf 正文、Embedding 或 BM25 文本；
5. A1-A6 不依赖 MinerU 私有 JSON Schema；
6. MinerU 原始产物完整保留用于审计，但不得被重复当作知识库正文；
7. 空白页不产 chunk、不导致整文档失败；非空页静默无输出必须 fail-closed；
8. 迁移期间不得静默回退到 Docling，避免同一语料混用两个解析口径。

---

## 2. 冻结配置

第一版采用 MinerU 默认主后端能力，除“强制全页面 OCR”外不调整内部模型阈值：

```text
backend = hybrid-engine
parse_method = ocr
effort = medium
lang = ch
formula_enable = true
table_enable = true
MINERU_TABLE_MERGE_ENABLE = false
image_analysis = false
figure_region_ocr = true
figure_region_ocr_engine = mineru_pipeline_ocr
return_md = true
return_middle_json = true
return_content_list = true
return_images = true
response_format = zip
```

约束：

- `parse_method=ocr` 是强制项，不得使用 MinerU 默认 `auto`；
- `image_analysis=false` 是强制项，本阶段不使用 VLM 补写图片语义；
- `figure_region_ocr=true` 是强制项，它与 `image_analysis` 是两项不同能力：前者只识别图片/图表裁剪中的可见文字，后者会让 VLM 分析图片/图表内容；
- MinerU 内部 OCR、检测、布局和表格阈值保持上游默认；
- 固定 `MINERU_TABLE_MERGE_ENABLE=false`：当前 MinerU 跨页合并会把合并结果锚定在前页的 `content_list.page_idx`，不能稳定表达 `page_end`；第一版保持原 A0 的“按物理页独立表格”语义；
- `pipeline + method=ocr` 必须保留为验收对照配置，但不得自动成为线上 fallback；
- 中英混合第一版统一使用 `lang=ch`，是否调整由 Golden 结果决定；
- 现有独立整页方向规范化阶段暂时保留，MinerU 接收规范化后的 PDF。

### 2.1 已确认的 MinerU 能力边界

不得假设上述主请求会自动返回图内 OCR 文字。当前固定版本中：

- Hybrid `effort=medium` 会把有效 `image_analysis` 强制设为 `false`；
- `image_analysis=false` 时，图片/图表保留版面位置，但跳过 VLM 图片/图表分析；
- OCR sidecar 只针对正文/标题类块，且该步骤使用 `rec=False`，只提供检测框，不识别图片/图表内部文字；
- 因此，主任务产物中的 image/chart `content` 允许为空，不能据此判定图片中没有文字。

第一版必须增加显式的 `figure-region OCR` 子阶段，不能把它留作后续优化，也不能通过打开 `image_analysis` 替代。

---

## 3. 进程边界

A0 不得直接 import `mineru.backend.*` 内部模块。正式链路通过本地 HTTP 服务运行：

```text
source PDF
  -> existing page-orientation normalization
  -> A0 MinerUAdapter
  -> mineru-api async /tasks
  -> result ZIP
  -> ZIP integrity/security validation
  -> raw MinerU artifact preservation
  -> figure/chart body crop OCR through project service shim
  -> project-owned semantic projection
  -> project quality checks
  -> OutputTransaction publish
  -> A1
```

要求：

- 开发烟测可使用同步接口；批量正式运行使用异步 `/tasks`；
- MinerU 使用独立 Python/容器环境，不能与现有 A0 环境混装重依赖；
- 服务地址、任务超时、轮询间隔和重试次数必须可配置；
- 只允许对传输/服务瞬时错误重试；解析结果质量失败不得通过无条件重试掩盖；
- ZIP 解压必须拒绝绝对路径、`..` 路径穿越、符号链接和超出配置上限的文件数量/总大小。

### 3.1 `figure-region OCR` 服务边界

MinerU 当前公开任务接口没有独立的图片区域 OCR API。下游必须在 **MinerU 独立运行环境内** 增加一个项目自有的批量 HTTP shim；A0 仍只调用 HTTP，不得加载 MinerU 模型或内部模块。

服务要求：

- 输入为 MinerU 已裁出的 image/chart body 图片及包含 `block_id/page/bbox` 的 manifest；不得把 caption/footnote 一并裁入；
- 只处理 `image|chart`，table 继续走 MinerU table 结果，不做重复区域 OCR；
- 服务端复用该固定 Git ref 中的 MinerU/Paddle OCR 检测与识别模型，并记录实际模型、设备和版本；
- 使用 MinerU 上游默认 OCR 接受阈值，第一版不增加项目自定义置信度阈值；
- 输出必须区分 `success_with_text`、`success_empty` 和 `error`；“成功但无文字”不是失败，调用异常或缺少某个请求 region 的结果才是失败；
- 每条 OCR line 至少返回 `text/score/bbox/order`，同时返回按阅读顺序拼接的规范化 `ocr_text`；
- 请求和响应都必须用 `block_id` 一一对应，重复、缺失或未知 ID 必须 fail-closed；
- 支持批处理，不允许正式链路对每一个 region 单独启动模型；
- A0 超时、重试、健康检查和设备核验规则与 MinerU 主服务一致；
- 允许服务 shim 在固定 MinerU 环境内封装 `mineru.backend.*`，但该耦合必须隔离在一个文件并由 Git ref 锁定测试覆盖。

主链集成前必须用至少一张“有图内文字”和一张“纯图片无文字”的样例完成真实模型 smoke test。未通过时不得把 figure OCR 标记为已实现。

---

## 4. 正式产物合同

每个文档仍发布到：

```text
<canonical_root>/<document_id>/
```

第一版必须包含：

```text
document.md
document.json
regions.json
quality_report.json
tables/index.json
orientation_report.json
normalized/oriented.pdf
raw/mineru/markdown.md
raw/mineru/middle.json
raw/mineru/content_list.json
raw/mineru/images/**
raw/project/figure_region_ocr.json
```

规则：

- `document.md` 是 A1 的唯一正式正文输入；
- `raw/mineru/markdown.md` 只用于人工 diff 和审计；
- `middle.json` 是详细审计证据，不作为第二份正文入库；
- 不得把 MinerU `middle.json` 直接改名为项目 `document.json`；
- 不得伪造 Docling 的 `self_ref/body/texts/pictures/tables` Schema；
- `document.json` 使用项目自有、带 `schema_version` 的稳定规范化 Schema；
- 所有产物继续通过 staging + atomic publish 事务发布。

### 4.1 `document.json` 最低字段

```text
schema_version
document_id
source
source_sha256
backend.name
backend.git_ref
backend.runtime_version
backend.parameters
pages[]
blocks[]
```

`pages[]` 最低字段：

```text
page_no                 # 1-based
width
height
page_status             # content | blank | visual_only | warning | error
eligible_for_indexing
quality_reasons[]
```

`blocks[]` 最低字段：

```text
block_id
content_kind            # text | title | list | table | figure | equation | code
source_order
page_start
page_end
bbox
section_path
text
caption[]
footnote[]
asset_path
table_html
quality
raw_mineru_ref
```

`block_id` 必须由稳定输入确定性生成，禁止使用随机 UUID。

figure block 的 `quality` 还必须记录 `region_ocr_status`、OCR engine/model/device、line count 和 score 统计；`raw_mineru_ref` 与 `raw/project/figure_region_ocr.json` 的引用必须都可追溯。

### 4.2 `quality_report.json` 兼容字段

以下字段必须保持当前 A2 兼容：

```text
document_id
source
source_sha256
```

同时新增：

```text
backend = mineru
mineru_git_ref
mineru_runtime_version
mineru_parameters
source_page_count
output_page_count
page_status_summary
block_count_by_kind
soft_warning_summary
raw_artifact_checksums
```

删除或停止生成 Docling/RapidOCR/TableFormer 专属的运行时质量字段，但不得影响上述 identity 字段。

### 4.3 `regions.json`

由 `document.json.blocks` 中的 `table/figure` 派生，最低字段：

```text
region_id
content_kind            # table | figure
page_start
page_end
bbox
caption
ocr_text
asset_path
source_order
quality
```

不得继续表达“所有 visual body 永远不得进入正式正文”。新语义是：visual OCR 文字不得混入普通正文，必须路由到独立 visual chunk。

如果 MinerU 同时把位于 image/chart body 内的文字识别为普通 text block，投影层按以下确定性规则去重：同页 text block 的中心点落在 visual body bbox 内时，该 text block 不进入普通正文投影，只保留在 raw 审计；figure 的图内文字只采用 `figure-region OCR` 结果。该决策及被隔离的 block ID 必须写入 figure `quality`，不得按字符串模糊匹配静默删除。

### 4.4 `tables/`

第一版保留兼容目录，但取消 native/OCR 双轨策略：

```text
source_kind = mineru_ocr
extractor = mineru
```

每张表保存 caption、footnote、HTML、纯文本回退、页码、bbox 和质量告警。`tables/index.json` 只保存摘要和单表 artifact 路径，不内嵌全部单元格。

---

## 5. 页码链路

页码标记由 A0 MinerUAdapter 自动生成，不允许人工编辑：

```text
content_list[].page_idx (0-based)
  -> <!-- PDF page N --> (1-based)
  -> document.md
  -> A1 lossless load
  -> A3 page range derivation（原子块同时读取自身 page_start/page_end）
  -> Leaf.page_start/page_end
```

规则：

- 每个有正式语义 block 的页面在 `document.md` 中恰好出现一个页码标记；
- 页码标记必须位于该页第一个正式 block 之前，不得插入 table HTML、pipe table、公式或其他原子块内部；
- 页码必须严格按源 PDF 递增；
- 页码标记不得进入 `Leaf.content`、Embedding、BM25、Section recovery 文本；
- LLM Evidence 使用结构化 `Page:` 字段，不把页码注释拼进正文；
- blank 页面可以不写入 `document.md`，但必须出现在 `document.json.pages` 和 `quality_report.json`；
- visual-only 页面即使不产文本 Leaf，也必须在审计 JSON 中保留原始页号和 region。

---

## 6. 正文与 Markdown 投影

正式 `document.md` 必须从 `content_list.json`/规范化 block 投影，不得直接复制 MinerU Markdown。

映射：

| MinerU 内容 | 项目投影 |
|---|---|
| `type=text, text_level>0` | 对应级别 Markdown heading |
| 普通 `text` | 正文段落 |
| `list` | Markdown list |
| `equation` | 保留 LaTeX，公式类型单独处理 |
| `table` | 独立 table block |
| `image/chart` | 独立 figure block |
| header/footer/page_number | 默认不进入正式正文，保留审计 |

这里的“所有可识别文字”不包括结构噪声：`page_number` 只进入页码 provenance，重复 header/footer 只进入 raw 审计；正文脚注 `page_footnote` 仍进入正文，已关联到 table/figure 的 footnote 只进入对应原子块。不得仅因 OCR score 较低删除其他正文文字。

MinerU 已转义的内容不得再次盲目转义。适配器必须对以下情况建立测试：

```text
正文以 #、>、+、-、*、1. 开头
正文含 * _ ` ~ $ [ ] | \
标题正文自身含 Markdown 特殊字符
代码、公式、HTML table 不得使用普通正文转义器
```

---

## 7. 原子 table/figure block

为了保持 A1 继续只读 `document.md`，A0 在正式 Markdown 中加入项目保留的 block 边界标记：

```text
<!-- A0 block-start kind=table block_id=<stable-id> page_start=N page_end=N -->
...
<!-- A0 block-end block_id=<stable-id> -->
```

或：

```text
<!-- A0 block-start kind=figure block_id=<stable-id> page_start=N page_end=N -->
...
<!-- A0 block-end block_id=<stable-id> -->
```

约束：

- 标记必须独占一行；
- 只允许 `table|figure`；
- `page_start/page_end` 为 1-based 闭区间且必须与 `document.json` 对应 block 一致；
- start/end 必须配对且 ID 相同；
- 禁止嵌套、交叉和无结束标记；
- 第一版禁止一个原子 block 跨物理页；跨页表格已关闭自动合并，每页分别生成 table block；
- A3 对普通正文从 PDF page marker 推导页码，对 table/figure 优先采用 block-start 的明确页码并交叉校验前置 PDF page marker；
- A3/A4 必须识别并剥离这些保留标记；
- 保留标记不得进入 Leaf、Section recovery、Embedding、BM25 或 LLM Evidence 正文；
- 普通用户文档中的其他 HTML comment 必须继续原样保留。

### 7.1 表格

每个物理页内的表格 region 无论大小都生成独立 table Leaf：

```text
caption
table body
footnote
```

规则：

- 数字 PDF 表格与扫描表格一视同仁；
- 第一版不自动合并跨页表格；相邻页的续表分别成 Leaf，`continuation_group_id=null`；
- 优先保留 MinerU HTML；无 rowspan/colspan 且能无损转换时允许使用 GFM pipe table；
- 如果表格结构缺失但 OCR 文本存在，仍生成 table Leaf，并记录 `structure_missing` warning；
- caption/footnote 只进入对应 table Leaf 一次，不得同时复制到普通正文；
- 表格不参与 overlap；
- 768 A3 tokens 是普通 Leaf 软上限，表格超过 768 仍保持单一 oversize table Leaf；
- A5 单输入硬上限为 8192 tokens。表格超过 8192 时第一版 fail-closed，禁止静默截断、摘要或拆分；在真实语料出现后另立设计处理超巨型表格。

### 7.2 图片/图表

每个有文字语义的图片或图表生成独立 figure Leaf：

```text
caption
OCR text inside figure/chart
footnote
```

规则：

- 不使用 `image_analysis` 生成图片语义；
- 图内文字必须来自第 3.1 节的 `figure-region OCR`，不得把缺失的 image/chart `content` 当作无文字证据；
- caption、图内 OCR 和 footnote 只属于 figure Leaf，不复制到普通正文；
- `success_empty` 且没有 caption/footnote 时不生成空 Leaf，只保留 region/asset 审计；
- region OCR 返回 `error`、缺少请求 region、ID 不一致或未执行时整文档 fail-closed，禁止降级为“无文字图片”；
- figure Leaf 不与相邻正文打包；
- 图内 OCR 较长时第一版仍以完整 figure 为原子块；超过 A5 8192-token 硬上限时 fail-closed；
- 原文顺序必须保持为“前正文 -> figure -> 后正文”。

---

## 8. A3/A4 最小扩展

第一版不得修改现有 `Leaf` 七字段 Schema：

```text
chunk_id
document_id
section_id
chunk_index
page_start
page_end
content
```

A3 新增平行的 `LeafMetadata`，以 `chunk_id` 关联：

```text
content_kind
source_block_id
source_order_start
source_order_end
previous_chunk_id
next_chunk_id
```

要求：

- table/figure block 各自生成独立 Leaf；
- table/figure 的 `section_path` 继承同一 source order 中最近的前置 heading；若此前没有 heading，则归入 document root；不得读取 caption 文本猜章节；
- 普通正文继续使用 768/96 规则；
- `chunk_index` 仍按 document-local 连续编号；
- neighbor fallback 仍可使用 `document_id + chunk_index`；
- 新增 explicit previous/next 关系，不能把 `chunk_index` 作为唯一结构依据；
- A4 Section recovery 必须保留 table/figure 正文，但剥离 A0 block marker 和 PDF page marker；
- A5 继续只嵌入 `Leaf.content`；
- A6 analyzer 继续只分析 `Leaf.content`；
- A5/A6 不得因为新增平行 metadata 改变向量或分词结果。

---

## 9. 空白页与质量策略

MinerU 内部阈值第一版保持默认。项目层只冻结结构硬失败和软告警，不复制旧 Docling/RapidOCR 的 0.75/0.90 页面阈值。

页面必须分类为：

| 状态 | 条件 | 正文行为 | 发布行为 |
|---|---|---|---|
| `blank` | 独立渲染证据确认近空白 | 不产 chunk | 正常发布 |
| `visual_only` | 无文字，但存在 table/figure/chart region | 有语义文字才产 visual chunk | 正常发布 |
| `content` | 存在有效 block | 正常切片 | 正常发布 |
| `warning` | 有输出但质量统计异常 | 保留并标记 | 正常发布、进入审计 |
| `error` | 页面非空且无任何 text/visual block，或结构损坏 | 不入库 | fail-closed |

空白判定继续使用独立页面渲染证据；不得仅以“MinerU 返回空列表”判断 blank。现有默认可作为第一版起点：

```text
blank_gray_threshold = 245
blank_ink_ratio_threshold = 0.0005
```

硬失败至少包括：

- MinerU 任务失败或结果 ZIP 不完整；
- 任一 image/chart region 未完成 `figure-region OCR`，或 OCR 响应 ID/状态非法；
- 源页数与规范化页面记录数不一致；
- 非空页无任何可定位 block；
- page/bbox/source_order 非法；
- block ID 重复或 block marker 不配对；
- 正式 Markdown/JSON 无法解析；
- table/figure Leaf 超过 A5 8192-token 硬上限；
- OutputTransaction 缺少必需产物。

软告警至少记录：

- OCR score 分布偏低；
- 单字符、乱码、异常标点比例；
- 标题层级跳变；
- table 缺少结构但存在 OCR 文本；
- caption 缺失；
- bbox 缺失但仍有可用文本；
- 同页 block 阅读顺序异常。

软告警阈值在 Golden 评测前不得用于整页自动删除。

---

## 10. Docling 迁移策略

- MinerU 是新主链；
- Docling/RapidOCR 代码第一版不立即删除，保留显式 legacy CLI 入口用于回归对照；
- 线上或批量正式运行不得在 MinerU 失败后静默自动回退 Docling；
- `quality_report.json` 必须明确记录实际 backend；
- 同一 canonical root 的一次构建不得混合 MinerU 与 Docling 产物；
- MinerU Golden 验收通过后，另行提交删除 Docling 运行依赖和旧模型资产的变更。

---

## 11. 实现交付物

下游至少交付：

```text
code/preprocessing/mineru_adapter.py
code/preprocessing/mineru_projection.py
code/preprocessing/mineru_quality.py
code/preprocessing/mineru_region_ocr_service.py
code/preprocessing/tests/test_mineru_adapter.py
code/preprocessing/tests/test_mineru_projection.py
code/preprocessing/tests/test_mineru_quality.py
code/preprocessing/tests/test_mineru_region_ocr_service.py
```

并修改：

```text
code/preprocessing/pdf_preprocess.py 或新增统一 backend dispatcher
code/preprocessing/verify_pdf_preprocess_server.py
code/knowledge_base/structure_parser.py
code/knowledge_base/leaf_chunker.py
code/knowledge_base/section_hierarchy.py
对应 A1-A6 regression tests
docs/preprocessing.md
docs/knowledge-base.md
```

同时提供：

- MinerU 独立环境/容器启动说明；
- 模型文件 manifest：来源、大小、SHA-256；
- 固定服务配置；
- 一条单文档 smoke 命令；
- 一条批量异步运行命令；
- 产物目录样例；
- Golden 对照报告。

---

## 12. 必须通过的测试

### 12.1 单元测试

1. `page_idx=0` 生成 `<!-- PDF page 1 -->`；
2. 页码标记只出现一次且严格递增；
3. 页码/A0 block marker 不进入 Leaf.content；
4. 普通 HTML comment 不被误删；
5. table/figure marker 缺失、嵌套、交叉、页码属性非法或与前置页码不一致时 fail-closed；
6. 小表格独立成 Leaf；
7. 769-token 表格保持单一 oversize Leaf；
8. 普通 769-token 正文按 768/96 切分；
9. table/figure 不参与 overlap；
10. HTML table 被识别为原子 table；
11. caption/footnote 不重复；
12. figure OCR 不进入前后正文 Leaf；
13. `image_analysis=false` 且主产物 image content 为空时仍调用 region OCR；
14. region OCR 的 `success_empty` 不生成空 Leaf；
15. region OCR 的 `error`、漏 ID、重复 ID 和未知 ID 均 fail-closed；
16. 图内 text block 按 bbox 中心点规则从普通正文隔离，且审计记录完整；
17. caption/footnote 不得被 region OCR 重复裁入；
18. `MINERU_TABLE_MERGE_ENABLE=false`，相邻页续表保持两个独立 Leaf 和各自页码；
19. blank page 无 chunk 且文档成功；
20. 非空页无 block 时 fail-closed；
21. Markdown 特殊字符不制造错误 heading/list/table；
22. A2 仍只依赖 quality identity 字段；
23. A5 dense vectors 对相同 Leaf.content 保持确定性；
24. A6 analyzer 只消费 Leaf.content；
25. OutputTransaction 不发布半成品。

### 12.2 集成验收

同一批中英文、数字/扫描混合 PDF 至少运行：

```text
hybrid-engine + method=ocr + effort=medium
pipeline      + method=ocr
```

记录并比较：

- 中英文 CER/WER；
- 标题识别与层级准确率；
- 多栏阅读顺序；
- 表格结构、空单元格、rowspan/colspan、跨页表格；
- 图片 OCR 与图注归属；
- 有字示意图/流程图/截图的 region OCR 召回，以及纯照片的 `success_empty` 误报；
- 页码 provenance 完整率；
- blank page 误识别/幻觉率；
- 单页和整文档耗时；
- 显存峰值；
- 失败率；
- 同输入重复运行的一致性；
- A1-A6 全量回归。

### 12.3 验收红线

以下任一发生即不得切换主链：

- 页码错位或页数丢失；
- page/A0 block marker 泄漏到 Leaf.content；
- 图内 OCR 混入普通正文；
- 任一 figure region 未执行 OCR 却以“无文字”正常发布；
- 表格被普通滑窗切断；
- MinerU raw JSON 被重复入库；
- 非空页面静默消失；
- backend/模型版本无法复现；
- 失败后静默回退 Docling；
- A1-A6 现有不相关行为发生回归。

---

## 13. 实施顺序

1. 部署并固定 MinerU 服务、模型和 `/health`；
2. 实现 HTTP Adapter、任务轮询、ZIP 安全校验和 raw artifact 保存；
3. 实现并真实烟测批量 `figure-region OCR` 服务 shim；
4. 实现 `content_list + figure OCR -> document.json/document.md/regions/tables/quality` 投影；
5. 更新 A0 事务和服务器验收，不再要求 Docling JSON Schema；
6. 实现 A3/A4 table/figure block marker 与平行 LeafMetadata；
7. 运行 A1-A6 回归；
8. 运行 hybrid/pipeline Golden 对照；
9. 审计失败样本并只针对真实 failure mode 调参；
10. 通过红线后将 MinerU 切为默认 A0 backend；
11. 另行决定是否删除 Docling legacy 链。

---

## 14. 文档生效与重新冻结

本文件是交给下游实现的迁移增量合同，不表示当前代码已经切换完成。实现和 Golden 验收通过前，旧 A0 冻结文档仍描述当前生产事实。

切换主链的同一个变更必须同步更新：

```text
docs/proceeding/2026-08-15-a0-freeze.md
docs/preprocessing.md
docs/knowledge-base.md
docs/architecture.md
docs/manufacturing-rag-v0.1-spec.md
```

并新增一次带日期的 A0 MinerU 重新冻结记录。不得只改代码而让主规范继续声称 Docling 是默认后端。
