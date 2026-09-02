# Manufacturing RAG

面向制造业公开技术文档的检索增强生成（RAG）实验项目。

当前发布版本为 `ver-0.5`：A0 预处理仍以冻结的 Docling/RapidOCR 主链运行；A1 Markdown
Loading、A2 Document Registry、A3 Structure-aware Leaf chunking、A4 Section hierarchy、
A5 Dense Embedding 已实现，并在正式 NASA 语料上通过 Linux 验收。B1 Query Router 与
B2 Query Processing 的 v1 需求已冻结，实现尚未开始。仓库内包含 MinerU 3.4.5
源码快照，供后续将 A0 重构为全量 OCR 主链使用；本版 A0 尚未切换。A6 Milvus 及之后模块
尚未实现。它不是已完成的生产系统。

仓库起点仍是 `ver-0`：项目方向定义、首批公开工程文档收集和原始 All-in-RAG 参考代码整理。

## ver-0 范围

- 建立制造业公开技术文档 RAG 的 V0.1 设计规范。
- 收录 24 份来自 NASA、NIST 官方来源的材料、加工、连接制造、制造规范与质量检测文档。
- 保留原始 All-in-RAG 的 C1-C9 示例代码，供后续审计和重构时对照。
- 内置 Docling 源码快照，供当前 A0 解析方案对照。
- 内置 MinerU 3.4.5 源码快照，供后续 A0 全量 OCR 迁移研究。

当前 C1-C9 中仍存在教程、菜谱和 Graph RAG 示例逻辑；部分脚本依赖已从本仓库移除的
原教程数据，因此不保证能够直接运行。这些文件是重构参考，不代表最终业务实现。

## 目录结构

```text
.
├── code/
│   ├── C1-C9/                    # All-in-RAG 历史/教程参考代码
│   ├── knowledge_base/           # A1–A5：Loading / Registry / Leaf / Hierarchy / Embedding
│   └── preprocessing/            # A0 正式实现（本版仍为 Docling 主链）
├── data/engineering_docs/        # 制造业公开文档及可校验清单
├── docling/                      # Docling 2.115.0 源码快照
├── mineru/                       # MinerU 3.4.5 源码快照（下一版 A0 OCR 迁移）
├── docs/                         # 稳定需求、模块文档、验收与 proceeding
└── models/                       # 模型目录占位符，不提交模型权重
```

## 数据集

数据清单位于
[`data/engineering_docs/manifest.csv`](data/engineering_docs/manifest.csv)，记录每份文档的
官方来源、相对路径、文件大小和 SHA-256。当前共 24 份 PDF，约 75 MiB。

这些文档来自公开的 NASA 和 NIST 官方站点。仓库中的副本仅用于研究与可复现实验；
各文档的权利和使用条件仍以其发布机构及原始页面为准。

## 开发状态

详细目标、数据方案、Metadata Schema、分块策略、检索链路和评测规划见
[`docs/manufacturing-rag-v0.1-spec.md`](docs/manufacturing-rag-v0.1-spec.md)。

`ver-0.5` 之后的下一阶段是把 A0 从 Docling/RapidOCR 主链重构为 MinerU 全量 OCR
（数字 PDF 与扫描 PDF 一律走 OCR），接入评估与实现需求见
[`docs/Preprocessing/mineru-a0-integration.md`](docs/Preprocessing/mineru-a0-integration.md)
与 [`docs/Preprocessing/mineru-a0-migration-implementation-requirements-v1.md`](docs/Preprocessing/mineru-a0-migration-implementation-requirements-v1.md)。
在该迁移完成并冻结前，不得把 MinerU 当作当前生产 A0。

## PDF 预处理入口

`code/preprocessing/pdf_preprocess.py` 将整页方向标准化置于 Docling 之前，并生成正式索引 Markdown 与完整审计产物。正式 `document.md` 只包含通过页面准入的正文，以及显式关联且可信的表注/图注；表格、图片及其内部 OCR 保留在 `document.json`，不会进入后续分块。

运行前确认 `models/PageOrientation/PP-LCNet_x1_0_doc_ori/` 中的真实 `model.onnx` 与 `manifest.json` 一致；当前验收基线的 SHA-256 为 `af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92`。仓库只跟踪模型配置、来源、大小和 SHA-256 清单，不提交模型权重；请按 [`models/preprocessing-models.manifest.json`](models/preprocessing-models.manifest.json) 中固定的上游地址在本地配置资产。例如：

```powershell
python code/preprocessing/pdf_preprocess.py `
  data/engineering_docs/raw/material/01_nasa_materials_aluminum_2014.pdf `
  --device cuda `
  --page-range 38 38 `
  --output-root outputs/preprocessing-smoke `
  --overwrite
```

`--page-range` 同时约束方向分类和 Docling，页码仍对应原 PDF。CUDA 模式下 PP-LCNet 与 RapidOCR Det/Cls/Rec 均采用 CUDA-preferred mixed execution：`CUDAExecutionProvider` 必须排在首位且实际激活，ONNX Runtime 可将不支持的少量节点交由 CPU；CUDA 不可用或会话整体退化为 CPU 时会明确失败。RapidOCR 会通过其可序列化配置入口自行建立内部会话，随后输出三个阶段的真实 Provider 列表。只有显式 `--device cpu` 才会创建 CPU-only 会话。冻结后的语义投影、产物字段与验收口径见 [`docs/preprocessing.md`](docs/preprocessing.md)，服务器验收命令见 [`docs/Preprocessing/pdf-preprocessing-server-acceptance.md`](docs/Preprocessing/pdf-preprocessing-server-acceptance.md)。

数字版表格使用项目本地、固定 `v2.3.0` 的 TableFormer V1 `accurate` 与 cell matching。原生文字必须一对一追溯并按 source cell 引用守恒；rowspan/colspan 及显式表头标记在 Canonical JSON 中保留，Markdown 则以左上锚点保留文字、覆盖格置空。标题和多级分组表头按原顺序位于 pipe table 前，表体分组行保持原行序；没有可信显式表头时使用空 Markdown 表头，不把首行数据臆造为表头。Accepted 表格通过 `TableItem` 原位置占位符一对一替换，可信关联脚注紧随表格保留；多页 provenance 的单个 TableItem 在 V0.1 统一降级为一个 `deferred` 审计记录，绝不复制 cells。OCR、混合、无文字及不可信结构表仍仅保留题注和审计信息。`models/docling-project--docling-models/model_artifacts/tableformer/accurate/` 的 `tm_config.json` 和 `tableformer_accurate.safetensors` 均按大小及 SHA-256 完整性清单校验，缺失或不一致时程序 fail-fast，禁止运行时下载。

服务器首次同步后优先执行不依赖 pytest 的真实运行面验收：

```bash
CUDA_VISIBLE_DEVICES=0 python code/preprocessing/verify_pdf_preprocess_server.py \
  --input-pdf data/engineering_docs/raw/material/01_nasa_materials_aluminum_2014.pdf \
  --page 38 --device cuda --num-threads 8 \
  --document-timeout 7200 \
  --output-root outputs/server-acceptance
```

该命令会校验本地模型 SHA-256、构造真实 RapidOCR 3.9.2 Det/Cls/Rec 会话、执行真实页面 OCR，并跑同一页端到端流程。正式输出采用事务目录：仅完整成功且至少有一页可入库时替换稳定目录；`partial_success` 或固定质量门禁拒绝的审计产物保存在 `.failed/`，不会覆盖上一次成功产物。

## A5 Dense Embedding 入口

`code/knowledge_base/dense_embedding.py` 对 A3 Leaf 的 `content` 做 document-side 编码：Qwen3-Embedding-4B、left padding、last-token pooling、L2 normalize、2560-d float32。不添加 document instruction，不实现 query-side 与 A6。超过 8192 inference token 必须失败，禁止 silent truncation。验收口径见 [`docs/proceeding/2026-08-18-a5-dense-embedding.md`](docs/proceeding/2026-08-18-a5-dense-embedding.md)。

```bash
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
python -m knowledge_base.dense_embedding \
  --canonical-root "$PWD/outputs/preprocessing" \
  --model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 4 \
  --max-input-tokens 8192 \
  --output /tmp/a5-embeddings.npz \
  --report /tmp/a5-embedding-report.json
```

## 环境提示

- 建议使用 Python 3.10-3.12。
- 通用参考依赖位于 `code/requirements.txt`。
- C8、C9 分别提供独立的 `requirements.txt`。
- C9 的环境变量模板位于 `code/C9/.env.example`；请勿提交真实 API Key 或密码。
- Milvus 的本地容器配置位于 `code/docker-compose.yml`。

由于 `ver-0` 保留了不同阶段的参考实现，目前不建议一次性安装并运行所有模块。
应根据正在重构的模块选择对应依赖。

## 来源与许可证

本项目基于 Datawhale 的
[all-in-rag](https://github.com/datawhalechina/all-in-rag) 修改，原项目采用
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)。
本仓库的衍生内容继续遵循该许可；第三方组件按其各自许可证使用。

完整说明见 [`LICENSE.md`](LICENSE.md) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
