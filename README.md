# Manufacturing RAG

面向制造业公开技术文档的检索增强生成（RAG）实验项目。

当前发布版本为 `ver-0.6`。

当前状态：

```text
A0  PDF Preprocessing                 已实现，生产主链仍为 Docling/RapidOCR
A1  Markdown Loading                  已实现并验收
A2  Document Registry                 已实现并验收
A3  Structure-aware Leaf Chunking     已实现并验收
A4  Section Hierarchy                 已实现并验收
A5  Dense Embedding                   已实现并验收
A6  Milvus Production Collection      已实现并完成真实语料生产验收

B1  Query Router                      v1 requirements FROZEN，尚未实现
B2  Query Processing                  v1 requirements FROZEN，尚未实现
B3  Hybrid Retrieval                  v1 requirements FROZEN，尚未实现
B4-B6                                 高层设计继续推进，尚未实现
```

A6 当前真实生产验收基线为：17 篇可检索文档、2034 个 Leaf，Milvus 2.5.14 单一 Leaf Collection，Dense 使用 `FLAT + IP`，Sparse 使用 Native BM25；真实 `verify`、Dense smoke、BM25 smoke 均已通过。A6 的详细实现与验收记录见 `docs/proceeding/`。

本项目仍处于分模块重构与验证阶段，不代表整条线上 RAG 已经完成。

仓库起点仍是 `ver-0`：项目方向定义、首批公开工程文档收集和原始 All-in-RAG 参考代码整理。

## 文档层级

```text
docs/*.md
→ 高层设计、模块边界和整体方案

docs/Preprocessing/*.md
→ 各模块冻结后的具体设计合同 / 实现要求

docs/proceeding/*.md
→ 实现过程、测试、真实运行与验收记录
```

发生细节冲突时，以最新模块 Frozen Requirements 与对应 proceeding 验收事实为准；根目录高层设计和 README 负责保持当前状态与总体方向一致。

## 目录结构

```text
.
├── code/
│   ├── C1-C9/                    # All-in-RAG 历史/教程参考代码
│   ├── knowledge_base/           # A1-A6：Loading / Registry / Leaf / Hierarchy / Embedding / Milvus
│   └── preprocessing/            # A0 正式实现（当前仍为 Docling/RapidOCR 主链）
├── data/
│   ├── EN/                       # 英文原始 PDF
│   ├── CHN/                      # 中文原始 PDF
│   └── engineering_docs/         # manifest 等语料清单
├── docling/                      # Docling 2.115.0 源码快照
├── mineru/                       # MinerU 3.4.5 源码快照，供下一阶段 A0 迁移研究
├── docs/                         # 高层设计、Frozen Requirements 与 proceeding 验收记录
└── models/                       # 模型目录占位符，不提交大模型权重
```

## 数据集

数据清单位于 [`data/engineering_docs/manifest.csv`](data/engineering_docs/manifest.csv)。当前 manifest 注册 27 份 PDF 候选：24 份 NASA/NIST 英文工程文档，以及 3 份中文电力设备技术文档。

当前 `ver-0.6` 真实生产 Collection 使用其中经过当前 canonical pipeline 处理并治理为可检索状态的 17 篇文档（14 EN + 3 CHN），产生 2034 个 Leaf。

这些文档仅用于研究与可复现实验；各文档的权利和使用条件仍以其发布机构及原始来源为准。

## 开发状态与设计入口

高层设计：

- [`docs/knowledge-base.md`](docs/knowledge-base.md)：A1-A6 知识库链路。
- [`docs/retrieval.md`](docs/retrieval.md)：B1-B6 在线检索链路。
- [`docs/manufacturing-rag-v0.1-spec.md`](docs/manufacturing-rag-v0.1-spec.md)：项目早期总体规格与规划。

当前已冻结的 B 系列具体合同位于：

- [`docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`](docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md)
- [`docs/Preprocessing/2026_09_02_B2_Query_Processing_Frozen_Requirements_v1.md`](docs/Preprocessing/2026_09_02_B2_Query_Processing_Frozen_Requirements_v1.md)
- [`docs/Preprocessing/2026-09-10-B3_Hybrid_Retrieval_Frozen_Requirements_v1.md`](docs/Preprocessing/2026-09-10-B3_Hybrid_Retrieval_Frozen_Requirements_v1.md)

A6 真实生产实现与验收记录：

- [`docs/proceeding/2026-09-08-a6.0-analyzer-profile.md`](docs/proceeding/2026-09-08-a6.0-analyzer-profile.md)
- [`docs/proceeding/2026-09-09-a6-production-collection.md`](docs/proceeding/2026-09-09-a6-production-collection.md)
- [`docs/proceeding/2026-09-10-a6-production-real-ingestion.md`](docs/proceeding/2026-09-10-a6-production-real-ingestion.md)

## A0 当前状态

当前生产 A0 仍使用冻结的 Docling/RapidOCR 主链。仓库包含 MinerU 3.4.5 源码快照，但在迁移完成并冻结前，不得把 MinerU 当作当前生产 A0。

MinerU 迁移设计与实现要求见：

- [`docs/Preprocessing/mineru-a0-integration.md`](docs/Preprocessing/mineru-a0-integration.md)
- [`docs/Preprocessing/mineru-a0-migration-implementation-requirements-v1.md`](docs/Preprocessing/mineru-a0-migration-implementation-requirements-v1.md)

## A5 Dense Embedding

`code/knowledge_base/dense_embedding.py` 对 A3 Leaf 的 `content` 做 document-side 编码：

```text
Qwen3-Embedding-4B
left padding
last-token pooling
L2 normalize
2560-d float32
```

不添加 document instruction；超过 8192 inference tokens 必须失败，禁止 silent truncation。

## A6 Production Collection

正式实现位于：

```text
code/knowledge_base/a6_document_metadata.py
code/knowledge_base/a6_collection.py
code/knowledge_base/a6_prepare_rows.py
```

当前 baseline：

```text
Milvus 2.5.14
Leaf-only Collection
dense_vector: FLAT + IP
sparse_vector: Native BM25 + SPARSE_INVERTED_INDEX
Analyzer: UNIFIED_ICU
BM25: k1=1.2, b=0.75, DAAT_MAXSCORE
```

Collection 只保留 `ACTIVE` 与 `VERSION_CONFLICT` 文档进入默认检索集合；`DUPLICATE` 与 `HISTORICAL` 不进入默认检索。

生命周期已实现并验收：

```text
build
ingest-update
rebuild
verify
Dense smoke
BM25 smoke
```

## 环境提示

- 建议使用 Python 3.10-3.12。
- 通用参考依赖位于 `code/requirements.txt`。
- C8、C9 分别提供独立的 `requirements.txt`。
- C9 的环境变量模板位于 `code/C9/.env.example`；请勿提交真实 API Key 或密码。
- Milvus 本地容器配置位于 `code/docker-compose.yml`。

由于仓库保留不同阶段的参考实现，不建议一次性安装并运行所有模块。应根据当前正在开发的模块选择对应依赖。

## 来源与许可证

本项目基于 Datawhale 的 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 修改，原项目采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)。

本仓库衍生内容继续遵循该许可；第三方组件按其各自许可证使用。完整说明见 [`LICENSE.md`](LICENSE.md) 和 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
