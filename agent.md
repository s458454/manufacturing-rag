# Manufacturing RAG — Repository Agent Guide

> 本文件是仓库级 Agent 行为规范、仓库地图和文档索引。  
> 详细需求、接口、参数与实现约束下沉到 `docs/`。  
> `docs/proceeding/` 独立记录阶段过程、烟测、故障、验收与交接，不作为稳定需求的主来源。

## 1. Project

本仓库面向制造业文档 RAG。当前主线只处理 **A0 预处理后正式发布的 Markdown 文档**，不在知识库链路中重新解析 raw PDF。

主链：

```text
Raw PDF
  ↓
A0 Preprocessing
  ↓
trusted document.md
  ↓
A Knowledge Base
  ↓
B Retrieval
  ↓
C Generation
  ↓
D Evaluation
```

当前重点是工业文档知识库与问答，不试图复刻完整图纸/ERP/MES/库存智能体平台。

## 2. Decision Status

文档统一使用：

- `[FROZEN]`：已经确认；不得由 coding agent 自行替换。
- `[PROVISIONAL]`：已有明确候选方案，但尚未最终确认。
- `[DEFERRED]`：当前明确不做或以后再定。
- `[CURRENT IMPLEMENTATION]`：描述当前代码/组件事实，不等于永久架构约束。

Agent 不得把 `PROVISIONAL` 擅自变成 `FROZEN`，也不得实现 `DEFERRED` 功能来“补完整架构”。

## 3. Repository Map

```text
agent.md
docs/
├── README.md
├── architecture.md
├── data-corpus.md
├── preprocessing.md
├── knowledge-base.md
├── retrieval.md
├── generation.md
├── evaluation.md
├── manufacturing-rag-v0.1-spec.md
├── Preprocessing/
│   ├── pdf-preprocessing-server-acceptance.md
│   └── 2026-08-14-region-ocr-isolation-and-native-table-empty-cell.md
└── proceeding/
    └── ...
```

模块阅读索引：

| 修改内容 | 先读 |
|---|---|
| 总体链路、边界、全局不变量 | `docs/architecture.md` |
| 当前语料、数据范围、语言与数据角色 | `docs/data-corpus.md` |
| PDF→正式 Markdown、OCR/Layout/Table、质量门禁 | `docs/preprocessing.md` |
| Markdown Loading、Metadata、Chunk、Hierarchy、Embedding、Milvus | `docs/knowledge-base.md` |
| Router、Query Processing、Hybrid、RRF、Rerank、Context Recovery | `docs/retrieval.md` |
| Evidence Assembly、Qwen2.5-7B、Citation | `docs/generation.md` |
| Retrieval Evaluation 与后续 Generation Evaluation 状态 | `docs/evaluation.md` |
| 当前阶段日志、烟测、问题、验收和交接 | `docs/proceeding/` |

## 4. Source of Truth

“应该怎么做”：

```text
当前对应模块 docs 中的 [FROZEN] 要求
> 旧 spec / 旧教程
```

“代码现在实际上怎么做”：

```text
真实代码
> proceeding / README / 历史说明
```

旧 Datawhale/C8 教程和旧制造业 spec 只是历史参考。如果与当前模块文档冲突，以当前模块文档为准。

## 5. Agent Working Rules

1. 修改前先读对应模块文档，再读真实入口及直接上下游代码。
2. 不根据文件名、README 一句话或旧教程猜当前调用链。
3. 不扩大 scope；没有冻结的模块不得“顺手实现”。
4. 每新增一个持久化 Metadata 字段，必须能指出当前明确消费者；否则不加。
5. 参数未经过真实数据或评测确定时，不得声称“最佳值”。
6. 参数应集中配置，不得散落硬编码。
7. A0 的解析/OCR/Table/Figure 问题必须在 A0 修，A1 以后不得用清洗绕过。
8. 修改公共接口前必须检查上下游影响。
9. 如果文档要求与库/API能力冲突，先报告证据，再讨论修改；不得静默替换技术方案。
10. Proceeding 记录“发生过什么”，稳定模块文档规定“系统应该怎样工作”；两者不得混用。

## 6. Global Invariants

以下是不允许在模块实现中破坏的端到端原则：

1. 知识库正式输入是 A0 发布的 `document.md`，不是 raw PDF。
2. A0 的 audit JSON 不得作为正文重复入库。
3. A1 只扫描明确的正式输出根目录，不扫描 smoke / `.failed` / `.staging`。
4. Markdown heading 与 PDF page provenance 必须在知识库链路中可恢复。
5. Chunking 先结构、后长度；Leaf 是第一阶段唯一 Retriever Candidate。
6. Parent 使用多级 Section hierarchy，不固定 H2/H3。
7. Dense 与 BM25 检索同一批 Leaf。
8. Vector DB 使用 Milvus；当前 Dense baseline 为精确 FLAT，Sparse 为 Milvus Native BM25。
9. Hybrid → RRF → Cross-Encoder → Top Leaf → Context Recovery 的顺序不得反转。
10. LLM 采用 Evidence-Bounded Generation：允许基于证据总结、解释、比较和推理，但不得用参数知识补企业事实。
11. D1 必须能分阶段观察 Dense / BM25 / RRF / Reranker，而不是只看最终答案。
12. D2 当前仍为 Deferred。

## 7. Current Non-goals

当前不默认实现：

```text
GraphRAG
复杂多 Agent
VLM / CAD / STEP / STL 理解
ERP / MES / SQL Structured RAG
自动业务 Metadata 抽取
raw PDF knowledge loader
Parent 参与第一阶段 Retrieval
默认 Query Rewrite / Decomposition / Step-Back / HyDE
WeightedRanker baseline
HNSW baseline
MRL embedding 降维
无证据 general free-answer route
```

具体 exclusions 以对应模块文档为准。
