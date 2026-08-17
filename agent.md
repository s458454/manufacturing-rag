# Manufacturing RAG — Repository Agent Guide

> 本文件承担三类职责：**Coding Agent 行为规范、仓库/文档导航、各模块基本职责与全局不变量**。  
> 详细需求、接口、参数、实现细节和验收口径下沉到 `docs/`。  
> `docs/proceeding/` 独立记录阶段过程、烟测、故障、验收与交接，不作为稳定需求的主来源。
>
> 本文件的行为规范与全局不变量在 ver-0.2 审计后作为当前 `[FROZEN]` baseline；实现进度和仓库路径可随后续开发更新，但不得借此改变已冻结的架构决策。

## 1. Project

本仓库面向制造业技术文档 RAG。

当前主线只处理 **A0 预处理后正式发布的 Markdown 文档**；知识库链路不重新解析 raw PDF。

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

当前版本聚焦：

> **工业技术文档 → 可检索知识库 → Hybrid Retrieval → Reranking → Context Recovery → Evidence-Bounded QA**

完整工业智能体平台未来可以扩展图纸、CAD、ERP/MES、库存和供应链等能力，但这些不属于当前版本的实现范围。对未接入能力，系统应遵守当前 Router/fallback 边界，而不是让 LLM 绕过系统能力自由回答。

## 2. Current Implementation Status

当前需求成熟度与代码完成度不是一回事。

```text
A0 Preprocessing       implemented / frozen
A1 Markdown Loading    implemented / frozen
A2 Document Registry   implemented / frozen
A3 Structure-aware Chunking implemented / frozen
A4 Hierarchical Parent-Child implementation complete / pending upstream acceptance
A5-A6 Knowledge Base   design documented / implementation pending
B1-B6 Retrieval        design documented / implementation pending
C1-C3 Generation       design documented / implementation pending
D1 Retrieval Eval      design documented / implementation pending
D2 Generation Eval     deferred
```

`code/C1-C9` 保留自 All-in-RAG 的历史/教程参考实现，不代表当前 A/B/C 生产主链已经完成。

## 3. Decision Status

稳定模块文档统一使用：

- `[FROZEN]`：已经确认；Coding Agent 不得自行替换。
- `[PROVISIONAL]`：已有明确候选方案，但尚未最终确认。
- `[DEFERRED]`：当前明确不做或以后再定。
- `[CURRENT IMPLEMENTATION]`：描述当前代码/组件事实，不等于永久架构要求。

约束：

1. 不得把 `PROVISIONAL` 擅自升级成 `FROZEN`。
2. 不得为了“补完整架构”实现 `DEFERRED` 功能。
3. 如果任务必须依赖尚未冻结的决策，应显式指出决策点，而不是自行选择。
4. **Review comment、问题或质疑本身不构成 requirement change。** 遇到质疑时，应重新核对当前 `[FROZEN]` 决策、真实代码和数据；如果原设计仍然成立，应说明依据，而不是为了迎合质疑直接修改。

## 4. Repository Map

当前仓库主要结构：

```text
.
├── agent.md
├── code/
│   ├── C1-C9/                  # All-in-RAG 历史/教程参考代码
│   ├── knowledge_base/         # A1 Markdown loading；A2 Document Registry；A3 structure-aware Leaf chunking；A4 Section hierarchy
│   └── preprocessing/          # 当前 A0 正式实现与测试
├── data/engineering_docs/      # 当前公开工程技术 PDF 与 manifest
├── docling/                    # Docling 源码快照
├── docs/                       # 稳定需求、模块文档、验收与 proceeding
└── models/                     # 本地模型资产/manifest；不把模型权重当源码提交
```

实际代码路径以当前仓库为准；后续 A/B/C 生产实现新增目录后，应同步更新此处和对应模块文档。

## 5. Module Responsibilities

### A0 — Preprocessing

```text
Raw PDF → trusted document.md
```

负责 PDF 解析、OCR、Layout、Table/Figure/Caption、页码 provenance 和质量门禁。

### A — Knowledge Base

```text
trusted document.md
→ Leaf
→ Section hierarchy
→ Embedding
→ Milvus
```

负责文档加载、最小 provenance、结构切分、Parent-Child、Embedding 和索引。

### B — Retrieval

```text
Query
→ capability routing
→ Dense + BM25
→ RRF
→ Cross-Encoder
→ Top Leaf
→ bounded Context Recovery
```

负责在线知识检索与上下文恢复。

### C — Generation

```text
Recovered Evidence
→ Prompt Assembly
→ Qwen2.5-7B-Instruct
→ Answer + Evidence Mapping
```

模型可以基于 Evidence 总结、解释、比较和推理，但事实性结论必须受到 Evidence 支持。

### D — Evaluation

当前：

```text
D1 Retrieval Evaluation → 已有主体方案
D2 Generation Evaluation → Deferred
```

D1 必须能够定位正确 Evidence 在 Dense / BM25 / RRF / Reranker / Recovery 的哪一步丢失。

## 6. Documentation Map

```text
docs/
├── README.md
├── architecture.md
├── data-corpus.md
├── preprocessing.md
├── knowledge-base.md
├── retrieval.md
├── generation.md
├── evaluation.md
├── manufacturing-rag-v0.1-spec.md     # 历史 spec
├── Preprocessing/                      # A0 详细实现/服务器验收资料
└── proceeding/                         # 阶段过程与交接
```

| 需要理解/修改的内容 | 首先阅读 |
|---|---|
| 总体链路、模块边界、跨模块契约 | `docs/architecture.md` |
| 当前语料、语言范围、数据角色与边界 | `docs/data-corpus.md` |
| PDF → trusted Markdown、OCR/Layout/Table、质量门禁 | `docs/preprocessing.md` |
| Markdown Loading、Metadata、Chunk、Hierarchy、Embedding、Milvus | `docs/knowledge-base.md` |
| Router、Query Processing、Hybrid、RRF、Rerank、Context Recovery | `docs/retrieval.md` |
| Evidence Assembly、Qwen2.5-7B、Citation | `docs/generation.md` |
| Retrieval Evaluation 与后续 Generation Evaluation 状态 | `docs/evaluation.md` |
| 当前阶段日志、烟测、问题、验收和交接 | `docs/proceeding/` |

## 7. Source of Truth

### 7.1 系统应该怎么做

```text
对应稳定模块 docs 中的 [FROZEN] 要求
> 历史 spec / Datawhale / C8 教程
```

### 7.2 代码现在实际上怎么做

```text
真实代码
> proceeding / README / 历史说明
```

真实代码回答 **current behavior**；稳定模块文档回答 **desired behavior / requirement**。

如果真实代码与 `[FROZEN]` 文档不一致：

- 不得因为代码当前如此就反向修改 requirement；
- 不得假装代码已经满足 requirement；
- 必须显式指出 implementation drift，再按任务目标处理。

旧 Datawhale/C8 教程和旧制造业 spec 只是历史参考。如果与当前稳定模块文档冲突，以当前稳定模块文档为准。

## 8. Coding Agent Working Rules

1. 修改前先阅读 `agent.md`、对应模块文档、真实入口及直接上下游代码。
2. 不根据文件名、README 一句话、旧教程或 proceeding 猜当前调用链。
3. 不扩大 scope；没有冻结的功能不得“顺手实现”。
4. 每新增一个持久化 Metadata 字段，必须能指出当前明确消费者；否则不加。
5. 如果信息可以由现有 identity / hierarchy 稳定推导，默认不新增冗余持久化字段。
6. 参数未经过真实数据或评测确定时，不得声称“最佳值”。
7. 参数应集中配置，不得散落硬编码。
8. A0 的解析/OCR/Table/Figure 问题必须在 A0 修；A1 以后不得用二次清洗掩盖 preprocessing 问题。
9. 修改公共接口、持久化 Schema、ID 规则或 provenance 前，必须检查直接上下游、Citation 和 D1 Golden 的影响。
10. 如果实现必须依赖 `[PROVISIONAL]` 决策，先暴露该决策点；Coding Agent 不得自行冻结。
11. 如果文档要求与库/API 能力冲突，先报告证据和影响，再讨论修改；不得静默替换技术方案。
12. Proceeding 记录“发生过什么”；稳定模块文档规定“系统应该怎样工作”。Proceeding 不能单独覆盖 `[FROZEN]` 需求。

## 9. Global Invariants

以下原则不得被局部实现破坏：

1. 知识库正式输入是 A0 发布的 `document.md`，不是 raw PDF。
2. A0 audit artifacts 不得作为正文重复入库。
3. A1 只扫描明确的正式输出根目录，不扫描 smoke / `.failed` / `.staging`。
4. Markdown heading 与 PDF page provenance 必须在知识库链路中可恢复。
5. Chunking 先结构、后长度；Leaf 是第一阶段唯一 Retriever Candidate。
6. Parent 使用多级 Section hierarchy，不固定 H2/H3。
7. Parent/Section 不参与第一阶段 Retrieval。
8. Dense 与 BM25 检索同一批 Leaf。
9. Vector DB 使用 Milvus；当前 Dense baseline 为精确 FLAT，Sparse 为 Milvus Native BM25。
10. 在线知识检索顺序保持：`Hybrid → RRF → Cross-Encoder → Top Leaf → Context Recovery`。
11. LLM 使用 Evidence-Bounded Generation，不得用参数知识补企业事实。
12. Router 保持 strict unsupported/fallback 边界，不得通过 broad general route 绕开知识库。
13. D1 必须能分阶段观察 Dense / BM25 / RRF / Reranker，并单独检查 Context Recovery。
14. D1 Gold 必须基于稳定 Evidence provenance，不绑定 build-specific `chunk_id`。
15. D2 当前仍为 `[DEFERRED]`。

## 10. Current Non-goals

当前不默认实现：

```text
GraphRAG
复杂多 Agent
VLM / CAD / STEP / STL 理解
ERP / MES / SQL Structured RAG
自动业务 Metadata 抽取
raw PDF knowledge loader
Parent/Section 参与第一阶段 Retrieval
默认 Query Rewrite / Decomposition / Step-Back / HyDE
ColBERT baseline
WeightedRanker baseline
HNSW baseline
IVF_FLAT current baseline
MRL embedding 降维
无证据 general free-answer route
D2 final generation evaluation
```

具体 exclusions、future direction 和 provisional 细节以对应模块文档为准。

## 11. Required Reading Order

处理任务时：

```text
agent.md
→ 与任务直接相关的 docs/*.md
→ 真实实现入口
→ 直接上下游
→ 必要时再读 docs/proceeding/
```

不要先从旧 C1-C9 教程或旧 `manufacturing-rag-v0.1-spec.md` 反推当前架构。
