# Architecture

## 1. Purpose

当前项目目标是构建一个能够对制造业技术文档进行可靠知识检索、上下文恢复和证据约束问答的 RAG baseline。

参考的工业智能体产品形态可以同时连接图纸、工艺库、业务系统、库存和供应链数据，但本仓库当前只实现其中的**工业文档知识库与问答链路**。图纸/CAD/ERP/MES/库存等能力不在本阶段 scope。

## 2. System Boundary

```text
Raw PDF
  ↓
A0 Preprocessing
  ↓
trusted document.md
  ↓
A1 Markdown Loading
  ↓
A2 Minimal Provenance / Identity
  ↓
A3 Structure-aware Chunking
  ↓
A4 Hierarchical Parent-Child
  ↓
A5 Embedding
  ↓
A6 Milvus Index
  ↓
B1 Query Router
  ↓
B2 Query Processing
  ↓
B3 Dense + BM25 Retrieval
  ↓
B4 RRF Fusion
  ↓
B5 Cross-Encoder Reranking
  ↓
B6 Hierarchical Context Recovery
  ↓
C1 Evidence-Bounded Prompt Assembly
  ↓
C2 Qwen2.5-7B-Instruct
  ↓
C3 Citation / Evidence Mapping
  ↓
D1 Retrieval Evaluation
```

D2 Generation Evaluation 尚未冻结。

## 3. Module Responsibilities

### A0 — Preprocessing

负责：

```text
Raw PDF → trusted Markdown
```

它处理 OCR、Layout、Table、Figure/Caption、页码 provenance 和质量门禁。

A0 的错误必须在 A0 修复。

### A — Knowledge Base

负责：

```text
trusted document.md
→ Leaf
→ Section hierarchy
→ Embedding
→ Milvus
```

不重新解释 raw PDF。

### B — Retrieval

负责：

```text
Query
→ capability route
→ Dense/BM25
→ RRF
→ Cross-Encoder
→ Top Leaf
→ bounded context recovery
```

### C — Generation

负责：

```text
Evidence
→ structured prompt
→ evidence-bounded answer
→ citation mapping
```

### D1 — Retrieval Evaluation

负责判断：

```text
正确 evidence 到底在哪一步丢失？
```

分别观察 Dense、BM25、RRF、Reranker 和 B6 Recovery。

## 4. Cross-module Contracts

### A0 → A1

唯一正式正文输入：

```text
document.md
```

Audit artifacts 不得作为正文重复入库。

### A3/A4 → B6

A3/A4 必须提供：

```text
Leaf identity
document identity
section identity
document-local chunk order
page provenance
Section hierarchy
```

使 B6 可以：

```text
Leaf → Section → parent Section
```

以及：

```text
Leaf ± neighbor
```

### B5 → B6

B5 输出 Leaf，而不是 Parent。

Context Recovery 发生在 Rerank 之后。

### B6 → C1

B6 输出去重后的 bounded evidence context。

C1 不重新检索。

### C1 → C2

C2 只能基于 Evidence 回答。

Evidence 不足必须明确退化。

## 5. Architectural Principles

### 5.1 Child Retrieval, Larger Context Generation [FROZEN]

小 Leaf 负责 Retrieval/Rerank；较大的语义上下文负责 Generation。

### 5.2 Structure Before Length [FROZEN]

Markdown 结构优先，固定窗口只是末级 Section 过长时的 fallback。

### 5.3 Retrieval Before Recovery [FROZEN]

```text
Leaf Retrieval
→ Rerank
→ Context Recovery
```

禁止先把长 Parent 展开再 Rerank。

### 5.4 Evidence-Bounded Generation [FROZEN]

“Grounded”不等于逐字复述。

模型可以：

```text
总结
解释
比较
基于 Evidence 推理
```

但不能：

```text
凭参数记忆补企业事实
编造标准条款
编造工艺参数
```

### 5.5 Data-driven Tuning [FROZEN]

Chunk size、未来 IVF 参数、query instruction wording 等必须由实际语料和 D1 支持。

## 6. Decision Status

### Frozen

- Milvus
- Dense + Native BM25 Hybrid
- RRF fusion
- Cross-Encoder baseline
- Multi-level Section hierarchy
- Qwen3-Embedding-4B baseline
- Qwen2.5-7B-Instruct generation
- Strict capability boundary + Evidence-Bounded answer
- D1 stage-wise retrieval evaluation

### Provisional

- B6 精确 ancestor expansion policy
- C3 claim-level citation 粒度
- D1 精确 hit/coverage rule
- BM25 中文 analyzer/tokenizer
- Section Tree 的具体持久化介质

### Deferred

- D2 最终 Generation Evaluation
- Query Rewrite / Decomposition / Step-Back / HyDE
- ColBERT baseline
- GraphRAG
- VLM/CAD RAG
- Structured business-system RAG
