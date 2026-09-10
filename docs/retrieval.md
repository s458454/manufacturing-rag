# Retrieval — B1 to B6

## 1. Online Retrieval Chain

```text
User Query
  ↓
B1 Router
  ↓
B2 Query Processing
  ↓
B3 Dense + BM25
  ↓
B4 RRF
  ↓
B5 Cross-Encoder
  ↓
B6 Context Recovery
```

---

# B1 — Query Router

状态：`[FROZEN]`

v1 详细合同见 `docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`。实现尚未开始。

## B1.1 Routes

只定义：

```text
knowledge
system
unsupported
```

### knowledge

进入工业文档 RAG。

### system

处理系统自身：

```text
能力
使用方式
当前支持范围
状态
```

### unsupported

当前能力范围外任务，例如本版本没有接入的：

```text
图纸理解
库存查询
ERP/MES实时数据
```

## B1.2 Strict Capability Baseline

```text
unsupported
→ 明确 fallback
```

而不是：

```text
unsupported
→ 绕过 RAG
→ LLM 自由回答外部事实
```

当前不提供 broad `general` route。

## B1.3 Safety

Capability Router 与 safety/policy gate 分离，不混成一个分类体系。

Router 具体实现手段（规则/轻量模型/LLM）当前未冻结，要求是行为契约而不是强制某种分类器。

---

# B2 — Query Processing

状态：`[FROZEN baseline]`

v1 详细合同见 `docs/Preprocessing/2026_09_02_B2_Query_Processing_Frozen_Requirements_v1.md`。实现尚未开始。Identity Processor（`NO TRANSFORMATION`）是当前 baseline。

## B2.1 Baseline

```text
original query
```

直接进入 Retrieval。

## B2.2 Extension Interface

保留统一 QueryProcessor/等价接口，但 baseline 不启用：

```text
LLM Rewrite
Query Decomposition
Step-Back
HyDE
LLM-as-a-Judge pre-rerank
```

这些只有 D1/真实 failure mode 证明有必要时才加入。

---

# B3 — Hybrid Retrieval

状态：`[FROZEN]`

v1 详细合同见 `docs/Preprocessing/2026-09-10-B3_Hybrid_Retrieval_Frozen_Requirements_v1.md`。实现尚未开始；A6 已提供其依赖的生产 Collection、Dense 索引与 Native BM25 能力。

## B3.1 Dense

```text
original query
→ frozen English retrieval instruction
→ Qwen3-Embedding-4B
→ Milvus FLAT/IP
→ Top-20 Leaf
```

## B3.2 Sparse

```text
original query text
→ Milvus Native BM25
→ UNIFIED_ICU analyzer
→ Top-20 Leaf
```

BM25 baseline：

```text
k1 = 1.2
b = 0.75
DAAT_MAXSCORE
```

Analyzer/BM25 常量由 A6/A6.0 负责，B3 不在应用层重新分词或维护第二套 BM25。

## B3.3 Baseline Candidate Count

```text
Dense Top-20
BM25 Top-20
```

这不是永久最佳值，但当前作为 v1 baseline。

## B3.4 Production / D1 Observability

线上生产路径优先使用一次 Milvus `hybrid_search`，不要求为了日志额外执行两次单路检索。

生产至少保留：

```text
request_id
task_id
dense encode latency
hybrid search status / latency
fused output count
```

D1 / profiling 为了 stage-wise evaluation，可以对同一 Query 额外执行：

```text
Dense-only Top-20
BM25-only Top-20
Hybrid + RRF Top-20
```

并记录 Dense/BM25 原始 rank 与 score。

---

# B4 — RRF Fusion

状态：`[FROZEN baseline]`

## B4.1 Method

```text
Milvus Native RRF
```

v1 线上优先使用：

```text
hybrid_search + RRFRanker
```

B3/B4 是逻辑模块边界，不要求拆成多次 Milvus 网络调用。

## B4.2 Baseline Parameter

```text
k = 60
```

使用 Milvus 2.5.x 官方默认值作为 v1 baseline；不声称 `60` 是当前语料最优参数。

## B4.3 Identity / Dedup

RRF 合并的实体身份是：

```text
chunk_id
```

同一 Leaf 被 Dense/BM25 同时召回时作为一个 Candidate 融合，而不是两条重复结果。

## B4.4 Output

```text
Dense Top-20
BM25 Top-20
↓
RRFRanker(k=60)
↓
Fused Top-20
```

Milvus 返回顺序即 v1 融合顺序；应用层不再计算第二套 RRF 或基于 raw score 做二次融合。

## B4.5 WeightedRanker

当前：

```text
not baseline
```

只有后续证据显示需要人为偏置 Dense/BM25 权重时再评估。

---

# B5 — Cross-Encoder Reranking

状态：`[FROZEN baseline]`

## B5.1 Model

```text
Qwen3-Reranker-0.6B
```

升级候选：

```text
Qwen3-Reranker-4B
```

只有 D1 明确表明 reranking 是主要瓶颈时升级。

## B5.2 Input / Output

```text
RRF Top-20 Leaf
→ Cross-Encoder
→ Top-5 Leaf
```

Top-5 是**最大优先候选数**，不是要求最终 Prompt 一定塞满 5 个不同 Context。

## B5.3 Why Cross-Encoder, not ColBERT baseline

当前第一阶段已经有：

```text
Dense + BM25 + RRF
```

其职责是高 Recall 候选获取。

B5 的目标是：

> 对少量 Candidate 做更充分的 Query-Document 联合语义判断。

ColBERT 属于 late-interaction retrieval/ranking 路线，当前：

```text
[DEFERRED / not baseline]
```

## B5.4 Reranker Input Serialization

`[PROVISIONAL]`

具体 Query/Chunk template、是否附带 heading context、长 Chunk 截断方式尚未冻结。

Baseline 不得在没有确认的情况下额外拼一套复杂 metadata 文本。

## B5.5 Ordering Constraint

禁止：

```text
Leaf Retrieval
→ Parent Recovery
→ Rerank Parent
```

必须保持：

```text
Leaf Retrieval
→ RRF
→ Rerank Leaf
→ Recovery
```

---

# B6 — Hierarchical Context Recovery

状态：主行为 `[FROZEN]`；exact ancestor selection `[PROVISIONAL]`

## B6.1 Input

```text
Reranker Top-5 Leaf
```

## B6.2 Recovery

对每个 Leaf：

```text
Leaf.section_id
→ nearest Section
→ parent Section
→ ...
```

Parent 不固定 H2/H3。

## B6.3 Shared-parent Grouping

多个高排名 Leaf 可能属于同一个 Section。

必须：

```text
group
deduplicate
```

避免同一大段文本重复进入 C1。

多 Child 命中同一 Parent 时最终 group score/rank aggregation 方式：

```text
[PROVISIONAL]
```

## B6.4 Context-recovery Budget

必须受 budget 控制，不能一路向上恢复整篇文档。

独立 recovery budget 的具体数值：

```text
[PROVISIONAL]
```

总 Evidence Budget 由 C1 baseline 约 8K tokens 控制。

## B6.5 Parent-too-large Fallback

如果语义 Parent 无法合理使用：

```text
matched Leaf + neighbor chunks
```

Neighbor 由：

```text
document_id + chunk_index
```

定位。

## B6.6 Exact Ancestor Selection

`[PROVISIONAL]`

已确认：

- 从命中 Leaf 的语义 Section 出发；
- 动态使用多级 Section；
- 受 budget 限制；
- 不固定 Heading level；
- Parent 不适用则 neighbor fallback。

尚未冻结：

```text
nearest fitting ancestor 即停止
```

还是：

```text
在预算允许时继续向上扩大到更高祖先
```

coding agent 不得自行选择一种后当作最终行为。

---

# Retrieval Invariants

1. Router 只允许 knowledge/system/unsupported。
2. B2 baseline 不 Rewrite。
3. Dense/BM25 默认同时开启。
4. 两支均取 Top-20。
5. BM25 使用 A6 冻结的 `UNIFIED_ICU + k1=1.2 + b=0.75 + DAAT_MAXSCORE`。
6. RRF baseline `k=60`。
7. RRF 输出 Top-20。
8. Qwen3-Reranker-0.6B → Top-5 Leaf。
9. Parent Recovery 必须发生在 Rerank 之后。
10. Context Recovery 使用多级 Section + neighbor fallback。
11. 生产路径不为 observability 强制增加单路检索；D1 可单独运行 Dense/BM25 做 stage-wise evaluation。
