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

## B3.1 Dense

```text
original query
→ English retrieval instruction
→ Qwen3-Embedding-4B
→ Milvus FLAT/IP
→ Top-20 Leaf
```

## B3.2 Sparse

```text
original query text
→ Milvus Native BM25
→ Top-20 Leaf
```

## B3.3 Baseline Candidate Count

```text
Dense Top-20
BM25 Top-20
```

这不是永久最佳值，但当前作为 baseline。

## B3.4 Branch Observability

必须分别保留/记录：

```text
Dense rank
Dense score

BM25 rank
BM25 score

chunk_id
document/source identity
```

不能只保存融合结果。

这既用于 debug，也用于 D1 stage-wise evaluation。

## B3.5 BM25 Analyzer

中文 BM25 analyzer/tokenizer：

```text
[PROVISIONAL]
```

当前不能由 coding agent 随意选择后宣称完成生产中文检索。

---

# B4 — RRF Fusion

状态：`[FROZEN baseline]`

## B4.1 Method

```text
RRF
```

优先使用 Milvus 的 RRF 能力。

## B4.2 Baseline Parameter

```text
k = 55
```

必须可配置。

## B4.3 Identity / Dedup

RRF 合并的实体身份是：

```text
chunk_id
```

同一 Leaf 被 Dense/BM25 同时召回时应作为一个 Candidate 融合，而不是两条重复结果。

## B4.4 Output

```text
Dense Top-20
BM25 Top-20
↓
RRF(k=55)
↓
Fused Top-20
```

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
5. RRF baseline `k=55`。
6. RRF 输出 Top-20。
7. Qwen3-Reranker-0.6B → Top-5 Leaf。
8. Parent Recovery 必须发生在 Rerank 之后。
9. Context Recovery 使用多级 Section + neighbor fallback。
10. Retrieval 每阶段必须对 D1 可观察。
