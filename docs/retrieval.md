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

v1 详细合同见 `docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`。
确定性内核与 vLLM 模型路径已实现。`b1-router-v3` 是 Qwen2.5-7B-Instruct 的 **reference baseline**，不是 production-accepted Router。B1 **NOT CLOSED**。停止 v4 及针对当前测试集的 prompt patch。14B+v3 本轮否定、不切换；未来重开须独立评估。regression 64 与 blind 64 均保留 gold；blind 对 v3 有效但已 **exposed**，不得再当 unseen CLOSED set。只改 prompt 升 `router_prompt_version`；换模型或合同级行为才升 contract version。B3–B5 继续独立验收；B1→B5 E2E 不得宣称 PASS/CLOSED。见 `docs/proceeding/2026-09-15-b1-experiment-closeout.md`。

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

v1 Router 实现已冻结为：

```text
Qwen2.5-7B-Instruct
+
vLLM OpenAI-compatible API
+
xgrammar JSON Schema constrained decoding
```

生产客户端必须发送 `guided_decoding_backend=xgrammar`，与 vLLM 0.8.5 V1 启动参数 `--guided-decoding-backend xgrammar` 逐字一致。禁止 `auto`（会换约束库）和请求级 `xgrammar:no-fallback`（该版本 V1 不接受）。发给 xgrammar 的 `guided_json` 是冻结合同 Schema 的兼容投影：去掉其不支持的 `minItems` / `maxItems` / `minLength`，**不用 `oneOf` 编码业务组合**。`ROUTER_JSON_SCHEMA` 与 business validation 仍保留完整限制；非法 mode/字段组合由二次校验打成 `RouterProtocolError`，不自动修复、不重试。当前实现 prompt 为 `b1-router-v3`（7B reference baseline，非 production-accepted）。合同与 Schema 版本仍为 `b1-contract-v1` / `b1-schema-v1`。B1 不 CLOSED。

---

# B2 — Query Processing

状态：`[FROZEN baseline]`；实现 `[CLOSED]`

v1 详细合同见 `docs/Preprocessing/2026_09_02_B2_Query_Processing_Frozen_Requirements_v1.md`。Identity Processor（`NO TRANSFORMATION` / `identity-v1`）已实现并关闭。

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

v1 详细合同见 `docs/Preprocessing/2026-09-10-B3_Hybrid_Retrieval_Frozen_Requirements_v1.md`。

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

中文 BM25 analyzer/tokenizer v1 集成 baseline：

```text
UNIFIED_ICU
tokenizer = icu
filters = lowercase + removepunct
```

Analyzer 的正式 Collection / Function 物理配置属于 A6。B3 不得在应用层维护第二套 BM25。

---

# B4 — RRF Fusion

状态：`[FROZEN baseline]`

v1 详细合同见 `docs/Preprocessing/2026-09-10-B4_RRF_Fusion_Frozen_Requirements_v1.md`。

## B4.1 Method

```text
RRF
```

优先使用 Milvus 的 RRF 能力。

## B4.2 Baseline Parameter

```text
k = 60
```

v1 使用 Milvus 2.5.x `RRFRanker` 官方默认值。这不是已证明的语料最优值；只有 D1 证明融合是稳定瓶颈时才重新打开 k 调参。

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
RRF(k=60)
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

v1 详细合同见 `docs/Preprocessing/2026-09-10-B5_Cross_Encoder_Reranking_Frozen_Requirements_v1.md`。

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

v1 使用官方 Qwen3-Reranker 模板 + 固定英文 instruction，文档侧只使用 `Leaf.content`。正式修改模板必须升级 B5 contract version。

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
5. RRF baseline `k=60`。
6. RRF 输出 Top-20。
7. Qwen3-Reranker-0.6B → Top-5 Leaf。
8. Parent Recovery 必须发生在 Rerank 之后。
9. Context Recovery 使用多级 Section + neighbor fallback。
10. Retrieval 每阶段必须对 D1 可观察。
