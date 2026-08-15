# Evaluation

## 1. Status

```text
D1 Retrieval Evaluation
→ 主体框架已确定

D2 Generation Evaluation
→ [DEFERRED]
```

当前不要把旧 spec 中的 LLM-as-a-Judge 方案恢复成最终 D2 标准。

---

# D1 — Retrieval Evaluation

状态：framework `[FROZEN]`；精确 hit-rule `[PROVISIONAL]`

## D1.1 Goal

D1 不是只看最终答案。

需要定位：

```text
Dense 有没有找到 Gold？
BM25 有没有找到 Gold？
RRF 有没有融合好？
Reranker 有没有把 Gold 提到前面？
B6 Recovery 最终是否覆盖完整 answer-supporting evidence？
```

## D1.2 Stage-wise Evaluation

至少分别评：

```text
Dense
BM25
RRF
Reranker
```

并额外记录：

```text
B6 recovered context coverage
```

每阶段保留：

```text
ranked result identity
rank
score（若该阶段有）
source/page/section provenance
```

## D1.3 Core Metrics

第一版核心：

```text
Recall@5
Recall@10
Recall@20
MRR
```

可额外观察更大的 K 看 recall plateau，但不是第一版主指标。

## D1.4 nDCG

```text
[DEFERRED unless graded relevance]
```

当前没有工业专家的 3/2/1/0 graded relevance，因此不强行为了指标丰富引入 nDCG。

## D1.5 Precision / MAP / F1

当前不作为第一版核心验收指标。

如果后续分析需要，可以补，但不能代替 Recall/MRR 主体。

---

# 2. Benchmark Construction

## D1.6 Source

Benchmark 必须基于：

```text
真实 A0 处理后的实际文档
```

而不是用不相关公开 QA benchmark 替代项目检索验收。

## D1.7 First-stage Scale

`[FROZEN]`

第一阶段：

```text
1 representative document
≈ 50 queries
```

这是工程迭代 benchmark，不代表最终整个工业知识库的整体效果。

## D1.8 Query Generation Process

从真实 evidence 反向构造：

```text
Gold Evidence
→ LLM generates question/reference answer
→ human lightweight validation
```

人工不需要具备完整工业领域专家经验。

## D1.9 Query Variants

应覆盖：

```text
Direct
Paraphrase
Cross-lingual
```

尤其：

```text
Chinese Query → English Evidence
```

## D1.10 Gold Evidence Groups

`[PROVISIONAL construction pattern]`

当前建议：

```text
~16–17 Gold Evidence Groups
× 3 variants
≈ 48–51 queries
```

用户已冻结的是：

```text
单文档约 50 Query
```

具体 group 数与三类 Query 的精确配比仍可调整。

---

# 3. Golden Data

## D1.11 Gold Must Not Bind to chunk_id

`[FROZEN]`

Golden 不使用：

```text
relevant_chunk_ids
```

作为稳定定义。

因为调整：

```text
chunk_size
overlap
heading split
```

以后 chunk_id 可能变化。

## D1.12 Stable Provenance

每条 Query 至少保存：

```text
query
source_document
page / page range
section_path
evidence_text
```

如果可稳定获得：

```text
text offsets
```

建议保存。

还可以保存：

```text
reference_answer
variant_type
gold_evidence_group_id
```

其中：

```text
variant_type ∈ {direct, paraphrase, cross_lingual}
```

## D1.13 Gold Evidence Definition

`[PROVISIONAL]`

当前 proposal：

> Gold 尽量定义为 minimum answer-supporting span，即支持答案所需的最小充分证据。

避免：

```text
整页/整章过宽
```

也避免：

```text
只标一个无法独立支持答案的关键词
```

## D1.14 Human Validation

`[FROZEN]`

人工至少检查：

```text
1. Question 是否确实能从 Gold Evidence 回答？
2. Reference Answer 是否确实被 Gold Evidence 支持？
```

不要求人工逐条标注所有 Retriever Candidate 的相关度等级。

---

# 4. Hit Rule

## D1.15 Retrieval Hit

`[PROVISIONAL]`

当前讨论方案：

```text
same source
+
Gold evidence span coverage 达到阈值
```

初始候选阈值：

```text
~80%
```

80% 尚未冻结。

## D1.16 Union Coverage

`[PROVISIONAL]`

如果 Gold Evidence 被当前 A3 切到多个 Leaf：

```text
Top-K union
```

可以共同覆盖 Gold span。

否则会把“需要两个相邻 Leaf 才完整覆盖”的正确检索误判为 miss。

最终 union coverage 公式/实现仍需确认。

## D1.17 MRR Relevance Unit

`[PROVISIONAL]`

MRR 应以：

> 第一个能够独立构成 answer-supporting / relevant evidence unit 的结果

作为 first relevant result。

不能因为某个 Chunk 只擦到一个关键词就算 relevant。

精确规则仍需确认。

## D1.18 No Semantic-model-defined Gold

`[FROZEN]`

禁止用：

```text
Embedding similarity
LLM semantic score
```

直接定义 Retrieval Candidate 是否命中 Gold。

Hit 应基于可审计：

```text
source
page
section
evidence span
coverage
```

关系。

---

# 5. B6 Recovery Evaluation

除了 Leaf Retrieval，还必须记录：

```text
B6 recovered context
是否最终完整/充分覆盖 Gold evidence
```

因为：

```text
Leaf Retrieval success
```

与：

```text
最终送给 LLM 的 Context success
```

不是同一个指标。

---

# 6. D1 Drives Tuning

以下参数不提前声称“最佳”，应由 D1 支持：

```text
chunk_size
overlap
query instruction wording
heading-enhanced retrieval（如未来实验）
reranker 0.6B → 4B upgrade
future IVF_FLAT nlist/nprobe
```

---

# D2 — Generation Evaluation

状态：`[DEFERRED]`

当前未冻结：

```text
Faithfulness judge
Answer relevance judge
Citation correctness metric
LLM-as-a-Judge model
Judge prompt
aggregation
acceptance threshold
```

在 D2 正式讨论前，不由 coding agent 自行补充最终标准。
