# Generation — C1 to C3

## 1. Generation Principle

当前使用：

> **Evidence-Bounded Generation**

这不是逐字复制模式。

模型允许：

```text
总结
解释
比较
基于提供 Evidence 的合理推理
```

但禁止：

```text
凭模型记忆补企业事实
编造工艺参数
编造标准条款
编造来源
```

Evidence 不足时必须明确说明不足。

---

# C1 — Prompt / Evidence Assembly

状态：`[FROZEN main behavior]`

## C1.1 Input

输入来自 B6 的：

```text
grouped
deduplicated
bounded recovered evidence
```

C1 不重新检索。

## C1.2 Evidence Blocks

使用结构化 Evidence ID，例如：

```text
[E1]
Source: ...
Page: ...
Section: ...
Content:
...

[E2]
...
```

Evidence ID 只在本次回答上下文中需要稳定唯一。

## C1.3 Ordering

Parent grouping / dedup 后，保持与 Reranker relevance 对应的优先顺序。

不要按文件名/页码重新洗牌。

## C1.4 Evidence Budget

baseline：

```text
~8K tokens
```

可配置。

它表示**检索 Evidence 的预算**，不是模型完整 context window。

System prompt、User Query、回答空间仍需另外预留。

精确 accounting：

```text
[PROVISIONAL]
```

## C1.5 Over-budget Strategy

超过 budget 时优先：

```text
降低 Recovery 粒度
```

例如：

```text
larger Section
→ smaller Section / subsection
→ matched Leaf + neighbors
→ drop low-priority evidence
```

禁止简单在语义 Section 中间硬截一刀作为主策略。

## C1.6 Evidence Block Fields

`[PROVISIONAL detail]`

当前需要支持：

```text
Evidence ID
source/document identity
page range
section identity/path
content
```

具体序列化模板可调整，但不得让模型自己猜 source/page。

---

# C2 — LLM Generation

状态：`[FROZEN baseline]`

## C2.1 Model

```text
Qwen2.5-7B-Instruct
```

它负责阅读 Evidence、归纳和组织答案，而不是替 Retrieval 补知识。

## C2.2 Deterministic / Evaluation Profile

```text
do_sample = False
greedy decoding
```

用于：

```text
regression
ablation
evaluation
```

目标是减少生成随机性对比较的干扰。

## C2.3 Demo / Interactive Profile

baseline：

```text
do_sample = True
temperature = 0.2
top_p = 0.8
top_k = 20
repetition_penalty = 1.05
```

事实边界仍由 Evidence-Bounded policy 约束，不由 temperature 保证。

## C2.4 max_new_tokens

```text
[PROVISIONAL / not frozen]
```

保持配置化。

## C2.5 Knowledge Boundary

C2 不承担：

```text
外部知识补全
无证据自由回答
替 Retriever 猜答案
```

---

# C3 — Citation / Evidence Mapping

状态：`[PROVISIONAL]`

当前已有清晰方案，但 citation 粒度尚未最终冻结。

## C3.1 Evidence ID

模型使用：

```text
[E1]
[E2]
```

引用。

禁止引用 Prompt 中不存在的 Evidence ID。

## C3.2 Backend Mapping

Backend 负责：

```text
Evidence ID
→ document_id
→ document title/source
→ section
→ page_start/page_end
```

模型不得自己生成/猜：

```text
文件路径
页码
source URL
```

## C3.3 Claim-level Citation

当前 proposal：

以下关键事实应尽量/必须有直接 Evidence 支持：

```text
数值
标准要求
工艺条件
限制条件
材料/制造事实
影响最终结论的技术性事实
```

纯组织语言不必每句都标 citation。

但以下仍 `[PROVISIONAL]`：

```text
claim granularity
“必须引用”的严格覆盖范围
最终 citation UI 形式
```

---

# Generation Invariants

1. C1 输入是 B6 已恢复的 Evidence，不重新检索。
2. Evidence Block 有稳定 ID。
3. 总 Evidence baseline 约 8K tokens，可配置。
4. 超预算优先降低 Recovery 粒度。
5. C2 = Qwen2.5-7B-Instruct。
6. Evaluation 用 Greedy；Demo 用低温 sampling baseline。
7. 模型不得利用无 Evidence 参数知识补企业事实。
8. C3 backend mapping 不让模型猜来源。
9. C3 citation 严格粒度仍为 Provisional。
