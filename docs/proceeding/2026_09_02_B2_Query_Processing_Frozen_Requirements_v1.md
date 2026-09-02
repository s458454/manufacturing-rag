# B2 Query Processing — Frozen Requirements v1

> Status: **FROZEN**
>
> Contract version: `b2-contract-v1`  
> Processor version: `identity-v1`
>
> 本文档是 B2 Query Processing baseline 的唯一权威需求。
>
> B2 v1 的设计目标不是“尽可能改写 Query”，而是建立一个**可复现、无语义漂移、可作为后续检索增强对照基线**的 Identity Query Processor。
>
> Query Translation、Retrieval Decomposition、Query Expansion、HyDE、Step-Back 等能力只有在 D1/真实检索 failure mode 证明必要时才允许进入后续版本。

---

# 1. B2 在整条链路中的位置

```text
B1 Query Router
    ↓
knowledge RoutedTask
    ↓
B2 Query Processing
    ↓
ProcessedQuery
    ↓
B3 Hybrid Retrieval
├── Dense
└── BM25
```

B2 的职责是：

> 接收 B1 已经确认语义完整、可执行的 knowledge Query，并决定是否需要为 Retrieval 改变其表达形式。

B2 v1 的决策固定为：

```text
NO TRANSFORMATION
```

即：

```text
input query
=
output retrieval query
```

---

# 2. B1 与 B2 的边界

B1 负责：

```text
用户到底想做什么
Conversation Context 理解
上下文缺失时生成 candidate_query
用户确认 Context Rewrite
业务级多意图拆分
knowledge / system / unsupported 路由
```

B2 负责：

```text
一个已经语义完整的 knowledge Query，
是否需要为了 Retriever 的特性生成其他检索表达
```

当前 baseline：

```text
不需要
```

必须严格区分：

```text
B1 Context Rewrite
= ON

B2 Retrieval Rewrite
= OFF
```

---

# 3. B2 输入

B2 只接受：

```text
B1 direct
+
route = knowledge
```

以及多任务请求中被 Dispatcher 分派出的单个 `knowledge` task。

以下不得进入 B2：

```text
system
unsupported
confirm_rewrite
clarify
task_limit
```

## 3.1 输入结构

固定：

```python
@dataclass(frozen=True)
class QueryProcessingRequest:
    request_id: str
    task_id: str
    query: str
```

`request_id` 继承当前用户请求，B2 不生成、不修改。

`task_id` 由 Orchestrator / Dispatcher 在 B1 `direct` 结果分派时生成。

固定生成规则：

```text
task_id = "{request_id}:{task_index}"
```

其中 `task_index` 按 B1 `tasks` 原始顺序，从 `0` 开始。

例如：

```text
request_id = req-123

B1 tasks:
0 → knowledge
1 → system
2 → knowledge
```

则进入 B2 的 task_id 分别为：

```text
req-123:0
req-123:2
```

B2 不重新编号。

`query` 必须等于 B1 对应 knowledge `RoutedTask.query`。

B2 不再读取：

```text
B1 route
Conversation History
raw_user_query
candidate_query
```

---

# 4. B2 Baseline Runtime

固定：

```text
B2 runtime = NONE
```

B2 本身不调用：

```text
LLM
Embedding Model
Reranker
Milvus
Web
External Tool
```

B2 是纯确定性程序。

---

# 5. Identity Processing

唯一处理行为：

```python
retrieval_query = request.query
```

禁止修改文本。

---

# 6. 禁止的 Query Normalization

B2 baseline 不允许：

```text
strip
lowercase
uppercase
简繁转换
拼写修正
标点删除
内部空格修改
换行修改
数字格式归一化
单位归一化
标准号归一化
缩写展开
停用词删除
关键词抽取
分词后重组
```

原因：

> B1 已经完成输入校验和必要的 Context Rewrite。B2 无权再次“纠正”用户语义或技术标识。

例如禁止自动：

```text
NASA STD 5006A
→ NASA-STD-5006A
```

除非未来有独立 failure evidence 和明确的确定性规范化合同。

---

# 7. 禁止的 Retrieval Transformation

B2 v1 全部关闭：

```text
LLM Rewrite
Query Translation
Query Expansion
Retrieval Query Decomposition
Step-Back
HyDE
LLM-as-a-Judge pre-retrieval / pre-rerank
Abbreviation Expansion
Identifier Normalization
Synthetic Answer Generation
```

不得因为实现者认为“可能有帮助”而自行开启。

---

# 8. Query Translation

Baseline：

```text
OFF
```

行为：

```text
中文 Query
→ 原中文 Query

English Query
→ 原 English Query

中英混合 Query
→ 原混合 Query
```

B2 不做语言识别。

未来只有在 D1 / 真实 failure mode 证明：

```text
中文 Query 检索英文文档时稳定漏召回
且
人工/离线翻译后 Recall 明显提升
```

时，才允许评估 Translation。

---

# 9. Retrieval Query Decomposition

Baseline：

```text
OFF
```

例如：

```text
比较 6061 与 7075 的焊接性能差异。
```

B2 v1 必须保持为一个 Query。

禁止自动生成：

```text
6061 的焊接性能是什么？
7075 的焊接性能是什么？
```

未来只有在 D1 / 真实 failure mode 证明：

```text
复合 knowledge Query 因多个信息需求共享一个有限 Top-K Candidate Pool，
产生稳定的候选竞争 / 子主题漏召回，
且拆分后 Recall 明显提升
```

时，才允许评估 Retrieval Decomposition。

---

# 10. B1 Multi-Task Decomposition 与 B2 Retrieval Decomposition

必须严格区分。

B1 处理多个业务目标，例如：

```text
查 5006A
+
询问助手能力
+
查询 ERP
```

B2 未来可能处理：

```text
一个业务目标
```

为了 Retrieval 召回完整性，在内部产生多个 retrieval query。

例如：

```text
比较 6061 与 7075 的焊接性能
```

仍然是一个业务 task，但未来可能产生多个 retrieval query。

B2 v1 不启用该能力。

---

# 11. Dense Instruction 不属于 B2

B2 输出的 Query 永远保持用户语义文本。

例如：

```text
NASA-STD-5006A 对铝合金焊接有什么要求？
```

进入 B3 后：

Dense Branch 可以根据 Qwen3-Embedding 模型要求拼接：

```text
retrieval instruction
+
query
```

这是：

```text
Dense model input serialization
```

属于 B3。

BM25 Branch 直接使用：

```text
原 retrieval query text
```

因此：

> 模型专属 retrieval instruction 禁止写回 B2 ProcessedQuery，也禁止污染 BM25 Query。

---

# 12. B2 输出结构

固定：

```python
@dataclass(frozen=True)
class ProcessedQuery:
    request_id: str
    task_id: str
    original_query: str
    retrieval_queries: tuple[str, ...]
```

Baseline 唯一输出：

```python
ProcessedQuery(
    request_id=request.request_id,
    task_id=request.task_id,
    original_query=request.query,
    retrieval_queries=(request.query,),
)
```

---

# 13. Baseline Invariants

必须同时满足：

```python
len(processed.retrieval_queries) == 1
processed.retrieval_queries[0] == processed.original_query
processed.original_query == request.query
processed.request_id == request.request_id
processed.task_id == request.task_id
```

Coding agent 不得因为 `retrieval_queries` 是复数字段就自行生成多个 Query。

---

# 14. 为什么保留 retrieval_queries 复数字段

它是未来 Query-adaptation 的接口扩展点。

如果未来基于实验升级：

```text
identity-v1
→ translation-v2
或 decomposition-v2
```

可能形成：

```text
original_query
+
多个 retrieval_queries
```

但 baseline 固定：

```text
retrieval_query_count = 1
```

未来是否以及如何让 B3 消费多个 Retrieval Query，需要与对应 B2 新版本一起重新冻结。

当前 B3 v1 不得提前实现未冻结的 multi-query fusion 策略。

---

# 15. 为什么保留 original_query

当前：

```text
original_query
==
retrieval_queries[0]
```

仍必须同时保留两个概念，用于：

```text
D1 stage-wise evaluation
debug
未来 Rewrite/Translation/Decomposition 对照
```

未来即使 Retrieval Query 发生变化，也必须能追溯进入 B2 的原始 knowledge Query。

---

# 16. Conversation History

固定：

```text
B2 reads conversation history = NO
```

Conversation Context 已在 B1 处理。

禁止：

```text
B2 再读取 history
B2 根据 history 重新解释用户意图
B2 根据 history 修改 retrieval query
```

---

# 17. 输入契约保护

理论上所有进入 B2 的 Query 已经通过 B1 校验。

B2 仍执行最小内部 contract check：

```text
request_id 为非空 str
task_id 为非空 str
query 为非空 str
```

不得：

```text
strip
自动修复
替换默认值
```

调用者违反合同：

```text
→ QueryProcessingContractError
```

属于内部 programming / orchestration error，不是用户 Query 错误。

不新增 B2 用户侧固定回复，由 Orchestrator 按内部服务异常处理。

---

# 18. B2 不产生正常业务失败路径

由于 v1 是 Identity Processor，正常合法输入必须成功产生 `ProcessedQuery`。

不存在：

```text
RewriteUnavailable
TranslationFailed
DecompositionFailed
ModelUnavailable
```

这些错误在 baseline 中不得存在。

---

# 19. Observability

每次 B2 处理至少记录：

```text
request_id
task_id
query_processor_contract_version
query_processor_version
retrieval_query_count
processing_latency_ms
```

Baseline 固定：

```text
retrieval_query_count = 1
```

B2 observability log 不再次保存完整 Query 正文。

通过 `request_id` / `task_id` 与 Request / Conversation / Retrieval logs 关联。

---

# 20. 版本

固定：

```text
query_processor_contract_version = "b2-contract-v1"
query_processor_version = "identity-v1"
```

任何 Query transformation 行为变化必须升级 `query_processor_version`。

不兼容接口变化必须升级 B2 contract version。

---

# 21. Failure-Driven Extension Principle

B2 后续能力必须由真实 Retrieval Failure 驱动。

不得依据：

```text
论文中常见
框架提供该功能
技术看起来先进
实现方便
```

直接开启。

一个 B2 Extension 至少应满足：

```text
1. D1 或真实 Query 中出现可重复 Retrieval Failure。
2. 定位表明 Failure 与 Query Representation 直接相关，而不是 B3/B4/B5/B6 的问题。
3. 手工或离线 Oracle 变换后 Retrieval 指标明显改善。
4. 自动化 B2 策略在目标 Failure Set 上稳定改善。
5. 普通 Query 不产生不可接受的回归。
6. 新策略具有独立开关、版本和 stage-wise observability。
```

---

# 22. 典型未来 Extension 触发条件

## Query Translation

```text
中文 Query → 英文 corpus 稳定漏召回
人工英文 Query → Recall 明显改善
```

→ Translation candidate。

## Retrieval Decomposition

```text
一个复合 knowledge Query 中包含多个证据子主题
共享一个 Top-K Candidate Pool
→ 某些子主题长期被挤出
```

手工拆解后明显改善：

```text
→ Retrieval Decomposition candidate
```

## Abbreviation Expansion

例如：

```text
GMAW
vs
Gas Metal Arc Welding
```

若长期匹配不足且人工展开稳定改善：

```text
→ Abbreviation Expansion candidate
```

## Identifier Normalization

若标准号 / 工艺编号格式差异导致稳定漏召回，且存在确定性规范化规则：

```text
→ Identifier Normalization candidate
```

## Query Expansion

若原 Query 语义过于抽象，增加合理检索表达后 Recall 稳定改善：

```text
→ Query Expansion candidate
```

---

# 23. Failure Ownership

发现 Retrieval Failure 时必须先定位责任模块：

```text
对话指代无法理解
→ B1

中文问英文内容长期不召回，人工翻译有效
→ B2 Translation candidate

复合 Query 的多个证据子主题争抢 Top-K
→ B2 Retrieval Decomposition candidate

完整 Query 已正确，但 Dense 召回弱
→ B3 Dense

精确术语 / 标准号 Sparse 召回弱
→ B3 BM25 / Analyzer

正确 Candidate 已被两支召回，但融合后掉出
→ B4

正确 Leaf 在 RRF Top-K，但 Reranker 排错
→ B5

正确 Leaf 已命中，但恢复上下文不完整
→ B6
```

禁止：

```text
Recall 不好
→ 默认给 B2 加 Rewrite
```

---

# 24. B2 Final Baseline

```text
Input:
    one B1 knowledge task

Runtime:
    NONE

History:
    NONE

Normalization:
    NONE

Rewrite:
    OFF

Translation:
    OFF

Retrieval Decomposition:
    OFF

Expansion:
    OFF

Step-Back:
    OFF

HyDE:
    OFF

Keyword Extraction:
    OFF

Output:
    original_query = input query
    retrieval_queries = (input query,)

Invariant:
    retrieval_query_count = 1
```

---

# 25. B2 Final Pipeline

```text
B1 direct
knowledge RoutedTask
       ↓
Dispatcher assigns:
request_id
task_id = request_id:task_index
       ↓
B2 QueryProcessingRequest
       ↓
Identity Query Processor
       ↓
ProcessedQuery
├── original_query = Q
└── retrieval_queries = (Q,)
       ↓
B3 Hybrid Retrieval
```

---

# 26. Explicitly Deferred

```text
LLM Query Rewrite
Query Translation
Retrieval Query Decomposition
Query Expansion
Abbreviation Expansion
Identifier Normalization
Step-Back
HyDE
LLM-as-a-Judge preprocessing
Multi-query Retrieval Fusion
```

这些能力只有新的 B2 版本明确冻结后才允许进入正式链路。

---

# 27. B2 Frozen Invariants

```text
1. B2 只接收 knowledge task。
2. B2 不读取 Conversation History。
3. B2 runtime = NONE。
4. B2 v1 是 Identity Query Processor。
5. B2 不修改任何 Query 字符。
6. original_query == B1 knowledge task.query。
7. retrieval_queries 只能有一个元素。
8. retrieval_queries[0] == original_query。
9. Dense retrieval instruction 属于 B3，不属于 B2。
10. BM25 获取未被 Dense instruction 污染的原 Query。
11. Retrieval Rewrite / Translation / Decomposition 全部 OFF。
12. 所有未来 Query adaptation 必须由 D1 / 真实 Failure 驱动。
13. B2 不替 B3/B4/B5/B6 的错误背锅。
14. request_id / task_id 必须贯穿后续 Retrieval observability。
15. Coding agent 不得自行增加任何 Query enhancement。
```

---

# 28. Acceptance

B2 v1 为纯确定性 Identity Processor。

其 baseline 验收主要是：

```text
接口
不变量
traceability
无 Query mutation
```

而不是单独建立模型准确率 benchmark。

B2 Query Transformation 的效果评估只在未来开启对应 Extension 时增加。
