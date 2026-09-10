# B3 Hybrid Retrieval — Frozen Requirements v1

> Status: **FROZEN**
>
> Contract version: `b3-contract-v1`
>
> Dense query serializer version: `b3-dense-query-v1`
>
> 本文档是 B3 Hybrid Retrieval v1 的权威逻辑需求。
>
> **2026-09-10 修订：** 删除“应用层必须先独立执行 Dense/BM25 再自行 RRF”的物理实现限制；v1 允许并优先使用 Milvus 2.5.x 原生 `hybrid_search + RRFRanker`。
>
> **2026-09-10 参数修订：** RRF 平滑参数由无明确依据的 `55` 改为 Milvus 2.5.x 官方默认值 `60`。
>
> B3 负责把 B2 输出的单个知识检索 Query 构造成 Dense 与 Native BM25 两路检索请求。
> B4 负责定义 RRF 融合规则。v1 线上物理实现允许通过一次 Milvus `hybrid_search`
> 同时执行 B3 两路检索与 B4 RRF，不要求应用层先物化两份独立候选列表。
>
> 本文不定义正式 Milvus Collection 的物理 Schema、字段类型、Collection 命名、版本管理、
> rebuild/upsert、Section Store 等 A6 实现细节。A6 必须满足本文所依赖的逻辑检索合同，
> B3 不得反向自行修改 A6 的物理实现合同。

---

# 1. B3 在整条链路中的位置


```text
B1 Query Router
    ↓
B2 Query Processing
    ↓
B3 Dense + BM25
    ↓
B4 RRF
    ↓
B5 Cross Encoder Reranker
    ↓
B6 Context Recovery
```

逻辑职责：

```text
B3：
构造 Dense 检索请求
+
构造 BM25 检索请求

B4：
定义 RRF 融合规则
```

v1 线上物理执行允许合并为一次 Milvus 原生混合检索：

```text
B2 query
   ↓
B3 构造两路 AnnSearchRequest
├── Dense limit = 20
└── BM25 limit = 20
   ↓
B4 提供 RRFRanker(k=60)
   ↓
Milvus hybrid_search(..., limit=20)
   ↓
融合 Top-20
   ↓
B5 Cross Encoder
```

中文说明：

- `AnnSearchRequest`：Milvus 的单路向量/稀疏检索请求对象。
- `RRFRanker`：Milvus 内置的倒数排名融合器。
- `hybrid_search`：Milvus 一次执行多路检索并按指定融合器重排的接口。
- 外层 `limit=20`：融合完成后最多返回 20 条候选。

因此：

> **B3/B4 保持逻辑模块边界，但 v1 不强制把它们拆成多次 Milvus 网络调用。**

B3 不负责：

```text
业务意图识别
Conversation Context 理解
业务任务拆分
Query Rewrite
Query Translation
Retrieval Query Decomposition
Query Expansion
RRF 数学规则选择
Cross Encoder 重排
Parent / Section 上下文恢复
最终回答生成
```

# 2. 与 B1 / B2 的边界

## 2.1 B1 业务任务拆分

B1 可以拆分明确、独立的多个业务目标。

例如：

```text
查 NASA-STD-5006A 的焊接要求，
再告诉我系统支持哪些功能。
```

可以拆成：

```text
knowledge task
+
system task
```

但 B1 不是“能拆就拆”。

例如：

```text
比较 6061 和 7075 的焊接性、热处理要求和检验要求。
```

仍然可以是一个完整 knowledge task。

## 2.2 B2 Retrieval Decomposition

B2 v1：

```text
Retrieval Decomposition = OFF
```

B3 v1 只接受：

```text
len(processed.retrieval_queries) == 1
```

B3 不得自行把一个 Query 再拆成多个检索 Query。

---

# 3. B3 输入

B3 直接消费 B2 的：

```python
@dataclass(frozen=True)
class ProcessedQuery:
    request_id: str
    task_id: str
    original_query: str
    retrieval_queries: tuple[str, ...]
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `request_id` | 当前用户请求的唯一编号，用于跨模块追踪 |
| `task_id` | 当前 knowledge 子任务编号 |
| `original_query` | 进入 B2 时的原始完整知识问题 |
| `retrieval_queries` | 实际用于检索的问题集合；B2 v1 固定只有一个 |

B3 v1 必须满足：

```python
len(processed.retrieval_queries) == 1
processed.retrieval_queries[0] == processed.original_query
```

实际检索 Query：

```python
query = processed.retrieval_queries[0]
```

其中：

| 字段 | 中文含义 |
|---|---|
| `query` | B3 当前真正用于 Dense 和 BM25 检索的问题文本 |

B3 不读取：

```text
Conversation History
B1 route
raw_user_query
candidate_query
```

---

# 4. 检索实体

B3 第一阶段唯一检索实体：

```text
Leaf
```

A3 已冻结：

```text
Dense document text = Leaf.content
BM25 indexed text   = Leaf.content
```

禁止 B3 自行构造：

```text
heading + content
document title + content
section path + content
page marker + content
```

Leaf 的检索唯一身份：

```text
chunk_id
```

---

# 5. RetrievalCandidate

B3 两路统一使用：

```python
@dataclass(frozen=True)
class RetrievalCandidate:
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str
    rank: int
    score: float
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `chunk_id` | 当前 Leaf 的唯一检索编号，也是两路融合时识别同一候选的主身份 |
| `document_id` | 当前 Leaf 属于哪一份文档 |
| `section_id` | 当前 Leaf 属于哪个章节 |
| `chunk_index` | 当前 Leaf 在所属文档中的顺序编号 |
| `page_start` | 当前 Leaf 对应原文的起始页 |
| `page_end` | 当前 Leaf 对应原文的结束页 |
| `content` | 当前 Leaf 的正文，也是 Dense/BM25 baseline 使用的正文 |
| `rank` | 当前候选在本检索分支中的排名，从 1 开始 |
| `score` | 当前检索分支返回的原始相关性分数，越大越相关 |

B3 不返回：

```text
dense_vector
sparse_vector
Analyzer token list
BM25 中间稀疏表示
```

---

# 6. Candidate 不变量

## 6.1 rank

固定：

```text
rank = 1-based
```

即：

```text
第一名 rank = 1
第二名 rank = 2
...
```

B3 按底层检索接口返回顺序赋 rank，不重新根据跨分支分数排序。

## 6.2 score

`score` 保留分支原始相关性分数。

Dense：

```text
score = IP similarity score
```

中文含义：

> 查询向量与文档向量的内积相似度，越大越相似。

BM25：

```text
score = native BM25 relevance score
```

中文含义：

> BM25 根据词项匹配计算的原始相关性分数，越大越相关。

禁止：

```text
Dense score 与 BM25 score 直接比较
Dense score + BM25 score
跨分支 min-max normalization
跨分支 z-score
跨分支 softmax 后相加
```

两路融合由 B4 RRF 基于 rank 完成。

## 6.3 chunk_id 重复规则

同一分支内部：

```text
duplicate chunk_id = protocol error
```

中文含义：

> 同一次 Dense 或同一次 BM25 结果里不允许同一个 Leaf 重复出现。

两路之间：

```text
same chunk_id across Dense/BM25 = expected
```

中文含义：

> 同一个 Leaf 同时被 Dense 与 BM25 命中是正常情况，后续由 B4 按 chunk_id 融合。

如果同一 `chunk_id` 同时出现在两支，则以下字段必须一致：

```text
document_id
section_id
chunk_index
page_start
page_end
content
```

只有：

```text
rank
score
```

允许不同。

---

# 7. Dense 稠密向量检索

## 7.1 Dense 输入

Dense 使用：

```text
B2 retrieval query
```

但在真正送入 Embedding Model 前，需要增加固定检索任务说明。

这只是：

```text
Dense model input serialization
```

中文含义：

> 为向量模型说明“这个 Query 的向量将用于什么类型的检索”。

它不是：

```text
Query Rewrite
Query Translation
Query Expansion
```

不得写回 B2，也不得污染 BM25 Query。

---

# 8. Dense Query Instruction

v1 固定英文任务说明：

```text
Given a manufacturing and engineering query, retrieve relevant passages from technical documents that contain the information needed to answer the query.
```

中文含义：

> 给定一个制造业或工程技术问题，从技术文档中检索包含回答该问题所需信息的相关段落。

固定序列化格式：

```text
Instruct: Given a manufacturing and engineering query, retrieve relevant passages from technical documents that contain the information needed to answer the query.
Query:{query}
```

实现规则：

```python
serialized_query = (
    "Instruct: Given a manufacturing and engineering query, "
    "retrieve relevant passages from technical documents that contain "
    "the information needed to answer the query.\n"
    f"Query:{query}"
)
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `query` | B2 传入的真实检索问题 |
| `serialized_query` | 添加固定任务说明后，真正送入 Qwen3-Embedding-4B 的完整文本 |

版本：

```text
dense_query_serializer_version = "b3-dense-query-v1"
```

中文含义：

> `dense_query_serializer_version` 用于标识当前 Query 拼接方式和任务说明版本。

正式修改任务说明或序列化格式时必须升级该版本。

---

# 9. Dense Query Encoder

模型固定：

```text
Qwen3-Embedding-4B
```

查询端与 A5 文档端必须保持同一数值合同：

```text
Qwen3-Embedding-4B
→ left padding
→ last-token pooling
→ L2 normalize
→ 2560-d
→ float32
```

中文说明：

| 项 | 中文含义 |
|---|---|
| `left padding` | 批量编码时在较短文本左侧补齐 |
| `last-token pooling` | 使用最后一个有效位置的隐藏状态作为整段文本向量 |
| `L2 normalize` | 把向量长度归一化为 1 |
| `2560-d` | 最终查询向量包含 2560 个浮点数 |
| `float32` | 最终向量使用 32 位浮点数 |

查询和文档均 L2 normalize，因此检索相似度固定：

```text
IP
```

中文含义：

> Inner Product，内积。由于两侧向量都已经 L2 归一化，其排序与余弦相似度一致。

---

# 10. Dense 输入长度

完整 `serialized_query` 的最大模型输入预算：

```text
8192 inference tokens
```

检查对象必须是：

```text
instruction + Query 标签 + 实际 query
```

禁止只检查原始 Query 后忽略任务说明。

超过上限：

```text
→ DenseQueryEncodingError
```

中文含义：

> 查询无法按照冻结的稠密向量编码合同完成。

禁止：

```text
silent truncation
```

即禁止静默截断 Query 后继续检索。

---

# 11. Dense Encoder 复用原则

B3 不应重新复制一套：

```text
tokenizer
pooling
normalize
dimension check
```

应尽量复用 A5 已验收的 `Qwen3DenseEncoder` 底层数值实现。

推荐逻辑：

```text
B3 serialize_query()
        ↓
A5-compatible dense encoder core
        ↓
2560-d query vector
```

A5 的 document-side `encode_documents()` 不得被修改为自动添加 Query instruction。

---

# 12. Dense 模型生命周期

Qwen3-Embedding-4B 应在服务生命周期内常驻。

禁止：

```text
每个 Query 重新加载模型
```

v1 不要求拆成独立 HTTP Embedding 服务。

模型加载失败或推理失败时：

```text
禁止自动切换 Qwen3-Embedding-0.6B
禁止自动切换其他 Embedding 模型
禁止自动 CPU fallback
```

---

# 13. BM25 关键词检索

BM25 直接使用：

```text
B2 retrieval query 原文
```

禁止 B3 在应用层进行：

```text
语言识别
中文分词
英文分词
lowercase
去标点
关键词抽取
停用词删除
Query Rewrite
Query Translation
Query Expansion
```

链路：

```text
B2 retrieval query
        ↓
Milvus Native BM25
        ↓
按 BM25 相关性返回候选
```

---

# 14. BM25 与 Analyzer 的责任边界

B3 的逻辑要求：

```text
Native BM25
```

当前 v1 集成 baseline 使用 A6.0 profiling 推荐的：

```text
UNIFIED_ICU
```

其当前配置为：

```text
tokenizer = icu
filters   = lowercase + removepunct
```

中文说明：

| 配置 | 中文含义 |
|---|---|
| `tokenizer=icu` | 使用 ICU 规则把正文和 Query 切成检索词 |
| `lowercase` | 英文字母统一转为小写 |
| `removepunct` | 分词时移除标点符号 |

重要边界：

> Analyzer 的正式 Collection / Function 物理配置属于 A6。B3 不得在应用层重新实现 ICU 分词，也不得维护第二套 BM25。

B3 不做语言判断，也不根据中文/英文切换 Analyzer。

---

# 15. BM25 参数 baseline

当前 v1 integration baseline：

```text
k1 = 1.2
b  = 0.75
```

中文说明：

| 参数 | 中文含义 |
|---|---|
| `k1` | 控制一个词在同一 Leaf 中重复出现时，词频继续增加还能带来多少额外相关性收益 |
| `b` | 控制 Leaf 长短差异对 BM25 分数的修正强度 |

这些值是：

```text
v1 baseline
```

不是：

```text
已证明的最优参数
```

如果未来 D1 明确证明 BM25 参数是召回瓶颈，再重新打开参数评估。

当前 A6.0 profiling 使用：

```text
SPARSE_INVERTED_INDEX
BM25
DAAT_MAXSCORE
```

其中具体物理索引配置仍由 A6 负责，B3 不重复维护。

---

# 16. 两路并行执行


B3 在逻辑上包含两路：

```text
Dense
BM25
```

但 v1 线上不要求应用层分别启动两个异步任务。

正式物理执行允许：

```text
一个 Milvus hybrid_search
    ├── Dense AnnSearchRequest
    └── BM25 AnnSearchRequest
```

两路请求在同一次 Hybrid Search 中参与候选生成，再由 B4 的 `RRFRanker` 完成融合。

因此冻结：

```text
逻辑上：
Dense + BM25 两路同时开启

物理上：
优先使用 Milvus 原生 hybrid_search
```

禁止因为使用单次 `hybrid_search` 而改变：

```text
Dense candidate limit = 20
BM25 candidate limit = 20
```

两路仍是独立的候选预算。

# 17. 分支运行状态


v1 线上生产路径不要求 Milvus 向应用层分别暴露 Dense/BM25 的独立运行状态。

生产路径只要求整体状态：

```python
class HybridSearchStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"
```

字段中文说明：

| 状态 | 中文含义 |
|---|---|
| `SUCCESS` | Milvus 混合检索正常完成，并返回至少一条融合候选 |
| `EMPTY` | Milvus 混合检索正常完成，但融合后返回 0 条候选 |
| `FAILED` | 混合检索调用没有正常完成 |

必须严格区分：

```text
EMPTY ≠ FAILED
```

Dense/BM25 单路的 `SUCCESS / EMPTY / FAILED` 只属于：

```text
D1 分阶段评估
Analyzer profiling
单路调试
```

不是每个线上请求必须物化的生产字段。

# 18. 单路结果结构


线上生产路径不要求构造：

```text
BranchRetrievalResult
dense_result
bm25_result
```

因为 Milvus `hybrid_search` 可以在数据库内部直接完成：

```text
Dense Top-20
+
BM25 Top-20
+
B4 RRFRanker
```

如果 D1 / profiling 需要单独观察两路，则允许使用独立 search 调用输出：

```python
@dataclass(frozen=True)
class BranchRetrievalResult:
    status: RetrievalStatus
    candidates: tuple[RetrievalCandidate, ...]
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `status` | 单路评估时该检索分支的执行状态 |
| `candidates` | 单路评估时返回的有序 Leaf 候选列表 |

该结构属于：

```text
evaluation / profiling interface
```

不是生产 `hybrid_search` 的强制中间对象。

# 19. 两路状态汇合规则


生产 v1 使用 Milvus 原生 Hybrid Search，因此故障合同按整个调用判断：

```text
hybrid_search 正常并有结果
→ SUCCESS

hybrid_search 正常但 0 结果
→ EMPTY

hybrid_search 抛错 / 超时 / 返回非法结构
→ FAILED
```

任何 `FAILED`：

```text
→ 当前 knowledge retrieval 整体失败
→ 不进入 B5
→ 不进入 B6
```

禁止：

```text
Hybrid Search 失败
→ 再偷偷改成 Dense-only

Hybrid Search 失败
→ 再偷偷改成 BM25-only
```

也就是说：

> v1 不提供单路降级检索。

如果 D1 / profiling 使用独立 Dense/BM25 调用，则任一路基础能力失败都必须记录为对应分支故障；但这不改变生产路径的原子 Hybrid Search 合同。

# 20. 并发失败后的另一支任务


由于 v1 线上优先使用单次 Milvus `hybrid_search`，应用层不存在必须管理的：

```text
Dense future
BM25 future
取消另一支任务
```

因此原先“某一路失败后取消另一支”的应用层要求删除。

由 Milvus 负责单次 Hybrid Search 内部执行。

应用层只处理：

```text
hybrid_search 成功 / 空结果 / 失败
```

# 21. 自动重试


v1：

```text
B3/B4 application retry = OFF
```

中文含义：

> 应用层不会因为一次 `hybrid_search` 失败而自动重新执行相同混合检索。

以后如果确实需要瞬时网络故障重试，应由基础设施客户端策略单独冻结，不能由 B3/B4 隐式增加。

# 22. 超时


生产 v1 至少必须支持：

```python
hybrid_search_timeout_ms: int
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `hybrid_search_timeout_ms` | 一次 Milvus 混合检索允许执行的最长时间，单位毫秒 |

当前：

```text
timeout mechanism = REQUIRED
timeout concrete value = NOT FROZEN
```

具体数值在真实部署验收后确定。

超时：

```text
→ FAILED
```

Dense Query Encoder 自身如需要独立推理超时，可以由 Dense 编码服务配置单独控制；该数值不在本轮冻结。

# 23. B3 正常输出


B3 的生产逻辑输出不是两份已经物化的候选列表，而是两路检索请求计划：

```python
@dataclass(frozen=True)
class HybridSearchPlan:
    request_id: str
    task_id: str
    query: str
    dense_candidate_limit: int
    bm25_candidate_limit: int
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `request_id` | 当前用户请求唯一编号 |
| `task_id` | 当前 knowledge 子任务编号 |
| `query` | B2 传入的实际检索问题 |
| `dense_candidate_limit` | Dense 分支参与 Hybrid Search 的最大候选数，v1 为 20 |
| `bm25_candidate_limit` | BM25 分支参与 Hybrid Search 的最大候选数，v1 为 20 |

实际运行时，还必须包含：

```text
Dense query vector
BM25 raw query text
A6 提供的检索字段绑定
```

这些属于执行对象，不要求永久序列化到业务日志。

B4 随后为同一次 `hybrid_search` 提供：

```text
RRFRanker(k=60)
fusion_candidate_limit=20
```

其中：

- `RRFRanker(k=60)`：使用 Milvus 2.5.x 官方默认的 RRF 平滑参数 60 进行排名融合；
- `fusion_candidate_limit=20`：融合完成后最多保留 20 条候选交给 B5。

v1 选择 `k=60` 的理由：

```text
1. Milvus 2.5.x 官方默认值。
2. 当前项目没有实验依据支持自定义为 55。
3. v1 优先使用成熟组件默认参数建立可复现 baseline。
4. 不声称 k=60 是当前语料最优值。
5. 只有 D1 证明 RRF 融合是稳定瓶颈时才重新打开 k 调参。
```

因此生产链路中：

> **B3 与 B4 可以由一个 Retrieval Service 在一次 Milvus `hybrid_search` 调用中共同完成。**

D1 / profiling 仍可以单独调用 Dense 和 BM25，以得到两路原始 rank / score。

# 24. B3 异常


## 24.1 DenseQueryEncodingError

```text
DenseQueryEncodingError
```

中文含义：

> 查询向量编码失败。

包括但不限于：

```text
Qwen3-Embedding-4B 不可用
GPU 推理失败
完整输入超过 8192 token
输出不是 2560 维
向量出现 NaN / Inf
归一化结果不合法
```

Dense Query 编码失败时：

```text
→ 不发起 hybrid_search
→ 当前 knowledge retrieval 失败
```

## 24.2 HybridSearchError

```text
HybridSearchError
```

中文含义：

> Milvus 原生混合检索没有正常完成。

包括但不限于：

```text
Milvus 连接失败
Dense/BM25 所需索引或 Function 不可用
Collection 未加载
hybrid_search 超时
返回结构非法
```

正常返回 0 条候选不属于 `HybridSearchError`。

B3/B4 只报告结构化内部错误，不负责生成用户侧固定提示语。

# 25. v1 候选数量

第一版正式冻结：

```text
Dense candidate limit = 20
BM25 candidate limit  = 20
```

建议配置：

```python
dense_candidate_limit = 20
bm25_candidate_limit = 20
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `dense_candidate_limit` | Dense 分支按自身排名最多返回 20 条 Leaf |
| `bm25_candidate_limit` | BM25 分支按自身排名最多返回 20 条 Leaf |

两支当前都为 20，但必须保持两个独立配置字段。

禁止合并成：

```text
top_k = 20
```

原因：

> 后续版本可能分别调整两路候选数量。

---

# 26. v1 分数阈值

固定：

```python
dense_score_threshold = None
bm25_score_threshold = None
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `dense_score_threshold` | Dense 候选最低相似度阈值；v1 不启用 |
| `bm25_score_threshold` | BM25 候选最低相关性分数阈值；v1 不启用 |

因此 v1：

```text
只按分支自身排名取 Top-20
不根据绝对 score 再删候选
```

以后只有 D1 / 真实检索 Failure 证明低分候选存在稳定、可复现噪声模式时，才允许评估 score threshold。

---

# 27. 与 B4 / B5 的已确认边界

虽然 B4/B5 的独立实现合同另行冻结，但当前已确认第一版预算：

```text
Dense Top-20
+
BM25 Top-20
        ↓
B4 RRF
        ↓
RRF Top-20
        ↓
B5 Cross Encoder
```

建议下游配置名称：

```python
fusion_candidate_limit = 20
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `fusion_candidate_limit` | B4 RRF 融合后最多保留 20 条候选交给 B5 Cross Encoder |

注意：

```text
dense_candidate_limit
bm25_candidate_limit
fusion_candidate_limit
```

虽然 v1 都是 20，但必须是三个独立参数。

`fusion_candidate_limit` 属于 B4，不由 B3 实现。

---

# 28. Observability


生产 v1 每次运行至少记录：

```text
request_id
task_id
retrieval_contract_version
dense_encode_latency_ms
hybrid_search_status
hybrid_search_latency_ms
hybrid_output_count
error_type
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `request_id` | 当前用户请求编号 |
| `task_id` | 当前 knowledge 子任务编号 |
| `retrieval_contract_version` | 当前使用的 B3 检索合同版本 |
| `dense_encode_latency_ms` | Query 生成 Dense 向量所花时间 |
| `hybrid_search_status` | 本次 Milvus 混合检索整体状态 |
| `hybrid_search_latency_ms` | Milvus `hybrid_search` 调用耗时 |
| `hybrid_output_count` | RRF 后实际返回候选数量 |
| `error_type` | 如果失败，记录内部错误类型 |

固定：

```text
retrieval_contract_version = "b3-contract-v1"
```

线上默认不要求额外执行单路 search 只是为了获得：

```text
Dense rank / score
BM25 rank / score
```

这些字段在：

```text
D1 stage-wise evaluation
profiling
debug mode
```

中通过独立单路检索获得并保存。

B3 observability 默认不重复保存完整 Query 正文，通过：

```text
request_id
task_id
```

与 B1/B2/D1 日志关联。

# 29. A6 依赖边界

B3 逻辑合同依赖 A6 最终提供：

```text
1. Leaf 可按 chunk_id 唯一识别。
2. 能返回 B3 RetrievalCandidate 所需 Leaf 字段。
3. Dense document vector 与 A5 完全一致：
   Qwen3-Embedding-4B / 2560-d / L2 normalized。
4. Dense 检索语义为 IP。
5. Native BM25 基于 Leaf.content。
6. v1 集成 baseline 使用统一 ICU Analyzer。
7. 能分别执行 Dense 与 BM25 查询并返回各自有序结果。
```

但以下内容不是 B3 的权威定义：

```text
Milvus Collection 名称
物理字段名
Milvus datatype
VARCHAR max_length
Primary Key 物理配置
BM25 Function schema
Collection build version
rebuild / upsert
重复入库防护
Section Store
文档版本/有效期等 A6 文档级 metadata schema
```

这些由 A6 合同单独冻结。

---

# 30. Explicitly Forbidden

B3 v1 禁止：

```text
1. 自行拆解 B2 Query。
2. 自行 Rewrite / Translation / Expansion。
3. Dense instruction 写回 B2。
4. Dense instruction 用于 BM25。
5. 应用层自行做 ICU 分词。
6. 维护第二套 BM25 主索引。
7. Dense/BM25 score 直接加权融合。
8. 任一路 FAILED 后使用另一支单路结果。
9. Dense 模型失败后自动换 0.6B。
10. 自动 CPU fallback。
11. silent truncation。
12. B3 application-level 自动重试。
13. B3 不得自行更改 B4 的 RRF 方法或参数。
14. 在 B3 内实现 Cross Encoder。
15. 在 B3 内实现 Parent / Section Recovery。
```

---

# 31. B3 Final Baseline


```text
Input:
    one B2 ProcessedQuery
    retrieval_query_count = 1

Retrieval Entity:
    Leaf

Identity:
    chunk_id

Dense:
    Query instruction = frozen
    Qwen3-Embedding-4B
    left padding
    last-token pooling
    L2 normalize
    2560-d float32
    IP
    candidate limit = 20
    score threshold = OFF

BM25:
    original B2 retrieval query
    Milvus Native BM25
    Leaf.content
    unified ICU integration baseline
    k1 = 1.2
    b = 0.75
    candidate limit = 20
    score threshold = OFF

Logical Execution:
    Dense + BM25 两路同时开启

Physical Runtime:
    one Milvus hybrid_search
    ├── Dense AnnSearchRequest(limit=20)
    ├── BM25 AnnSearchRequest(limit=20)
    └── B4 RRFRanker(k=60)
        ↓
    final limit = 20

Failure:
    Dense Query 编码失败
    → 不发起 hybrid_search

    hybrid_search FAILED
    → retrieval 整体失败
    → no single-branch fallback
    → do not enter B5/B6

Evaluation:
    D1 / profiling 可独立执行 Dense/BM25 search
    以保存两路原始 rank / score
```

# 32. B3 Frozen Invariants


```text
1. B3 v1 只消费一个 B2 retrieval query。
2. B3 不负责 Query 拆解、改写、翻译或扩展。
3. 第一阶段 Retriever Candidate 只能是 Leaf。
4. Retrieval identity = chunk_id。
5. Dense/BM25 文档正文 baseline 都是 Leaf.content。
6. Dense instruction 只属于 Dense model input serialization。
7. Dense Query encoder 与 A5 document encoder 保持同一数值合同。
8. Dense 向量固定 2560-d、float32、L2 normalized。
9. Dense metric = IP。
10. BM25 直接使用原 B2 query，不做应用层预处理。
11. Native BM25 的 Analyzer 由 A6 物理配置负责；B3 不自行分词。
12. 逻辑上 Dense 与 BM25 两路同时启用。
13. v1 物理实现优先使用一个 Milvus hybrid_search。
14. Dense AnnSearchRequest candidate limit = 20。
15. BM25 AnnSearchRequest candidate limit = 20。
16. Dense/BM25 score threshold 均关闭。
17. B3 不要求线上物化两份独立 BranchRetrievalResult。
18. D1 / profiling 可通过独立单路 search 保存 Dense/BM25 rank 与原始 score。
19. Dense Query 编码失败时不得发起 Hybrid Search。
20. hybrid_search FAILED → 整体 retrieval FAILED。
21. 失败后禁止 Dense-only / BM25-only fallback。
22. EMPTY 是合法空结果，不等于 FAILED。
23. B4 负责定义 RRF；v1 使用 Milvus RRFRanker(k=60)。
24. B4 v1 融合后 final limit = 20，交给 B5 Cross Encoder。
25. B3/B4 application-level 自动重试关闭。
26. B3 不生成用户侧错误文案。
27. B3 不复制 A6 Collection 物理 schema。
28. 所有行为变化必须升级 b3 contract / component version。
```

# 33. 当前待后续模块处理

不属于 B3 未完成项：

```text
A6：
正式 Collection schema
文档 metadata
字段 datatype
Analyzer 的正式物理绑定
Collection version
rebuild / upsert
Section Store

B4：
Milvus RRFRanker(k=60) 的正式调用合同
fusion_candidate_limit = 20
RRF 输出结构与运行日志

B5：
Cross Encoder 模型
20 → 最终多少条
重排接口和失败合同

B6：
Section / Parent Context Recovery
```

B3 v1 至此冻结。
