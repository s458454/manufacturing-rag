# B4 RRF Fusion — Frozen Requirements v1

> Status: **FROZEN**
>
> Contract version: `b4-contract-v1`
>
> 本文档是 B4 RRF Fusion v1 的权威需求。
>
> B4 负责定义 Dense 与 BM25 两路候选如何通过 Reciprocal Rank Fusion（RRF）
> 进行融合，并输出最多 20 个 Leaf 给 B5 Cross Encoder。
>
> v1 线上物理实现优先使用 Milvus 2.5.x 原生：
>
> ```text
> hybrid_search + RRFRanker
> ```
>
> B3 与 B4 保持逻辑模块边界，但不强制拆成多次 Milvus 网络调用。

---

# 1. B4 在整条链路中的位置

```text
B1 Query Router
    ↓
B2 Query Processing
    ↓
B3 Dense + BM25
    ↓
B4 RRF Fusion
    ↓
B5 Cross Encoder
    ↓
B6 Context Recovery
```

逻辑上：

```text
B3：
构造 Dense 检索请求
+
构造 BM25 检索请求

B4：
定义两路如何通过 RRF 融合
```

v1 线上物理执行：

```text
Dense AnnSearchRequest(limit=20)
+
BM25 AnnSearchRequest(limit=20)
        ↓
Milvus hybrid_search
        ↓
RRFRanker(k=60)
        ↓
final limit = 20
        ↓
B5 Cross Encoder
```

---

# 2. Fusion Method

v1 唯一 baseline：

```text
RRF
```

全称：

```text
Reciprocal Rank Fusion
```

中文含义：

> 倒数排名融合。只根据一个候选在各检索分支中的名次进行融合，
> 不直接比较 Dense 与 BM25 的原始分数尺度。

禁止 v1 使用：

```text
Dense score + BM25 score
min-max score fusion
z-score fusion
softmax score fusion
手工加权 score fusion
```

---

# 3. RRF 公式

RRF 逻辑公式：

\[
S(d)=\sum_i \frac{1}{k+\operatorname{rank}_i(d)}
\]

其中：

| 符号 | 中文含义 |
|---|---|
| `d` | 某个 Leaf 候选 |
| `i` | Dense 或 BM25 某一路检索 |
| `rank_i(d)` | 当前 Leaf 在该分支中的排名 |
| `k` | RRF 平滑参数 |
| `S(d)` | 当前 Leaf 的 RRF 融合分数 |

规则：

```text
rrf_score 越大
→ 融合排名越靠前
```

如果某 Leaf 只出现在一个分支：

```text
只计算已命中分支的贡献
```

未命中的分支：

```text
不虚构 rank
不按 rank=21 处理
不额外补分
```

---

# 4. rrf_k

v1 固定：

```python
rrf_k = 60
```

中文含义：

> `rrf_k`：RRF 的排名平滑参数，控制不同名次之间贡献差异的强弱。

选择理由：

```text
1. Milvus 2.5.x 官方默认值。
2. 当前项目没有实验依据支持自定义为 55。
3. v1 优先采用成熟组件默认参数建立可复现 baseline。
4. 不声称 k=60 是当前语料最优值。
5. 只有 D1 证明 RRF 融合是稳定瓶颈时才重新打开参数调优。
```

---

# 5. Candidate Budget

v1 固定：

```python
dense_candidate_limit = 20
bm25_candidate_limit = 20
fusion_candidate_limit = 20
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `dense_candidate_limit` | Dense 分支最多提供 20 个 Leaf 候选参与融合 |
| `bm25_candidate_limit` | BM25 分支最多提供 20 个 Leaf 候选参与融合 |
| `fusion_candidate_limit` | RRF 融合后最多保留 20 个 Leaf 交给 B5 |

虽然三个值当前都是 20，但必须保留三个独立参数。

禁止合并为：

```text
top_k = 20
```

原因：

> 后续版本可能分别调整 Dense、BM25 和融合后候选预算。

---

# 6. 生产物理实现

v1 优先使用：

```python
hybrid_search(
    reqs=[
        dense_request,
        bm25_request,
    ],
    ranker=RRFRanker(60),
    limit=20,
)
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `dense_request` | Dense 稠密向量检索请求，内部候选上限 20 |
| `bm25_request` | BM25 Native BM25 检索请求，内部候选上限 20 |
| `RRFRanker(60)` | Milvus 原生 RRF 融合器，参数 60 |
| `limit=20` | 融合后最终最多返回 20 条 |

线上 v1：

```text
Milvus RRFRanker 的返回结果
= B4 权威融合结果
```

应用层禁止再次重新计算一套 RRF。

---

# 7. WeightedRanker

v1：

```text
WeightedRanker = OFF
```

中文含义：

> 不人为规定 Dense 或 BM25 哪一路更重要。

当前没有证据支持：

```text
Dense > BM25
```

或：

```text
BM25 > Dense
```

因此两路保持对称融合。

只有 D1 / 真实 Failure 证明需要显式偏置某一路时，才允许重新评估 WeightedRanker。

---

# 8. 输出候选结构

v1 固定：

```python
@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str
    rrf_score: float
    fused_rank: int
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `chunk_id` | 当前 Leaf 的唯一检索编号 |
| `document_id` | 当前 Leaf 所属文档编号 |
| `section_id` | 当前 Leaf 所属章节编号 |
| `chunk_index` | 当前 Leaf 在所属文档中的顺序编号 |
| `page_start` | 当前 Leaf 对应原文起始页 |
| `page_end` | 当前 Leaf 对应原文结束页 |
| `content` | 当前 Leaf 正文，供 B5 Cross Encoder 使用 |
| `rrf_score` | Milvus RRF 返回的融合相关性分数 |
| `fused_rank` | 当前 Leaf 在 RRF 最终返回列表中的排名，从 1 开始 |

B4 不要求生产结果携带：

```text
dense_rank
dense_score
bm25_rank
bm25_score
```

这些原始分支信息由 D1 / profiling 单独获取。

---

# 9. fused_rank

固定：

```text
Milvus RRFRanker 返回顺序
= B4 最终排序顺序
```

因此：

```python
for index, hit in enumerate(hits, start=1):
    fused_rank = index
```

中文含义：

> Milvus 返回第几个，该候选的 `fused_rank` 就是多少。

v1 不在应用层增加第二套 tie-break 规则。

禁止：

```text
Milvus RRF 后再按 chunk_id 排序
Milvus RRF 后再按 Dense score 排序
Milvus RRF 后再按 BM25 score 排序
Milvus RRF 后再人为优先 Dense/BM25
```

如果未来 D1 证明完全同分候选在 Top-20 边界存在实际不稳定问题，再单独重新冻结 tie-break 机制。

---

# 10. FusionResult

固定：

```python
@dataclass(frozen=True)
class FusionResult:
    request_id: str
    task_id: str
    query: str
    candidates: tuple[FusedCandidate, ...]
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `request_id` | 当前用户请求编号 |
| `task_id` | 当前 knowledge 子任务编号 |
| `query` | 当前实际检索问题，继续传给 B5 |
| `candidates` | RRF 融合后的 Leaf 候选列表，最多 20 条 |

`query` 必须保留，因为 B5 Cross Encoder 需要：

```text
Query + Leaf.content
```

做联合相关性判断。

---

# 11. 输出数量

固定：

```text
0 <= len(candidates) <= 20
```

如果 Dense/BM25 两路高度重合：

```text
融合后可能不足 20 个不同 Leaf
```

此时：

```text
返回实际数量
```

禁止：

```text
复制候选补满 20
返回空占位符
重新向 Retriever 要更多结果
```

---

# 12. EMPTY

如果 Milvus Hybrid Search 正常完成，但最终返回：

```text
0 candidate
```

则：

```text
EMPTY
```

是合法业务结果。

允许：

```python
FusionResult(
    request_id=...,
    task_id=...,
    query=...,
    candidates=(),
)
```

必须严格区分：

```text
EMPTY ≠ FAILED
```

EMPTY 不进入正常 B5 rerank 流程。

具体用户侧回复由更上层模块定义，B4 不生成业务文案。

---

# 13. FAILED

v1 生产路径中 B3+B4 共享同一次 Hybrid Search。

以下均归为：

```text
HybridSearchError
```

中文含义：

> Dense + BM25 + RRF 的混合检索没有正常完成。

包括但不限于：

```text
Milvus 连接失败
Collection 不可用
Dense 索引不可用
BM25 Function 不可用
RRFRanker 执行失败
请求超时
返回结构非法
```

失败后：

```text
→ 当前 knowledge retrieval 失败
→ 不进入 B5
→ 不进入 B6
```

禁止：

```text
Hybrid Search 失败
→ Dense-only fallback

Hybrid Search 失败
→ BM25-only fallback
```

---

# 14. Application Retry

v1：

```text
B3/B4 application retry = OFF
```

中文含义：

> 应用层不会因为一次 `hybrid_search` 失败而自动重新执行相同检索。

未来如果需要基础设施级瞬时重试，必须单独冻结。

---

# 15. Observability

生产 v1 每次至少记录：

```text
request_id
task_id
fusion_contract_version
rrf_k
dense_candidate_limit
bm25_candidate_limit
fusion_candidate_limit
hybrid_search_latency_ms
fused_candidate_count
fusion_status
error_type
```

字段中文说明：

| 字段 | 中文含义 |
|---|---|
| `request_id` | 当前用户请求编号 |
| `task_id` | 当前 knowledge 子任务编号 |
| `fusion_contract_version` | 当前使用的 B4 融合合同版本 |
| `rrf_k` | 当前 RRF 平滑参数，v1 为 60 |
| `dense_candidate_limit` | Dense 最大候选数，v1 为 20 |
| `bm25_candidate_limit` | BM25 最大候选数，v1 为 20 |
| `fusion_candidate_limit` | RRF 最大输出数，v1 为 20 |
| `hybrid_search_latency_ms` | 整个 Milvus 混合检索调用耗时 |
| `fused_candidate_count` | 实际返回的融合候选数 |
| `fusion_status` | 本次融合结果状态：成功、空结果或失败 |
| `error_type` | 如果失败，记录内部错误类型 |

固定：

```text
fusion_contract_version = "b4-contract-v1"
```

生产日志默认不额外保存完整 Query 正文。

---

# 16. D1 / Profiling Observability

线上生产路径不要求为了观测额外执行：

```text
Dense-only search
BM25-only search
```

D1 / profiling 模式允许针对同一 Query 分别运行：

```text
Dense-only Top-20
BM25-only Top-20
Hybrid Search + RRF Top-20
```

并记录：

```text
Dense rank
Dense score
BM25 rank
BM25 score
RRF rank
RRF score
```

用途：

```text
1. 判断相关 Leaf 是否根本没有被 Dense/BM25 召回。
2. 判断正确 Leaf 是否已被单路召回但在 RRF 后掉出。
3. 定位瓶颈属于 B3 还是 B4。
4. 决定未来是否需要调整 candidate limit / rrf_k / WeightedRanker。
```

这些字段不是生产 `FusionResult` 的强制字段。

---

# 17. Explicitly Forbidden

B4 v1 禁止：

```text
1. 应用层重新实现第二套 RRF 并覆盖 Milvus 排序。
2. Dense/BM25 原始 score 直接融合。
3. score normalization 后再融合。
4. WeightedRanker。
5. 人为优先 Dense。
6. 人为优先 BM25。
7. RRF 后应用层再次重排。
8. Hybrid Search 失败后单路降级。
9. 为了补满 20 条而复制/伪造候选。
10. B4 内部执行 Cross Encoder。
11. B4 内部执行 Context Recovery。
12. B4 生成用户侧回答。
```

---

# 18. B4 Final Baseline

```text
Fusion Method:
    Milvus Native RRF

Physical API:
    hybrid_search

Dense Input:
    Top-20

BM25 Input:
    Top-20

Ranker:
    RRFRanker

rrf_k:
    60

WeightedRanker:
    OFF

Final Output:
    Top-20 Leaf

Ranking:
    Milvus returned order

fused_rank:
    1-based

Failure:
    HybridSearchError
    → no single-branch fallback

Empty:
    legal
```

---

# 19. B4 Frozen Invariants

```text
1. Fusion method = RRF。
2. v1 使用 Milvus Native RRFRanker。
3. rrf_k = 60。
4. 60 是 Milvus 官方默认 baseline，不声称为当前语料最优值。
5. Dense candidate limit = 20。
6. BM25 candidate limit = 20。
7. Fusion candidate limit = 20。
8. Dense/BM25 两路等权。
9. WeightedRanker = OFF。
10. 应用层不得自行重新计算 RRF。
11. 应用层不得根据 Dense/BM25 raw score 再做二次融合。
12. Milvus 返回顺序即最终 fused order。
13. fused_rank 从 1 开始，按 Milvus 返回列表赋值。
14. 最多输出 20 个 Leaf。
15. 不足 20 个时返回实际数量。
16. 0 个候选是合法 EMPTY，不是 FAILED。
17. Hybrid Search 失败后禁止单路降级。
18. B4 输出必须保留 Leaf.content，供 B5 Cross Encoder 使用。
19. B4 输出必须继续保留 request_id / task_id / query。
20. Dense/BM25 原始 rank/score 不属于生产路径强制输出，由 D1 profiling 单独获取。
21. 所有不兼容行为变化必须升级 B4 contract version。
```

---

# 20. Explicitly Deferred

```text
WeightedRanker
自定义 Dense/BM25 权重
rrf_k 调优
应用层 tie-break
其他 rank fusion 算法
多于两路的 Retrieval Fusion
```

只有 D1 / 真实 Failure 提供明确证据后才允许进入后续版本。

B4 v1 至此冻结。
