# B3 / B4 / B5 实现交接（2026-09-11）

> 本文是实现验收记录，不是需求主文档。  
> B3 权威需求：`docs/Preprocessing/2026-09-10-B3_Hybrid_Retrieval_Frozen_Requirements_v1.md`  
> B4 权威需求：`docs/Preprocessing/2026-09-10-B4_RRF_Fusion_Frozen_Requirements_v1.md`  
> B5 权威需求：`docs/Preprocessing/2026-09-10-B5_Cross_Encoder_Reranking_Frozen_Requirements_v1.md`  
> `docs/retrieval.md` 只是高层设计摘要；与冻结合同冲突时以 Preprocessing 合同为准。

## 当前验收状态

```text
B3 DESIGN
FROZEN / PASS
- 只消费一个 B2 identity retrieval query
- Dense adapter 复用 A5 encode_documents()，不复制 pooling / normalize
- BM25 使用 B2 原文；Dense instruction 不得进入 BM25
- 无 status filter；默认检索集合 invariant 由 A6 负责

B4 DESIGN
FROZEN / PASS
- 一次 Milvus hybrid_search + RRFRanker(k=60)
- 应用层不重算 RRF，不二次排序
- fused_rank = Milvus 返回顺序（1-based）
- output_fields 一次拿齐 Leaf 字段

B5 DESIGN
FROZEN / PASS
- Qwen3-Reranker-0.6B
- 官方 prefix / Instruct-Query-Document body / suffix
- truncation=False；超 4096 → RerankerInputTooLongError
- 排序：rerank_score DESC，fused_rank ASC；Top-5

B3/B4/B5 IMPLEMENTATION
PASS（机制级）
- code/retrieval/b3, b4, b5
- 未改 A0–A6 行为合同；未进入 B6

B3/B4/B5 UNIT TEST
PASS with documented skip
- Windows: 68 passed, 2 skipped
- skip = 两条 B5 打分排序用例需要 torch；本机 retrieval 环境无 torch
- 空结果 / 超长输入 / 官方模板 / hybrid_search 合同用例均不 skip

B3/B4/B5 CLOSED
NOT CLOSED
- 真实 Milvus hybrid_search smoke 未跑
- 真实 Qwen3-Reranker-0.6B 服务器验收未跑
```

---

## 1. 完成结论

| 项 | 结果 |
|---|---|
| 范围 | B3 Dense adapter + B3/B4 一次 hybrid_search + B5 Cross-Encoder |
| 代码位置 | 新建 `code/retrieval/b3`、`b4`、`b5`；**未修改** A0–A6 检索行为 |
| 本机 pytest | **68 passed / 2 skipped / 0 failed**（`python -m pytest retrieval/tests -q`，cwd=`code/`） |
| 真实 hybrid_search smoke | 未执行 |
| 真实 reranker smoke | 未执行。`python retrieval/verify_b5_reranker.py --check-manifest-only` → `manifest_check=ok` |
| CLOSED | **不申请** |

开工前批复已落实：

1. B4/B5 Frozen Requirements 已在 `docs/Preprocessing/`，`docs/README.md` 已索引 B4。
2. B3 **不**新增 `status` filter。
3. B3 Query Encoder 是 A5 public `encode_documents()` adapter。
4. `hybrid_search_timeout_ms / 1000.0` 显式换成 PyMilvus seconds。
5. B5 复制官方 token template，禁止官方示例里的 `truncation='longest_first'`。
6. Qwen3-Reranker-0.6B revision pin：`e61197ed45024b0ed8a2d74b80b4d909f1255473`。

---

## 2. 修改 / 新增文件

新增：

```text
code/retrieval/b3/
code/retrieval/b4/
code/retrieval/b5/
code/retrieval/tests/test_b3_serialize.py
code/retrieval/tests/test_b3_plan.py
code/retrieval/tests/test_b3_encoder.py
code/retrieval/tests/test_b3_config.py
code/retrieval/tests/test_b3_b4_hybrid.py
code/retrieval/tests/test_b4_parse.py
code/retrieval/tests/test_b5_template.py
code/retrieval/tests/test_b5_rerank.py
code/retrieval/tests/test_b5_config.py
code/retrieval/verify_b5_reranker.py
docs/proceeding/2026-09-11-b3-b5-implementation.md
```

改过（状态/索引/pin）：

```text
agent.md
docs/README.md
docs/retrieval.md          # B4 节补权威合同路径
models/retrieval-models.manifest.json
code/retrieval/__init__.py
code/retrieval/tests/helpers.py
```

明确未改：

```text
code/knowledge_base/**
code/preprocessing/**
code/C8/**
code/C9/**
docs/Preprocessing/2026-09-10-B3_Hybrid_Retrieval_Frozen_Requirements_v1.md
docs/Preprocessing/2026-09-10-B4_RRF_Fusion_Frozen_Requirements_v1.md
docs/Preprocessing/2026-09-10-B5_Cross_Encoder_Reranking_Frozen_Requirements_v1.md
```

本轮不做 `SPLIT-MANIFEST.md` / `architecture.md` / `evaluation.md` 的 stale 清理；不阻塞本实现。

---

## 3. 生产边界

```text
ProcessedQuery
  ↓
B3 serialize_query()
  ↓
Qwen3DenseEncoder.encode_documents([serialized_query])[0]
  ↓
Milvus hybrid_search(
    Dense AnnSearchRequest(limit=20),
    BM25 AnnSearchRequest(limit=20),
    ranker=RRFRanker(60),
    limit=20,
    output_fields=Leaf fields,
    timeout=hybrid_search_timeout_ms/1000.0,
)
  ↓
B4 FusionResult（Milvus 顺序）
  ↓
B5 RerankingResult（最多 5 个 Leaf）
```

禁止：

```text
B3 status filter
应用层第二套 RRF
hybrid_search 失败后 Dense-only / BM25-only
B5 silent truncation
B5 失败后拿 RRF Top-5 进 B6
给 A5 新增 encode_query
调用 A5 私有 _encode_batch()
```

公开配置：

| 环境变量 | 含义 |
|---|---|
| `B3_MILVUS_URI` | Milvus URI |
| `B3_COLLECTION_NAME` | 生产 Leaf Collection |
| `B3_HYBRID_SEARCH_TIMEOUT_MS` | 必填；毫秒。代码转换成秒再传给 PyMilvus |
| `B5_MODEL_NAME_OR_PATH` | Qwen3-Reranker-0.6B 本地路径或 hub id |
| `B5_MODEL_REVISION` | 可选，默认 pin `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| `B5_BATCH_SIZE` | 可选，默认 8；OOM 不得自动降 batch |

`hybrid_search_timeout_ms` 没有算法最优默认值，必须显式配置。

---

## 4. 行为要点

### 4.1 B3 adapter

```python
text = serialize_query(query)
vector = base_encoder.encode_documents([text])[0]
```

instruction 只属于 Dense serialization。BM25 `query_text` 仍是 B2 原文。

### 4.2 Hybrid Search

`output_fields` 一次返回：

```text
chunk_id
document_id
section_id
chunk_index
page_start
page_end
content
```

`hit distance/score → rrf_score`；返回顺序 → `fused_rank`。

A6 负责默认 Collection 只保留 `ACTIVE + VERSION_CONFLICT`。B3 不写 `status` / `expr` / `filter`。

### 4.3 Timeout

```python
timeout_seconds = hybrid_search_timeout_ms / 1000.0
```

`5000` 表示 5 秒，不是 5000 秒。

### 4.4 B5 template

官方 prefix / suffix，含 `<think></think>`。完整 tokenization `truncation=False`。`total_tokens > 4096` 失败，不截断。

score = 官方 yes/no logits 的 `P(yes)`。

---

## 5. 测试矩阵（Windows 本机）

```text
python -m pytest retrieval/tests -q
# 68 passed, 2 skipped
```

| 文件 | 覆盖 |
|---|---|
| `test_b3_serialize.py` | 冻结 instruction、不改写 Query |
| `test_b3_plan.py` | 单 query、独立 20/20 limit |
| `test_b3_encoder.py` | adapter 只调 encode_documents；不调 `_encode_batch` |
| `test_b3_config.py` | timeout 必填；5000ms → 5.0s |
| `test_b3_b4_hybrid.py` | 无 status filter、完整 output_fields、BM25 原文、空结果、不重试、不按 score 重排 |
| `test_b4_parse.py` | fused_rank、duplicate、缺 content、超过 20 |
| `test_b5_template.py` | 官方 prefix/suffix/body；revision pin |
| `test_b5_rerank.py` | 空输入不打模型；超长失败且 truncation=False；score 排序需 torch |
| `test_b5_config.py` | 模型路径必填；manifest revision |

---

## 6. 明确未做

```text
真实 Milvus hybrid_search smoke
真实 Qwen3-Reranker-0.6B CUDA 加载 / 打分验收
B6 Context Recovery
D1 单路 Dense/BM25 profiling 接口物化
SPLIT-MANIFEST / architecture.md / evaluation.md stale 清理
```

---

## 7. 上游验收口径

```text
B3：实现完成，NOT CLOSED
B4：实现完成，NOT CLOSED
B5：实现完成，NOT CLOSED
```

真实 smoke 通过后可记 `B3/B4 IMPLEMENTATION COMPLETE` 与 `B5 IMPLEMENTATION COMPLETE`，仍不 CLOSED，直到 D1 / 服务器验收集冻结。
