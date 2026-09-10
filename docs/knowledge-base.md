# Knowledge Base — A1 to A6

## 1. Purpose

本模块把 A0 正式发布的 `document.md` 建成可检索知识库。

```text
document.md
→ A1 Loading
→ A2 Minimal Provenance
→ A3 Leaf Chunk
→ A4 Section Hierarchy
→ A5 Dense Embedding
→ A6 Milvus
```

知识库不重新解析 raw PDF。

---

# A1 — Markdown Loading

状态：`[FROZEN / IMPLEMENTED]`

## A1.1 Canonical Input Root

A1 接收明确指定的正式 A0 输出根目录，并批量发现：

```text
<canonical_root>/<document_id>/document.md
```

禁止扫描 `.failed/`、`.staging/`、smoke / acceptance / temporary output。

## A1.2 Lossless Loading

A1 保留：

```text
Markdown headings
<!-- PDF page N -->
paragraph order
lists
trusted captions
admitted Markdown tables
```

A1 不做 PDF parse、OCR、正文重写、Chunk、Embedding 或 table reparse。

---

# A2 — Minimal Provenance / Identity

状态：Document Registry `[FROZEN / IMPLEMENTED]`；Leaf identity 字段契约由 A3/A4 实现。

## A2.1 Document Registry

A2 的最小 Registry 仍只保存：

```text
document_id
document_title
source
```

这条设计没有因为 A6 增加生产治理字段而失效。

A6 的：

```text
file_name
document_number
document_version
finalized_at
effective_from
effective_to
ingested_at
status
```

属于**生产索引与文档治理字段**，不等于把 A2 Registry 扩展为同一张大表。A2 继续负责最小 provenance / identity；A6 负责可检索版本治理与 Collection 侧持久化。

## A2.2 Join

显式提供 manifest 时：

```text
quality_report.source_sha256 == manifest.sha256
```

必须完整 SHA-256 精确匹配，不得用 filename 或 document_id hash 前缀模糊 join。

未提供 manifest 时，A2 才从 `quality_report.source` 做跨平台 basename fallback。

## A2.3 Leaf Identity / Provenance

Leaf 需要：

```text
chunk_id
document_id
section_id
chunk_index
page_start
page_end
```

用途：

| 字段 | 消费者 |
|---|---|
| `chunk_id` | Retrieval identity / dedup / log / D1 runtime result |
| `document_id` | B6 / C3 / D1 source scope |
| `section_id` | B6 hierarchy lookup |
| `chunk_index` | B6 neighbor fallback |
| `page_start/page_end` | C3 citation / D1 provenance |

`document_id` 是 source-document stable；`section_id/chunk_id` 允许 build-specific，因此 D1 Gold 不得绑定 `chunk_id`。

---

# A3 — Structure-aware Chunking

状态：`[FROZEN / IMPLEMENTED]`

## A3.1 Principle

```text
Structure first
Length fallback
```

先按真实 Markdown heading 结构划分；只有末级 Section 过长才做 tokenizer-aware fallback。

## A3.2 Baseline

```text
chunk_size = 768
overlap_tokens = 96
```

Tokenizer：

```text
Qwen/Qwen3-Embedding-4B
transformers.AutoTokenizer
```

Markdown block-aware greedy packing；只有单个非-table block 本身超过 `chunk_size` 时才做 sliding fallback。Overlap 不跨 Section。

## A3.3 Leaf Content

`Leaf.content` 是第一阶段 Dense/BM25 Retrieval 与 B5 Rerank 的最小正文实体。

当前 baseline：

```text
Dense document text = Leaf.content
BM25 indexed text   = Leaf.content
```

禁止额外拼：

```text
heading
section path
document title
page marker
```

`<!-- PDF page N -->` 只用于 `page_start/page_end`，不进入 `Leaf.content`。

合法 Markdown table 不得因为 chunk_size 在任意行中间切断；单张超长 table 可以形成 table-only oversize Leaf。

---

# A4 — Hierarchical Parent-Child

状态：多级 hierarchy `[FROZEN / IMPLEMENTED]`；Section text materialization `[FROZEN / IMPLEMENTED]` source span recovery；B6 exact ancestor selection 与生产 Section Store 介质仍 `[PROVISIONAL]`。

## A4.1 Hierarchy

保持真实 Markdown 多级结构，不固定 H2/H3，不补 synthetic heading。

Leaf：

```text
section_id = nearest semantic Section
```

B6 必须能够：

```text
Leaf
→ nearest Section
→ parent Section
→ ...
```

## A4.2 Section Text Materialization

Section 使用 trusted `document.md` 的 source span 恢复，不从 Leaf 拼接 Parent，也不持久化完整 `Section.content`。

当前实现为 in-memory `SectionHierarchy`；JSON artifact 只用于 smoke/debug。生产 Section Store 介质仍未冻结。

Neighbor 通过：

```text
document_id + chunk_index
```

推导，不要求持久化 prev/next chunk id。

---

# A5 — Dense Embedding

状态：document-side Dense Embedding `[FROZEN / IMPLEMENTED]`；query-side instruction wording 已由 B3 冻结。

实现入口：`code/knowledge_base/dense_embedding.py`。

## A5.1 Model

```text
Qwen3-Embedding-4B
Transformers AutoTokenizer + AutoModel
```

文档端：

```text
Leaf.content
→ left padding
→ last-token pooling
→ L2 normalize
→ 2560-d float32
```

不添加 document instruction。

`Qwen3-Embedding-0.6B` 不是 runtime fallback；若切换模型，必须重新定义维度并全量 rebuild。

## A5.2 Input Budget

```text
max_input_tokens = 8192
```

禁止 silent truncation；超过上限必须失败。

Baseline 部署：

```text
device = cuda
forward dtype = FP16
output dtype = FP32
batch_size = 4
```

## A5.3 Query Side

Query serialization 属于 B3。当前 B3 已冻结英文 retrieval instruction，并复用 A5-compatible encoder 数值核心；A5 `encode_documents()` 不得因此自动给 document 添加 instruction。

---

# A6 — Production Milvus Collection

状态：`[FROZEN / IMPLEMENTED / REAL-CORPUS-ACCEPTED]`

正式实现：

```text
code/knowledge_base/a6_document_metadata.py
code/knowledge_base/a6_collection.py
code/knowledge_base/a6_prepare_rows.py
```

真实生产验收记录见：

```text
docs/proceeding/2026-09-08-a6.0-analyzer-profile.md
docs/proceeding/2026-09-09-a6-production-collection.md
docs/proceeding/2026-09-10-a6-production-real-ingestion.md
```

## A6.1 Vector DB / Collection

固定：

```text
Milvus
Leaf-only retrieval collection
Dense + Native BM25 in the same collection
```

当前真实验收环境：

```text
Milvus 2.5.14
production collection: a6_leaf_v1
17 retrievable documents
2034 Leaf rows
```

Section/Parent 不作为第一阶段 Dense/BM25 Candidate。

## A6.2 Production Schema

生产 Collection 包含 Leaf 检索字段和文档治理字段。

Leaf / retrieval 核心：

```text
chunk_id
document_id
section_id
chunk_index
page_start
page_end
content
dense_vector
sparse_vector
```

文档治理 / 可检索控制：

```text
file_name
document_title
document_number
document_version
finalized_at
effective_from
effective_to
ingested_at
status
```

`sparse_vector` 由 Milvus BM25 Function 自动生成，不由应用层手工写入。

## A6.3 Document Governance

A6 文档治理支持：

```text
ACTIVE
VERSION_CONFLICT
HISTORICAL
DUPLICATE
```

默认 Collection 只保留：

```text
ACTIVE
VERSION_CONFLICT
```

`HISTORICAL` 与 `DUPLICATE` 不进入默认检索集合。

治理依据包括：

```text
source_sha256
document_content_hash
document_number
finalized_at
title fallback（仅在 document_number 缺失时）
```

字段抽取来自正文显式证据与固定规则，不允许从文件名推断 document number / version。

## A6.4 Dense Baseline

```text
FLAT + IP
2560-d
```

Embedding 已在 A5 L2 normalize，A6 不二次归一化。

当前 FLAT 用作无 ANN 近似误差的准确率 baseline。未来数据规模或 latency 成为问题时，优先评估：

```text
FLAT → IVF_FLAT
```

不默认迁移 HNSW。

## A6.5 Sparse Baseline

```text
Milvus Native BM25
SPARSE_INVERTED_INDEX
UNIFIED_ICU
k1 = 1.2
b = 0.75
DAAT_MAXSCORE
```

这些参数已由 A6.0 profiling 冻结并被正式 A6 直接复用，不再是 `[PROVISIONAL]`。

B3 应直接查询 Native BM25，不在应用层维护第二套主 BM25 或重新分词。

## A6.6 Strict Join

A6 入库前必须严格校验：

```text
Leaf ↔ Dense embedding
Leaf document_id ↔ retrievable metadata
```

missing / extra / duplicate 任一不为 0 都 fail，不做部分插入。

## A6.7 Lifecycle

以下生命周期已经实现并验收：

```text
build
ingest-update
rebuild
```

规则：

- `build`：Collection 已存在则失败，不静默覆盖。
- `ingest-update`：只删除并重插受影响 document_id。
- `rebuild`：必须显式确认，防止误全量重建。
- 从可检索状态转为 `HISTORICAL/DUPLICATE` 时删除旧 Leaf。
- `VERSION_CONFLICT` 文档双方都可保留检索。

因此旧版“Rebuild / Upsert `[PROVISIONAL]`”已失效。

## A6.8 Verification

真实生产验收已经跑通：

```text
verify_collection
Dense smoke
BM25 smoke
```

当前真实基线：

```text
17 documents
2034 Leaf
join_missing_count = 0
join_extra_count = 0
verification_passed = True
Dense smoke = PASS
BM25 smoke = PASS
```

## A6.9 Section Store

Section hierarchy 不参与第一阶段向量检索。

生产 Section Store 的具体持久化介质仍：

```text
[PROVISIONAL / implementation-open]
```

A6 Production Collection 完成不等于该项已经冻结。

---

# Knowledge-base Invariants

1. A1 只吃正式 `document.md`。
2. A2 Registry 保持最小 provenance；A6 文档治理字段属于独立生产索引职责。
3. 先 Heading structure，末级过长才长度切分。
4. `chunk_size=768`、`overlap_tokens=96` 为当前 baseline，Overlap 不跨 Section。
5. Leaf 是唯一第一阶段 Retriever Candidate。
6. 多级 Section hierarchy 必须可恢复。
7. Dense/BM25 baseline 都使用 `Leaf.content`，不额外拼 heading；page marker 不进入 content。
8. Qwen3-Embedding-4B + last-token pooling + left padding + 2560d + L2 normalize；8192 inference budget，禁止 silent truncation。
9. A6 使用 Milvus 单一 Leaf Collection：Dense `FLAT/IP` + Native BM25。
10. Native BM25 使用 `UNIFIED_ICU + k1=1.2 + b=0.75 + DAAT_MAXSCORE`。
11. 默认检索文档状态仅 `ACTIVE + VERSION_CONFLICT`。
12. A6 build / ingest-update / rebuild 已冻结实现并通过真实 Milvus 验收。
13. 当前不使用 HNSW；future ANN 优先 IVF_FLAT。
14. 生产 Section Store 介质仍未冻结。
