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

状态：`[FROZEN]`

## A1.1 Canonical Input Root

A1 接收明确指定的正式 A0 输出根目录，并批量发现：

```text
<canonical_root>/<document_id>/document.md
```

禁止：

```text
glob("outputs/**/document.md")
```

扫描全部测试/历史产物。

## A1.2 Exclusions

不得读取：

```text
.failed/
.staging/
smoke output
acceptance output
temporary output
```

测试产物是否物理删除不是正确性前提。

## A1.3 Lossless Loading

A1 必须保留：

```text
Markdown headings
<!-- PDF page N -->
paragraph order
lists
trusted captions
admitted Markdown tables
```

A1 不做：

```text
PDF parse
OCR
Markdown→HTML
纯文本化导致结构丢失
正文重写
Chunk
Embedding
table reparse
```

---

# A2 — Minimal Provenance / Identity

状态：`[FROZEN principle]`

## A2.1 Metadata Rule

只有当前链路存在明确消费者的持久化字段才加入。

当前不要求：

```text
organization
material
year
domain
document_type
```

自动抽取或人工强制维护。

## A2.2 Document Registry

文档级信息只保存一次。

至少需要：

```text
document_id
document_title
source
```

其中 `document_id` 是跨建库稳定的文档身份。

`document_title/source` 的可靠来源规则：`[PROVISIONAL]`，实现时不得随意由 LLM 推断。

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

## A2.4 ID Stability

`document_id`：

```text
source-document stable
```

`section_id/chunk_id`：

```text
build-specific is acceptable
```

因为改变 Chunk 参数后，Section/Chunk 边界可能变化。

因此 D1 Gold **不得绑定 chunk_id**。

## A2.5 Derived Data

默认不冗余保存：

```text
prev_chunk_id
next_chunk_id
完整 ancestry path
grandparent content
完整业务 metadata
```

如果可通过：

```text
document_id + chunk_index
section_id + Section hierarchy
```

得到，则不在每条 Leaf 重复写。

Section path 必须可恢复，但不要求复制到每条 Leaf。

---

# A3 — Structure-aware Chunking

状态：`[FROZEN]`

## A3.1 Principle

```text
Structure first
Length fallback
```

不允许 baseline 直接：

```text
whole Markdown
→ fixed 400-token sliding windows
```

## A3.2 Section-first Split

先解析：

```text
# H1
## H2
### H3
...
```

如果存在更低级标题，优先继续使用结构边界。

只有：

```text
末级 Section + 过长
```

才使用 Recursive Token Split。

## A3.3 Leaf Definition

Leaf 是：

> 第一阶段 Dense/BM25 Retrieval 与 Reranking 的最小文本实体。

```text
short terminal Section → 1 Leaf
long terminal Section → N Leaf
```

## A3.4 Chunk Size Profiling

不提前冻结 400/600/800。

必须先统计真实末级 Section token length：

```text
P50
P75
P90
P95
Max
```

再确定：

```text
chunk_size
overlap
```

两者必须配置化。

Tokenizer 口径：`[PROVISIONAL]`。应与实际模型/预算管理保持可解释一致，但当前未冻结具体 tokenizer。

## A3.5 Overlap

只发生在：

```text
same terminal Section
→ Recursive Split
→ adjacent Leaf
```

禁止跨 Heading Section overlap。

## A3.6 Table Atomicity

A0 已决定哪些 table 可以进入正式 Markdown。

A3 baseline：

> 一个合法 Markdown table 不得因为 chunk_size 在任意行中间切断。

异常超长 table 记录为单独 case；当前不设计复杂 table chunker。

## A3.7 Leaf Content

Baseline Dense/BM25 都直接使用：

```text
Leaf.content
```

不默认构造：

```text
retrieval_text = heading + content
```

标题增强只作为未来实验。

`[PROVISIONAL]`：Leaf.content 最终序列化是否包含“当前 Section 自身 heading”，以及 `<!-- PDF page N -->` 是否只转 provenance 而不进入 Retrieval text，需在实现前再确认。

## A3.8 Page Range

A3 必须能从 page markers 推导：

```text
page_start
page_end
```

一个跨页 Leaf 需要正确记录页范围。

---

# A4 — Hierarchical Parent-Child

状态：多级 hierarchy `[FROZEN]`；精确恢复策略 `[PROVISIONAL]`

## A4.1 Hierarchy

```text
Document
└── H1 Section
    └── H2 Section
        └── H3 Section
            ├── Leaf
            └── Leaf
```

不固定：

```text
Parent = H2
```

或：

```text
Parent = H3
```

## A4.2 Relationship

Leaf：

```text
section_id = nearest semantic Section
```

Section：

```text
section_id
parent_section_id
heading
recoverable text/span
```

B6 必须能够：

```text
Leaf
→ nearest Section
→ parent Section
→ ...
```

## A4.3 Section Text Materialization

`[PROVISIONAL]`

系统必须能恢复一个 Section 的完整语义范围，但当前不冻结 Section Store 必须直接存：

```text
content
```

还是存：

```text
source span / child references
```

只要 B6 能可靠恢复：

```text
heading
section text
parent link
page range/path
```

即可。

## A4.4 Neighbor

Neighbor 使用：

```text
document_id + chunk_index
```

推导。

不要求持久化：

```text
prev_chunk_id
next_chunk_id
```

---

# A5 — Embedding

状态：`[FROZEN baseline]`

## A5.1 Model

```text
Qwen3-Embedding-4B
```

Fallback：

```text
Qwen3-Embedding-0.6B
```

只有明确出现加载/显存/吞吐/部署压力时才考虑 fallback。

## A5.2 Capability Requirement

必须兼顾：

```text
Chinese
English
Cross-lingual Retrieval
```

尤其：

```text
Chinese Query → English technical document
```

## A5.3 Document Side

```text
Leaf.content
→ Qwen3-Embedding-4B
→ L2 normalize
→ 2560-d dense vector
```

baseline 不添加 query instruction。

## A5.4 Query Side

```text
English retrieval instruction
+
original query
→ Qwen3-Embedding-4B
→ L2 normalize
```

要求：

- instruction 可配置；
- wording 暂不冻结；
- instruction 不改变 Query 的事实语义；
- instruction 不是 B2 Query Rewrite。

## A5.5 Dimension

```text
2560
```

当前不启用 MRL 降维。

---

# A6 — Milvus Index

状态：`[FROZEN baseline]`

## A6.1 Vector DB

固定使用：

```text
Milvus
```

当前仓库部署为 Milvus Standalone 系列，真实版本以 `code/docker-compose.yml` 为当前实现事实。

## A6.2 Leaf-only Retrieval Collection

Milvus 第一阶段只索引 Leaf。

Section/Parent 不作为 Dense/BM25 Candidate。

## A6.3 Logical Leaf Fields

至少需要：

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

`content` 同时作为 Milvus Native BM25 的输入文本。

## A6.4 Dense + Sparse Same Collection

当前使用单一 Leaf Collection：

```text
dense_vector
+
BM25 sparse field
```

不拆成两套独立主索引。

## A6.5 Dense Baseline

```text
FLAT + IP
```

Embedding 先 L2 normalize。

当前使用 FLAT 是为了获得无 ANN 近似误差的检索 baseline。

## A6.6 Sparse Baseline

```text
Milvus Native BM25
+
SPARSE_INVERTED_INDEX
```

不在应用层维护第二套主 BM25 库。

BM25 analyzer/tokenizer，特别是中文处理：`[PROVISIONAL]`。

## A6.7 Future ANN

后续数据规模或 latency 成为问题后：

```text
FLAT → IVF_FLAT
```

优先评估 IVF_FLAT，不默认迁移 HNSW。

未来调：

```text
nlist
nprobe
```

并以 FLAT 的 Recall/MRR 为准确率基准。

## A6.8 Section Store

Section hierarchy 不参与第一阶段向量检索。

具体持久化介质：

```text
[PROVISIONAL / implementation-open]
```

可以是轻量结构存储，但当前不额外绑定 SQLite/PostgreSQL/第二 Milvus Collection。

## A6.9 Rebuild / Upsert

`[PROVISIONAL]`

full rebuild、upsert、重复入库防护、schema/build versioning 尚未冻结。实现前必须单独确认，coding agent 不得自行假设“重复 insert 即可”。

---

# Knowledge-base Invariants

1. A1 只吃正式 `document.md`。
2. Metadata 最小化。
3. 先 Heading structure，末级过长才长度切分。
4. Chunk size 由 Section profiling 决定。
5. Overlap 不跨 Section。
6. Leaf 是唯一第一阶段 Retriever Candidate。
7. 多级 Section hierarchy 必须可恢复。
8. Dense/BM25 baseline 都索引 Leaf.content。
9. Qwen3-Embedding-4B + 2560d + L2 normalize。
10. Milvus FLAT/IP + Native BM25。
11. 当前不使用 HNSW；future ANN 优先 IVF_FLAT。
