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

状态：Document Registry `[FROZEN]`；Leaf identity 字段契约见 A2.3，由 A3/A4 实现

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

状态：`[FROZEN]`

文档级信息只保存一次。持久化字段只有：

```text
document_id
document_title
source
```

`document_id` 是跨建库稳定的文档身份：沿用 A1 `LoadedMarkdownDocument.document_id`，必须与同目录 `quality_report.json` 的 `document_id` 一致。A2 不重新生成、不从文件名二次推导。

`source` 是面向 citation / UI 的原始文档逻辑来源标识，不是当前服务器文件系统绝对路径。C3 从 Registry 取 title/source，模型不得自己猜路径或 URL。

`quality_report.json` 只作为 identity / provenance metadata 读取：

```text
document_id
source
source_sha256
```

不得把 audit JSON、OCR text、`document.json`、`regions.json` 作为正文入库。

禁止用以下方式得到 title/source：

```text
first Markdown heading
LLM title extraction
正文正则猜标题
按文件名 / 目录名模糊匹配 manifest
```

### Join

显式提供 corpus manifest 时，join 键为完整 SHA-256：

```text
quality_report.source_sha256 == manifest.sha256
```

精确匹配（strip 后大小写不敏感）。不得用 `document_id` 的 16 位 hash 前缀或文件名单独 join。

Manifest 语义：

```text
本次 canonical corpus ⊆ manifest
```

多余未使用行不失败。manifest 内 `sha256` 必须全局唯一；重复则整次 fail fast。

### Manifest mode

`manifest_path` 是本次 build 的 metadata authority。每一篇 LoadedMarkdownDocument 都必须唯一命中一行。任一失败整次失败，禁止对其余文档做 filename fallback。

```text
document_title = manifest.title

candidate_source =
  strip(source_url)  若非空
  否则 strip(local_path)

若 candidate_source 是机器绑定文件系统路径 → fail fast
否则 source = candidate_source
```

对 strip 后的最终 candidate（不是只检查 `local_path` 字段）拒绝：

```text
1. 以 "/" 开头
2. 以 "\" 开头
3. 匹配 ^[A-Za-z]:[\\/]
4. 大小写不敏感以 "file:" 开头
```

不做通用 URI allowlist。下列只要不被上述规则命中即允许：

```text
raw/joining/foo.pdf
joining/foo.pdf
https://...
http://...
s3://...
dms://...
oss://...
```

title 为空，或 `source_url` 与 `local_path` 都为空，fail fast。

### No-manifest mode

未传 `manifest_path` 时，全部文档统一从 `quality_report.source` 做跨平台 filename fallback。不要把 manifest 模式的绝对路径拒绝规则套到这条路径上：A0 source 只是提取输入。

```text
document_title = basename(quality_report.source).stem
source         = basename(quality_report.source)
```

basename/stem 必须同时识别 POSIX `/` 与 Windows `\`，不依赖建库机 OS。例如：

```text
/foo/bar/manual.pdf   → title=manual  source=manual.pdf
D:\foo\bar\manual.pdf → title=manual  source=manual.pdf
```

`quality_report.source` 缺失或提取结果为空 → fail fast。

### 非持久化

下列字段只用于构建期 join/校验，不写入 RegistryEntry：

```text
source_sha256
local_path
organization
category
year
pages
bytes
```

接口不得硬编码 `data/engineering_docs/manifest.csv`：

```python
build_document_registry(
    documents: list[LoadedMarkdownDocument],
    manifest_path: Path | None = None,
) -> dict[str, DocumentRegistryEntry]
```

## A2.3 Leaf Identity / Provenance

本阶段 Document Registry **不**持久化下列字段；它们由 A3/A4 在 Leaf/Section 上写入。

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

`[FROZEN baseline]`

Leaf.content 为 Leaf 的正文内容。Dense 与 BM25 baseline 直接使用：

```text
Leaf.content
```

Section heading **不作为额外 retrieval prefix 拼入 Leaf.content**。

当前 baseline 不构造：

```text
retrieval_text = heading + content
```

`heading + content` 只作为后续 retrieval enhancement / D1 ablation；只有 D1 证明 Chunk 脱离标题后明显影响 Retrieval 时再启用。

`[PROVISIONAL]`：`<!-- PDF page N -->` 是否从 retrieval text 中剥离、仅转换为 provenance，当前尚未最终冻结。

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
8. Dense/BM25 baseline 都索引 Leaf.content，Section heading 不作为额外 retrieval prefix。
9. Qwen3-Embedding-4B + 2560d + L2 normalize。
10. Milvus FLAT/IP + Native BM25。
11. 当前不使用 HNSW；future ANN 优先 IVF_FLAT。
