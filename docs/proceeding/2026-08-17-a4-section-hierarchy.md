# A4 Hierarchical Parent-Child 实现冻结与交接（2026-08-17）

> 本文是 A4 **实现交接**，不是需求主文档。  
> 需求合同仍以 `docs/knowledge-base.md` A4 `[FROZEN]` 与上游批准的开工口径为准。  
> 写本文的目的：上游在无法打开仓库时，仍能根据下面的实现口径、pytest 记录和 smoke 记录验收 A4。  
> 本轮结束后不再继续改 A4。**`A4 CLOSED` 只能由上游确认。**

## 当前验收状态

```text
A4 DESIGN
FROZEN / PASS
- SectionNode.source_start/source_end = A3 0-based half-open Markdown 行号
- SectionRef span = identity anchor；SectionNode span = semantic recovery
- A4 不修改 SectionRef，不重生 section_id
- children_by_parent key = (document_id, parent_section_id)
- 不补 synthetic heading；document_root 不是假 H1

A4 IMPLEMENTATION
PASS
- build_section_hierarchy() / SectionHierarchy API / recover_section_text()
- CLI: python -m knowledge_base.section_hierarchy
- 允许的 A3 表面：chunk_parsed_documents()；section_semantic_end()
- 未修改 A1 / A2 / A0 / C8；未实现 A5/A6/B6

A4 UNIT TEST
PASS
- Linux: 161 passed, 0 skipped（含 A1 T8 真实 PASSED；含 T48–T80）

A4 REAL CORPUS
PASS
- canonical root = /public/zhangkairan/MyMethod/all-in-rag-main/outputs/preprocessing
- document_count = 3（NASA-STD-4003A / 6033 / 5009C）
- section_node_count = 415 = 414 heading + 1 document_root
- heading_level_distribution = 2:18,3:200,4:157,5:24,6:15（与 A3.1 完全一致）
- leaf_count = 271（与 A3.2 完全一致）
- 全部 invariant = 0
- recovered_section_count = 415；PDF marker violation = 0
- manual recovery PASS

A4 CLOSED
READY FOR UPSTREAM CLOSE
- 实现工作到此结束；申请关闭 A4，进入 A5
```

Linux 服务器硬条件已满足。申请关闭 A4，进入 A5。A5 只消费 `Leaf.content` 做 Qwen3-Embedding-4B dense vector；不得借 A5 改 A4 hierarchy，也不得提前实现 B6 ancestor selection。

审查已通过、无需再改的部分：行号坐标系、identity span ≠ semantic span、`chunk_parsed_documents` 抽取、document-scoped children index、page marker 只剥离、最小 `SectionNode` schema。

---

## 1. 完成结论

| 项 | 结果 |
|---|---|
| 范围 | A4 Section hierarchy + source-span recovery。**未实现 A5–A6 / B6** |
| 输入 | 一次 `parse_markdown_document()` 同时供给 A3.2 与 A4 |
| 代码位置 | 新增 `section_hierarchy.py`；A3 仅加法接口 |
| Tokenizer / 768/96 | 只为重建同一套 A3 Leaf；A4 不重选 |
| **Linux pytest** | **161 passed, 0 skipped** |
| **A4 smoke** | 415 node；414 heading；1 document_root；271 Leaf；invariant 全 0 |
| 未使用 | `outputs/final-acceptance-*` / smoke / `.failed` / `.staging` |
| A4 CLOSED | **申请关闭**（服务器硬条件已满足） |

---

## 2. 修改 / 新增文件

新增：

```text
code/knowledge_base/section_hierarchy.py
code/knowledge_base/tests/test_section_hierarchy.py
docs/proceeding/2026-08-17-a4-section-hierarchy.md
```

改过：

```text
code/knowledge_base/leaf_chunker.py          # 公开 chunk_parsed_documents；chunk_documents 改为 parse 后调用
code/knowledge_base/structure_parser.py      # 导出 section_semantic_end（行为同原 _section_end）
code/knowledge_base/tests/test_leaf_chunker.py  # chunk_documents vs chunk_parsed_documents 逐字段回归
code/knowledge_base/__init__.py
docs/knowledge-base.md                       # A4.3 升 FROZEN；parent / document_root / span / recovery
docs/SPLIT-MANIFEST.md                       # A4 materialization 移出 provisional checklist
agent.md                                     # pending upstream acceptance
```

明确未改：

```text
code/knowledge_base/markdown_loader.py
code/knowledge_base/document_registry.py
code/knowledge_base/leaf_ids.py
code/knowledge_base/chunking_config.py
code/knowledge_base/token_count.py
code/knowledge_base/tests/test_markdown_loader.py
code/knowledge_base/tests/test_document_registry.py
code/knowledge_base/tests/test_section_profile.py
code/C8/**
code/preprocessing/**
```

未把 `/tmp/a4-sections.json` 写成生产 Section Store。A6.8 持久化介质仍为 `[PROVISIONAL]`。

---

## 3. API

```python
class A4HierarchyError(Exception): ...

@dataclass(frozen=True)
class SectionNode:
    section_id: str
    document_id: str
    parent_section_id: str | None
    kind: str                     # heading | document_root
    heading_level: int | None
    heading: str | None
    source_start: int             # semantic, 0-based line
    source_end: int               # half-open
    page_start: int
    page_end: int

def build_section_hierarchy(
    documents,
    parsed_documents,
    chunking_result,
) -> SectionHierarchy

class SectionHierarchy:
    def get_section(section_id) -> SectionNode
    def get_parent(section_id) -> SectionNode | None
    def get_ancestors(section_id) -> tuple[SectionNode, ...]   # nearest first, 不含自身
    def recover_section_text(section_id, documents) -> str
    def child_section_ids(document_id, parent_section_id) -> tuple[str, ...]
```

`children_by_parent` 是 derived in-memory index，key 为 `(document_id, parent_section_id)`。`None` 只表示该文档的 top-level，不跨文档聚合。

A3 加法接口（行为不变）：

```python
chunk_parsed_documents(documents, parsed_documents, tokenizer, ...) -> ChunkingResult
chunk_documents(...):
    parsed = parse_markdown_document(...)
    return chunk_parsed_documents(...)
```

回归：`chunk_documents(input)` 与 `parse + chunk_parsed_documents(input)` 的全部 Leaf/SectionRef 字段与 ID 一致（Linux pytest 中该测试 PASSED）。

---

## 4. 正式验收环境

```text
host:         swift
repository:   /public/zhangkairan/MyMethod/all-in-rag-main
environment:  mfg-rag-preprocess
Python:       3.11.15
transformers: 5.15.0
tokenizer:    Qwen/Qwen3-Embedding-4B
canonical:    /public/zhangkairan/MyMethod/all-in-rag-main/outputs/preprocessing
corpus:       NASA-STD-4003A / 6033 / 5009C（当前 A0 hash identity）
```

```bash
cd /public/zhangkairan/MyMethod/all-in-rag-main
conda activate mfg-rag-preprocess

PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" \
python -m knowledge_base.section_hierarchy \
  --canonical-root "$PWD/outputs/preprocessing" \
  --tokenizer Qwen/Qwen3-Embedding-4B \
  --chunk-size 768 \
  --overlap 96 \
  --output /tmp/a4-sections.json

python -m pytest code/knowledge_base/tests -v
```

---

## 5. Linux pytest（原文口径）

```text
collected 161 items
161 passed in 0.84s
0 skipped
```

含 A1 T8 symlink 真实 PASSED，含 T48–T80 与 `test_chunk_parsed_documents_matches_chunk_documents`。

开头出现的 `122 passed in 0.62s` 是部署 A4 测试前的旧 A3 套件，不以它为验收口径。

---

## 6. Linux real corpus smoke（stdout 原文口径）

```text
canonical_root=/public/zhangkairan/MyMethod/all-in-rag-main/outputs/preprocessing
output=/tmp/a4-sections.json
tokenizer=Qwen/Qwen3-Embedding-4B
transformers_version=5.15.0
chunk_size=768
overlap_tokens=96
document_count=3
section_node_count=415
heading_section_count=414
document_root_count=1
heading_level_distribution=2:18,3:200,4:157,5:24,6:15
top_level_section_count=78
leaf_count=271
leaf_section_resolution_failures=0
parent_link_count=337
missing_parent_count=0
cross_document_parent_count=0
hierarchy_cycle_count=0
invalid_heading_level_relation_count=0
section_page_invalid_count=0
section_span_invalid_count=0
recovered_section_count=415
recovered_pdf_marker_violation_count=0
max_hierarchy_depth=5
hierarchy_depth_p50=2.0
hierarchy_depth_p95=3.0
```

### 6.1 硬条件

| 条件 | 结果 |
|---|---|
| `document_count = 3` | PASS |
| `leaf_section_resolution_failures = 0` | PASS |
| `missing_parent_count = 0` | PASS |
| `cross_document_parent_count = 0` | PASS |
| `hierarchy_cycle_count = 0` | PASS |
| `invalid_heading_level_relation_count = 0` | PASS |
| `section_page_invalid_count = 0` | PASS |
| `section_span_invalid_count = 0` | PASS |
| `recovered_pdf_marker_violation_count = 0` | PASS |
| A4 `leaf_count` = A3.2 `leaf_count` = 271 | PASS |
| `parent_link_count + top_level_section_count` = `section_node_count`（337+78=415） | PASS |
| `recovered_section_count` = `section_node_count` = 415 | PASS |

### 6.2 Structural cross-check

| 条件 | 结果 |
|---|---|
| A4 `heading_section_count` = 414 | PASS |
| A3.1 heading 分布 `2:18,3:200,4:157,5:24,6:15` = 18+200+157+24+15 = 414 | PASS |
| A4 未新造 heading Section | PASS |
| A4 `document_root_count` = 1（仅 NASA-STD-5009C 有 heading 前 lead） | PASS |
| 无 H1（分布从 H2 起） | PASS（T52 真实主路径） |

### 6.3 Spotcheck（10 项，确定性选取）

| 合同类别 | 实际选取 |
|---|---|
| 2 个 top-level | H3 `NASA TECHNICAL STANDARD`、H3 `METRIC/SI`，`parent=None`（NASA 封面级 heading，合法无 H1） |
| 2 个 H2→H3→H4 | 两个 H4 `APPROVED FOR PUBLIC RELEASE...`，各有 parent `sec_...`，跨页 3–4 / 4–5 |
| 2 个 level-jump | H5 `4.1.3 Shock and Fault Protection`；H6 `4.1.3.1 Hazardous Area Bonding` |
| 2 个跨页 Parent | H2 `ELECTRICAL BONDING...` page 1–30；H3 `DOCUMENT HISTORY LOG` page 2–3 |
| 1 个无直接 Leaf 的 parent Heading | H3 `1. SCOPE`，leaves=0，children=3 |
| 1 个 document_root | NASA-STD-5009C，`kind=document_root`，`heading=None`，span `[0,11)`，page 1–1 |

H2 `ELECTRICAL BONDING...`：`leaves=0`、`children=38`、span `[18,759)`、page 1–30。空 preface 的真实 parent heading 被保留，semantic span 覆盖 descendants。

preview 中出现 `<!-- TABLE ... -->` 是合法保留的普通 comment，不是 PDF page marker。

### 6.4 Manual recovery

```text
document_id=15_nasa_std_4003a_electrical_bonding-5c5271413bd05f61
heading='ELECTRICAL BONDING FOR NASA LAUNCH VEHICLES, SPACECRAFT, PAYLOADS, AND FLIGHT EQUIPMENT'
contains_own_heading=True
contains_child_headings=True
contains_next_same_or_higher_heading=False
contains_pdf_page_marker=False
```

这是 A4 最关键人工验收项：Parent H2 恢复含自身 heading 与 child headings，不含下一个同级/更高级 heading，不含 `<!-- PDF page N -->`。

---

## 7. Depth 观察（不冻结最佳值）

```text
max hierarchy depth = 5
P50 = 2.0
P95 = 3.0
```

与 heading 分布一致：大量 H3，少量 H5/H6。这只用于观察真实 NASA 结构，不冻结“最佳 depth”。

---

## 8. Scope statement

```text
是否修改 A3 behavior？ NO
是否修改 Leaf？        NO
是否重生成 section_id/chunk_id/chunk_index？ NO
是否实现 A5？          NO
是否实现 A6？          NO
是否实现 B6 ancestor selection / budget / shared-parent / neighbor fallback？ NO
是否冻结 Section Store 生产介质？ NO
是否把 JSON artifact 写成长期架构？ NO
```

---

## 9. A4 DoD

```text
A4 DESIGN                         FROZEN / PASS
MULTI-LEVEL HIERARCHY             PASS
PARENT RELATION                   PASS
LEVEL JUMP                        PASS
DOCUMENT ROOT                     PASS
EMPTY HEADING PRESERVATION        PASS
LEAF→SECTION RESOLUTION           PASS
SEMANTIC SOURCE SPAN              PASS
SECTION TEXT RECOVERY             PASS
PAGE PROVENANCE                   PASS
PDF MARKER REMOVAL                PASS
MINIMAL SECTION SCHEMA            PASS
A3 IMMUTABILITY                   PASS
UNIT TEST                         PASS   # Linux 161 passed, 0 skipped
A1/A2/A3 REGRESSION               PASS
REAL CORPUS                       PASS
```

之后：

```text
A4 CLOSED
```

必须由上游确认。确认后 `agent.md` 可改为 `A4 implemented / frozen`，下一模块为 **A5 — Dense Embedding**。
