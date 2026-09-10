# A6 Production Collection 开工合同 v1 — 实现与测试交接（2026-09-09）

> 本文是 **正式 A6 实现交接**，覆盖合同 P0–P9、P14–P16；P10–P13（真实语料满量入库 + 生产 smoke）在本机曾被阻塞，见 §7。
> 未改 A0–A5 行为合同。Analyzer/BM25 常量直接复用 A6.0 冻结结论（`a6_profile_config.py`），未重新选型。
>
> **更新（2026-09-10）**：§7 记录的阻塞已解除——在 `shiyuan` 上补跑真实语料，通过 Xftp（不是 git，见下方更正）把产出传回本机并完成真实生产入库 + 验收。完整过程、真实命令与结果见新文档
> [`docs/proceeding/2026-09-10-a6-production-real-ingestion.md`](./2026-09-10-a6-production-real-ingestion.md)。
> 本文档保留原状作为"实现与机制级测试"的交接记录，仅修正下方发现的一处 CLI 用法笔误（`a6_document_metadata.py` 没有 `extract` 子命令，见 §7.1/§8.4）；
> **§10（同步到 shiyuan）整节基于错误假设（以为跨机器同步走 git push/pull），实际同步方式是 Xftp 直传文件，§10 的具体命令未被执行，请勿参照，以新文档为准。**

## 当前验收状态

```text
A6 PRODUCTION PURPOSE
PASS
- 新增 a6_document_metadata.py（P0-P5：抽取/去重/版本治理）
- 新增 a6_collection.py（P6-P9：strict join/schema/BM25/生命周期）
- 不重新实现 A0-A5；不重选 Analyzer；不做 Hybrid/RRF/Query Rewrite

P0-P5 DOCUMENT METADATA
PASS
- 106 个新增单测全部通过（61 metadata + 44 collection-fake + 1 collection-Milvus）
- T1-T6 合同验收用例（dedup ×2 + version governance ×3 + filename-lookalike ×1）全部覆盖

P6-P9 COLLECTION
PASS
- schema/BM25 Function/FLAT+IP dense index 在真实 Milvus 2.5.14 建出来并可检索
- build / ingest-update / rebuild 三条生命周期路径均用真实 Milvus 跑通一次完整闭环

P10-P13 REAL CORPUS INGESTION
BLOCKED（本文原始记录；2026-09-10 已解除，见下方"更新"提示和新文档）
- 本机 .venv 与 conda base 均无 torch / transformers；无法跑 A3 分块、A5 embedding
- 仓库当前没有任何 document.md（A1 canonical 语料未落盘在本机）
- 用合成数据在真实 Milvus 上做了机制级集成验证代替 P10 的正式生产验证

P15 REGRESSION
PASS
- code/knowledge_base/tests 全量：294 passed, 3 skipped（skip = 需要 torch 的既有用例，与本次改动无关）

A6 OFFICIAL
MECHANISM COMPLETE / DATA-BLOCKED（本文原始记录，2026-09-10 已升级为 COMPLETE，见新文档）
- 代码、schema、生命周期、审计链路均已实现并测试
- 生产级 17 篇语料的真实入库+验收仍需在具备 Qwen 权重+torch 的机器上补跑
```

---

## 1. 完成结论

| 项 | 结果 |
|---|---|
| 范围 | 正式 A6：文档级治理（P0-P5）+ Milvus Collection（P6-P9）。**不含**真实 17 篇语料的生产入库（P10-P13 见 §7） |
| 新增文件 | `a6_document_metadata.py`、`a6_collection.py`、`a6_prepare_rows.py`（GPU 机器专用胶水脚本）+ 4 个测试文件 |
| 未改 | A0-A5 全部行为合同；`a6_profile_config.py` / `a6_analyzer_profile.py`（A6.0 产物，只读引用，未修改） |
| Analyzer/BM25 | 直接复用 A6.0 冻结值：`UNIFIED_ICU` + `k1=1.2 b=0.75 DAAT_MAXSCORE`，未重新测 |
| Dense 索引 | `FLAT + IP`，`dim=2560`（复用 `embedding_config.EMBEDDING_DIMENSION`） |
| 测试 | 单元测试 105 个（fake store / 无 Milvus 依赖）+ 1 个真实 Milvus 机制级集成测试，全部通过 |
| 生产入库 | 本文撰写时**未执行**（本机无 Qwen 权重、无 torch/transformers、无真实 `document.md` 语料）；**2026-09-10 已在 shiyuan + 本机真实跑通**，见 [`2026-09-10-a6-production-real-ingestion.md`](./2026-09-10-a6-production-real-ingestion.md) |

---

## 2. 运行时拆分（现状，非最终方案）

| 机器 | 现状 | 本轮做了什么 |
|---|---|---|
| 本机 Windows（`.venv`） | 有 `pymilvus==2.5.11`；补装了 `pytest`、`markdown-it-py`（`numpy` 已有）；**无 torch/transformers** | 全部单元测试 + 真实 Milvus 机制级集成测试 |
| 本机 conda base | 有 `pytest`/`numpy`/`markdown-it-py`；**无 pymilvus，无 torch/transformers** | 未使用（`.venv` 已够用，避免污染 base） |
| Milvus | Docker `milvusdb/milvus:v2.5.14` standalone，`http://localhost:19530`，运行正常 | schema 建表、BM25 Function、FLAT/IP 索引、build/ingest-update/rebuild 全流程验证 |
| GPU/Qwen 机器（如 `shiyuan`） | 未在本轮涉及 | **P10 正式入库需要在此类机器上补跑**：A1 canonical 语料 + A3 分块 + A5 embedding，再把产出的 rows 传到能连 Milvus 的机器执行 `a6_collection.py build` |

`a6_document_metadata.py` 和 `a6_collection.py` 都刻意不 import `torch`/`transformers`：

- `a6_document_metadata.py` 只用 `re`/`hashlib`/`unicodedata`/`datetime`，加上 A1/A2 的轻量导入（`markdown_loader.py`、`document_registry.py`，两者都不依赖 torch）。
- `a6_collection.py` 不 import `dense_embedding.py`（会拉 torch），改为对 `DenseEmbeddingResult` 做鸭子类型（只读 `.chunk_ids` / `.vectors`），也不 import `a6_analyzer_profile.py`（profiling-only 工具），只 import `a6_profile_config.py`（纯常量）避免正式实现反向依赖 profiling 脚本。

这使得两个模块都能在"只有 Milvus、没有 GPU"的机器上完整开发和测试，与本机现状对齐。

---

## 3. 新增文件

```text
code/knowledge_base/a6_document_metadata.py
code/knowledge_base/a6_collection.py
code/knowledge_base/a6_prepare_rows.py
code/knowledge_base/tests/test_a6_document_metadata.py
code/knowledge_base/tests/test_a6_collection.py
code/knowledge_base/tests/test_a6_collection_milvus_integration.py
code/knowledge_base/tests/test_a6_prepare_rows.py
docs/proceeding/2026-09-09-a6-production-collection.md
.gitignore（新增一行 .pytest_tmp/）
```

`a6_prepare_rows.py` 是本轮为了让 shiyuan 能跑 P10 而补上的第三个模块：把 A1 载入 → A3 分块 → A5 embedding → A6 strict join 串成一条命令，只有它允许（通过延迟 import）依赖 torch/transformers。详见 §7.1。

未改：A0-A5 任何文件；`a6_profile_config.py`；`a6_analyzer_profile.py`。

（`chunking_config.py` / `leaf_chunker.py` / `leaf_ids.py` / `test_leaf_chunker.py` 在 `git status` 中显示为 modified，核对后是纯换行符差异（CRLF/LF），`git diff` 无实际内容变化，非本轮改动。）

---

## 4. P0-P5：文档级元数据治理（`a6_document_metadata.py`）

### 4.1 SS7 扫描窗口

前 300 + 后 300 个**有效文本行**（跳过空白行）。实现上直接取 `effective_lines[:300]` 和 `effective_lines[-300:]` 的并集：当文档 ≤600 有效行时两段自然重叠覆盖全文，无需为"不足 600 行扫全文"单独分支。用一条长文档（900 有效行）+ 中段隐藏的"批准日期"验证扫描窗口确实跳过中间、命中首尾。

### 4.2 字段抽取（SS8-SS12）

| 字段 | 识别方式 | 关键设计 |
|---|---|---|
| `document_number` | 显式标签（文件编号/文档编号/标准号/规范号/编号/Document No. 等）+ 冒号 | 支持纯文本行和两列 Markdown 表格行（`\| 标准号 \| GB/T 1234-2020 \|`）；**绝不看 file_name** |
| `document_version` | 显式标签（版本号/版本/版次/修订号/修订版/Version/Revision/Rev.） | "2025版"/"第3版"这类值本身就是通过标签抓到的，不需要额外的启发式规则 |
| `finalized_at` | 三级优先：批准>签发>发布 | 只用最高非空优先级；同级冲突→NULL+`conflict=True`；不跨级回退冲突 |
| `effective_from` | 显式标签 + "自 DATE 起实施/生效"无标签句式 + "自发布之日起实施"（=finalized_at） | 三种候选合并进同一个 tier 做一致性校验，而不是互相覆盖 |
| `effective_to` | 显式标签 + "自 DATE 起实施，有效期N年"复合句 | 只认同一行内显式给出的起始日期；`有效期3年`单独出现（无起始日期）→ NULL，不允许用已解析的 `effective_from` 去拼接跨行推断 |

日期解析支持中文 `YYYY年MM月DD日`、`YYYY-MM-DD`、`YYYY/MM/DD`、`YYYY.MM.DD`、英文 `Month DD, YYYY` / `DD Month YYYY`。`add_years` 对闰年 2/29 做 `day=28` 回退。

审计记录（`MetadataAuditRecord`）字段严格对齐合同给出的 7 个字段：`document_id/field/value/method/evidence/line_start/line_end/conflict`，`method` 取值限定为 `explicit_label` / `not_found` / `derived_from_finalized_at` / `duration_calculation` / `conflicting_candidates`。

### 4.3 去重 + 版本治理（SS17-SS24）

`govern_documents(existing_catalog, new_drafts)` 按 `new_drafts` 传入顺序逐个判定（约定调用方按 `document_id` 排序，与 A1 loader 一致，保证可复现）：

1. `source_sha256` 命中任何非 DUPLICATE 条目 → `DUPLICATE`
2. 否则 `document_content_hash` 命中 → `DUPLICATE`
3. 否则若 `document_number` 命中同号条目：内容不同 → 按 `finalized_at` 排序（新胜旧，旧转 `HISTORICAL`）；日期任一缺失或相等 → 双方都转 `VERSION_CONFLICT` 并写一条 `DocumentConflict`
4. 否则若 `document_number` 缺失但标题（NFC + 折叠空白）命中：内容不同 → 双方 `VERSION_CONFLICT`
5. 否则 `ACTIVE`

`HISTORICAL` 条目不再参与后续比较（已被判定为被取代，不重复判定）；重复处理同一个 `document_id`（同名重跑）是幂等的：`ingested_at` 保留首次值，且该条目在比较池中排除自身。

`document_content_hash` 的规范化（SS18）：CRLF→LF、Unicode NFC、逐行 `rstrip()`、整行删除 `<!-- PDF page N -->`（不留空行占位）。用 T1/T2 + CRLF/NFC/尾随空白/大小写共 6 个用例验证。

### 4.4 T1-T6 合同验收用例

全部在 `test_a6_document_metadata.py` 中以合同原始编号命名并通过：`test_t1_identical_source_sha256_is_duplicate`、`test_t2_different_source_same_content_hash_is_duplicate`、`test_t3_same_number_different_content_reliable_dates_orders_versions`（含增量场景）、`test_t4_same_number_missing_dates_is_version_conflict_both_retrievable`（含 equal-date 变体）、`test_t5_title_match_only_is_version_conflict_neither_deleted`、`test_t6_filename_lookalike_is_null_and_ignores_filename`。

另有一条端到端用例 `test_run_governance_end_to_end_with_real_a1_a2`：真实调用 A1 `load_markdown_documents` + A2 `build_document_registry`（不 mock），验证 `file_name`/`document_title` 的联表来自 A2、`document_number` 只来自正文标签。

---

## 5. P6-P9：Collection（`a6_collection.py`）

### 5.1 Schema（SS28-SS30）

17 个字段：`chunk_id`(PK) / `document_id` / `file_name` / `document_title` / `document_number`(nullable) / `document_version`(nullable) / `finalized_at`(nullable INT64) / `effective_from`(nullable INT64) / `effective_to`(nullable INT64) / `ingested_at` / `status` / `section_id` / `chunk_index` / `page_start` / `page_end` / `content` / `dense_vector` / `sparse_vector`（BM25 Function 自动生成，**从不手工写入**）。

`chunk_id`/`document_id` 的 VARCHAR 上限直接从 `a6_profile_config.py` import（而不是重新声明常量），并写了一条防漂移测试 `test_chunk_id_and_document_id_lengths_match_a6_0_frozen_values`：只要 A6.0 常量以后变了，这条测试会立刻炸，而不是让两处悄悄不一致。BM25 `k1/b/DAAT_MAXSCORE` 和 ICU analyzer 参数同样是直接 import + 防漂移断言，不是复制粘贴。

`nullable=True` 在 Milvus 2.5.14 上是真的可用（本机在动手写 schema 前先用一个一次性探针 collection 验证过 insert `None` + query 回读）。

### 5.2 Strict Join（SS31-SS32）

- `join_leaves_with_embeddings(leaves, embedding_result)`：`missing/extra/duplicate_leaf/duplicate_embedding` 任一非零就 FAIL，不做部分插入。`embedding_result` 只要求 `.chunk_ids`/`.vectors` 两个属性（鸭子类型），不强制是 `DenseEmbeddingResult`。
- `join_leaves_with_metadata(leaves, metadata_by_document_id)`：**对称**集合相等（不是子集检查）。调用方必须先用 `select_retrievable_metadata()` 把 catalog 收窄到本次真正要插入的 `document_id` 集合再传进来——这是本轮对合同 SS31 "A3 document_id 集合 == metadata document_id 集合"的具体化解释：等式是相对"本批要建的可检索文档集合"而言，不是相对全量历史 catalog（否则每次增量更新都会因为历史文档不在场而误报）。
- `validate_dense_vectors`：只检查 `dim==2560` 和有限性（NaN/Inf），**不重新做 L2 归一化或范数校验**——A5 已经做过，A6 不得重复或质疑这一步（合同原话）。

### 5.3 生命周期（SS34）

`build_collection`（已存在则 FAIL，不静默覆盖）/ `ingest_update_collection`（Collection 不存在则 FAIL，只删除+插入受影响 `document_id`）/ `rebuild_collection`（必须显式 `confirm_rebuild=True`，防止误走全量重建分支）。

`document_ids_to_delete_for_update(previous_catalog, updated_catalog)` 把"哪些文档要从默认 Collection 里删除"这条规则做成纯函数：只有从 `{ACTIVE, VERSION_CONFLICT}` 转出（转 `HISTORICAL` 或 `DUPLICATE`）才需要删旧 Leaf；`VERSION_CONFLICT` 之间的转换不删除任何一方，两个版本都留着可检索（SS26/SS27/SS35）。

### 5.4 验证 + Smoke（SS37-SS39）

`verify_collection` 用 `query_iterator`（而不是一次性 `limit=1000000` 的 `query`）拉全量行——**这是本轮在真实 Milvus 上跑出来的真实 bug**：Milvus 单次 `query` 的 `offset+limit` 硬上限是 16384，超过直接报错，必须用游标式 `query_iterator` 分批拉取。

另一个真实环境细节：`delete()` 之后如果不显式 `flush()`，`get_collection_stats()` 返回的 `row_count` 在 compaction 完成前不会反映删除（本轮实测：先 5 行，删 3 插 2，`get_collection_stats` 仍报 7，直到加了 `flush`）。即便加了 `flush`，`get_collection_stats` 的计数器语义仍然是"尽力而为"，所以 `VerificationResult` 把"真正查询到的行数"（`row_count`，用于校验通过/失败）和"stats 接口报告的行数"（`stats_row_count`，仅供参考）分开成两个字段，校验只信任前者。

`dense_smoke_test`/`bm25_smoke_test` 分别单独调用 Dense-only / BM25-only 检索，不借助 RRF/Reranker 兜底（合同要求）。

---

## 6. 测试结果

```text
.\.venv\Scripts\python.exe -m pytest code\knowledge_base\tests -q -p no:cacheprovider --basetemp=.pytest_tmp

300 passed, 4 skipped in ~32s
```

> **2026-09-10 更新**：`a6_collection.py` 新增了 `report` 子命令（见 §8.4 第 4 步），并为其补了 7 个测试（`derive_bm25_query` 的纯函数测试 + `report` 子命令的 argparse 及 fail-fast 测试）。现在跑同一条命令是 **307 passed, 4 skipped**。4 个 skip 的构成不变（见下表说明）。

拆分：

| 文件 | 用例数 | 依赖 |
|---|---|---|
| `test_a6_document_metadata.py` | 61 | 纯 stdlib + A1/A2（无 Milvus，无 torch） |
| `test_a6_collection.py` | 44（2026-09-10 新增 `report` 子命令测试后为 51） | `FakeCollectionStore`（无 Milvus，无 torch） |
| `test_a6_collection_milvus_integration.py` | 1（内部覆盖 build/verify/dense-smoke/bm25-smoke/report/ingest-update/rebuild 全链路） | 真实 Milvus 2.5.14，合成数据 |
| `test_a6_prepare_rows.py` | 6 passed + 1 skipped（本机） | 6 个纯过滤/序列化逻辑（无 torch）；1 个 orchestration 用 `pytest.importorskip("torch")` 门控，本机 skip，**2026-09-10 已在 shiyuan 上真实 PASS**（见新文档） |
| 其余既有 A0-A5 套件 | 188 passed / 3 skipped | 既有基线，未受本轮改动影响 |

4 个 skip 里 3 个是既有 `test_dense_embedding.py` 需要 `torch` 的用例（与本轮无关，本机没有 torch 会一直 skip），1 个是本轮新增、需要在 shiyuan 上补跑的 `test_prepare_rows_wires_chunk_embed_and_join`（**已在 shiyuan 上验证为 PASS**，本机因为没有 torch 仍会显示 skip，这是预期的、非缺陷）。

集成测试跑完会 `drop_collection` 自清理；执行后用 `list_collections()` 确认 Milvus 上无残留 collection。

本机 `.venv` 补装（仅新增，未升级/替换已有版本）：`pytest==9.1.1`、`markdown-it-py==4.2.0`。

---

## 7. P10-P13 阻塞记录（环境限制，需要在别的机器补）

### 7.1 新增 `a6_prepare_rows.py`：给 shiyuan 用的胶水脚本

本轮补了第三个模块，专门解决"P10 需要在 shiyuan 上把 A1/A3/A5 串起来产出 Milvus 可插入的 rows"这件事：

```text
python -m knowledge_base.a6_prepare_rows \
  --canonical-root <A1 canonical 根目录> \
  --metadata-jsonl outputs/index_metadata/document_metadata.jsonl \
  --tokenizer Qwen/Qwen3-Embedding-4B \
  --model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --output outputs/index_metadata/a6_rows.jsonl
```

它做的事：读 `document_metadata.jsonl`（由 `a6_document_metadata.py` 先产出，**该 CLI 是扁平参数、没有子命令**，下方 §8.4 已修正此前误写的 `extract`）→ 用 `select_retrievable_metadata` 只保留 `ACTIVE`/`VERSION_CONFLICT` 的 `document_id` → 只对这些文档跑 A3 `chunk_documents`（不浪费 GPU 时间在 `DUPLICATE`/`HISTORICAL` 文档上）→ 跑 A5 `embed_leaves` → `join_leaves_with_embeddings` + `build_collection_rows` → 写 `rows.jsonl`。

这是本轮唯一允许依赖 `torch`/`transformers` 的新模块，而且是通过**函数体内延迟 import**做到的：`chunk_documents`/`embed_leaves`/`load_tokenizer` 只在 `prepare_rows()` 被真正调用时才 import，所以模块本身、以及"哪些文档要重新入库"这条纯过滤逻辑（`select_retrievable_documents`），在没有 torch 的本机也能正常 import 和测试。真正调用 A3/A5 那部分（`test_prepare_rows_wires_chunk_embed_and_join`）用 `pytest.importorskip("torch")` 门控，在本机会 skip，需要在 shiyuan 上才能真正跑起来验证。

### 7.2 详细阻塞记录

| 需要 | 本机现状 |
|---|---|
| 真实 17 篇 `document.md`（A1 canonical 语料） | 仓库里不存在，只有 A6.0 冻结的 `leaf_catalog.jsonl`（无 dense 向量，`section_id` 是占位符 `catalog`，不能当正式 A3 产物用） |
| Qwen3-Embedding-4B 权重 + `torch` + `transformers` | `.venv` 和 `conda base` 都没装；`token_count.py`/`dense_embedding.py` 都无法真正跑 |
| 生产级 `document_metadata.jsonl` / `document_conflicts.jsonl` / 审计 JSONL | 未生成（依赖上面两项） |
| 生产 Collection 里跑真实 Query 做 D1/D2 检索验收 | 未执行 |

本轮用**合成数据**在真实 Milvus 上把 P10-P13 要求的"机制"全部跑了一遍（build → verify → dense smoke → bm25 smoke → build report → 版本替换式 ingest-update → rebuild），证明 schema/索引/生命周期/校验代码本身是对的；但这**不能替代**用真实语料+真实向量做的生产验收，因为：

1. 合成向量是随机单位向量，不反映真实语义相似度，Dense smoke 的"命中"没有实际检索质量含义。
2. 合成文本只有几句英文，BM25 smoke 的"命中"同样不代表真实语料上的检索质量。
3. 没有真实的 `document_number`/`document_version`/日期抽取样本可供人工抽查审计记录的准确率。

**建议**：在具备 Qwen 权重的机器（如 `shiyuan`）先跑 `a6_document_metadata.py`（不需要 GPU，只读 `document.md` 正文和 `quality_report.json`；CLI 无子命令，直接接参数），再跑 `a6_prepare_rows.py`（需要 GPU/Qwen，见 §7.1）产出 `rows.jsonl`，最后把 `rows.jsonl` 传到能连 Milvus 的机器执行 `python -m knowledge_base.a6_collection build`。2026-09-10 已按此思路真实跑通，实际步骤/命令/结果见 [`2026-09-10-a6-production-real-ingestion.md`](./2026-09-10-a6-production-real-ingestion.md)（§10 的同步方案已被该文档取代）。

---

## 8. 复现命令

### 8.1 单元测试（不连 Milvus，不需要 torch）

```powershell
cd "D:\code backup\MyMethod\all-in-rag-main"
.\.venv\Scripts\python.exe -m pytest code\knowledge_base\tests\test_a6_document_metadata.py code\knowledge_base\tests\test_a6_collection.py -q -p no:cacheprovider --basetemp=.pytest_tmp
```

### 8.2 真实 Milvus 机制级集成测试

```powershell
cd "D:\code backup\MyMethod\all-in-rag-main\code"
docker compose up -d
cd "D:\code backup\MyMethod\all-in-rag-main"
.\.venv\Scripts\python.exe -m pytest code\knowledge_base\tests\test_a6_collection_milvus_integration.py -q -p no:cacheprovider --basetemp=.pytest_tmp -v
```

### 8.3 全量回归

```powershell
.\.venv\Scripts\python.exe -m pytest code\knowledge_base\tests -q -p no:cacheprovider --basetemp=.pytest_tmp
```

（`--basetemp=.pytest_tmp -p no:cacheprovider` 是本机特有的规避手段：Windows 用户临时目录 `%TEMP%\pytest-of-<user>` 在本环境下没有写权限，直接用默认设置会在几乎所有用例的 `tmp_path` fixture 上报 `PermissionError`。）

### 8.4 生产 CLI（需要真实语料 + Qwen 权重；2026-09-10 已在 shiyuan + 本机真实跑通，完整记录见 [`2026-09-10-a6-production-real-ingestion.md`](./2026-09-10-a6-production-real-ingestion.md)）

> **修正**：下方第 1 步此前误写成 `a6_document_metadata extract`——`a6_document_metadata.py` 的 CLI 是扁平参数，**没有子命令**，不需要 `extract` 这个词。已修正。

```powershell
# 第 1 步：文档级治理（不需要 GPU；shiyuan 或本机均可，只要能读到 canonical_root）
python -m knowledge_base.a6_document_metadata `
  --canonical-root <A1 canonical 根目录> `
  --manifest <manifest.csv 路径> `
  --metadata-out outputs/index_metadata/document_metadata.jsonl `
  --audit-out outputs/index_metadata/document_metadata_audit.jsonl `
  --conflicts-out outputs/index_metadata/document_conflicts.jsonl

# 第 2 步：A3 分块 + A5 embedding + strict join（需要 GPU + Qwen 权重，只能在 shiyuan 跑）
python -m knowledge_base.a6_prepare_rows `
  --canonical-root <A1 canonical 根目录> `
  --metadata-jsonl outputs/index_metadata/document_metadata.jsonl `
  --tokenizer Qwen/Qwen3-Embedding-4B `
  --model Qwen/Qwen3-Embedding-4B `
  --device cuda `
  --output outputs/index_metadata/a6_rows.jsonl

# 第 3 步：把 a6_rows.jsonl 传到能连 Milvus 的机器，执行建库
python -m knowledge_base.a6_collection build `
  --uri http://localhost:19530 `
  --collection a6_leaf_v1 `
  --rows-jsonl outputs/index_metadata/a6_rows.jsonl

# 第 4 步（2026-09-10 新增子命令）：对已建好的 Collection 跑验收报告
python -m knowledge_base.a6_collection report `
  --uri http://localhost:19530 `
  --collection a6_leaf_v1 `
  --metadata-jsonl outputs/index_metadata/document_metadata.jsonl `
  --rows-jsonl outputs/index_metadata/a6_rows.jsonl `
  --report-out outputs/index_metadata/a6_report.json
```

---

## 9. 上游需要确认的事

1. **`join_leaves_with_metadata` 的等价范围**：本轮解释为"相对本批要插入的可检索文档集合"，而不是"相对全量历史 catalog"。如果合同原意是后者，需要说明增量更新时如何处理历史文档不在本批的情况。
2. **`effective_to` 的复合日期计算**：只认同一行内"起始日期 + 有效期 N 年"同时出现的显式句式，不跨字段拼接已解析的 `effective_from`。如果需要支持"实施日期在一处、有效期在另一处"的跨行组合，需要额外的规则确认（本轮判断这样做会引入不受控的推断风险，偏保守）。
3. **`document_conflicts` 的 `reason` 取值**：本轮定义了三个受控字符串（`same_document_number_missing_finalized_at` / `same_document_number_finalized_at_tie` / `title_match_document_number_insufficient`），合同未给出枚举，需要确认是否符合下游消费预期。
4. **P10-P13 的真实执行机器**：需要拍板在哪台机器上补跑真实语料 + Qwen embedding + 正式入库验收（本机不具备条件，见 §7）。
5. **`ingest-update` CLI 的 `--delete-document-ids-file` 来源**：本轮 CLI 假设调用方已经算好要删除的 `document_id` 列表（通常是 `document_ids_to_delete_for_update()` 的输出落盘），CLI 本身不重新跑一遍全量治理比较。如果需要 CLI 自己从两份 `document_metadata.jsonl`（旧/新）算差异，需要再加一个子命令。
6. **（2026-09-10 新发现）真实 17 篇语料上 `document_number`/`finalized_at`/`effective_from`/`effective_to` 几乎全是 NULL**：真实生产结果显示 17 篇里 `document_number`=17/17 NULL、`finalized_at`/`effective_from`/`effective_to`=17/17 NULL、`document_version` 也只有 2 篇非 NULL（两篇 NASA-STD 文档从标题里的版本字母 "A" 抓到）。P0-P5 的抽取逻辑本身已用 61 个合成用例验证过（能正确抓取"标签: 值"这类显式模式），但这批真实 NIST/NASA/中文语料里这些字段大多不是以"标签: 值"这种显式可抓取的形式出现在正文里（可能在封面页版式、页眉页脚，或者 OCR/Markdown 转换后丢失了结构）。这不是本轮代码的 bug，但需要上游确认：这个近乎全 NULL 的结果是否符合预期，是否需要为这批具体语料补充额外的抽取规则，或者接受"大多数字段本来就该是 NULL"这个结论。详见 [`2026-09-10-a6-production-real-ingestion.md`](./2026-09-10-a6-production-real-ingestion.md) §4。

---

## 10. 同步到 shiyuan：具体步骤（本节已过时，仅作历史记录）

> **本节整体基于错误假设，已被 2026-09-10 的真实流程取代，请勿参照执行。**
> 撰写本节时，我错误地假设跨机器同步走 git push/pull（比照 A1-A5 的历史模式）。实际上用户与 `shiyuan` 之间的文件同步走 **Xftp（SFTP 客户端）直传**，不涉及 git。下方 §10.1-§10.5 描述的 git commit/push 流程**从未执行**，仓库里的 A6/A6.0 新文件在本节撰写时、以及 2026-09-10 完成真实生产入库时，始终是本地未提交状态（`git status` 一直显示为 untracked）。
>
> 真实执行的同步方式、需要传输的文件清单、以及 2026-09-10 的完整生产入库结果，见
> [`docs/proceeding/2026-09-10-a6-production-real-ingestion.md`](./2026-09-10-a6-production-real-ingestion.md) §2（Xftp 同步）与 §3-§5（生产命令与结果）。
>
> 是否要把这批文件提交进 git（以及提交粒度），仍是一个未决定的独立问题，与本轮的 Xftp 同步无关，留给用户自行决定。
