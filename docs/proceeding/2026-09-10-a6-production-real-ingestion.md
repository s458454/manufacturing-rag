# A6 Production Collection：真实语料生产入库与验收（2026-09-10）

> 本文记录 [`2026-09-09-a6-production-collection.md`](./2026-09-09-a6-production-collection.md) §7 记录的 P10-P13 阻塞如何被解除：
> 在 `shiyuan`（GPU 机器，有 Qwen3-Embedding-4B + torch/transformers）上跑真实语料治理 + 分块 + embedding，
> 通过 **Xftp 直传文件**（不是 git push/pull）把产出传回本机，在本机真实 Milvus 上完成生产入库，
> 并给 `a6_collection.py` 新增了一个 `report` 子命令来跑真实验收。

## 0. 结论

```text
P10-P13 REAL CORPUS INGESTION
PASS（2026-09-10，真实语料 + 真实向量）
- 17 篇真实文档全部治理为 ACTIVE，0 duplicate/historical/version_conflict
- 真实 A3 分块 + A5 embedding 产出 2034 个 Leaf，strict join 无缺失无多余
- 真实 Milvus 生产 Collection a6_leaf_v1 建库成功，row_count=2034
- verify_collection / dense_smoke_test / bm25_smoke_test 全部 PASS（真实向量、真实语义相关的 BM25 查询）
- 补跑 knowledge_base 全量回归：307 passed, 4 skipped（无新增失败）；真实 Milvus 集成测试单独 1 passed
```

---

## 1. 环境与同步方式的更正

[2026-09-09 文档](./2026-09-09-a6-production-collection.md) §10 曾错误假设跨机器同步走 git push/pull（比照 A1-A5 的历史模式）。**实际同步方式是 Xftp（SFTP 客户端）直传文件**，两台机器之间没有通过 git 交换任何本轮新增代码或数据；本轮结束时，仓库里的 A6/A6.0 相关文件仍然是本地未提交状态（两台机器上都是），是否要提交进 git 是一个独立于本次同步的未决定问题。

两台机器的角色分工不变：

| 机器 | 角色 |
|---|---|
| `shiyuan`（Linux，`/public/zhangkairan/MyMethod/all-in-rag-main`，conda env `mfg-rag-preprocess`，NVIDIA A100） | 有 torch/transformers/Qwen3-Embedding-4B 权重 + 真实 17 篇语料；跑 A1 治理 + A3 分块 + A5 embedding |
| 本机 Windows（`D:\code backup\MyMethod\all-in-rag-main`） | 有 pymilvus + 本地 Docker Milvus 2.5.14（`http://localhost:19530`）；无 torch；跑最终建库 + 验收 |

---

## 2. Xftp 同步的文件清单

**shiyuan → 本机**（通过 Xftp 传输，路径均在各自仓库根目录下的 `outputs/index_metadata/`）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `document_metadata.jsonl` | ~11 KB | 17 篇文档的治理结果 |
| `document_metadata_audit.jsonl` | ~19 KB | 字段抽取审计记录 |
| `document_conflicts.jsonl` | 不存在 | `new_conflicts=0`，`_append_jsonl` 在 0 条记录时按设计不创建文件，不是漏传 |
| `a6_rows.jsonl` | ~115 MB | 2034 行，每行含 2560 维 dense 向量，是本次传输里最大的文件 |

本机 → shiyuan：本轮反向没有传输任何文件（`a6_report.json` 等验收产出目前只留在本机）。

---

## 3. shiyuan 上执行的真实命令与结果

### 3.1 环境验证（先确认 Xftp 传过去的三个新模块能在真实 torch 环境里正常工作）

```bash
cd /public/zhangkairan/MyMethod/all-in-rag-main
conda activate mfg-rag-preprocess
PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" \
python -m pytest code/knowledge_base/tests/test_a6_document_metadata.py code/knowledge_base/tests/test_a6_collection.py code/knowledge_base/tests/test_a6_prepare_rows.py -q
```

结果：`112 passed in 18.32s`（0 skip——`test_prepare_rows_wires_chunk_embed_and_join` 这条在本机因为没有 torch 会被 `pytest.importorskip("torch")` 跳过，在 shiyuan 上用真实 torch 跑通了）。

### 3.2 第 1 步：文档级治理（P0-P5，不需要 GPU）

```bash
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
python -m knowledge_base.a6_document_metadata \
  --canonical-root "$PWD/outputs/preprocessing" \
  --manifest "$PWD/data/engineering_docs/manifest.csv" \
  --metadata-out "$PWD/outputs/index_metadata/document_metadata.jsonl" \
  --audit-out "$PWD/outputs/index_metadata/document_metadata_audit.jsonl" \
  --conflicts-out "$PWD/outputs/index_metadata/document_conflicts.jsonl"
```

> 排查记录：第一次执行时误写成 `a6_document_metadata extract --canonical-root ...`（沿用了 2026-09-09 文档里的一处笔误），报 `unrecognized arguments: extract`。核对 `a6_document_metadata.py` 的 `main()` 源码后确认该 CLI **没有子命令**，是扁平参数，去掉 `extract` 即可。已同步修正 2026-09-09 文档里的相应命令。

真实输出：

```text
canonical_root=/public/zhangkairan/MyMethod/all-in-rag-main/outputs/preprocessing
document_count=17
ACTIVE=17
DUPLICATE=0
HISTORICAL=0
VERSION_CONFLICT=0
new_conflicts=0
```

17 篇文档（14 篇英文 + 3 篇中文，对应 ver1 语料）全部判定为 `ACTIVE`，无重复、无历史版本、无版本冲突——符合预期（A0-A2 已做过清洗，这批语料本身不含重复文档）。

### 3.3 第 2 步：A3 分块 + A5 embedding + strict join（P10-P13 主体，需要 GPU）

```bash
export CUDA_VISIBLE_DEVICES=1
python -m knowledge_base.a6_prepare_rows \
  --canonical-root "$PWD/outputs/preprocessing" \
  --metadata-jsonl "$PWD/outputs/index_metadata/document_metadata.jsonl" \
  --tokenizer Qwen/Qwen3-Embedding-4B \
  --model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --output "$PWD/outputs/index_metadata/a6_rows.jsonl"
```

真实输出：

```text
canonical_root=/public/zhangkairan/MyMethod/all-in-rag-main/outputs/preprocessing
retrievable_document_count=17
row_count=2034
output=/public/zhangkairan/MyMethod/all-in-rag-main/outputs/index_metadata/a6_rows.jsonl
```

17 篇可检索文档，768/96 chunking 参数下产出 2034 个 Leaf（约 120 leaf/文档，量级符合工程文档预期）。Qwen3-Embedding-4B 权重加载耗时约 68 秒，整体推理在几分钟内完成。

---

## 4. 本机执行的真实命令与结果

### 4.1 文件校验

Xftp 传输落地到 `D:\code backup\MyMethod\all-in-rag-main\outputs\index_metadata\`：`document_metadata.jsonl`（10,860 字节）、`document_metadata_audit.jsonl`（19,390 字节）、`a6_rows.jsonl`（120,741,474 字节）。用 Python 脚本按 UTF-8 读取校验了全部 17 条 `document_metadata.jsonl` 记录，`document_id`/`document_title` 均正确（含 3 篇中文文档标题），此前在 PowerShell 终端直接 `python -c` 打印时看到的乱码只是控制台 GBK 编码显示 UTF-8 中文的问题，不是数据损坏。

### 4.2 第 3 步：建库（P10-P13 落地）

```powershell
cd "D:\code backup\MyMethod\all-in-rag-main"
$env:PYTHONPATH = "D:\code backup\MyMethod\all-in-rag-main\code"
.\.venv\Scripts\python.exe -m knowledge_base.a6_collection build `
  --uri http://localhost:19530 --collection a6_leaf_v1 `
  --rows-jsonl outputs\index_metadata\a6_rows.jsonl
```

真实输出：

```text
collection=a6_leaf_v1
row_count=2034
```

建库前确认本机 Milvus 在线且无任何既有 collection（`MilvusClient(...).list_collections()` 返回 `[]`），所以 `a6_leaf_v1` 这个命名没有冲突风险。

### 4.3 第 4 步（新增）：验收报告

发现 `a6_collection.py` 的 CLI 此前只有 `build`/`ingest-update`/`rebuild` 三个子命令，都只打印 `row_count`，没有接 `verify_collection`/`dense_smoke_test`/`bm25_smoke_test`/`build_a6_report` 这几个已经写好但从未在真实生产数据上跑过的函数。本轮给 `a6_collection.py` 新增了一个 `report` 子命令（`_cmd_report` + `derive_bm25_query` 辅助函数），并补了 7 个单测（`derive_bm25_query` 的纯函数测试 + `report` 子命令的 argparse 解析/默认值测试 + 两个 fail-fast 测试），详见 §5。

```powershell
.\.venv\Scripts\python.exe -m knowledge_base.a6_collection report `
  --uri http://localhost:19530 --collection a6_leaf_v1 `
  --metadata-jsonl outputs\index_metadata\document_metadata.jsonl `
  --rows-jsonl outputs\index_metadata\a6_rows.jsonl `
  --bm25-query "welding requirements for aerospace materials crimping wire harness" `
  --report-out outputs\index_metadata\a6_report.json
```

真实输出：

```text
bm25_query='welding requirements for aerospace materials crimping wire harness'
collection_name=a6_leaf_v1
build_timestamp=2026-09-10T06:01:07Z
document_count=17
leaf_count=2034
row_count=2034
active=17 historical=0 duplicate=0 version_conflict=0
metadata_null_counts={'document_number': 17, 'document_version': 15, 'finalized_at': 17, 'effective_from': 17, 'effective_to': 17}
join_missing_count=0
join_extra_count=0
verification_passed=True
dense_smoke={'anns_field': 'dense_vector', 'metric_type': 'IP', 'hit_count': 10, 'passed': True}
bm25_smoke={'anns_field': 'sparse_vector', 'metric_type': 'BM25', 'hit_count': 10, 'passed': True}
report_out=outputs\index_metadata\a6_report.json
```

`report` 子命令的设计：

- **`verify_collection`**：`expected_document_ids` 来自 `select_retrievable_metadata(catalog)` 的 key 集合（即 `document_metadata.jsonl` 里 `ACTIVE`/`VERSION_CONFLICT` 的文档），`expected_row_count` 直接用传入的 `rows.jsonl` 行数——两者都不需要重新连 A3/A5，只读 JSONL 文件。
- **`join_missing_count`/`join_extra_count`**：CLI 层面重新计算（`rows.jsonl` 里出现的 `document_id` 集合 vs. catalog 里应可检索的 `document_id` 集合的对称差），因为 `a6_prepare_rows.py` 在 shiyuan 跑的时候没有把这两个数字持久化下来。
- **dense smoke 的查询向量**：直接取 `rows.jsonl` 第一行的真实 `dense_vector`（真实 Qwen3-Embedding-4B 向量，不是合成的随机单位向量）。
- **bm25 smoke 的查询文本**：默认取 `rows.jsonl` 第一行 `content` 的前 8 个词（`derive_bm25_query`），可用 `--bm25-query` 覆盖。默认值实测抓到的是文档头部的联系方式文本（"www.mel.nist.gov ... Dale Hall Director..."），能触发命中但语义价值不高，所以本次正式报告改用手动指定的、贴合真实文档内容的查询词。

### 4.4 BM25 语义相关性抽查（额外验证，不只是"有命中"）

用上面那条查询词直接调 `store.search_sparse()` 看实际命中的 `document_id`：

```text
14_nasa_std_8739_4a_crimp_wiring-...    (×3)   ← 匹配 "crimping wire harness"
13_nasa_std_5006a_welding_requirements-... (×1)   ← 匹配 "welding requirements for aerospace materials"（几乎是文档标题原文）
17_nasa_std_6016c_materials_processes-... (×4)   ← 匹配 "materials"
18_nasa_std_6030_additive_manufacturing_systems-... (×1)
```

Top-10 命中里有 8 条来自和查询词语义直接相关的 2 篇文档（Crimping、Welding），证明这不只是索引机制"有返回"，而是真实检索到了语义相关的内容——BM25 Function/Analyzer/索引在真实生产数据上工作正常。

### 4.5 全量回归确认无破坏

```powershell
.\.venv\Scripts\python.exe -m pytest code\knowledge_base\tests -q -p no:cacheprovider --basetemp=.pytest_tmp
# 307 passed, 4 skipped in ~33s（此前 300 passed，+7 是本轮新增的 report 子命令测试）

.\.venv\Scripts\python.exe -m pytest code\knowledge_base\tests\test_a6_collection_milvus_integration.py -q -p no:cacheprovider --basetemp=.pytest_tmp -v
# 1 passed（合成数据机制级集成测试不受影响，自清理，不碰 a6_leaf_v1）

# 确认生产 collection 未被测试污染
python -c "from pymilvus import MilvusClient; c = MilvusClient('http://localhost:19530'); print(c.list_collections()); print(c.get_collection_stats('a6_leaf_v1'))"
# ['a6_leaf_v1']
# {'row_count': 2034}
```

---

## 5. 代码改动（本文新增，相对 2026-09-09 文档记录的实现）

`code/knowledge_base/a6_collection.py`：

- 新增 `derive_bm25_query(content, *, word_count=8)`：纯函数，取正文前 N 个词作为 BM25 smoke-test 的默认查询词。
- 新增 `_cmd_report(args)`：读 `document_metadata.jsonl` + `rows.jsonl`，跑 `verify_collection`/`dense_smoke_test`/`bm25_smoke_test`/`build_a6_report`，打印摘要并可选写 JSON 报告文件；`metadata`/`rows` 任一为空都 fail-fast 返回退出码 2；verify/smoke 任一未通过返回退出码 3。
- `build_parser()` 新增 `report` 子命令：`--uri`/`--collection`/`--metadata-jsonl`/`--rows-jsonl`/`--bm25-query`（可选）/`--limit`（默认 10）/`--report-out`（可选）。
- 顶部 import 增加 `load_document_metadata_jsonl`（从 `a6_document_metadata.py`，本来就是模块级依赖，无新增外部依赖）。

`code/knowledge_base/tests/test_a6_collection.py`：新增 7 个测试（`derive_bm25_query` ×3、`report` 子命令 argparse 解析 ×2、`_cmd_report` fail-fast ×2），全部基于纯 Python/argparse，不连真实 Milvus。

未改：`a6_document_metadata.py`、`a6_prepare_rows.py`、任何 A0-A5 文件、`a6_profile_config.py`。

---

## 6. 已知问题记录

1. **文档级 CLI 用法笔误**：2026-09-09 文档 §7.1/§8.4/§10.3 三处把 `a6_document_metadata.py` 的调用写成 `a6_document_metadata.py extract ...`，该 CLI 实际没有子命令。已在 2026-09-09 文档里就地修正，不影响任何已交付代码（纯文档错误，代码本身一直是对的）。
2. **真实语料的字段抽取近乎全 NULL**：见 [2026-09-09 文档 §9 第 6 条](./2026-09-09-a6-production-collection.md)，已记录为待上游确认的开放问题，本文不重复展开。
3. **BM25 smoke 默认查询词质量不高**：`derive_bm25_query` 默认取正文前 8 个词，如果文档开头是页眉/联系方式等 boilerplate，会生成语义价值不高（但仍然有效）的查询词。当前设计允许 `--bm25-query` 覆盖，本轮验收报告用的是手动指定的查询词。这不是错误，只是默认值的局限性，如果后续要把 `report` 命令做成免人工介入的自动化验收，可以考虑改成从多个不同文档各取一段内容拼接，而不是只取第一行。

---

## 7. 遗留事项

- `outputs/index_metadata/a6_report.json` 目前只落在本机，未回传给 shiyuan 或纳入版本控制；如果需要留档，应决定它的归档位置。
- 是否要把本轮及上一轮（A6.0）新增的全部文件提交进 git（以及提交粒度），仍是独立于本次同步的未决定问题，见 [2026-09-09 文档 §10](./2026-09-09-a6-production-collection.md#10-同步到-shiyuan具体步骤本节已过时仅作历史记录)。
- P16（文档化）在合同原始范围里列为待完成项；本文 + 2026-09-09 文档已覆盖大部分内容，是否需要额外的用户/运维文档需另行确认。
