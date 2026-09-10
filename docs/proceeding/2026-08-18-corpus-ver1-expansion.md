# 语料 ver1 扩容：中文文档接入与 A0–A5 重新验收（2026-08-18）

> 本文是**语料扩容记录**，不是模块实现交接。A0–A5 代码/数值合同均未改动，本轮只是把新增语料跑通并记录结果。
> 范围判定原则见 `docs/data-corpus.md` §2.1（ver1 = 数字原生 PDF，不依赖 OCR；ver2 = 需要 OCR 的扫描件，本轮不处理）。

## 当前状态

```text
CORPUS INTAKE
FROZEN / PASS
- ver1 全量 17 篇文档（3 已在产线 + 14 新增：11 EN + 3 CHN）
- ver2 延后 19 篇（10 EN 扫描件 + 9 CHN 扫描件/空文本层）
- manifest.csv 从 24 行扩到 27 行（新增 3 行 CHN，id=25/26/27）

A0 RE-RUN（14 篇新文档）
PASS
- 全部 status=success，均已发布到统一 canonical root
- RapidOCR 空结果告警仅出现在图片型元素上，不影响整体 status

A1 → A2 RE-RUN
PASS
- discovered=17 / loaded=17 / registry_size=17
- 无 "No manifest row for source_sha256" 报错

A3 RE-RUN（section_profile + leaf_chunker）
PASS
- document_count=17 / terminal_section_count=2254 / leaf_count=2034
- body_tokens_max=13786（09_nistir7218 单个 heading 正文，非表格）
- leaf_tokens_max=1019（表格 leaf，来自中文标书）；768/96 参数未变
- 全部不变量 = 0（duplicate_chunk_id / section_id_collision / non_contiguous_chunk_index / table_split_violation / zero_leaf_document）

A4 RE-RUN
PASS
- section_node_count=2433 / leaf_count=2034（与 A3 一致）
- 全部不变量 = 0（missing_parent_count / cross_document_parent_count / hierarchy_cycle_count / invalid_heading_level_relation_count）

A5 RE-RUN（real smoke）
PASS
- leaf_count=2034 / embedding_count=2034 / embedding_dimension=2560
- input_token_max=1020 << 8192；over_limit_leaf_count=0
- nan/inf/invalid_norm/duplicate/missing 全 0；determinism_failure_count=0
- vector_norm ∈ [0.9999998807907104, 1.0000001192092896]，mean=1.0

回归测试
PASS
- Linux pytest: 187 passed in 8.38s，0 skipped/failed
```

服务器：`shiyuan`，`/public/zhangkairan/MyMethod/all-in-rag-main`，conda 环境 `mfg-rag-preprocess`（沿用 A0–A5 历次验收同一环境）。

---

## 1. 背景与触发

`data/` 目录新增了一批中文企业技术规范书（`data/CHN/`，供电局/电网设备类，非公开），同时英文语料的物理目录做了调整（`data/EN/`）。需要把这批新语料纳入知识库构建，同时明确：

- A2 `document_registry` 靠 `sha256` 精确匹配 manifest 行做身份关联，不依赖文件路径/目录结构，因此英文文档搬家本身不影响正确性，只是 `local_path` 列的卫生问题（本轮未处理，不影响 join）。
- ver1 明确只处理**数字原生 PDF**（文本层可直接提取、不依赖 OCR）；任何需要 OCR 才能正确转换的文档（无论中英文）一律推迟到 ver2，ver2 还需先做中文 OCR 验证（`docs/preprocessing.md` §11）才能开工，本轮不涉及。

## 2. 语料分类依据

分类基于逐份 PDF 文本抽取的**第一轮筛查**（非 A0 真实 Docling/RapidOCR 流水线结果），用于圈定候选范围；每篇最终是否真正进入 ver1，以其实际跑通 A0 CLI 并人工核对 `document.md` 为准。

### 2.1 英文语料（24 份候选）

| 结果 | 文件 | 依据 |
|---|---|---|
| 已在产线（3） | `15_nasa_std_4003a_electrical_bonding`、`19_nasa_std_6033_am_equipment_facility`、`21_nasa_std_5009c_nde` | 此前已跑完 A1–A5，271 Leaf 生产事实（见 A5 real corpus 验收记录） |
| 新增入 ver1（11） | `09_nist_smart_machining_systems_nistir7218`（2005）、`10_nistir7734_machining_measurement_process_planning`（2010）、`13_nasa_std_5006a_welding_requirements`、`14_nasa_std_8739_4a_crimp_wiring`、`16_nasa_std_8739_1b_polymeric_bonding`、`17_nasa_std_6016c_materials_processes`、`18_nasa_std_6030_additive_manufacturing_systems`、`20_nasa_std_5017b_mechanisms`、`22_nasa_std_8739_12a_metrology_calibration`、`23_nasa_hdbk_8739_19_2_mte_specifications`、`24_nasa_hdbk_8739_19_3_measurement_uncertainty` | 文本抽取干净、无乱码；均为 2005 年后现代标准/手册，born-digital。`13_nasa_std_5006a` 此前已作为嵌套表格 Golden 测试验证过表格处理正确性，不是待验证项 |
| 推迟到 ver2（9） | `01`–`08` 全部 8 份 1970s Materials Data Handbook、`12_nasa_machining_titanium_chatter_tool_wear`（1974） | 文本抽取明显乱码（如 `07_stainless_301` 页面直接印有 "REPRODUCIBILITY OF THE ORIGINAL COPY IS POOR"），是扫描件走旧 OCR 得到的文本层 |
| 推迟到 ver2（1，保守判定） | `11_nasa_machine_tools_fixtures_1974` | 封面/文献编目页明显是扫描 OCR 乱码（`N74-30964`、`CSCL 131 Unclas`），正文虽干净但整体判定为扫描来源，保守起见与其他 1974 年代文档一并推迟 |

规律：`joining` / `manufacturing_guidelines` / `quality_inspection`（NASA-STD-*/HDBK-*，2005 年后）全部 born-digital；`material`（1970s 手册）与 1974 年的两份 `machining` 报告是扫描件。

### 2.2 中文语料（12 份候选）

| 结果 | 文件 | 依据 |
|---|---|---|
| 新增入 ver1（3） | `YH5WR-17_45型避雷器技术条件书`（5页）、`附件1：深圳供电局有限公司人才发展中心2026年水贝培训基地变电站仿真实训室设备购置项目技术规范书`（31页）、`附件1：电动汽车兆瓦级充电设备技术规范书`（20页） | 文本抽取完整，标题层级清晰（1/2.1/2.2.1 编号），带目录页 |
| 推迟到 ver2（9） | `标的1`–`标的7` 全部 7 份、`主变中性点直流电流测量装置...DCT-80A`、另 1 份无具体标题的 `技术规范书.pdf` | 全篇每页文本抽取为空，要么是扫描件要么是不可提取的字体编码；"标的1–7" 疑似出自同一套生成/扫描流程 |

### 2.3 汇总

```text
ver1 总计：17 篇（3 已有 + 14 新增：11 EN + 3 CHN）
ver2 延后：19 篇（10 EN + 9 CHN）
候选总量：36 篇（24 EN + 12 CHN）
```

## 3. Manifest 变更

`data/engineering_docs/manifest.csv` 从 24 行扩到 27 行，新增 3 行中文文档（`id=25/26/27`，`category=power_equipment`）：

| id | title | local_path | pages | sha256 |
|---|---|---|---|---|
| 25 | YH5WR-17/45型避雷器技术条件书 | `CHN/YH5WR-17_45型避雷器技术条件书.pdf` | 5 | `c0c3c9796c3ca7b39c3556f91137ef2886075708529949a7b09de7b2eeaa46f5` |
| 26 | 深圳供电局有限公司人才发展中心2026年水贝培训基地变电站仿真实训室设备购置项目技术规范书 | `CHN/附件1：深圳供电局有限公司人才发展中心2026年水贝培训基地变电站仿真实训室设备购置项目技术规范书.pdf` | 31 | `162d87245ef57765d554abe8aba83fa05ea8636178120b63d28881de5c138e84` |
| 27 | 电动汽车兆瓦级充电设备技术规范书 | `CHN/附件1：电动汽车兆瓦级充电设备技术规范书.pdf` | 20 | `61e95bb4c9d1c8a5a0ea76ff56eb8440f0009ca11bf4d9a37c732aa5af7624c7` |

这 3 篇没有公开可访问的 `source_url`（企业内部资料），`source_url` 留空，`document_registry._candidate_source` 会按优先级回退到 `local_path` 作为 citation source（`code/knowledge_base/document_registry.py`，逻辑未改动）。`data/CHN/` 保持平铺，未建子目录，`category` 统一标为 `power_equipment`。

被推迟到 ver2 的 19 篇（10 EN + 9 CHN）**未加入本次 manifest**，因为 `document_registry` 只对实际跑过 A0、存在 `document.md` 的文档做校验；等 ver2 真正处理这些文档时再补行。

## 4. 全量重跑结果

以下均为服务器 `shiyuan` 实测输出（非本地模拟）。

### 4.1 A0（14 篇新文档）

11 篇英文 + 3 篇中文全部 `status=success`，均发布到统一 canonical root（`outputs/preprocessing`）。处理过程中出现的 "RapidOCR returned empty result!" 告警来自文档内部分图片型元素，不影响整篇文档的最终 `status=success`。

### 4.2 A1 → A2

```text
discovered=17
loaded=17
registry_size=17
```

无 `No manifest row for source_sha256` 报错，17 篇文档（3 已有 + 14 新增）全部通过 manifest 关联，`document_title` / `source` 正确赋值。

### 4.3 A3 — section_profile

```text
document_count=17
terminal_section_count=2254
terminal_sections_with_body=2251 / terminal_sections_without_body=3
body_tokens_max=13786（09_nist_smart_machining_systems_nistir7218，单个 heading 正文，非表格）
table_count=419
table_token_p50=342.0 / table_token_p95=795.4
max_table_tokens=1019
anomalies: missing_page_marker=0 / malformed_page_marker=0 / page_order_error=0 / unheaded_document_body=0
```

`body_tokens_max=13786` 明显高于此前 271-Leaf 语料的量级，来自 `09_nistir7218` 一个跨 37 页（35-72）的 level-6 heading 正文（NIST 报告本身章节粒度较粗），不是异常；A3.2 chunking 会按 768/96 把它正确切成多个 normal leaf，不产生超限 leaf（见下）。

### 4.4 A3 — leaf_chunker

```text
document_count=17
terminal_section_count=2254 / terminal_sections_with_leaf=1726 / empty_terminal_sections=528
leaf_count=2034
leaf_tokens_p50=218.0 / p75=512.0 / p90=721.0 / p95=755.0 / max=1019
leaf_tokens_mean=304.323009
normal_leaf_count=2007 / normal_leaf_max_tokens=768
oversize_table_leaf_count=27 / oversize_table_max_tokens=1019
sections_split_count=117 / sections_unsplit_count=1609
不变量：page_invalid=0 / duplicate_chunk_id=0 / section_id_collision=0 / non_contiguous_chunk_index=0 / table_split_violation=0 / zero_leaf_document=0
```

`leaf_tokens_max=1019` 来自一个超过 768 但符合"表格不可拆分"规则的 oversize table leaf（27 个此类 leaf，均来自各文档表格密集章节），768/96 分块参数未因中文语料改变。

### 4.5 A4 — section_hierarchy

```text
document_count=17
section_node_count=2433
heading_section_count=2429 / document_root_count=4
top_level_section_count=214
leaf_count=2034（与 A3 一致）
leaf_section_resolution_failures=0
parent_link_count=2219
不变量：missing_parent_count=0 / cross_document_parent_count=0 / hierarchy_cycle_count=0 / invalid_heading_level_relation_count=0
```

`document_root_count=4` 表示 17 篇文档中有 4 篇存在首个 heading 之前的正文（document-root 级 section），其余 13 篇正文完全落在 heading 之下，属正常分布差异，不是异常。

### 4.6 A5 — dense_embedding real smoke

```text
leaf_count=2034 / embedding_count=2034
embedding_dimension=2560 / embedding_dtype=float32
input_token_min=3 / p50=219.0 / p95=756.0 / max=1020（<< max_input_tokens=8192）
over_limit_leaf_count=0
nan_vector_count=0 / inf_vector_count=0 / invalid_norm_count=0
duplicate_chunk_id_count=0 / missing_vector_count=0
vector_norm_min=0.9999998807907104 / max=1.0000001192092896 / mean=1.0
determinism_failure_count=0
```

`input_token_max=1020` 比 A3 leaf 层 `leaf_tokens_max=1019` 高 1，是 A5 推理分词器与 A3 profiling 分词器 `add_special_tokens` 口径差异导致（A5 会追加 1 个 EOS token），与 A5 冻结记录中的既有结论一致，不是新问题。A5 数值合同（4B / 2560-d / last-token pooling / L2 normalize / 无 silent truncation）未变。

### 4.7 回归测试

```text
187 passed in 8.38s, 0 skipped, 0 failed
```

含 A1–A4 回归与 A5 单测（T81–T106 等）全部通过。

## 5. 结论

- ver1 语料从 3 篇扩容到 17 篇（新增 11 EN + 3 CHN），A0→A5 全链路在合并语料上重新验收，硬条件（不变量、NaN/Inf/norm、determinism、pytest）全部为 0/PASS。
- 768/96 chunking 参数、A5 数值合同（4B/2560-d/last-token/L2/无截断）均未因本次语料扩容而改变。
- ver2 范围（10 EN + 9 CHN，共 19 篇）明确记录在案，后续需要先完成中文 OCR 验证（`docs/preprocessing.md` §11）才能开工，本文不覆盖 ver2 实现。
- 相关文档更新：`docs/data-corpus.md` §2/§2.1/§4 已同步反映中文语料接入与 ver1/ver2 准入原则；`data/engineering_docs/manifest.csv` 已扩至 27 行。
