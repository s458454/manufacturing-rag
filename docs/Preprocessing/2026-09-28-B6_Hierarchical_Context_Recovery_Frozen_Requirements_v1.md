# B6 Hierarchical Context Recovery — Frozen Requirements v1

> Status: **FROZEN / READY FOR IMPLEMENTATION**
>
> Contract version: `b6-contract-v1`
>
> 本文档是 B6 Hierarchical Context Recovery v1 的权威需求。
>
> B6 把 B5 Top-5 Leaf 恢复成去重、合并、受预算约束的 Evidence，交给 C1。
> B6 不重新检索，不重新打分，不分配 `[E1]` / `[E2]`。

---

# 1. 位置

```text
B5 Cross-Encoder Reranking
→ B6 Hierarchical Context Recovery
→ C1 Prompt / Evidence Assembly
```

B6 决定哪些 Evidence、正文、恢复粒度、去重与合并、排序，以及 Evidence 是否超过 Evidence Budget。

C1 分配 `[E1]` / `[E2]`，补 Source / document title，序列化 Prompt，并在包装之后做最终 context-window 校验。`[E1]` 不是 B6 字段。

---

# 2. 输入

输入是 `RerankingResult.candidates`，最多 5 个 `RerankedCandidate`。

每个 Candidate 至少包含：

```text
chunk_id
document_id
section_id
chunk_index
page_start
page_end
content
rerank_rank
rerank_score
rrf_score
fused_rank
```

B6 用 `rerank_rank` 决定优先级。B6 不重新计算 `rerank_score`，也不给 Parent 制造 `rerank_score`。

超过 5 条 fail-fast。0 条返回空 Evidence。

---

# 3. Section Store

生产存储只冻结到：

```text
sections.jsonl sidecar
+ 启动时载入内存
+ 与 A6 使用同一批 retrievable_documents 上的同一次 A3
```

`sections.jsonl` 每行一个 Section，字段仅限：

```text
section_id
document_id
parent_section_id
kind
heading_level
heading
source_start
source_end
page_start
page_end
```

Section 正文不写入 Section Store。正文用 `document_id + source_start / source_end` 从 canonical `document.md` 恢复，并剥掉 `<!-- PDF page N -->`。

构建链路：

```text
all canonical documents
→ A6 metadata filtering（ACTIVE + VERSION_CONFLICT）
→ retrievable_documents
→ parse_loaded_documents
→ chunk_parsed_documents          ← 唯一一次 A3
   ├─ leaves → A5 → A6 rows
   └─ same ChunkingResult → A4 hierarchy → sections.jsonl
```

`HISTORICAL` / `DUPLICATE` 不进入 Section Store。在线阶段禁止调用 `build_section_hierarchy()`，禁止另跑一套 hierarchy。现有 A4 debug JSON 仍然只是 debug artifact。

禁止修改 A4，使 `document_root` 强行成为顶层 heading 的 Parent。当前 A4 里顶层 heading 的 `parent_section_id` 通常是 `None`。若存储结果里 `document_root` 确实是某个 Section 的 direct parent，B6 按普通 direct parent 处理。

加载 sidecar 时复用 A4 已有结构约束：缺字段、重复 `section_id`、parent 不存在、cross-document parent、cycle、parent heading level 不低于 child，一律失败。绑定 canonical 文档时检查 source span 与 page range：span 落在文档行数内，页码与 A4 的 page provenance 一致。这些检查发生在服务初始化，不发生在每个 query。

---

# 4. Leaf catalog 与 runtime identity guard

`outputs/index_metadata/a6_rows.jsonl` 是 B6 的生产部署依赖，不只是 A6 的临时构建产物。启动时读取它，丢掉 `dense_vector`，在内存中保留叶子身份和 `content`。这一版不为此增加 SQLite，也不对 Milvus 做 scalar query。删掉该文件后，Milvus 与 B3–B5 仍可工作，B6 无法启动。

每个留下的 B5 `chunk_id` 必须能在当前加载的 rows catalog 中找到，否则立即失败。这叫 **matched-Leaf runtime identity guard**。它只证明该 Leaf 存在于当前 catalog。它不证明 Milvus、`sections.jsonl`、canonical `document.md`、`rows.jsonl` 来自同一次完整 build。

build-level consistency：

```text
[PROVISIONAL: fingerprint/manifest]
```

这是当前唯一已知且接受的完整性缺口。文档不得写成“已完成 same-build 校验”。

邻居用 `document_id + section_id + chunk_index ± 1` 从 catalog 推导。不持久化 prev/next chunk id。

---

# 5. 启动装载

生产服务初始化时一次性完成：

```text
load sections.jsonl
load A6 rows catalog（丢掉 dense_vector）
load canonical document.md
parse page markers / line starts
load Qwen2.5 tokenizer
```

请求阶段只查内存。`recover()` 接收已经注入的 store、catalog、documents、tokenizer。禁止每个 query 重新 `load_markdown_documents()` 或 `parse_markdown_document()`。

Tokenizer 是 C2 的 Qwen2.5-7B-Instruct，不是 A3/A5 的 Qwen3-Embedding tokenizer。计数使用 `encode(..., add_special_tokens=False)`。

逻辑 ID：`Qwen/Qwen2.5-7B-Instruct`。

Revision：`a09a35458c702b33eeacc393d103063234e8bc28`，与 `models/retrieval-models.manifest.json` 一致。运行时加载该 revision 的本地 artifact。禁止拉取 Hugging Face latest。

---

# 6. Evidence Budget

```text
generation_context_limit = 32768 tokens
YaRN = disabled
nominal_evidence_budget_ratio = 0.75
nominal_evidence_budget = 32768 × 0.75 = 24576 tokens
```

24576 是 Evidence ceiling，不是使用目标。恢复到更小的完整语义证据就停。禁止为了填满 75% 继续向祖先扩大。

B6 保证 recovered Evidence 的 token 总和不超过 nominal / configured Evidence Budget。

C1 在真正构造 Prompt 之后仍然检查 `input_tokens + max_new_tokens <= 32768`。该精确核算仍是 `[PROVISIONAL]`，不阻塞 B6。

不设置 `max_single_evidence_ratio`。coverage、核心 Leaf 与总预算都允许时，一个 14000 token 的 Parent 可以在 24576 内恢复。

---

# 7. 核心 Leaf

先按 `chunk_id` 去重，保留更优 `rerank_rank`。

然后做 runtime identity guard。

再做 Core admission：

```text
unique leaves <= budget → 全部保留
> budget → 从 Top5、Top4、… 依次丢弃
只剩最高优先级 Leaf 仍超过 budget → fail-fast
```

禁止 silent truncation。更高 `rerank_rank` 优先保留（数字更小者优先）。

---

# 8. 恢复顺序

正式顺序是两轮。禁止写成“每个 Evidence 沿 Leaf → Window → Section → Parent 升完，再处理下一个”。

```text
所有 Top-5 核心 Leaf 先保底
→ Pass 1：所有组先争取 Leaf → Window → Section
→ Pass 2：再考虑 eligible direct Parent
```

代价：低排名 Section 可能先占住预算，使高排名 Parent 最后升不上去。被保住的 Top1 Leaf / Section 仍然在。

升一级是替换，不是把 Leaf 与 Section 叠在一起。

```text
incremental_cost = new_evidence_token_count - token_count_of_evidence_being_replaced
```

只有 `current_total + incremental_cost <= budget` 才升级。

## 8.1 Pass 1

按 `evidence_priority` 从优到差处理每个 nearest Section 组。同一 `section_id` 的叶子是一组，priority 是组内最小 `rerank_rank`。一组做完 Window 与 Section 之后，才处理下一组。全部组完成 Pass 1 之后才进入 Pass 2。

对当前组：

1. 为每个 matched Leaf 构造同 `document_id`、同 `section_id` 的 `chunk_index ± 1`。
2. 先合并重叠或 `chunk_index` 相邻的窗口，再计一次预算。例如 `[9,10,11] ∪ [11,12,13] → [9,10,11,12,13]`，`incremental_cost = tokens(9~13) - tokens(matched leaves 10 与 12)`。同一个 chunk 只计一次。禁止先按 Leaf 逐个扣预算、最后再合并。
3. 中间有 gap 的窗口保持多个 Evidence，不补洞。每个已分离的窗口各做一次完整窗口的 incremental cost。
4. 整窗放得下则替换为 `leaf_window`。放不下则这些 matched Leaf 保持 `leaf`。
5. 然后尝试 nearest Section。`incremental_cost = section_tokens - 当前这一组证据的 token`。放得下则整组变成一个 Section。窗口失败不跳过这次 Section 尝试。

Section 是 A4 的真实 Markdown semantic Section，不按 heading level 数字硬编码。

同一 Section 只恢复一次。priority 是支持它的 matched Leaf 中最小的 `rerank_rank`。必须保留 supporting Leaf provenance。

### 窗口边界

Section 首尾本来就只有一侧邻居时，允许单侧窗口：

```text
只有 next → matched + next
只有 previous → previous + matched
```

previous 与 next 都存在时，必须尝试完整 `±1`。完整窗口放不下，就保持 matched Leaf。禁止改成只拿 previous 或只拿 next。

禁止跨 Section 寻找邻居。最大范围是 `chunk_index - 1 / chunk_index / chunk_index + 1`。v1 不扩展到 `±2`。

A3 `overlap_tokens = 96`。B6 只做 `chunk_id` 去重，不做模糊文本 overlap trimming。

## 8.2 Pass 2

Parent candidate 来自原始 matched Leaf 的 nearest Section groups，不看 Pass 1 结束后的 `recovery_mode`。

至少两个不同 nearest Section 共享同一个 `parent_section_id`，才形成 candidate。其中一个 Pass 1 后仍是 `leaf` 或 `leaf_window` 时，candidate 仍然成立。

v1 只检查 direct Parent。禁止 recursive ancestor expansion、arbitrary LCA，以及因为更高祖先相同而继续向上。

coverage：

```text
numerator = 所有 unique matched direct-child Section 的完整正文 token
denominator = 完整 Parent 正文 token
parent_coverage = numerator / denominator
```

分子规则：

> 对 unique matched direct-child Section 的 source spans 做真正的区间并集。重叠或紧邻、中间没有缺口的区间合并后恢复一次。不连续的区间分别恢复、分别计数，再求和。不得用最小 start 到最大 end 的 bounding span 代替并集。

Parent 下 Child A、B、C 依次排列且只命中 A 与 C 时，分子是 `tokens(A) + tokens(C)`。中间没命中的 B 不算进去。同一 Section 只计一次。两个 matched span 若重叠，重叠正文只计一次。叶子条数和 `rerank_score` 不参与。coverage 使用完整 Child Section，而不是 matched Leaf 的长度。

`parent_coverage >= 0.50` 才允许 Parent candidate。

升级成功时，Parent 替换这些 Child groups **当前留下的** Evidence。当前粒度是 `section`、`leaf_window` 或 `leaf` 都可以被替换。

```text
Parent.evidence_priority = 所有 supporting matched Leaf 的 best rerank_rank
incremental_cost = parent_tokens - 这些当前 Evidence 的 token
```

放不下则 Child 保持 Pass 1 的粒度。

Parent candidates 按 best `rerank_rank` 从优到差尝试。

---

# 9. 最终身份与排序

相同身份只留一条。身份按恢复类型定义：

```text
parent_section → section_id
section        → section_id
leaf           → chunk_id
leaf_window    → 该窗口实际包含的 ordered chunk_ids
```

同一 Section 里的 `leaf_window [9,10,11]` 与 `leaf_window [24,25,26]` 必须同时保留。禁止用 `recovery_mode + section_id` 把它们并成一条。

然后按 `evidence_priority` 升序。相同优先级按 `matched_chunk_ids` 字典序。禁止按文件名、页码或章节位置重排。

`matched_chunk_ids` 只含预算留下的 B5 Leaf。`context_chunk_ids` 只含窗口补进的邻居。Section 与 Parent 的 `context_chunk_ids` 为空。

`section_path` 只沿已存储的 `parent_section_id` 收集 heading，作为标签。这条链不把祖先正文放进 Evidence。

---

# 10. 输出

```python
@dataclass(frozen=True)
class RecoveredEvidence:
    document_id: str
    recovery_mode: str  # parent_section / section / leaf_window / leaf
    section_id: str | None
    section_path: tuple[str, ...]
    page_start: int
    page_end: int
    content: str
    evidence_priority: int
    matched_chunk_ids: tuple[str, ...]
    context_chunk_ids: tuple[str, ...]
    supporting_rerank_ranks: tuple[int, ...]
    token_count: int
```

不含 `[E1]`，不含 `rerank_score`。

---

# 11. 明确禁止

```text
再次 Dense/BM25 retrieval
再次 RRF
再次 Cross-Encoder rerank
给 Parent 伪造 rerank_score
无限向 ancestor 扩张
arbitrary LCA
跨 Section 找 neighbor
previous 与 next 都存在时因预算不足临时只选一侧
硬截 Section 中间正文
为了填满 75% Context 强行扩大
按页码或文件名重新排序
第二套向量库
第二套 hierarchy
overlap 文本裁剪
per-evidence 比例上限
修改 A4 让 document_root 强行成为顶层 heading 的 Parent
```

---

# 12. 仍未实现

```text
build fingerprint / manifest
C1 对 System / Query / metadata / max_new_tokens 的精确 token accounting
```
