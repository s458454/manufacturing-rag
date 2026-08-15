# A0 预处理剩余工作实现说明：Region-OCR Visual Isolation + Native Table Empty Logical Cell

> 完成时间：2026-08-14 22:10 (UTC+8)
>
> 修订：2026-08-15（UTC+8）——补上 **Task 2.5：TableFormer 格子 AABB 外的 source 视为页眉/页脚污染并排除**。
> 不改 Task 1。不引入 NASA / 页码 / 关键词特判。
>
> **A0 冻结（同日）：** V0.1 将嵌套/格子归属 0 或 ≥2 的表视为结构不可信，不收表体。
> 5006A 第 47 页属此条。Golden 已改为：36–46 与 48 必须 accepted；**第 47 页必须 `rejected` 且不得进入 Markdown**。
> 方向 / 乱码阈值不调参，不属于本冻结范围的剩余工作。
>
> 对应需求：用户在本轮会话中给出的「A0 当前剩余需求 v2」（唯一需求版本，替代此前讨论并废弃的
> VisualRegion Resolver / Composite Figure / reading-order 推断等方案）。
>
> 服务器验收（2026-08-15）：Task 1、空单元格、AABB 页眉污染已通过。第 47 页嵌套表按冻结口径 fail-closed。
> Golden 合同已改为不要求第 47 页 accepted，并要求该页必须 `rejected`。

## 1. 需求回顾（浓缩版）

产品行为要求（`document.md` 视角）：

| 原 PDF 内容 | `document.md` |
| --- | --- |
| 普通 native/OCR 正文 | 保留（继续走现有 quality gate） |
| 图片/工程图本体、图内文字 | 不保留，只保留可信图注 |
| 扫描/OCR/mixed/image-only 表格本体 | 不保留，只保留可信表注 |
| native 数字表格（结构可信） | 转 Markdown Table 保留，合法空单元格显示为空 |
| native 数字表格（结构不可信） | 不保留，只保留可信表注 |

拆成两个独立任务：

- **Task 1 — Region-OCR Visual Isolation**：修复 Figure 4（page 25）文字泄露。利用
  `get_ocr_rects()` 已经算好的、真正送给 RapidOCR 的矩形坐标，只对 `region_ocr` 页面生效：若某个
  OCR 矩形与 `PictureItem` 或 `source_kind ∈ {ocr, mixed, image_only}` 的 `TableItem` 有实际几何
  相交，则该矩形内任何普通 `TextItem`（无论其自身是不是 `from_ocr`）都不得进入 `document.md`，
  trusted caption 例外。`full_page_ocr` / `native_only` 页面不受影响。Native 表格不参与此判定。
- **Task 2 — Native Table Empty Logical Cell**：修复 Requirements Compliance Matrix（page 36-48）
  被整表拒收的问题。原因是 TableFormer 对合法空白单元格（如 "Applicable"/"Comments" 列的留白格）
  不产生结构 cell，旧逻辑一旦发现逻辑网格有坐标未被覆盖就直接拒收整表。新逻辑要求：先完成所有真实
  source text 的唯一归属与 100% 守恒校验，**只有这些都通过之后**，才允许对剩余的"缺失坐标"做
  row/column 结构证据判定；两个维度都有真实候选覆盖才算 `provably_empty`，补一个不参与 ownership/
  bbox/文本吸收的 synthetic 空单元格；否则整表继续 `rejected`。

完整逐条要求见本轮对话原文；本文档只记录**我实际做了什么、改了哪些文件、为什么这样改**。

## 2. Task 1 实现细节

### 2.1 `docling/docling/models/base_ocr_model.py`

`BaseOcrModel.get_ocr_rects()` 原本只把 `route_requested` 和 `ocr_rectangle_count` 写入
`page.ocr_audit`，矩形坐标本身没有被持久化。新增：

```python
page.ocr_audit.update(
    {
        "route_requested": route,
        "ocr_rectangle_count": len(selected_rects),
        "ocr_rectangles": [
            {"l": float(rect.l), "t": float(rect.t), "r": float(rect.r), "b": float(rect.b)}
            for rect in selected_rects
        ],
    }
)
```

- 没有重新计算任何几何：`selected_rects` 就是这一页真正被裁图送进 RapidOCR 的矩形（`region_ocr`
  时是逐图连通域，`full_page_ocr` 时是整页一个矩形，`native_only` 时是空列表）。
- 坐标沿用 `find_ocr_rects()` 已经采用的 `CoordOrigin.TOPLEFT`，与需求里给的示例 JSON 一致，不需要
  额外转换。
- 这是本次唯一改动的 Docling 快照文件；`RapidOcrModel.__call__()` 的 OCR 路由/裁图逻辑本身完全未动。

### 2.2 `code/preprocessing/pdf_preprocess.py`

#### 新增两个几何 helper（紧跟 `_item_page_numbers` 之后）

- `_item_page_bboxes(item)`：返回一个 item 每个 `prov` 条目的 `(page_no, 原始 bbox)`。
- `_boxes_intersect(a, b)`：两个已经归一化为 `[left, top, right, bottom]`（TOPLEFT，`left<=right`、
  `top<=bottom`）矩形的正面积重叠判定。必须 `max(left) < min(right)` 且 `max(top) < min(bottom)`；
  单纯边缘或角点接触不算相交。OCR 矩形已经做过膨胀，接边不足以证明它属于某张 Picture / deferred
  table。不引入任何 IoU / 面积阈值。

#### `SemanticProjection` dataclass 新增两个审计字段

```python
visual_ocr_rectangles_by_page: dict[int, list[list[float]]]
visual_ocr_isolated_refs: set[str]
```

只有最终、驱动 `document.md` 的那次 `build_semantic_projection` 调用会真正填充它们；其余调用
（table 抽取前的两次 provisional 调用）永远拿到空字典/空集合，行为与改动前完全一致。

#### `build_semantic_projection()` 新增 `page_heights` 形参与三段逻辑

签名变为：

```python
def build_semantic_projection(
    document: Any,
    pages: list[dict[str, Any]],
    table_results: dict[str, TableExtractionResult] | None = None,
    page_heights: dict[int, float] | None = None,
) -> SemanticProjection:
```

`page_heights` 缺省 `None`；只要缺省，Task 1 的全部新逻辑都是 no-op，`excluded_refs` 的计算公式
退化为改动前的原公式（保证对所有既有调用方 100% 向后兼容，包括三次 provisional/post-gate 调用和全部
既有单测）。

1. **收集视觉候选框**（在原有的 `document.iterate_items` 主循环里顺带做，不新增一次遍历）：对每个
   `PictureItem`，或 `source_kind != "native"` 的 `TableItem`（通过 `table_results` 按
   `docling_ref` 反查得到），把它每页的 bbox 转成 TOPLEFT 后存入 `visual_boxes_by_page[page_no]`。
   **`source_kind == "native"` 的表格永远不参与**（1.3 的显式要求），这是 Task 2 的责任边界。
2. **标记 `visual_ocr_rectangle`**：只遍历 `route_observed == "region_ocr"` 的页面，读它的
   `ocr_route_evidence["ocr_rectangles"]`（即 2.1 新落的字段），逐个矩形用 `_boxes_intersect` 和
   该页 `visual_boxes_by_page` 做正面积重叠测试（边缘接触不算），命中的矩形进入
   `visual_ocr_rectangles_by_page[page_no]`。`full_page_ocr` / `native_only` 页面因为
   `route_observed` 不匹配，天然被跳过（1.7 / 1.8）。
3. **按 bbox 中心点隔离普通 body TextItem**：只在 `visual_ocr_rectangles_by_page` 非空时才扫描
   `all_doc_items`，对每个 `TextItem` 的每个 `prov`，把 bbox 转 TOPLEFT 后取中心点，若落入该页任一
   `visual_ocr_rectangle`，加入 `visual_ocr_isolated_refs`。**不检查该 TextItem 自己是否
   `from_ocr`**（1.6 的显式要求——同一矩形内可能同时有 OCR 文字和 native overlay，两者都要隔离）。

排除公式从：

```python
excluded_refs = all_visual_descendants - accepted_caption_refs
```

改为：

```python
excluded_refs = (all_visual_descendants | visual_ocr_isolated_refs) - accepted_caption_refs
```

trusted caption 的例外复用同一个既有的 `accepted_caption_refs` 集合，**没有为本次改动重新设计
caption 判定**（1.5 的显式要求）。

#### `run_preprocessing()` 里的接线

- 在 `result = converter.convert(...)` 之后，直接从 `result.pages`（Docling 已经算好的每页
  `Page.size.height`）建一个 `page_heights: dict[int, float]`，复用的是 `extract_tables()` 早就
  在用的同一个数据源，不新增第二套几何查询路径。
- 只在**最后一次**（驱动 `document.md` 的那次）`build_semantic_projection(...)` 调用上传入
  `page_heights=page_heights`；其余两次 provisional/post-gate 调用保持原样不传，因为那两次连
  `table_results` 都还没有（无法判定表格 `source_kind`），传了也不会生效，不传更清楚地表达"这两次
  调用不驱动 Markdown"。

#### `quality_report.json` 新增统计字段

```python
"semantic_projection": {
    "eligible_pdf_pages": ...,
    "excluded_docling_ref_count": ...,
    "accepted_caption_ref_count": ...,
    "visual_ocr_rectangle_count": ...,        # 新增
    "visual_ocr_rectangle_pages": ...,        # 新增
    "visual_ocr_isolated_text_item_count": ...,  # 新增
},
```

满足 Task 3.3 里"至少统计 visual OCR rectangles 数量、被 visual OCR isolation 排除的 TextItem
数量"的最低要求，没有为此设计更复杂的 Evaluation。

### 2.3 已知残余风险（记录，不阻塞交付）

这两点是我在评审需求时就identify、现在补充记录在案的：

1. **膨胀边界**：`get_ocr_rects()` 内部对 bitmap 连通域做了 20×20 卷积膨胀再取连通域，OCR 矩形会比
   底层栅格图实际范围向外扩张约 10-20pt。理论上存在极小概率把紧贴图片/扫描表边缘（20pt 以内）的
   合法正文一并划入 `visual_ocr_rectangle`。这个风险有界，且正是这个膨胀行为解释了 Figure 4 那一页
   `ocr_rectangle_count=1`（四张图被合并成一个连通域，覆盖了图间空白区）——也就是本次修复能命中
   Figure 4 的关键机制,不是缺陷。建议第 6 节的语料级抽查里留意是否有误伤案例。
2. **两套"是否属于视觉区域"的判据不是同一套**：`add_nonvisual_ocr_evidence()`（决定
   `nonvisual_ocr_text_cells` 质量分数，不影响 Markdown 内容）用的是"OCR cell 中心点是否落入
   table/picture 紧凑 bbox"；Task 1 新增的判据用的是"OCR 矩形（膨胀后）与 Picture/非-native Table
   是否相交"。两者数字不会一致——修复后 `nonvisual_ocr_text_cells` **不会自动变成 0**，这是正常的,
   因为它们衡量的不是同一件事。

## 3. Task 2 实现细节

文件：`code/preprocessing/table_extraction.py`，只改了 `NativePdfTableExtractor.extract()` 和
`table_to_markdown()` 里的表头判定，`OcrTableExtractor`、`source_cells_in_region`、
`classify_source_kind`、`table_summary` 等均未改动。

### 3.1 `NativePdfTableExtractor.extract()` 的执行顺序重排

改动前的顺序是：

```text
校验 TableFormer cell 结构
  ↓
检查逻辑网格是否被完全覆盖 → 一旦有缺失坐标立即 reject
  ↓
source cell 唯一归属校验
  ↓
one-to-one / 确定性重建 / 文字守恒校验
```

问题：网格覆盖检查在 source 归属校验**之前**，任何合法空白单元格（TableFormer 没有为它输出结构
cell）都会导致整表在看到任何 source cell 之前就被拒收。

改动后：

```text
校验 TableFormer cell 结构（不变）
  ↓
记录 missing_coordinates，但不立即 reject
  ↓
source cell 归属校验（2026-08-15：owner>1 立即 reject；
  owner==0 且中心在结构格子 AABB 内 → reject；
  owner==0 且中心在 AABB 外 → 视为污染，剔除后继续）
  ↓
one-to-one / 确定性重建 / 文字守恒校验
  （守恒宇宙 = 剔除 AABB 外污染之后仍留在格子内的 native source）
  ↓
【新增】只有以上全部通过，才检查 missing_coordinates 是否 provably empty
```

这个顺序本身就是安全网：如果某个"缺失坐标"其实是 TableFormer 漏检的真实内容（不是空白），对应的
source text cell 的中心会落在结构格子 AABB **内部**却找不到 owner，归属校验会立刻
`source_cell_geometry_is_unassigned_or_ambiguous` reject，根本走不到"判断是否可证明为空"
这一步。**先证明所有真实文字全部安全归属，再允许补空单元格**（需求 2.2 的核心原则）不是一句口号，
是这个顺序保证的。页眉/页脚这类中心在 AABB **之外**的文字不再误伤整表，见 3.6。

### 3.2 Provably Empty 判定（新增代码块，插入在文字守恒校验通过之后、最终 accept 之前）

对每个 `missing_coordinate (r, c)`：

```python
has_row_evidence = any(row_start <= r < row_end for row_start, row_end in row_spans)
has_column_evidence = any(col_start <= c < col_end for col_start, col_end in column_spans)
```

`row_spans`/`column_spans` 来自**真实** `candidates` 列表（TableFormer 结构候选，不含 synthetic
cell），行证据和列证据允许来自不同的候选 cell（不要求同一个 cell 同时覆盖行和列）。两者都满足才是
`provably_empty`；否则 `unexplained`。整行/整列在 `candidates` 里完全没有出现过，则该维度证据
天然为空，对应坐标必然落入 `unexplained`（需求 2.4 的"禁止推断"由这个判定自动满足，不需要额外
特判代码）。

只要 `unexplained` 非空，整表继续 `rejected`，`failure_reasons=["logical_grid_has_uncovered_coordinates"]`
（字符串本身不变，语义收紧为"排除了可证明为空之后仍然无法解释的坐标"）。

### 3.3 Synthetic Empty Cell

```python
{
    "row_start": r, "row_span": 1,
    "column_start": c, "column_span": 1,
    "column_header": False, "row_header": False,
    "text": "", "source_cell_refs": [], "bbox": None,
    "inferred_empty": True,
}
```

不加入 `candidates`（因此不参与 ownership、不吸收任何 source text、不参与 bbox 插值），只在最终
`output_cells` 列表里追加。真实 cell 现在统一带 `"inferred_empty": False`，方便下游（尤其是
Markdown 表头判定）区分。

### 3.4 Schema：新增字段，旧字段语义不变

```python
"uncovered_logical_coordinates": [...]        # 保留：TableFormer 原始缺失坐标，accept/reject 都会填
"provably_empty_logical_coordinates": [...]   # 新增：可以安全补空的坐标
"unexplained_logical_coordinates": [...]      # 新增：无法证明为空的坐标
"structural_grid_bbox": [l, t, r, b] | None   # 2026-08-15：全部真实 TableFormer cell bbox 的轴对齐并集
"harvested_source_cell_count": int            # 2026-08-15：TableItem 外包框内采集到的 source 数
"excluded_outside_grid_source_cell_count": int
"excluded_outside_grid_source_cells": [...]   # 2026-08-15：AABB 外剔除的污染 source（全文案/bbox）
```

`validation["source_cells"]` 在归属校验之后被**改写成格子内守恒宇宙**（剔除 AABB 外污染后的
集合）。Golden 的 `_validate_accepted_table_provenance()` 要求 `source_cells` 与输出 cell 的
`source_cell_refs` 一一对应且 `text_conservation_ratio == 1.0`；如果把页眉留在 `source_cells`
里，表会被接受但 Golden 守恒检查会失败。完整收割痕迹改由
`harvested_source_cell_count` + `excluded_outside_grid_source_cells` 承担。

`logical_grid_has_uncovered_coordinates` 依然只出现在 `failure_reasons` 数组里（本来就不是独立
字段），没有改名或删除。为了 schema 一致性，即使一张表完全没有缺失坐标，`accept` 分支也会显式写
三个空单元格字段为 `[]`，而不是缺省不出现。

### 3.5 `table_to_markdown()` 表头判定对 synthetic cell 中性化

`is_explicit_complete_header(row)` 原来要求整行每一列的 owner 都必须是 `column_header=True`。改成：

- 遇到 `owner.get("inferred_empty")` 为真的列：跳过（既不算通过也不算失败，"中性"）。
- 遇到真实 cell 但 `column_header is not True`：立即判定该行不是表头。
- 循环结束后，只有至少出现过一个**真实、非 synthetic** 的 `column_header=True` cell，才判定整行是
  表头（需求 2.6："禁止整行 synthetic cell 自己创造 header"）。

`is_structural_preamble()` 未改动：synthetic cell 的 `column_span` 恒为 1，不会被误判为
"跨列的分组标题行"。

### 3.6 结构格子 AABB 外的 source 视为污染（2026-08-15）

**触发原因（5006A Golden 第二轮）：** Task 1 与 Task 2 空单元格已经通过。page 36/37/39/40/41/44/46/48
的 matrix 表 `accepted`。仍失败的是 **[38, 42, 43, 45, 47]**，全部在空单元格逻辑之前就被
`source_cell_geometry_is_unassigned_or_ambiguous` reject（`unexplained=None`）。

Harvest 用的是 **TableItem 外包框**：中心落在外包框里的每一个 native 文本都必须恰好属于一个
TableFormer cell。page 38/42/43/45 的问题文本是页眉 `NASA-STD-5006A W/CHANGE 2`，TOPLEFT 约
`[321, 75, 471, 83]`，中心 y≈79.3。这是 running header，不是表行。对照：

| 表 | TableItem 顶边（约） | 页眉中心 y≈79.3 | 结果 |
| --- | --- | --- | --- |
| 37（已 accept） | 79.31 | 刚好在外包框外 | 页眉从未被 harvest |
| 38 / 42 / 43 / 45 | 76.8–78.2 | 落在外包框内 | 0 个 cell owner → 整表 reject |

页眉仍在 `document.md` 正文里；被丢掉的只是 Markdown 表。

page **47 按 A0 冻结视为结构不可信**：问题文本是真实列表头 `Requirement in this Standard`。该页在
4.12.2.3a 里还有一张嵌套的 Figure 5 表，owner 为 0 或 ≥2。V0.1 不还原「格里还有一张表」；
唯一归属失败 → 整表 `rejected`，只留可信表注。这不是漏修，是规范里的非法格式。

**通解（禁止 NASA / 页码 / 关键词 / 固定内缩）：** 结构候选建好之后，对每个 harvest 到的 source：

1. 中心落在**恰好一个** cell bbox → 归属（不变）。
2. 中心落在**两个及以上** cell bbox → 整表 reject（不变；覆盖 47 的重叠几何）。
3. 中心落在**零个** cell：
   - 若也在全部真实 TableFormer cell bbox 的轴对齐并集（AABB）**之外** → **污染**，从表格
     source 集合剔除；表仍可 accept。页眉、页脚、侧注走这条路。
   - 若在该 AABB **之内**（格子之间的洞） → 仍整表 reject。

实现位置：`code/preprocessing/table_extraction.py`

- 新增 `_union_aabb()`，对 `candidates[*]["bbox"]` 取 min/max。
- 归属循环按上面 1–3 分流。
- 剔除之后重写守恒宇宙：`source_cells` / `native_character_count` / `source_text_cell_count`
  等只保留格子内 source。
- 审计字段见 3.4。`source_kind` 仍按**原始 harvest** 分类（若外包框里混进 OCR 页码，表仍会
  因 `mixed` 被 defer；本补丁不改分类时机）。

**不做什么：** 不用字符串识别页眉、不按页码特判、不对外包框做固定 inset、不推断阅读顺序。

## 4. 新增/修改的单元测试

### 4.1 `code/preprocessing/tests/test_table_extraction.py`（空单元格 6 个 + AABB 污染 2 个；原有测试未改断言）

- `test_native_table_infers_provably_empty_cell_with_row_and_column_evidence`：2×2 网格，
  `(1,1)` 缺失但行、列证据分别来自不同真实 cell → accept，校验 synthetic cell 字段和 Markdown
  渲染出的空单元格。
- `test_native_table_infers_multiple_provably_empty_cells_across_rows`：3×3 网格，直接复刻
  Requirements Compliance Matrix 的真实模式（表头行齐全，两个数据行各缺一个"备注"列）→ accept，
  两处补空。
- `test_native_table_rejects_when_any_missing_coordinate_is_unexplained`：整列在 `candidates`
  里从未出现过 → 两个缺失坐标都判定为 `unexplained` → 整表仍然 `rejected`。
- `test_ambiguous_source_cell_rejection_takes_priority_over_empty_cell_inference`：source cell
  几何归属歧义 → 在早于新逻辑的阶段就 reject，且新字段完全不出现在 `validation` 里（区分"字段缺失"
  与"字段为空列表"）。
- `test_synthetic_empty_cell_in_header_row_is_neutral_but_needs_a_real_header_cell`：表头行自己
  缺一格但能被下方数据行提供列证据补空 → accept，且表头判定仍然成立（真实 header cell + 1 个
  synthetic cell 共同构成合法表头）。
- 已有的 `test_native_table_rejects_incomplete_logical_grid` 未改动测试代码，但底层判定路径变了
  （现在会先跑到 provably-empty 判定再确认是 unexplained），用于确认重排后旧行为不回归。
- `test_native_table_excludes_source_outside_structural_grid_aabb`（2026-08-15）：外包框内、
  格子 AABB 上方/下方各放一段与 NASA 页眉同构的几何（**不用关键词规则**），格子内正文守恒
  `ratio==1.0`，表 `accepted`，页眉/页脚进入 `excluded_outside_grid_source_cells` 且不进 Markdown。
- `test_native_table_still_rejects_unassigned_source_inside_structural_grid_aabb`（2026-08-15）：
  两行格子之间有空隙，空隙里有一段中心落在 AABB 内、不属于任何 cell 的文字 → 仍
  `source_cell_geometry_is_unassigned_or_ambiguous`。已有的 cell bbox 重叠测试继续覆盖 `owner>1`。

### 4.3 `code/preprocessing/tests/test_server_acceptance.py`（2026-08-15 Golden 合同）

- 默认 fixture：第 47 页为 `rejected`，不注入 Markdown 表；36–46 与 48 仍为 accepted。
- `test_5006a_golden_rejects_admitted_nested_matrix_on_page_47`：第 47 页若被收成矩阵表 → Golden 失败。
- `test_5006a_golden_requires_page_47_rejected_artifact`：第 47 页缺少 native 拒绝产物 → Golden 失败。

### 4.2 `code/preprocessing/tests/test_semantic_projection.py`（新增 6 个测试）

- 扩展了测试替身 `_FakeProv`，新增可选 `bbox` 参数（默认 `None`，对全部已有测试 100% 向后兼容）。
- `test_region_ocr_rectangle_isolates_stray_visual_text_but_keeps_trusted_caption`：直接复刻
  NASA-STD-5006A page 25 Figure 4 的几何关系——一个 Picture、一段落在图片间隙里的"泄露正文"、一个
  bbox 恰好落在同一 OCR 矩形里的 trusted caption、一段真正在矩形之外的正常正文。断言泄露文字被排除、
  caption 依然保留、正常正文不受影响。
- `test_full_page_ocr_route_never_applies_visual_ocr_isolation` / 
  `test_native_only_route_never_applies_visual_ocr_isolation`：确认 1.7 / 1.8 的路由边界。
- `test_native_table_does_not_contribute_to_visual_ocr_rectangle_marking` /
  `test_non_native_table_contributes_to_visual_ocr_rectangle_marking`：确认 1.3 "native 表格不
  参与"的边界，以及非 native（ocr/mixed/image_only）表格确实参与。
- `test_visual_ocr_isolation_is_a_no_op_without_page_heights`：确认不传 `page_heights` 时（即
  table 抽取前的两次 provisional 调用）新逻辑是纯 no-op。

## 5. 本机已经做过的验证（仅供参考，不能替代服务器验收）

本机没有 `docling_core`/`torch`/RapidOCR，无法跑真实转换，做了两类轻量验证：

1. **`table_extraction.py` 的测试**（当时 10 原有 + 6 空单元格）在本机**直接用标准库 Python
   跑过**（该文件零外部依赖），全部通过。2026-08-15 又加了 2 个 AABB 污染测试；按你的要求本机
   **不再执行 pytest**，请在服务器上重跑 `python -m pytest code/preprocessing/tests -v`。

   ```text
   当时：16 passed, 0 failed, 16 total
   现在应收：18 个 table_extraction 测试（服务器确认）
   ```

2. **`test_semantic_projection.py`** 需要 `pydantic`/`pytest`，本机临时装了这两个包跑过一次
   （**遵照你的要求，之后不会再在本机执行任何测试**），33 个测试里 31 个通过；另外 2 个失败
   （`test_missing_tableformer_assets_fail_fast`、
   `test_output_transaction_preserves_previous_good_run_on_rejected_replacement`）是
   Windows 本机 `%TEMP%\pytest-of-*` 目录权限问题导致 `tmp_path` fixture 本身建不起来，跟本次
   改动无关（这两个测试都不涉及 `build_semantic_projection`，改动前同样会因为这个环境问题失败）。

3. 全部 5 个改动过 / 新增的 Python 文件都过了 `py_compile` 语法检查。

以上只能证明"新逻辑在人工构造的最小场景下行为符合预期、没有引入语法错误、没有破坏既有测试"，
**不能证明真实 5006A PDF 会被正确修复**——这必须用真实 Docling/TableFormer/RapidOCR 输出才能确认，
所以需要你在服务器上按下面的命令实际跑一遍。

## 6. 请在服务器上执行的验收命令

### 6.1 单元测试（先跑这个，最快，能确认代码本身没问题）

```bash
cd ~/MyMethod/all-in-rag-main
conda activate mfg-rag-preprocess
export PYTHONPATH="$PWD/docling:$PWD/code/preprocessing${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest code/preprocessing/tests -v
```

重点看 `test_table_extraction.py` 和 `test_semantic_projection.py` 里新增的那些测试名字
（见第 4 节列表）是否全部 `PASSED`。

### 6.2 NASA-STD-5006A 48 页 Golden 验收（关键：这是 P0-1/P0-2 的直接验收）

```bash
mkdir -p logs/preprocessing
set -o pipefail

CUDA_VISIBLE_DEVICES=0 python \
  code/preprocessing/verify_pdf_preprocess_server.py \
  --input-pdf data/engineering_docs/raw/joining/13_nasa_std_5006a_welding_requirements.pdf \
  --page 36 \
  --device cuda \
  --num-threads 8 \
  --document-timeout 7200 \
  --output-root outputs/final-acceptance-5006a \
  --golden-5006a \
  2>&1 | tee logs/preprocessing/final-5006a-$(date +%Y%m%d-%H%M%S).log

rc=${PIPESTATUS[0]}
echo "final_acceptance_exit_code=$rc"
test "$rc" -eq 0
```

这个脚本内置的 `validate_5006a_golden_artifacts()` 包含 P0-1（20/22/25 页图片本体隔离）和
P0-2（36–46 与 48 页 requirements matrix 必须 accepted；**第 47 页必须 `rejected` 且不进 Markdown**）。
嵌套/归属不唯一在 V0.1 是结构不可信，不再把第 47 页当成 Golden 失败。

跑完之后，除了看 `final_acceptance_exit_code=0`，建议人工再看一眼：

```bash
# Figure 4 泄露文字应该已经消失，只剩 caption
grep -n "Notes: Root of Joint" outputs/final-acceptance-5006a/*/document.md   # 应该无匹配
grep -n "Figure 4" outputs/final-acceptance-5006a/*/document.md              # 应该仍然存在

# page 36-46 与 48 应有 Markdown Table；第 47 页按冻结口径没有 TABLE
grep -n "TABLE id=.*page=3[6-9]\|page=4[0-8]" outputs/final-acceptance-5006a/*/document.md

# 看一下新增的审计统计是不是非零（证明 Task 1 真的生效了，不是没触发）
python -c "
import json
report = json.load(open('outputs/final-acceptance-5006a/<实际目录名>/quality_report.json'))
print(report['semantic_projection'])
"

# 看一下 page 36 表格 canonical JSON 里的空单元格字段
python -c "
import json, glob
for path in glob.glob('outputs/final-acceptance-5006a/*/tables/*-p0036-*.json'):
    record = json.load(open(path))
    print(path, record['decision'])
    print('provably_empty:', record['validation'].get('provably_empty_logical_coordinates'))
    print('unexplained:', record['validation'].get('unexplained_logical_coordinates'))
"

# 2026-08-15：38/42/43/45 应 accepted，且页眉被记入 excluded_outside_grid
python -c "
import json, glob
for page in (38, 42, 43, 45, 47):
    paths = glob.glob(f'outputs/final-acceptance-5006a/*/tables/*-p{page:04d}-*.json')
    for path in paths:
        record = json.load(open(path))
        v = record['validation']
        print(page, record['decision'], v.get('failure_reasons'),
              v.get('excluded_outside_grid_source_cell_count'),
              [c.get('text') for c in v.get('excluded_outside_grid_source_cells') or []])
"
```

**Golden 冻结后的预期（`final_acceptance_exit_code` 应为 0）：**

- Task 1 绿：`visual_ocr_isolated_text_item_count` 非零；`Notes: Root of Joint` /
  `Unequal leg fillet weld` 不在 `document.md`；`Figure 4` caption 仍在。
- page 36–46 与 48：`accepted` 矩阵表（空单元格 + AABB 页眉规则）。
- page **38 / 42 / 43 / 45**：`accepted`；页眉在 `excluded_outside_grid_source_cells`，不进表 Markdown。
- page **47**：必须 `rejected`（`source_cell_geometry_is_unassigned_or_ambiguous`），且 `document.md`
  该页没有 `<!-- TABLE`。这是冻结合同，不是回归。
- 同步时必须带上仓库里的 `docling/` 快照（Task 1 依赖 `ocr_rectangles`）。`PYTHONPATH` 仍是：

```bash
export PYTHONPATH="$PWD/docling:$PWD/code/preprocessing${PYTHONPATH:+:$PYTHONPATH}"
```

### 6.3 非 5006A 数字文档抽查（Task 3.2）

从 `data/engineering_docs/raw/` 里选 2-3 份非 5006A、覆盖"普通正文 + 图片 + native table"或
"数字 PDF 内夹扫描表格"的文档，跑一次普通预处理（不带 `--golden-5006a`）：

```bash
python code/preprocessing/pdf_preprocess.py \
  <某份非5006A的PDF路径> \
  --device cuda --num-threads 8 \
  --output-root outputs/preprocessing
```

人工检查生成的 `document.md`：正文还在、图片只剩 caption、扫描表只剩 caption、native table 是正常
Markdown Table。

### 6.4 24 份文档语料级 sanity（Task 3.3）

对 `data/engineering_docs/raw/` 下全部 PDF 跑一遍（可以写个简单的 shell 循环调用 6.3 的命令），
然后汇总每份的 `quality_report.json` 里的：

```text
region_counts.tables / region_counts.pictures
table_summary（accepted_native / deferred_* / rejected_structure）
semantic_projection.visual_ocr_rectangle_count
semantic_projection.visual_ocr_isolated_text_item_count
```

重点人工抽查 `visual_ocr_isolated_text_item_count` 异常高的文档（可能是膨胀边界误伤，见第 2.3 节
残余风险）和 `rejected_structure` 仍然很高的文档（可能是真实结构无法证明为空，行为正确但需要确认
不是新 bug）。

## 7. A0 冻结口径（2026-08-15）

下列四条同时成立后，A0 预处理剩余工作视为冻结，不再为 V0.1 继续改抽取算法或阈值：

1. Task 1（Region-OCR Visual Isolation）、Task 2（空单元格）、Task 2.5（AABB 外污染）已落地，
   并在 5006A 与三份非 5006A NASA-STD 抽查上验证过。
2. **嵌套表 / 格子归属 0 或 ≥2 = 结构不可信。** V0.1 不收表体、不另开通解。5006A 第 47 页与
   5009C 中同类拒绝表属于此条。A1 不得再洗。
3. 方向 `top1≥0.90`、中心裁剪墨量、`alphanumeric_ratio≥0.35` **不调参**。
4. Golden 与产品规范对齐：36–46 与 48 必须有 accepted matrix；第 47 页必须 rejected 且不进 Markdown。

不在本冻结内、也不阻塞 A0 关闭的事项：24 份全量 sanity、扫描 OCR 准确率小样本（原 P1）、
嵌套表还原（V0.1 之后如要做，必须另开几何通解，禁止 NASA/页码/栏名特判）。

服务器在同步本次 Golden 改动后应重跑 `python -m pytest code/preprocessing/tests -v` 与 6.2；
期望 `final_acceptance_exit_code=0`。
