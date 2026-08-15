# Preprocessing — A0

## 1. Status and Scope

A0 边界：`[FROZEN]`（2026-08-15：嵌套/归属不唯一的表不收；方向与乱码阈值不调）  
具体组件和阈值：除明确标注外属于 `[CURRENT IMPLEMENTATION]`。

A0 是知识库的上游标准化层：

```text
Raw PDF
→ orientation / layout / OCR / table / semantic projection / quality gate
→ trusted document.md
```

知识库从 A1 开始，不重新解析 PDF。

## 2. Supported Input

当前主线：

```text
PDF
```

必须覆盖：

```text
digital/native PDF
scanned PDF
mixed PDF
```

TXT / Markdown / DOCX / HTML 等未来如接入，应先经过 adapter/normalization 统一成标准 Markdown，再进入 A1。

## 3. Formal Output Contract

唯一允许进入 Chunk / Embedding / Milvus 的正文：

```text
document.md
```

以下文件只用于 audit/debug/future reprocessing：

```text
document.json
regions.json
quality_report.json
```

不得把 audit JSON 中的 raw text/OCR text 再次作为正文加入知识库。

## 4. Page Provenance

正式 Markdown 必须保留稳定 PDF 页码标记，例如：

```html
<!-- PDF page 25 -->
```

后续 A3、C3、D1 依赖该信息。

A1 不得删除。

## 5. Visual-region Policy

当前正式正文准入原则：

- 图片主体正文不进入 `document.md`；
- 图片内部 OCR 文本不进入正式正文；
- OCR table body 当前不进入正式正文；
- 可信 caption 可以保留；
- 当前不使用 VLM 对图片/图纸做语义理解。

A0 的 Figure/Caption 识别错误必须在 A0 修复，不能在 A1 以后用正则清洗。

## 6. Table Policy

### Native / digital table

只有通过当前 A0 的结构合法性、源文本守恒等准入检查后，才允许投影为正式 Markdown table。

V0.1 结构合法性包括：每个真实 native source 的中心必须落在恰好一个 TableFormer 格子里。
下列情况视为结构不可信，**不收表体**（只留可信表注，A1 不得再洗）：

- 格子归属为 0 且落在结构格子 AABB 内部；
- 格子归属 ≥ 2（重叠）；
- 嵌套表导致上述归属失败（5006A 第 47 页为此类冻结样本）。

AABB 外的页眉/页脚污染从表 source 剔除，不因此整表拒绝。

`[FROZEN]` 2026-08-15：不在 V0.1 还原嵌套表；不调方向 / 乱码阈值。

### OCR / mixed / image-only table

当前：

```text
[DEFERRED]
```

其 table body 不作为正式可索引文本进入 `document.md`。

## 7. Untrusted Parse

Raw/native parse 可以保留做：

```text
audit
debug
future reprocessing
```

但不能仅因为“解析出了文字”就绕过 quality gate 进入正式正文。

## 8. Document Identity

当前实现使用源文件身份与 raw PDF hash 构造稳定 `document_id`。

最低要求：

- 同一源 PDF 重跑时稳定；
- 不依赖 Chunk 参数；
- 不因重新建库随机变化。

Chunk/Section ID 的稳定性要求见 `knowledge-base.md`。

## 9. Transactional Publish

A0 应保持事务式输出语义：

```text
working/staging
→ validation
→ atomic publish
```

失败或部分结果：

```text
.failed/
```

不覆盖上一次正式成功产物。

A1 只允许读取正式 canonical output root。

## 10. Current OCR Routing

`[CURRENT IMPLEMENTATION]`

基于 page bitmap coverage：

```text
coverage > 0.75
→ full_page_ocr

0.05 < coverage <= 0.75
→ region_ocr

coverage <= 0.05
→ native_only
```

这些是当前实现参数，不等于永久最佳值。

## 11. Current OCR Language

当前开发 NASA/NIST 类英文语料使用：

```text
English
```

因此当前实现不能宣称已经验证中文 OCR。

未来中文/中英混合文档必须单独评估：

```text
OCR model
language/dictionary
quality threshold
mixed-language behavior
```

## 12. Current Components

`[CURRENT IMPLEMENTATION]`

当前主链包含/曾明确采用：

```text
PP-LCNet page orientation
Docling Layout Heron
RapidOCR
PP-OCRv6 Det + Rec
TableFormer V1 accurate
heading hierarchy recovery
```

当前不把以下能力作为主线：

```text
picture description
picture classification
chart semantic extraction
code enhancement
formula enhancement
VLM
```

## 13. Current Quality Gates

`[CURRENT IMPLEMENTATION]`

当前讨论过的主要门禁：

```text
native parse score >= 0.50
OCR mean confidence >= 0.75
short OCR (<20 alnum chars) confidence >= 0.90
```

如果真实代码与文档数值不一致，以真实代码作为“当前行为事实”，并显式报告差异；不要静默猜测。

## 14. A0 → A1 Acceptance Invariants

A0 正式发布的 `document.md` 应满足：

1. 页码 provenance 可恢复；
2. heading hierarchy 可被 Markdown parser 读取；
3. 不可信图像正文不混入正式文本；
4. 只有符合准入条件的 native table 才进入正式 Markdown；
5. audit-only 内容不混入正式正文；
6. 正文顺序在 A1 无需再次“修复”。

## 15. Proceeding Boundary

以下内容只放 `docs/proceeding/`：

```text
某次 smoke
某页 Golden failure
当前 P0/P1
临时 workaround
性能日志
阶段 acceptance
```

本文件不记录具体排查经过，只规定稳定边界和当前实现契约。
