# Documentation Index

`docs/` 保存稳定需求、模块职责、接口、参数、实现约束和验收口径。  
阶段过程、烟测、故障与交接独立放在 `docs/proceeding/`。

## Documents

| 文档 | 负责内容 |
|---|---|
| `architecture.md` | 项目定位、总链路、模块边界、跨模块契约 |
| `data-corpus.md` | 当前语料角色、语言范围、数据边界与扩展原则 |
| `preprocessing.md` | A0：Raw PDF → trusted `document.md` |
| `knowledge-base.md` | A1–A6：Loading / Metadata / Chunk / Hierarchy / Embedding / Milvus |
| `retrieval.md` | B1–B6：Router / Query / Hybrid / RRF / Rerank / Context Recovery |
| `generation.md` | C1–C3：Evidence Assembly / Generation / Citation |
| `evaluation.md` | D1 Retrieval Evaluation；D2 当前状态 |
| `Preprocessing/pdf-preprocessing-server-acceptance.md` | 服务器验收命令与 Golden 检查 |
| `Preprocessing/2026-08-14-region-ocr-isolation-and-native-table-empty-cell.md` | A0 Region-OCR / 空单元格 / AABB 实现说明 |
| `proceeding/` | 阶段日志、烟测、故障、验收、交接 |
| `proceeding/2026-08-15-a0-freeze.md` | A0 冻结补记：嵌套表不收、Golden 第 47 页合同 |
| `proceeding/2026-08-15-a1-freeze.md` | A1 实现交接：Linux 16 passed / 3-doc canonical smoke；申请关闭 |
| `proceeding/2026-08-16-a2-freeze.md` | A2 实现交接：Document Registry；Linux pytest + 3-doc NASA manifest smoke |
| `proceeding/2026-08-17-a3-freeze.md` | A3 实现交接：结构解析 + Leaf chunking；Linux 122 passed / 3-doc smoke |
| `proceeding/2026-08-17-a4-section-hierarchy.md` | A4 实现交接：多级 Section hierarchy + source-span recovery；Linux 161 passed / 3-doc smoke；申请关闭 |
| `manufacturing-rag-v0.1-spec.md` | 历史 V0.1 spec，与当前模块文档冲突时以模块文档为准 |

## Reading Rule

修改某模块时：

```text
agent.md
→ 对应 docs
→ 真实代码
→ 直接上下游代码
```

不要先从旧 C1–C9 教程或旧 spec 反推当前需求。

## Status Markers

- `[FROZEN]`：当前正式要求。
- `[PROVISIONAL]`：已有方案但仍需确认。
- `[DEFERRED]`：当前不做。
- `[CURRENT IMPLEMENTATION]`：当前代码/组件事实。
