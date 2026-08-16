# Split Manifest

> 临时迁移文档，用于后续把拆分结果与 `all-in-one-complete.md` 对照。  
> 最终稳定后可以删除。

## Ownership Map

| 母版内容 | Authoritative destination |
|---|---|
| Agent 行为规范 / Source of Truth / 阅读顺序 | `../agent.md` |
| 项目总链路 / 模块边界 / 全局原则 | `architecture.md` |
| 语料背景 / 语言 / 数据角色 | `data-corpus.md` |
| A0 | `preprocessing.md` |
| A1–A6 | `knowledge-base.md` |
| B1–B6 | `retrieval.md` |
| C1–C3 | `generation.md` |
| D1 / D2 状态 | `evaluation.md` |
| smoke / bug / acceptance / handoff | `proceeding/` |

## Single-owner Rules

以下详细参数只允许有一个主定义位置：

| 参数/规则 | 主定义 |
|---|---|
| OCR routing / quality gate | `preprocessing.md` |
| Leaf Metadata / Chunking / Embedding / Milvus | `knowledge-base.md` |
| Top-20 / RRF k=55 / Reranker / Top-5 | `retrieval.md` |
| ~8K Evidence / generation decoding | `generation.md` |
| ~50-query benchmark / Recall/MRR / Gold | `evaluation.md` |

`architecture.md` 与 `agent.md` 只能摘要或链接，不应重复详细参数作为第二份 authoritative copy。

## Provisional Checklist

拆分后仍必须保留为未冻结：

- A1/A6 rebuild/upsert/versioning
- A3 tokenizer
- A3 Leaf.content 是否包含当前 Section heading / page marker serialization
- A4 Section text materialization
- B3 BM25 Chinese analyzer/tokenizer
- B5 reranker input serialization
- B6 exact ancestor selection
- B6 shared-parent score aggregation
- B6/C1 budget accounting
- C3 claim-level citation granularity
- D1 minimum answer-supporting span
- D1 ~80% coverage threshold
- D1 union coverage rule
- D1 MRR relevant-unit rule
- D1 Direct/Paraphrase/Cross-lingual exact ratio

## Deferred Checklist

拆分后仍必须保持 Deferred：

- D2 final generation evaluation
- Rewrite / Decomposition / Step-Back / HyDE baseline
- ColBERT baseline
- WeightedRanker baseline
- GraphRAG
- VLM/CAD RAG
- Structured ERP/MES/SQL RAG
