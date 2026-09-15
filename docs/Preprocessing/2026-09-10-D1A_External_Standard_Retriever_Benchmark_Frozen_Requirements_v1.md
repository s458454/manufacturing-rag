# D1-A External Standard Retriever Benchmark — Frozen Requirements v1

> Status: **FROZEN**  
> Contract version: `d1a-retriever-benchmark-v1`  
> Target system baseline: `ver-0.6`  
> Scope: **External standard benchmark for the retrieval subsystem only**  
> Benchmark: **C-MTEB / T2Retrieval, full Chinese dataset**

---

# 1. Purpose

D1-A is not an end-to-end RAG benchmark.

Its only purpose is to answer:

> **How well does the retrieval subsystem selected in ver-0.6 perform on a standard Chinese retrieval benchmark, and what is the contribution of Dense, BM25, Hybrid RRF, and Reranker respectively?**

D1-A evaluates:

```text
Query
  ↓
Dense Retrieval
+
BM25 Retrieval
  ↓
Hybrid RRF
  ↓
Reranker
```

D1-A must not be used to claim that the complete RAG system has been validated.

---

# 2. Benchmark

Use:

```text
C-MTEB / T2Retrieval
```

Requirements:

```text
Language = Chinese only
Dataset = full benchmark
Sampling = OFF
Cross-lingual = OFF
Multi-hop = OFF
Custom Query generation = OFF
Custom Gold construction = OFF
```

The benchmark's official:

```text
corpus
queries
qrels
```

must be used directly.

No manually generated Query or manually generated relevance labels may replace the official benchmark data.

---

# 3. Evaluation Scope

D1-A evaluates four logical retrieval stages independently:

```text
Stage 1: Dense-only
Stage 2: BM25-only
Stage 3: Hybrid RRF
Stage 4: Reranker
```

These are evaluation stages, not a requirement to change the production online call structure.

The ver-0.6 online retrieval path remains:

```text
B2 Query
   ↓
Dense AnnSearchRequest Top-20
+
BM25 AnnSearchRequest Top-20
   ↓
Milvus hybrid_search
+
RRFRanker(k=60)
   ↓
Hybrid Top-20
   ↓
B5 Reranker
   ↓
Online Top-5 to B6
```

Dense-only and BM25-only execution in D1-A exist only for stage-wise diagnosis.

D1-A must not modify the production B3/B4 contract in order to make evaluation easier.

---

# 4. Benchmark Data Adapter

T2Retrieval already provides passage-level corpus data.

Therefore:

```text
1 official T2 passage
=
1 benchmark retrieval unit
```

The benchmark corpus must not be passed through the production PDF processing and chunking pipeline.

Forbidden:

```text
T2 passage → A0 PDF preprocessing
T2 passage → A3 768/96 chunking
T2 passage → A4 Section hierarchy construction
```

The official passage identity must remain stable so retrieval results can be compared directly with official qrels.

A thin benchmark adapter may be implemented only to convert the official benchmark corpus into a form accepted by the benchmark Milvus collection.

The adapter must not:

```text
rewrite passage content
split passages
merge passages
generate synthetic metadata
generate synthetic Gold
rewrite Query
translate Query
perform Retrieval Decomposition
```

---

# 5. Benchmark Collection

D1-A must use an independent benchmark collection.

It must not insert benchmark passages into the production `a6_leaf_v1` collection.

The production A6 collection contains real industrial-document semantics such as:

```text
document_id
section_id
page_start
page_end
document metadata
version governance
```

T2Retrieval does not provide equivalent PDF / Section / page provenance.

Therefore D1-A must not fabricate meaningless production metadata merely to reuse the production schema.

The benchmark collection should contain only fields with an actual retrieval or evaluation consumer.

Minimum logical fields:

```text
benchmark_passage_id
content
dense_vector
sparse_vector
```

Additional fields are allowed only when required by the actual Milvus implementation.

---

# 6. Retrieval Configuration

D1-A must evaluate the ver-0.6 retrieval configuration.

It must not tune a separate benchmark-specific retrieval stack.

## 6.1 Dense

Use:

```text
Model:
Qwen3-Embedding-4B

Output:
2560-d
float32
L2 normalized

Similarity:
IP
```

The query-side numerical contract must remain compatible with the production B3 Dense query encoder.

---

# 7. Dense Query Serialization

D1-A must reuse the production B3 Dense Query serialization contract.

Fixed instruction:

```text
Given a manufacturing and engineering query, retrieve relevant passages from technical documents that contain the information needed to answer the query.
```

Fixed format:

```text
Instruct: Given a manufacturing and engineering query, retrieve relevant passages from technical documents that contain the information needed to answer the query.
Query:{query}
```

D1-A must not replace it with a T2-specific, generic retrieval, Chinese-only, or benchmark-optimized instruction.

The purpose of D1-A is to measure the current ver-0.6 retrieval stack, not optimize specifically for T2Retrieval.

---

# 8. BM25

Use the ver-0.6 Native BM25 configuration:

```text
Engine:
Milvus Native BM25

Analyzer:
ICU

Analyzer filters:
lowercase
removepunct

Index:
SPARSE_INVERTED_INDEX

Metric:
BM25

inverted_index_algo:
DAAT_MAXSCORE

k1:
1.2

b:
0.75
```

No benchmark-specific analyzer may be introduced.

No application-side custom Chinese tokenizer may replace the production ICU Analyzer.

---

# 9. Candidate Depth

Fixed:

```text
Dense candidate limit = 20
BM25 candidate limit = 20
Hybrid final limit = 20
Reranker input = Hybrid Top-20
D1-A reranked result retention = Top-20
```

Although the online production path only forwards Reranker Top-5 to B6, D1-A must retain complete reranked Top-20 for evaluation.

---

# 10. Fusion

Use:

```text
Milvus RRFRanker
k = 60
```

Hybrid evaluation must reflect the ver-0.6 production RRF configuration.

D1-A must not restore historical `k=55`.

D1-A must not replace RRF with raw-score or normalized-score fusion unless a future contract explicitly changes B4.

---

# 11. Reranker

D1-A Stage 4 evaluates:

```text
Hybrid Top-20
→ production B5 Reranker
→ reranked Top-20
```

The benchmark must use the B5 production model and frozen inference configuration in force for ver-0.6.

D1-A must not introduce a benchmark-only reranker.

The reranker is expected to change ranking order.

Because it only reorders the same Hybrid Top-20 candidate set and does not introduce new candidates:

```text
Hybrid Recall@20
=
Reranker Recall@20
```

under a correct implementation.

The main value of Reranker evaluation is therefore reflected in:

```text
Recall@5
Recall@10
nDCG@10
MRR@10
```

---

# 12. Metrics

## 12.1 Primary External Metric

```text
nDCG@10
```

## 12.2 Core Engineering Metrics

```text
Recall@5
Recall@10
Recall@20
```

Interpretation:

```text
Recall@20
→ whether the candidate pool contains relevant passages

Recall@10
→ mid-front candidate coverage

Recall@5
→ quality of the highest-ranked results
```

## 12.3 Ranking Diagnostic

```text
MRR@10
```

Used to observe how early the first relevant passage appears.

## 12.4 Optional Logged Metric

```text
MAP@10
```

MAP@10 may be computed and persisted for later comparison, but is not a D1-A core acceptance metric in v1.

---

# 13. Stage-wise Metric Table

The final D1-A report must contain at least:

| Retrieval Stage | nDCG@10 | Recall@5 | Recall@10 | Recall@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| Dense |  |  |  |  |  |
| BM25 |  |  |  |  |  |
| Hybrid RRF |  |  |  |  |  |
| Reranker |  |  |  |  |  |

This table must make it possible to determine:

```text
Dense vs BM25
→ branch-level retrieval differences

Dense/BM25 → Hybrid
→ whether RRF fusion improves retrieval

Hybrid → Reranker
→ whether reranking improves front-of-list ordering
```

---

# 14. Per-Query Result Retention

D1-A must not store only aggregate metrics.

The complete per-query Top-20 ranking must be retained for all four stages.

Recommended logical files:

```text
dense_results.jsonl
bm25_results.jsonl
hybrid_results.jsonl
reranker_results.jsonl
```

Each query record must contain at least:

```text
query_id
query
status
results
```

Each result item must contain at least:

```text
doc_id
rank
score
```

Logical example:

```json
{
  "query_id": "q123",
  "query": "原始中文查询",
  "status": "COMPLETED",
  "results": [
    {
      "doc_id": "p456",
      "rank": 1,
      "score": 0.83
    }
  ]
}
```

`doc_id` must remain the official benchmark passage identity used by qrels.

---

# 15. Score Semantics

The `score` field has different semantics by stage:

```text
Dense
→ IP similarity score

BM25
→ Native BM25 relevance score

Hybrid
→ RRF output score

Reranker
→ Cross Encoder relevance score
```

Scores must not be compared numerically across stages.

Cross-stage quality comparison must use benchmark metrics, not raw score magnitude.

---

# 16. Query Status

D1-A must distinguish:

```text
COMPLETED
EMPTY
FAILED
```

Semantics:

```text
COMPLETED
→ retrieval stage executed normally and returned results

EMPTY
→ retrieval stage executed normally but returned no result

FAILED
→ retrieval stage did not complete successfully
```

`FAILED` must never be silently converted to `EMPTY`.

`FAILED` queries must never be silently skipped from reporting.

---

# 17. Failure Audit

The final report must include at least:

```text
total_queries
completed_queries
empty_queries
failed_queries
```

If a query fails, the corresponding per-query record should retain an auditable error type.

Examples:

```text
DenseQueryEncodingError
HybridSearchError
RerankerError
```

The evaluation layer must not hide production-path failure by automatically switching to another retrieval branch.

In particular:

```text
Hybrid failure
→ must not silently fall back to Dense-only

Hybrid failure
→ must not silently fall back to BM25-only
```

---

# 18. Gold / Relevance

D1-A uses official T2Retrieval qrels.

D1-A must not construct:

```text
custom Gold Evidence
Gold Set
Gold Alternative
chunk-id-based custom Gold
LLM semantic relevance labels
embedding-similarity relevance labels
```

for the external benchmark.

The relation:

```text
query_id
+
retrieved benchmark_passage_id
+
official qrels
```

defines benchmark relevance.

The custom Gold Set / Gold Alternative design belongs to a future industrial benchmark, not D1-A.

---

# 19. First-run Baseline Policy

D1-A v1 does not define an arbitrary absolute pass threshold such as:

```text
nDCG@10 >= X
Recall@20 >= Y
```

because ver-0.6 has not yet produced its first full T2Retrieval result under this contract.

The first complete D1-A run becomes:

```text
ver-0.6 D1-A retrieval baseline
```

Future versions may compare against this baseline for regression or improvement.

---

# 20. Diagnostic Warnings

D1-A may flag suspicious degradation patterns for investigation.

Examples:

```text
Hybrid is clearly worse than both Dense and BM25
```

or:

```text
Reranker causes both nDCG@10 and Recall@5 to drop materially
```

These are diagnostic warnings, not pre-defined absolute FAIL thresholds in D1-A v1.

A single metric is not required to monotonically improve at every stage.

---

# 21. What D1-A Can Validate

D1-A can evaluate:

```text
✓ Qwen3-Embedding-4B retrieval behavior
✓ Dense retrieval
✓ Milvus Native BM25
✓ ICU Analyzer under the production BM25 configuration
✓ Dense + BM25 Hybrid Retrieval
✓ RRF fusion with k=60
✓ B5 Reranker
✓ stage-wise retrieval quality
✓ per-query retrieval failure / weakness diagnosis
```

---

# 22. What D1-A Cannot Validate

D1-A does not validate:

```text
✗ A0 PDF preprocessing quality
✗ OCR quality
✗ Markdown projection quality
✗ A3 structure-aware chunking quality
✗ A4 Section hierarchy quality
✗ A6 production document metadata governance
✗ A6 real industrial Leaf schema quality
✗ document version conflict handling
✗ B6 Context Recovery
✗ final Evidence Assembly
✗ LLM answer correctness
✗ faithfulness
✗ citation correctness
✗ end-to-end RAG quality
```

Therefore a high D1-A score must not be described as:

```text
"The complete manufacturing RAG system performs well."
```

The valid interpretation is:

> **The ver-0.6 retrieval subsystem achieves the reported performance on a standard Chinese retrieval benchmark.**

---

# 23. Relationship with Other Evaluation Stages

Current evaluation structure:

```text
D1-A
External Standard Retriever Benchmark
→ FROZEN
→ T2Retrieval
→ retrieval subsystem only

D1-B
Industrial Retrieval Benchmark
→ DEFERRED / to be determined later
→ intended to evaluate the real industrial-document retrieval chain

D2
Generation Evaluation
→ DEFERRED
→ intended to evaluate final answer / evidence / citation behavior
```

D1-A must not absorb D1-B or D2 responsibilities.

---

# 24. Reproducibility Requirements

A D1-A run should persist enough configuration information to reproduce the result.

At minimum record:

```text
benchmark name/version or revision
retrieval contract version
Dense model
Dense query serializer version
BM25 analyzer
BM25 k1
BM25 b
BM25 inverted-index algorithm
Dense candidate limit
BM25 candidate limit
RRF k
Hybrid final limit
Reranker model/configuration identity
run timestamp
```

The purpose is to prevent later results from being compared under silently changed retrieval configurations.

---

# 25. Frozen Summary

```text
D1-A
External Standard Retriever Benchmark
Status = FROZEN

Benchmark:
C-MTEB / T2Retrieval
Chinese only
full dataset
official corpus + queries + qrels

No:
sampling
English evaluation
cross-lingual evaluation
multi-hop evaluation
custom Query generation
custom Gold construction
A0-A4 processing

Stages:
Dense-only
BM25-only
Hybrid RRF
Reranker

Dense:
Qwen3-Embedding-4B
2560-d
float32
L2 normalize
IP
production B3 query instruction/serialization

BM25:
Milvus Native BM25
ICU + lowercase + removepunct
k1=1.2
b=0.75
DAAT_MAXSCORE

Depth:
Dense Top-20
BM25 Top-20
Hybrid Top-20
Reranker Top-20 retained

Fusion:
Milvus RRFRanker(k=60)

Metrics:
Primary = nDCG@10
Core = Recall@5 / Recall@10 / Recall@20
Diagnostic = MRR@10
Optional logged = MAP@10

Output:
aggregate summary
+
per-query Top-20 for all four stages

Status:
COMPLETED / EMPTY / FAILED
must be distinguished

Threshold:
no arbitrary absolute quality threshold in first run

Baseline:
first complete run = ver-0.6 D1-A baseline

Claim boundary:
D1-A evaluates the retrieval subsystem only,
not the complete RAG system.
```
