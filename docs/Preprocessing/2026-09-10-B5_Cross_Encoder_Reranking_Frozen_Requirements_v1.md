# B5 Cross-Encoder Reranking — Frozen Requirements v1

> Status: **FROZEN**
>
> Contract version: `b5-contract-v1`
>
> 本文档是 B5 Cross-Encoder Reranking v1 的权威需求。
>
> B5 负责对 B4 RRF 输出的少量 Leaf Candidate 做 Query–Document 联合相关性判断，
> 并将最多 20 个 RRF Candidate 重排为最多 5 个高优先级 Leaf，交给 B6 Context Recovery。
>
> B5 不负责第一阶段召回、不恢复 Parent/Section、不重新切分 Leaf、不生成最终回答。

---

# 1. B5 在整条链路中的位置

```text
B1 Query Router
    ↓
B2 Query Processing
    ↓
B3 Dense + BM25
    ↓
B4 RRF Fusion
    ↓
B5 Cross-Encoder Reranking
    ↓
B6 Context Recovery
```

固定顺序：

```text
Leaf Retrieval
→ RRF
→ Rerank Leaf
→ Context Recovery
```

禁止：

```text
Leaf Retrieval
→ Parent Recovery
→ Rerank Parent
```

B5 唯一排序实体：

```text
Leaf
```

---

# 2. B5 职责

B5 的目标：

> 对 B4 已召回的少量 Leaf Candidate 做更充分的 Query–Document 联合相关性判断，提高候选精排质量。

B5 不负责：

```text
Query Router
Query Rewrite
Query Translation
Query Expansion
Dense Retrieval
BM25 Retrieval
RRF Fusion
Parent / Section Recovery
Neighbor Expansion
最终证据拼接
最终回答生成
```

---

# 3. Model

v1 固定：

```text
Qwen3-Reranker-0.6B
```

升级候选：

```text
Qwen3-Reranker-4B
```

4B 不是 runtime fallback。

只有 D1 明确表明：

```text
相关 Leaf 已经稳定进入 RRF Top-20
但 0.6B 经常把正确 Leaf 排错
```

才允许评估 4B。

禁止：

```text
0.6B 加载失败 → 自动切换 4B
0.6B 加载失败 → 自动切换其他 reranker
```

---

# 4. Backend

v1 固定：

```text
Transformers
AutoTokenizer + AutoModelForCausalLM
```

B5 v1 不要求：

```text
vLLM
TensorRT
ONNX Runtime
自定义 serving framework
```

这些属于未来性能优化，不属于 v1 correctness baseline。

---

# 5. Input Candidate Budget

```python
rerank_input_limit = 20
```

中文含义：B5 最多接收 20 个 RRF 融合后的 Leaf Candidate。

如果 B4 实际只返回 `N < 20`，则 B5 只处理实际 N 个 Candidate。

禁止复制 Candidate 补满 20，也禁止重新调用 B3/B4 获取更多 Candidate。

---

# 6. Output Candidate Budget

```python
rerank_output_limit = 5
```

中文含义：B5 重排以后最多保留 5 个 Leaf，交给 B6。

如果输入 Candidate 数量 `N <= 5`，则输出全部 N 个，不补齐。

Top-5 是最大优先候选数量，不代表最终 Prompt 一定包含 5 个不同 Context。

---

# 7. Ranking Entity

B5 唯一排序实体：

```text
Leaf
```

文档文本固定：

```text
Leaf.content
```

v1 不拼接：

```text
Parent Section
Section heading
heading path
document title
document number
document version
source
page marker
其他 metadata
```

以后只有 D1 证明 Leaf 脱离 heading 后导致稳定 reranking failure，才允许重新测试 heading/context enhancement。

---

# 8. Instruction

v1：

```text
Instruction = ON
Instruction language = English
```

固定 instruction：

```text
Given a manufacturing and engineering query, determine whether the document passage contains relevant information needed to answer the query.
```

中文含义：给定一个制造业或工程技术问题，判断该文档段落是否包含回答该问题所需的相关信息。

该 instruction 只定义 Query ↔ Document relevance，不要求模型回答 Query、总结 Document、生成解释、判断真假或判断版本有效性。

---

# 9. Input Serialization

B5 必须使用 Qwen 官方 Qwen3-Reranker 输入模板。

逻辑输入包含：

```text
System:
Judge whether the Document meets the requirements based on the Query and the Instruct provided.
The answer can only be "yes" or "no".

User:
<Instruct>: {instruction}
<Query>: {query}
<Document>: {leaf.content}
```

其中：

| 字段 | 中文含义 |
|---|---|
| `instruction` | B5 固定的制造业/工程相关性判断 instruction |
| `query` | B4 继续传入的知识检索问题 |
| `leaf.content` | 当前 Candidate Leaf 正文 |

禁止自行设计另一套 prompt。正式修改模板必须升级 B5 contract version。

---

# 10. Query

B5 使用：

```text
B4 FusionResult.query
```

不得重新读取 raw user query、自行改写、翻译或拆分 Query。

---

# 11. Maximum Input Length

v1 工程预算固定：

```python
reranker_max_input_tokens = 4096
```

一个完整 Query–Leaf pair，包括官方模板、instruction、Query、Leaf.content 和必要 special tokens，最多允许 4096 个 inference token。

4096 是项目 v1 运行预算，不是模型能力上限。

长度检查必须覆盖完整序列化输入：

```text
official template
+
instruction
+
query
+
Leaf.content
+
special tokens
```

---

# 12. Silent Truncation

```text
silent truncation = OFF
```

若完整输入 `total_tokens > 4096`：

```text
→ RerankerInputTooLongError
```

禁止静默截断 Query、Leaf 或 instruction。

---

# 13. Leaf Re-splitting

B5 禁止重新切分 Leaf。

如果 oversize Leaf 频繁出现，应回到 A3 / table chunking 合同处理，而不是由 B5 静默修复。

---

# 14. Tokenizer / Padding

v1：

```text
padding_side = left
```

Tokenizer 必须与实际 Qwen3-Reranker-0.6B 模型匹配。

禁止使用 Embedding tokenizer 代替 Reranker tokenizer，也禁止用字符数估算 token 数。

---

# 15. Inference Mode

v1 固定：

```text
model.eval()
torch.inference_mode()
sampling = OFF
gradient = OFF
```

B5 不是文本生成任务。

禁止使用 temperature、top_p、top_k sampling、beam search 或 `model.generate()` 作为生产打分主路径。

---

# 16. Score Definition

Qwen3-Reranker 使用 yes/no logits 计算相关性 score：

\[
s = \frac{e^{z_{\mathrm{yes}}}}{e^{z_{\mathrm{yes}}}+e^{z_{\mathrm{no}}}}
\]

其中：

| 符号 | 中文含义 |
|---|---|
| `z_yes` | 模型最后判断位置上 `yes` token 的 logit |
| `z_no` | 模型最后判断位置上 `no` token 的 logit |
| `s` | 当前 Query–Leaf 的 rerank score |

固定：

```text
0 <= rerank_score <= 1
rerank_score 越大 → 当前 Leaf 与 Query 越相关
```

---

# 17. Threshold

```python
rerank_score_threshold = None
```

即 absolute threshold = OFF。

B5 只做 score 全量计算 → 排序 → Top-5。

---

# 18. Sorting

排序规则固定：

```text
1. rerank_score 降序
2. fused_rank 升序
```

实现逻辑：

```python
sort_key = (-rerank_score, fused_rank)
```

若两个 Candidate 完全同分，则保留 B4 RRF 原有更高排名的 Candidate 在前。

---

# 19. Output Structure

```python
@dataclass(frozen=True)
class RerankedCandidate:
    chunk_id: str
    document_id: str
    section_id: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str

    rrf_score: float
    fused_rank: int

    rerank_score: float
    rerank_rank: int
```

必须保留 `rrf_score` / `fused_rank`，便于 D1 分析 B5 是否纠正了 B4 排名。

---

# 20. RerankingResult

```python
@dataclass(frozen=True)
class RerankingResult:
    request_id: str
    task_id: str
    query: str
    candidates: tuple[RerankedCandidate, ...]
```

B6 必须消费 `RerankingResult`，而不是直接重新读取 B4 Candidate。

---

# 21. Empty Input

如果 B4 返回 0 candidate，则 B5 不执行模型推理，返回空 `RerankingResult`。

EMPTY 不是 B5 模型错误。

---

# 22. Device

```text
device = CUDA
```

禁止 CUDA 不可用时自动 CPU fallback。

---

# 23. Forward Dtype

```text
forward dtype = FP16
```

最终 `rerank_score` 持久化为 Python float / FP32-equivalent scalar。

BF16 / FP32 属于后续部署 profile，不属于当前 v1 baseline。

---

# 24. Batch Inference

```text
batch inference = ON
```

默认：

```python
reranker_batch_size = 8
```

`reranker_batch_size` 属于部署参数，可显式配置，但不得改变排序算法。

如果 batch=8 OOM：

```text
→ RerankerInferenceError
```

自动 batch fallback = OFF。

---

# 25. Model Lifecycle

Qwen3-Reranker-0.6B 应在服务启动阶段：

```text
load
→ move to CUDA
→ eval
→ warm
```

正常请求路径禁止每个 Query 或每个 Candidate 重新 load model。

---

# 26. Failure Policy

```text
reranker failure fallback = OFF
```

如果 B5 执行失败：

```text
→ 当前 knowledge task 失败
→ 不进入正常 B6
```

禁止跳过 B5、直接拿 RRF Top-5 进入 B6，或自动切换其他模型。

---

# 27. Exception Contract

```python
class RerankerError(Exception):
    pass

class RerankerModelError(RerankerError):
    pass

class RerankerInputTooLongError(RerankerError):
    pass

class RerankerInferenceError(RerankerError):
    pass
```

---

# 28. Observability

生产 v1 每次至少记录：

```text
request_id
task_id
reranker_contract_version
reranker_model
reranker_input_count
reranker_output_count
max_pair_tokens
rerank_latency_ms
rerank_status
error_type
```

固定：

```text
reranker_contract_version = "b5-contract-v1"
```

生产日志默认不重复保存完整 Query / Leaf 正文。

---

# 29. D1 / Profiling Observability

D1 至少可以保存：

```text
chunk_id
document_id
fused_rank
rrf_score
rerank_rank
rerank_score
```

用途包括：判断 B5 是否提升正确 Leaf、是否错误压低正确 Leaf、测量 reranker 对 Recall@5 / MRR / nDCG 的真实贡献，以及决定是否需要 4B、heading/context enhancement 或 absolute threshold。

---

# 30. Explicitly Forbidden

B5 v1 禁止：

```text
1. 在 B5 重新执行 Dense/BM25。
2. 在 B5 重新计算 RRF。
3. 在 B5 进行 Query Rewrite / Translation / Decomposition。
4. 在 Rerank 前恢复 Parent/Section。
5. Rerank Parent/Section。
6. 给 Leaf.content 拼接 heading/document metadata。
7. 静默截断完整 Query–Leaf 输入。
8. 在 B5 重新切分 Leaf。
9. 使用 model.generate() 作为生产主打分逻辑。
10. 使用 temperature/top_p 等采样参数。
11. 使用 absolute score threshold。
12. OOM 后自动减小 batch。
13. CUDA 失败后自动 CPU fallback。
14. 0.6B 失败后自动切换 4B 或其他模型。
15. B5 失败后跳过 Reranker 进入 B6。
16. 根据 Dense/BM25 raw score 修改 Reranker 排序。
```

---

# 31. B5 Final Baseline

```text
Model:
    Qwen3-Reranker-0.6B

Backend:
    Transformers
    AutoTokenizer + AutoModelForCausalLM

Ranking Entity:
    Leaf

Document Text:
    Leaf.content

Instruction:
    ON
    English

Input:
    official Qwen3-Reranker template
    + fixed instruction
    + Query
    + Leaf.content

Input Candidate Limit:
    20

Max Input:
    4096 inference tokens

Silent Truncation:
    OFF

Leaf Re-splitting:
    OFF

Padding:
    left

Inference:
    model.eval()
    torch.inference_mode()
    no sampling
    no generation-based scoring

Score:
    P(yes) from yes/no logits

Sorting:
    rerank_score DESC
    fused_rank ASC

Absolute Threshold:
    OFF

Output Candidate Limit:
    5

Device:
    CUDA

Forward Dtype:
    FP16

Batch:
    ON
    default = 8
    configurable
    no automatic OOM batch fallback

CPU Fallback:
    OFF

Model Fallback:
    OFF

Reranker Failure Fallback:
    OFF

Next Stage:
    B6 Context Recovery
```

---

# 32. B5 Frozen Invariants

```text
1. B5 v1 Model = Qwen3-Reranker-0.6B。
2. 4B 只作为 D1 证据驱动的升级候选，不是 runtime fallback。
3. Backend = Transformers。
4. B5 唯一排序实体 = Leaf。
5. 输入 Document text = Leaf.content。
6. v1 不拼 heading / Parent / document metadata。
7. Instruction = ON，语言 = English。
8. 使用官方 Qwen3-Reranker template。
9. Query 必须来自 B4 FusionResult.query。
10. rerank_input_limit = 20。
11. rerank_output_limit = 5。
12. reranker_max_input_tokens = 4096。
13. 长度检查必须覆盖完整模板 + instruction + Query + Leaf.content + special tokens。
14. silent truncation = OFF。
15. Leaf re-splitting = OFF。
16. padding_side = left。
17. model.eval() + torch.inference_mode()。
18. B5 不使用生成采样。
19. rerank_score = P(yes) from yes/no logits。
20. rerank_score 越大越相关。
21. absolute threshold = OFF。
22. 排序 = rerank_score DESC，然后 fused_rank ASC。
23. 输出保留 rrf_score / fused_rank。
24. rerank_rank 从 1 开始。
25. device = CUDA。
26. forward dtype = FP16。
27. automatic CPU fallback = OFF。
28. batch inference = ON。
29. reranker_batch_size 默认 8，可显式配置。
30. automatic batch fallback = OFF。
31. B5 failure fallback = OFF。
32. Parent / Section Recovery 必须发生在 B5 之后。
33. 所有不兼容行为变化必须升级 b5 contract version。
```

---

# 33. Explicitly Deferred

以下不是 v1 baseline：

```text
Qwen3-Reranker-4B
vLLM backend
BF16 profile
FP32 profile
heading + Leaf.content
section path + Leaf.content
document metadata augmentation
absolute rerank threshold
dynamic batch sizing
automatic batch fallback
CPU fallback
alternative Cross Encoder
ColBERT / late interaction
Parent reranking
rerank-after-recovery
```

只有 D1 / 真实 Failure 提供明确证据后，才允许进入后续版本。

B5 v1 至此冻结。
