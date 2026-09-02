# A5 Dense Embedding 实现冻结与交接（2026-08-18）

> 本文是 A5 **实现交接**，不是需求主文档。  
> 需求合同仍以 `docs/knowledge-base.md` A5 `[FROZEN baseline]` 与上游批准的 A5 检查合同为准。  
> 写本文的目的：上游在无法打开仓库时，仍能根据下面的实现口径、绑定点证据、pytest 命令和 smoke 命令验收 A5。  
> 本轮结束后不再继续改 A5。**`A5 CLOSED` 只能由上游确认。**

## 当前验收状态

```text
A5 DESIGN
FROZEN / PASS
- Qwen3-Embedding-4B + Transformers AutoTokenizer / AutoModel
- Leaf.content only；无 document instruction
- left padding + last-token pooling + L2 normalize
- 2560-d float32；不启用 MRL
- max_input_tokens=8192；禁止 silent truncation
- CUDA FP16 前向；FP32 输出
- 0.6B 不是 runtime fallback

A5 IMPLEMENTATION
PASS
- embed_leaves() / Qwen3DenseEncoder.encode_documents()
- CLI: python -m knowledge_base.dense_embedding
- 未修改 A1 / A2 / A3 Leaf / A4 / A0 / C8
- 未实现 A6 / BM25 / query instruction / MRL

A5 BINDING POINTS
PASS（shiyuan，2026-08-17/18 实测）
- Leaf API 可直接输入
- 4B 权重已缓存；hidden_size=2560
- GPU = NVIDIA A100-PCIE-40GB
- CUDA FP16 短文本前向 PASS；alloc≈7.55GB
- batch_size=4 保持默认
- 271 Leaf A5 inference token max=969 << 8192

A5 UNIT TEST
PASS
- Linux: 187 passed in 10.46s，0 skipped
- 含 A1–A4 回归与 T81–T106

A5 REAL CORPUS
PASS
- 271 Leaf / 271 vectors / 2560-d float32
- input_token_max=969；over_limit=0；NaN/Inf/norm 异常全 0
- batch_size=4 在 A100 40GB 上未 OOM
- determinism_failure_count=0（同一 10 条连跑两次 allclose）

A5 CLOSED
READY FOR UPSTREAM CLOSE
- 实现工作到此结束；申请关闭 A5，进入 A6
```

Linux pytest 与 271 CUDA smoke 硬条件均已满足。申请关闭 A5。A6 不得改 A5 数值合同（4B / 2560 / last-token / L2 / 无 truncation）。

---

## 1. 完成结论

| 项 | 结果 |
|---|---|
| 范围 | A5 document-side Dense Embedding。**未实现 A6 / BM25 / query instruction** |
| 输入 | 复用 A1 `load_markdown_documents()` + A3 `chunk_documents()` 重建同一套 Leaf |
| 代码位置 | 新增 `embedding_config.py` / `dense_embedding.py` / `tests/test_dense_embedding.py` |
| 未修改 | A3 Leaf schema、768/96、`token_count.py`、A4 hierarchy |
| 验收机 | `shiyuan`；`CUDA_VISIBLE_DEVICES=3` → A100-PCIE-40GB |
| **Linux pytest** | **187 passed, 0 skipped**（10.46s） |
| **A5 smoke** | 271 Leaf；2560-d float32；token max 969；norm≈1；invariants 全 0；`batch_size=4`；determinism=0 |
| A5 CLOSED | **申请关闭**（服务器硬条件已满足） |

---

## 2. 修改 / 新增文件

新增：

```text
code/knowledge_base/embedding_config.py
code/knowledge_base/dense_embedding.py
code/knowledge_base/tests/test_dense_embedding.py
docs/proceeding/2026-08-18-a5-dense-embedding.md
```

改过：

```text
code/knowledge_base/__init__.py          # 导出 A5 API
code/knowledge_base/dense_embedding.py   # 第一次 smoke 后：determinism 只比较同一 10 条的两次前向，不再跟 271 batch 对行
docs/knowledge-base.md                   # A5 段：无 document instruction、8192、禁止 truncation、0.6B 非 runtime fallback
agent.md                                 # A5 implementation complete / pending upstream acceptance
```

明确未改：

```text
code/knowledge_base/markdown_loader.py
code/knowledge_base/document_registry.py
code/knowledge_base/leaf_chunker.py
code/knowledge_base/chunking_config.py
code/knowledge_base/token_count.py
code/knowledge_base/leaf_ids.py
code/knowledge_base/structure_parser.py
code/knowledge_base/section_hierarchy.py
code/knowledge_base/tests/test_markdown_loader.py
code/knowledge_base/tests/test_document_registry.py
code/knowledge_base/tests/test_section_profile.py
code/knowledge_base/tests/test_leaf_chunker.py
code/knowledge_base/tests/test_section_hierarchy.py
code/C8/**
code/preprocessing/**
```

未把 `/tmp/a5-embeddings.npz` 写成 A6 生产存储。`batch_size=4`、具体 GPU index、FlashAttention、`/tmp` artifact 不是长期架构冻结项。

---

## 3. API

```python
class A5EmbeddingError(Exception): ...

@dataclass(frozen=True)
class DenseEmbeddingConfig:
    model_name_or_path: str = "Qwen/Qwen3-Embedding-4B"
    max_input_tokens: int = 8192
    batch_size: int = 4
    device: str = "cuda"

@dataclass(frozen=True)
class DenseEmbeddingResult:
    chunk_ids: tuple[str, ...]
    vectors: np.ndarray          # (N, 2560) float32
    model_name: str
    dimension: int               # 2560

class Qwen3DenseEncoder:
    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Leaf.content 原样编码。不加 instruction。"""

def embed_leaves(
    leaves: Sequence[Leaf],
    *,
    config: DenseEmbeddingConfig,
    encoder: Qwen3DenseEncoder | None = None,
) -> DenseEmbeddingResult
```

不暴露 `normalize=False` / `dimension=1024` / `pooling="mean"`。不实现 `encode_query` / query instruction serializer / `EmbeddedLeaf`。

`dimension`、last-token pooling、L2 normalize 不是用户可关 option。

---

## 4. 正式验收环境与命令

```text
host:         shiyuan
repository:   /public/zhangkairan/MyMethod/all-in-rag-main
environment:  mfg-rag-preprocess
Python:       3.11.15（与 A3/A4 freeze 同一 conda env）
torch:        2.6.0+cu126
transformers: 5.15.0
model:        Qwen/Qwen3-Embedding-4B
              cache snapshots/5cf2132abc99cad020ac570b19d031efec650f2b
GPU:          NVIDIA A100-PCIE-40GB（CUDA_VISIBLE_DEVICES=3 → cuda:0）
canonical:    /public/zhangkairan/MyMethod/all-in-rag-main/outputs/preprocessing
corpus:       NASA-STD-4003A / 6033 / 5009C（当前 A0 hash identity）
```

单元测试（stub，不加载 4B）：

```bash
cd /public/zhangkairan/MyMethod/all-in-rag-main
conda activate mfg-rag-preprocess
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"

python -m pytest code/knowledge_base/tests -v
```

Linux 已跑：**187 passed in 10.46s，0 skipped。**

271 Leaf CUDA smoke：

```bash
cd /public/zhangkairan/MyMethod/all-in-rag-main
conda activate mfg-rag-preprocess
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"

python -m knowledge_base.dense_embedding \
  --canonical-root "$PWD/outputs/preprocessing" \
  --model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 4 \
  --max-input-tokens 8192 \
  --output /tmp/a5-embeddings.npz \
  --report /tmp/a5-embedding-report.json
```

正式 271 smoke 已用 `batch_size=4` 跑通：`determinism_failure_count=0`，未 OOM。

`--canonical-root` 必填。CLI 复用 A1 loader + A3 `chunk_documents()`（768/96），用同一 `--model` 标识符加载 A3 tokenizer 与 A5 权重。A4 不参与。不得另写 chunker。

---

## 5. 六个现实绑定点（已测，不是猜测）

| # | 项 | 结果 |
|---|---|---|
| 1 | 真实 `Leaf` / `ChunkingResult` | **PASS**。七字段 frozen `Leaf` 与合同一致；A5 只读 `chunk_id`+`content` |
| 2 | 4B 权重 | **PASS**。HF cache 8.04GB；`hidden_size=2560`；`padding_side=left` |
| 3 | GPU | **PASS**。`NVIDIA A100-PCIE-40GB`。探测用过 2080 Ti 11GB，**验收卡是 A100** |
| 4 | AutoModel 加载 4B | **PASS**。CUDA FP16 前向 `cuda_forward_ok`；`shape (1,2560)` `float32` `norm=1.0`；`mem_alloc_GB=7.55` |
| 5 | `batch_size=4` | **PASS**。271 Leaf A100 smoke 未 OOM |
| 6 | 271 Leaf A5 token max | **PASS**。`leaf_count=271`；A5 inference `min=3` `max=969`；`over_8192=0` |

A3 profiler max=968 使用 `add_special_tokens=False`。A5 inference max=969，差 1 个 special token（EOS）。这不是 A3 冲突，A5 不以 A3 数字放行。

无接口冲突，未改 A3/A4。

---

## 6. Unit tests（T81–T106）

普通 pytest 用 stub tokenizer/model，验证 pipeline semantics，**不加载 4B**。

| ID | 断言 |
|---|---|
| T81 | 送入 tokenizer 的文本 `== Leaf.content` |
| T82 | 输入顺序 A,B,C → 输出 A,B,C |
| T83 | 重复 `chunk_id` → `A5EmbeddingError` |
| T84 | 空输入 → fail |
| T85 | 空 / 空白 `Leaf.content` → fail |
| T86 | actual tokens = 8192 → PASS |
| T87 | 8193 → fail，且未 truncation、未调用 model |
| T88 | tokenizer `truncation=False`，未传截断用 `max_length` |
| T89 | `padding_side == left`；短序列 pad 在左侧 |
| T90 | 合成 hidden states：left/right padding 都取最后有效 token |
| T91 | `[3,4]` → 约 `[0.6,0.8]`；先 float 再 L2 |
| T92 | 全部 \|v\|₂ ≈ 1 |
| T93 | dim = 2560 |
| T94 | dtype = float32 |
| T95 | NaN / Inf → fail |
| T96 | model 输出条数不匹配 → fail |
| T97 | `chunk_ids[i] ↔ vectors[i]` |
| T98 | A5 前后 Leaf 字段完全相同；无 `dense_vector` |
| T99 | 无 query instruction serializer |
| T100 | 4B load 失败 → `A5EmbeddingError`，不加载 0.6B |
| T101 | 指定 CUDA 但不可用 → fail，不自动 CPU |
| T102 | `batch_size` ≤0 → fail |
| T103 | `max_input_tokens` ≤0 → fail |
| T104 | fake hidden=1024 → fail，不 pad |
| T105 | 两次微小浮点差 `allclose` PASS，不要求 bitwise |
| T106 | A1–A4 测试文件仍在 |

Linux 已跑：

```text
187 passed in 10.46s
0 skipped
T81–T106: all PASSED
A1–A4 regression: all PASSED
```

---

## 7. Real CUDA smoke 硬条件

跑完 CLI 后，`/tmp/a5-embedding-report.json` 必须满足：

```text
leaf_count = embedding_count = 271
dimension = 2560
dtype = float32
over_limit_leaf_count = 0
nan_vector_count = 0
inf_vector_count = 0
invalid_norm_count = 0
duplicate_chunk_id_count = 0
missing_vector_count = 0
input_token_max < 8192          # 绑定点已测 969
abs(norm - 1.0) <= 1e-4
determinism_failure_count = 0
```

正式 Linux smoke（2026-08-18 第二次，`shiyuan`，A100，`batch_size=4`）原文口径：

```text
model_name=Qwen/Qwen3-Embedding-4B
transformers_version=5.15.0
torch_version=2.6.0+cu126
device=cuda
gpu_name=NVIDIA A100-PCIE-40GB
forward_dtype=float16
max_input_tokens=8192
batch_size=4
leaf_count=271
embedding_count=271
embedding_dimension=2560
embedding_dtype=float32
input_token_min=3
input_token_p50=132.0
input_token_p95=711.5
input_token_max=969
over_limit_leaf_count=0
nan_vector_count=0
inf_vector_count=0
invalid_norm_count=0
duplicate_chunk_id_count=0
missing_vector_count=0
vector_norm_min=0.9999998807907104
vector_norm_max=1.0000001192092896
vector_norm_mean=1.0
determinism_failure_count=0
```

| 硬条件 | 结果 |
|---|---|
| leaf_count = embedding_count = 271 | PASS |
| dimension=2560 dtype=float32 | PASS |
| over_limit / nan / inf / invalid_norm / duplicate / missing = 0 | PASS |
| input_token_max=969 < 8192 | PASS |
| abs(norm-1) ≤ 1e-4 | PASS（min/max 偏离约 1.2e-7） |
| batch_size=4 未 OOM | PASS |
| determinism_failure_count = 0 | PASS |

第一次 smoke 曾报 `determinism_failure_count=1`：当时检查器把 10 条子集向量跟 271 大 batch 对应行比较（不同 left-padding 几何 + FP16）。失败计数是 1 不是 2，同一 10 条连跑两次已 allclose。检查器按合同删掉跨 batch 比较后重跑，count=0。

Spotcheck 10 条（正式 smoke stdout，与第一次选取相同）：

| 类 | chunk_index | page | A5 tokens |
|---|---:|---|---:|
| 短 | 0 | 1–1 | 10 |
| 短 | 1 | 1–1 | 46 |
| 接近普通 max | 8 | 8–8 | 766 |
| 接近普通 max | 20（5009C） | 13–14 | 755 |
| 普通 table | 2 | 2–2 | 395 |
| 普通 table | 3 | 3–3 | 361 |
| oversize table | 59 | 31–31 | 922 |
| oversize table | 62 | 32–32 | 833 |
| 跨页 | 56 | 29–30 | 305 |
| 跨页 | 75 | 36–38 | 383 |

与 A3.2 freeze spotcheck 同类。A5 token 比 A3 profiler 大约 +1（EOS）。

npz：`/tmp/a5-embeddings.npz`（smoke artifact，不是 A6 store）。`vectors.shape` 对应 271×2560 float32。

---

## 8. Scope statement

```text
是否修改 A3 Leaf？                NO
是否修改 A3 chunking？            NO
是否修改 A4 hierarchy？           NO
是否实现 A6？                     NO
是否实现 BM25？                   NO
是否实现 query instruction？      NO
是否启用 MRL？                    NO
是否自动 fallback 0.6B？          NO
是否静默 truncation？             NO
```

---

## 9. A5 DoD

```text
A5 DESIGN                         FROZEN / PASS
QWEN3-EMBEDDING-4B                PASS（权重已缓存；hidden_size=2560）
LEAF.CONTENT ONLY                 PASS（实现 + T81）
LEFT PADDING                      PASS（实现 + T89；加载探测 padding_side=left）
LAST-TOKEN POOLING                PASS（实现 + T90；短文本 [:, -1] 前向）
MAX INPUT VALIDATION              PASS（实现；corpus max=969）
NO SILENT TRUNCATION              PASS（实现 + T87/T88）
2560 DIMENSION                    PASS
L2 NORMALIZATION                  PASS（短文本 norm=1.0）
FP32 OUTPUT                       PASS
CHUNK↔VECTOR MAPPING              PASS（实现 + T82/T97）
LEAF IMMUTABILITY                 PASS（实现 + T98）
NO RUNTIME MODEL FALLBACK         PASS（实现 + T100）
NUMERIC INVARIANTS                PASS（271 smoke：norm min/max ≈ 1 ± 1.2e-7）
DETERMINISM TOLERANCE             PASS（同一 10 条连跑两次 allclose；count=0）
UNIT TEST                         PASS（187 passed, 0 skipped）
A1–A4 REGRESSION                  PASS
REAL CUDA CORPUS                  PASS（271 / 2560 / fp32 / over_limit=0 / batch=4）
```

之后：

```text
A5 CLOSED
```

只能由上游确认。实现侧不自行把 `agent.md` 写成 frozen。

---

## 10. 给上游的冻结申请

A5 已按检查合同落到现有 A3 `Leaf` 上，六个绑定点无冲突。Linux pytest **187 passed, 0 skipped**。271 CUDA smoke 硬条件全部 PASS，含 `determinism_failure_count=0`。

申请关闭 A5，进入 A6。A6 消费 `chunk_id ↔ dense_vector` 做 FLAT+IP，不再二次 normalize。
