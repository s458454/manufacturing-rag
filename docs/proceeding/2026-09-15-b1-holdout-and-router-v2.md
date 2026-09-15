# B1 holdout 冻结与 router v2（2026-09-15）

> 权威需求仍是 `docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`。  
> 本文当时冻结 **64 条 CLOSED 门禁集** 与 `b1-router-v2`。  
> **2026-09-15 晚间上游批复：** 该 64 条对 v2 的结论仍有效，但从 v3 起降为 immutable regression/diagnostic，不再当 CLOSED holdout。新闭环候选集见 `docs/proceeding/2026-09-15-b1-router-v3.md`。  
> 不把 B1 标成 CLOSED。

## 1. 上游状态（绑定）

```text
B2 CLOSED
B1 DETERMINISTIC CORE ACCEPTED
B1 IMPLEMENTATION COMPLETE
B1 PROTOCOL PATH PASS
B1 SEMANTIC ACCURACY FAIL
B1 OVERALL NOT CLOSED
```

38 条草稿（`b1_draft_acceptance_set.jsonl`）只是 DEV/DIAGNOSTIC，**不能**升级为 CLOSED holdout。xuanwu 上 7B + `b1-router-v1` + guided `oneOf` 的摸底是 `mode_route_score=3/38`：几乎所有 `direct` 被压成 `task_limit` + 空 tasks。

## 2. Guided schema

`ROUTER_GUIDED_JSON_SCHEMA` 只做 vLLM 0.8.5 xgrammar 兼容投影：去掉 `minItems` / `maxItems` / `minLength`。

**已删除 `oneOf` / `const`。** 冻结合同 `ROUTER_JSON_SCHEMA` 与 business validation 不变。非法 mode/字段组合 → `RouterProtocolError`。不自动修复，不重试，不把 `task_limit` 改成 `direct`。

引擎/请求 backend 仍是 `xgrammar`。禁止 `auto` 与请求级 `xgrammar:no-fallback`。

## 3. 64 条 holdout 冻结

文件：

```text
code/retrieval/eval/b1_holdout_v1.jsonl
code/retrieval/eval/run_b1_holdout.py
```

分配：

| bucket | n | 门禁 |
|---|---|---|
| control | 4 | 确定性 100% |
| knowledge_single | 12 | 语义 ≥80%；false `task_limit` = 0 |
| long_tail | 5 | 同上 |
| zh_en | 5 | 同上 |
| system | 5 | 同上 |
| unsupported | 5 | 同上 |
| mixed | 6 | 同上 |
| multi_2_to_8 | 8 | 同上；含一条可数的 8 独立任务 |
| task_limit_gt8 | 4 | `task_limit` 100% |
| confirm_rewrite | 5 | 语义 ≥80%；candidate 逐字 |
| clarify | 5 | 语义 ≥80% |
| **合计** | **64** | |

防泄漏：当前问句不与 38 条草稿逐字相同；不用 `b1-router-v2` 正例原句。语料仍是 ver1 已入库 17 篇，换了问法。

评分分三路，禁止混进一个准确率：

1. **协议**：健康 serving 下必须得到 `RouterResult`，不能是 `RouterProtocolError` → CLOSED 要求 100%。
2. **语义 exact**：mode、route 序列、单任务 `task.query == current`、`confirm_rewrite` 的 `candidate_query` 逐字。总体 ≥57/60（95%）；每个语义 bucket ≥80%。
3. **基础设施**：`RouterUnavailableError` 不进语义分母。健康 serving 下必须为 0。`h-034`「随便处理一下」那次 unavailable 要单独复放，不进本题集。

另外：

- ≤8 任务误标 `task_limit` = 0
- >8 个明确独立任务 → `task_limit` = 100%
- P50/P95 只记录，`latency_frozen=no`

跑：

```bash
cd /public/zhangkairan/MyMethod/all-in-rag-main/code
conda activate mfg-rag-preprocess
export B1_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export B1_MODEL_ALIAS=Qwen/Qwen2.5-7B-Instruct
export B1_TOKENIZER_NAME_OR_PATH=/public/zhangkairan/Qwen2.5-7B-Instruct
export B1_REQUEST_TIMEOUT_MS=60000
python retrieval/eval/run_b1_holdout.py
```

脚本写死 `b1_closed=no`。即使 `holdout_gates_met=yes` 也要上游确认后才能关 B1。

实测报告：`docs/proceeding/2026-09-15-b1-router-v2-holdout-result.md`（v2 未过门禁，不申请 CLOSED）。

## 4. Prompt `b1-router-v2`

在 holdout 文件冻结之后，实现短 prompt：明确 mode 优先级，并给 direct / confirm_rewrite / clarify / task_limit 各一条正例。

```text
ROUTER_CONTRACT_VERSION = b1-contract-v1
ROUTER_PROMPT_VERSION   = b1-router-v2
ROUTER_SCHEMA_VERSION   = b1-schema-v1
```

模型仍是 Qwen2.5-7B-Instruct。本轮不升 14B；若 v2 过不了 holdout，再讨论换模型，那时要 bump B1 **合同**版本。

未改：business validation、冻结合同 Schema、confirmation lexicons、Memory 方案 A、B3–B5、vLLM pin。

## 5. 本轮仍不声称

- B1 CLOSED
- B1→B5 生产 E2E CLOSED
- 38 条草稿是 holdout
- P50/P95 数字已冻结
