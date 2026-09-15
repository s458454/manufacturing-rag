# B1 实验结论冻结（ver-0.7 / 2026-09-15）

> 权威需求：`docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`  
> 上游已接受本轮 B1 实验结论。本文是 **ver-0.7 绑定口径**，不是关闭申请。

## 1. 上游接受的 8 条

```text
1. B1 保持 NOT CLOSED。
2. b1-router-v3 是 Qwen2.5-7B-Instruct 当前最终 reference baseline；
   不称为 production-accepted Router。
3. 停止 b1-router-v4，以及针对当前测试集的继续 prompt patch。
4. 14B + v3 本轮否定，不切换模型。
   该结果不等价于永久否定 14B；未来若重开须作为独立方案评估。
5. 旧 regression 64 与本轮 blind 64 均保留原题与 gold。
   本轮 blind 结果对 v3 有效，但从现在起该集也视为 exposed；
   未来新版本不得再把它作为 unseen CLOSED set。
6. 若未来只修改 Prompt，应升级 router_prompt_version；
   若更换冻结模型、route/mode/interface 或其他合同级行为，
   才升级 B1 contract version。
7. B3–B5 继续独立服务器验收，不等待 B1 CLOSED。
8. 完整 B1→B5 production E2E 仍不得宣称 PASS/CLOSED。
```

## 2. 当前冻结事实

```text
B2 CLOSED
B1 DETERMINISTIC CORE ACCEPTED
B1 IMPLEMENTATION COMPLETE
B1 CONTRACT / SCHEMA
  b1-contract-v1 / b1-schema-v1
B1 PROMPT
  b1-router-v3 = 7B reference baseline（非 production-accepted）
B1 MODEL
  Qwen2.5-7B-Instruct（本轮不切 14B）
B1 OVERALL
  NOT CLOSED
```

guided JSON 仍是 vLLM 0.8.5 xgrammar 兼容投影（无 `oneOf`）。非法组合 → `RouterProtocolError`。不自动修复、不重试。

## 3. 实测矩阵（全部保留）

Serving：`vllm==0.8.5`，`--guided-decoding-backend xgrammar`。评测分协议 / 语义 exact / `unavailable`。

| 轮次 | 模型 | Prompt | 集 | 协议 | 语义 | 假 task_limit | >8 | 口径 |
|---|---|---|---|---|---|---|---|---|
| A | 7B | v1 + `oneOf` | 38 草稿 | smoke 过 | 3/38 | 几乎全空 `task_limit` | — | `oneOf` 已删除 |
| B | 7B | v2 | 64（当时 holdout） | 52/60 | 49/60 | 0 | 2/4 | 对 v2 有效；现为 regression |
| C | 7B | v3 | 同 64，现 regression | 56/60 | 51/60 | 0 | 4/4 | 已知 family 未消干净 |
| D | 7B | v3 | 新 64 blind | 55/60 | 45/60 | 2 | 3/4 | 对 v3 有效；即日起 **exposed** |
| E | 14B | 同一 v3 | regression 64 | 59/60 | 42/60 | 7 | 4/4 | 本轮否定切换 |

CLOSED 门禁均未过。P50/P95 只记录，未冻结。

分集角色：

| 文件 | 角色（ver-0.7 起） | 改题/gold |
|---|---|---|
| `code/retrieval/eval/b1_draft_acceptance_set.jsonl` | DEV/DIAGNOSTIC | 不升级为 holdout |
| `code/retrieval/eval/b1_holdout_v1.jsonl` | immutable regression | 否 |
| `code/retrieval/eval/b1_blind_acceptance_v1.jsonl` | v3 已暴露的诊断集，**不再是 unseen CLOSED set** | 否 |

## 4. 版本升级规则（绑定）

- **只改 System Prompt** → 升 `router_prompt_version`（例如 `b1-router-v4` 须另开，且不得针对当前两套 64 条做 patch）。合同可保持 `b1-contract-v1`。
- **换冻结模型、改 route/mode/接口、改身份不变量或其它合同级行为** → 升 B1 **contract version**。
- 本轮 14B + v3 失败 **不是** 永久否决 14B。未来重开须当独立方案：新 prompt 和/或新合同、新的 unseen 关闭集，不得复用已暴露的 blind 64 当 unseen。

## 5. 本里程碑仍不声称

- B1 CLOSED
- `b1-router-v3` 为 production-accepted Router
- 14B 永久不可用
- B1→B5 production E2E PASS/CLOSED
- 当前测试集上继续 prompt 迭代

B3–B5 代码与合同可继续独立服务器验收。详细分轮记录见：

- `docs/proceeding/2026-09-15-b1-router-v2-holdout-result.md`
- `docs/proceeding/2026-09-15-b1-router-v3-regression.md`
- `docs/proceeding/2026-09-15-b1-router-v3-blind.md`
