# B1 `b1-router-v3` blind 基线（2026-09-15）

> 集：`code/retrieval/eval/b1_blind_acceptance_v1.jsonl`  
> **是 64 条**（与 regression 同分配：control 4 + 语义 60）。  
> 模型：Qwen2.5-7B-Instruct + `b1-router-v3`  
> 只作 7B 基线，**不得据此 CLOSED，不得据此改 prompt（禁止 v4）。**

## 汇总

```text
case_count=64
control=4/4
protocol_healthy=55/60
semantic_exact=45/60
unavailable=0
false_task_limit=2
task_limit_gt8=3/4
blind_gates_met=no
b1_closed=no
```

相对 regression（51/60 语义、false `task_limit`=0），blind 更差：45/60，并且出现 **单任务误标 `task_limit`**。这正是 unseen 集该暴露的；不要回头用这些 FAIL 去写 v4。

CLOSED 门禁全部未过（除 control 与 unavailable）。

## 15 条 FAIL（gold 不改）

主路径仍在：knowledge 12/12、long_tail 5/5、2–8 任务 8/8、中文 9/11 条 `task_limit` PASS。

失败与 regression 同族，并多了 over-trigger：

| 类型 | 题 | 现象 |
|---|---|---|
| 英译中 | ba-018、ba-019 | 英文问句变成中文 task.query |
| 英文 9 任务 | ba-050 | 拆成 9 条 knowledge，`maxItems=8` |
| system 错 route | ba-026 | 问「会不会自己上网」→ unsupported（应 system） |
| system 错 route | ba-027 | 离线文档还是实时库 → knowledge（应 system） |
| 过澄清 | ba-025、ba-029、ba-032 | 问法已独立可懂，却 `clarify` |
| **假 `task_limit`** | ba-028、ba-031 | 单条 SAP 下单 / SolidWorks 出图 → `task_limit` |
| mixed 执行当能力 | ba-035 | 「查询 ERP 工单进度」标 system，应为 unsupported |
| 静默执行 | ba-052、ba-059 | 短 follow-up 直接 `direct` |
| confirm 文本不逐字 | ba-053 | mode 对，candidate 不等于 gold |
| 虚构拆任务 | ba-058 | 「别的方法也要」猜出第二条 knowledge |

## 不做什么

- 不改 blind / regression 题或 gold
- 不升 `b1-router-v4`，不针对本集做 prompt patch
- 不宣称 B1 或 B1→B5 E2E CLOSED
- **本集已 exposed**：未来新版本不得再把它当作 unseen CLOSED set

14B + v3 本轮否定，不切换。该结果不是永久否决 14B；重开须独立方案。绑定口径：`docs/proceeding/2026-09-15-b1-experiment-closeout.md`。
