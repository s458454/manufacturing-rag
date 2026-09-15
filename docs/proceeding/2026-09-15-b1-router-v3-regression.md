# B1 `b1-router-v3` regression 实测（2026-09-15）

> 集：`code/retrieval/eval/b1_holdout_v1.jsonl`（immutable regression，不是 CLOSED 集）  
> 模型：Qwen2.5-7B-Instruct + `b1-router-v3`  
> **不申请 CLOSED。不进入 `b1-router-v4`。**

## 1. 结论

v3 消掉了 v2 上最硬的两类：**>8 任务拆成 9 条 task**、**unsupported 只丢句号**。  
但批复要消的已知 family **没有消干净**：英译中、system 标成 knowledge、短 follow-up 静默执行、clarify 虚构实体仍在。另外出现英文压缩改写、confirm 文本不逐字、以及一条 unsupported 被过澄清。

按 ver-0.7 上游绑定：

```text
不再堆 b1-router-v4
14B+v3 本轮否定，不切换；未来重开须独立评估
只改 prompt → 升 router_prompt_version；换模型/合同级行为 → 升 contract version
B1 OVERALL NOT CLOSED
B1→B5 E2E 不得宣称 CLOSED
```

regression 分数不得解释为关闭证据。`perfect_score_closes_b1=no`。

## 2. 与 v2 对照

| 项 | v2 | v3 |
|---|---|---|
| 协议 | 52/60 | **56/60** |
| 语义 exact | 49/60 | **51/60** |
| `task_limit_gt8` | 2/4 | **4/4** |
| false `task_limit` | 0 | 0 |
| unavailable | 0 | 0 |
| unsupported 丢句号 h-028/029/030 | FAIL | **PASS** |
| 静默执行 h-053 | FAIL | **PASS** |
| 需 ≥57/60 | 未到 | 未到 |

```text
control=4/4
protocol_healthy=56/60
semantic_exact=51/60
unavailable=0
false_task_limit=0
task_limit_gt8=4/4
regression_gates_met=no
b1_closed=no
latency_p50_ms=601.2
latency_p95_ms=1901.8
```

## 3. v3 仍 FAIL 的 9 条

### 已知 family，未消除

| id | family | 原始输出 |
|---|---|---|
| h-019 | a 英译中 | `How does NASA-STD-4003A define class S bonding?` → 中文问句 |
| h-027 | e system→knowledge | JSON 合法，route=`knowledge` |
| h-055 | c 静默 direct | `持续运行电压呢？` 被补全后当 knowledge 执行 |
| h-058 | d 虚构 | 「另一种也要。」拆成 2 个 knowledge，第二条「另一种标准」 |
| h-059 | d 虚构 | 历史只有 6030，candidate 写入 **6016C** |

### 同类身份问题 / 回归

| id | 说明 |
|---|---|
| h-002 | 英文单任务被压缩：丢掉 `Please summarize` 和句号。v2 此题 PASS，v3 回归。 |
| h-018 | 英文避雷器问句被译成中文。v2 此题 PASS，v3 回归。route 仍是 knowledge。 |

### 新现象（不是 v2 那 11 条的表面重复）

| id | gold | 模型 | 说明 |
|---|---|---|---|
| h-032 | unsupported | clarify | 「用今天的天气写一段工厂开工祝福。」被过澄清。写作任务本身已可独立理解。 |
| h-054 | confirm_rewrite | confirm_rewrite，candidate 不逐字 | gold：`NASA-STD-6030 镍基高温合金粉末床熔融有哪些工艺控制？`；模型：`NASA-STD-6030 钛合金粉末床熔融的工艺控制是否同样适用于镍基高温合金？`。mode 已对，不是静默执行；exact 门禁仍判 FAIL。 |

## 4. 明确不做

- 不改 `b1-router-v3` 正文（那会变成 v4）
- 不改 regression / blind 的题或 gold
- 不放宽单任务逐字相同、不自动修协议错误
- 不把 51/60 说成已达 95% 门禁
- 不在 7B 上继续堆 prompt 版本

## 5. 后续（ver-0.7 已冻结）

14B + 同一 v3 已在 regression 上测过（42/60，假 `task_limit`=7），**本轮否定切换**。这不是永久否决 14B。绑定口径：`docs/proceeding/2026-09-15-b1-experiment-closeout.md`。
