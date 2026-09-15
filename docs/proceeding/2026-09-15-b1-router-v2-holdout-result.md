# B1 `b1-router-v2` holdout 实测报告（2026-09-15）

> 给上游的现状报告，不是关闭申请。  
> 权威需求：`docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`  
> Holdout 冻结说明：`docs/proceeding/2026-09-15-b1-holdout-and-router-v2.md`  
> 本文只报告 **已冻结的 64 条 holdout** 在 xuanwu / Qwen2.5-7B-Instruct / `b1-router-v2` 上的一次完整复测。

## 1. 绑定状态

```text
B2
  CLOSED

B1 DETERMINISTIC CORE
  ACCEPTED

B1 IMPLEMENTATION
  COMPLETE
  - 合同 b1-contract-v1 / Schema b1-schema-v1 未改
  - prompt 已升级为 b1-router-v2（holdout 文件冻结之后才改 prompt）
  - guided JSON 仅为 vLLM 0.8.5 xgrammar 投影（无 minItems/maxItems/minLength，无 oneOf）
  - 非法组合 → RouterProtocolError；不自动修复、不重试、不把 task_limit 改成 direct

B1 PROTOCOL PATH（实现）
  PASS / fail-closed
  - 二次校验按合同拒绝改写与 >8 tasks
  - 健康 serving 下 RouterUnavailableError = 0

B1 PROTOCOL GATE（CLOSED 门禁：健康 serving 协议 100%）
  FAIL
  - 52/60

B1 SEMANTIC GATE（CLOSED 门禁：exact ≥95% 且各语义桶 ≥80%）
  FAIL 
  - 49/60（需 ≥57/60）

B1 OVERALL
  NOT CLOSED
  - 不申请关闭
  - 不申请 B1→B5 生产 E2E CLOSED
```

38 条草稿仍是 DEV/DIAGNOSTIC，**不是** holdout。v1 + guided `oneOf` 时草稿摸底为 `3/38`。本文数字只来自 64 条冻结集。

## 2. 测量条件

| 项 | 值 |
|---|---|
| 机器 | xuanwu |
| 模型 | Qwen2.5-7B-Instruct（未换 14B，未改 artifact revision） |
| Serving | vLLM 0.8.5，`--guided-decoding-backend xgrammar` |
| Prompt | `b1-router-v2` |
| 合同 / Schema | `b1-contract-v1` / `b1-schema-v1` |
| 集 | `code/retrieval/eval/b1_holdout_v1.jsonl`（64，已冻结） |
| 客户端 | `mfg-rag-preprocess` |
| 复测 | 两次；第二次打开 FAIL 原始 JSON dump，分数相同 |

第二次汇总（与第一次一致）：

```text
control=4/4
protocol_healthy=52/60
semantic_exact=49/60
all_cases_incl_control=53/64
unavailable=0
false_task_limit=0
task_limit_gt8=2/4
latency_p50_ms=605.6
latency_p95_ms=2894.4
latency_frozen=no
holdout_gates_met=no
b1_closed=no
```

## 3. 对照 CLOSED 门禁

| 门禁 | 要求 | 实测 | 结果 |
|---|---|---|---|
| 确定性 / control | 100% | 4/4 | 过 |
| 健康 serving 协议合法 | 100% | 52/60 | 未过 |
| 语义 exact 总体 | ≥57/60（95%） | 49/60 | 未过 |
| 各语义桶 | ≥80% | 见下表 | 未过 |
| ≤8 任务误标 `task_limit` | 0 | 0 | 过 |
| >8 明确独立任务 → `task_limit` | 100% | 2/4 | 未过 |
| 健康 serving `RouterUnavailableError` | 0 | 0 | 过 |
| P50 / P95 | 记录，本阶段不冻结 | 605.6 / 2894.4 ms | 只记录 |

分桶（语义 exact / 健康 serving）：

| bucket | 实测 | 80% 线 | 结果 |
|---|---|---|---|
| knowledge_single | 12/12 | 10/12 | 过 |
| long_tail | 5/5 | 4/5 | 过 |
| zh_en | 4/5 | 4/5 | 过 |
| system | 4/5 | 4/5 | 过 |
| unsupported | 2/5 | 4/5 | 未过 |
| mixed | 6/6 | 5/6 | 过 |
| multi_2_to_8 | 8/8 | 7/8 | 过 |
| task_limit_gt8 | 2/4 | 须 4/4 | 未过 |
| confirm_rewrite | 3/5 | 4/5 | 未过 |
| clarify | 3/5 | 4/5 | 未过 |
| control | 4/4 | 4/4 | 过 |

相对 v1/`oneOf` 草稿 3/38：主路径已经从「几乎总是空的 `task_limit`」转到「会 `direct` / 会拆 1–8 任务 / 不再用 `task_limit` 表示不确定」。这是真实进步，**不是**关闭证据。

## 4. 11 条 FAIL（gold 未改，评分未误伤）

协议失败 8 条 + 协议合法但语义错 3 条。FAIL 题都抓了原始 JSON。

### 4.1 单任务身份：只丢句号（route 已对）

合同：`direct` 且仅 1 个 task 时，`task.query` 必须与当前 Query **逐字相同**。

| id | gold | 原始输出 | 校验 |
|---|---|---|---|
| h-028 | unsupported | route=`unsupported`，query 少了结尾 `。` | `direct single task must equal current query` |
| h-029 | unsupported | 同上，少 `。` | 同上 |
| h-030 | unsupported | 同上，少 `。` | 同上 |

这三题 **route 正确**。h-031 / h-032 同桶带句号且 PASS，说明不是不会留句号，只是不稳定。  
`unsupported` 桶 2/5 未过 80%，三票全来自标点，不是 route 选错。

不建议为这三题放宽身份不变量。要过门禁应改 prompt，或上游另开合同变更（本轮未申请）。

### 4.2 单任务身份：英译中（route 已对）

| id | 当前 Query | 模型 task.query |
|---|---|---|
| h-019 | `How does NASA-STD-4003A define class S bonding?` | `NASA-STD-4003A 定义 class S 绑定方式怎么规定？` |

route=`knowledge` 正确。同桶英文 h-018 逐字保留后 PASS。失败原因是翻译，不是跨语言能力整体崩溃。

### 4.3 `direct` 拆出 9 条（必须 `task_limit`）

| id | 现象 | `error_detail` |
|---|---|---|
| h-047 | 中文 9 个独立问题 → 9 条 knowledge task | `Router tasks exceed maxItems=8` |
| h-050 | 英文 9 个独立问题 → 9 条 knowledge task | 同上 |

模型把 9 件事都认出来了，但 mode 应用错：合同要求 `mode=task_limit`、`tasks=[]`，禁止只做前 8 条。  
h-048 / h-049（中文 9 条列举、中文 10 条编号）已经 PASS。h-047 几乎是刚过的 h-046（8 条）再加 1 条，属于惯性继续拆。

### 4.4 该确认却静默执行（补全文本本身对）

| id | 当前 Query | 模型当 `direct` 执行的 query | gold `candidate_query` |
|---|---|---|---|
| h-053 | 胶粘剂贮存呢？ | NASA-STD-6016C 胶粘剂贮存条件怎么规定？ | 同一句 |
| h-055 | 持续运行电压呢？ | YH5WR-17/45 避雷器持续运行电压是多少？ | 同一句 |

二次校验按身份规则拒绝（task.query ≠ 当前 Query）是对的。语义上应走 `confirm_rewrite` + 空 tasks。h-051 / h-052 / h-054 已会确认；短「X呢？」仍会漏。

### 4.5 协议合法，语义错

| id | gold | 模型 | 说明 |
|---|---|---|---|
| h-027 | system | knowledge | 「你现在接入了哪些类型的技术文档？」问的是助手当前数据/文档类型，不是检索某一篇。JSON 合法，route 错。 |
| h-058 | clarify | direct，2 个 knowledge | 「另一种也要。」猜出「另一种标准」，对象是编的。 |
| h-059 | clarify | confirm_rewrite | 历史只有 NASA-STD-6030，candidate 却写成「6030 **和 6016C**」。6016C 是幻觉。 |

这三题 gold 保持 `clarify` / `system`，不改 holdout。

## 5. 已稳定的能力（本轮可记，不可当关闭）

- 普通 / 长尾 knowledge 单任务：17/17
- 中英混合 knowledge：4/5（唯一失败是英译中）
- system 四条能力问法正确；失败仅 h-027
- mixed knowledge+unsupported：6/6
- 2–8 个可数独立任务（含恰好 8 条）：8/8
- 无历史含糊句 clarify：h-056 / h-057 / h-060 PASS
- 明确「同样适用吗 / 铝导线呢 / 不锈钢同样要做吗」类 confirm：3/5 PASS
- **不再**用 `task_limit` 表示不确定或单任务（`false_task_limit=0`）
- 健康 serving 无 `router_unavailable`（holdout 内）

多任务 PASS 只核 mode + route 序列；不核每条 `task.query` 文案质量。单任务 PASS 才核逐字相同。

## 6. 明确不声称 / 未做

- 不把 49/60 说成已达准确率门禁
- 不把 8 条协议失败说成 xgrammar 故障；它们是业务二次校验按合同拒绝
- 不改 holdout gold，不把 38 条草稿升级为 holdout
- 不放宽「单任务逐字相同」去迁就丢句号
- 不自动修复 `task_limit` ↔ `direct`
- 未换 14B；未 bump 合同版本
- holdout **未包含**「随便处理一下」那次基础设施复放，该条仍待单独复放，不进语义分母
- P50/P95 未冻结
- B3–B5 服务端验收可继续，但不能声称 B1→B5 生产 E2E 已 CLOSED

## 7. 请上游裁定

实现侧判断：v2 已把 7B 从「协议合法但几乎全错」拉到「主路径可用、剩余 11 题类型清楚」。剩余问题集中，不必为这 11 题改 holdout 或改校验。

请上游在下列中选一（本轮 **不自行** 改 prompt / 换模型）：

1. **授权 `b1-router-v3`**（合同仍 `b1-contract-v1`，模型仍 7B，holdout 不动）。只补四条：单任务连标点也逐字保留、禁止英译中、>8 先数再 `task_limit`、短 follow-up 只确认不执行 / 不能唯一则 `clarify`；system 不要标 knowledge。v3 仍用同一 64 条复测。
2. **停在 v2**，接受 B1 本阶段 NOT CLOSED，先推进 B3–B5，B1 准确率以后再开。
3. **换 14B**。按此前约定，换模型要 bump B1 **合同**版本，不是只改 prompt。

不申请：改 holdout、放宽身份不变量、自动修复协议错误、现在关闭 B1。

## 8. 上游裁定（同日）

批准选项 1，并加严：

- 授权 `b1-router-v3`；合同 / Schema / 7B 不变。
- 旧 64 条对 **v2** 的本报告结论有效；从 v3 起该集降为 immutable regression，即使 64/64 也不得关闭 B1。
- v3 prompt 冻结后另建 unseen blind acceptance set 作为 CLOSED 候选。
- 若 v3 regression 仍有明显已知错误或 blind 未达标，不再堆 v4，下一步评估 14B 并 bump 合同版本。

落地见 `docs/proceeding/2026-09-15-b1-router-v3.md`。
