# B1-Jev 候选 Router 现状（2026-09-28，2026-09-29 冻结）

> 状态：`CLOSED`
>
> 2026-09-29 关闭。权威合同是 `docs/Preprocessing/2026-09-29-B1_Jev_Router_Frozen_Requirements_v1.md`。下文在关闭前写成，其中 `B1 CLOSED = no` 的段落是当时记录。
>
> 现网判断核是 Jev。Qwen2.5-7B-Instruct 只做拆分和改写。旧 `b1-router-v3` 的 `QueryRouter` 已退出项目。
>
> 标签实验的历史记录仍是：`docs/proceeding/2026-09-23-b1-jev-v0-experimental.md`（现状截止 2026-09-27）

## 0. 一句话

候选包已经能从 `QueryRequest` 走到现有 `RouterResult`，并交给现有 `knowledge_processing_requests()`。2026-09-29 晚些时候，主路径改为 `build_jev_router()` → `JevQueryRouter.route()` → `route_for_b2()` → `dispatch_knowledge_tasks()`。旧 `QueryRouter` 已移出项目，副本在 gitignore 的 `local_backup/b1-router-v3/`。提示仍然停手。旧 128 条仍不是 unseen CLOSED set。

```text
B1-Jev candidate implementation = FROZEN
b1-contract-v2 = CLOSED
B1 CLOSED = yes
main_path = connected
prompt tuning = stopped
old_router = local_backup/b1-router-v3 (gitignored)
```

## 1. 代码落点

```text
主路径：code/retrieval/b1/pipeline.py
  build_jev_router()     SystemOneJudge + Qwen 文本生成
  route_for_b2()         JevQueryRouter.route() → dispatch_knowledge_tasks()

判断与生成：code/retrieval/b1_jev/
  router.py              JevQueryRouter.route() → RouterResult
  questions.py           与 v0 相同的两问
  jev_client.py          POST /v1/systemone，一次调用，无重试
  multi_splitter.py      只在 knowledge_mode=multi 时调用
  rewrite_generator.py   只在 knowledge_mode=confirm_rewrite 时调用

仍在项目里、给 Jev 和 B2 共用：
  code/retrieval/b1/dispatcher.py
  code/retrieval/b1/prompt.py
  code/retrieval/b2/
  b1_holdout_v1.jsonl
  b1_blind_acceptance_v1.jsonl

已移出项目，只留本地备份：
  local_backup/b1-router-v3/   gitignore
```

离线单测：`code/retrieval/tests/test_b1_jev_router.py`。假 Jev、假 Qwen，不连服务。接入后增加了 `route_for_b2` 把 knowledge 任务送进 B2 的断言。

链路与已冻结的替换顺序一致：

```text
B1.0
→ Jev primary_route
   system / unsupported → 直接 RouterResult，不再问第二问
   knowledge → Jev knowledge_mode
      direct / task_limit / clarify → 确定性组装，不调 Qwen
      multi → Qwen 拆分
      confirm_rewrite → Qwen 生成 candidate_query
→ 现有 RouterResult
```

旧 `QueryRouter`、`VLLMRouterClient`、旧 Router prompt 不进这条链路，也不做 fallback。`mixed` 没有新方案。

## 2. 服务

```text
Jev：兔子 jev-1.13，POST https://api.tu-zi.com/v1/systemone
Qwen：http://10.50.2.6:8000/v1 ，模型 Qwen/Qwen2.5-7B-Instruct
拆分对照用过的远端模型：LiaoBots gpt-4o-mini，https://ai.liaobots.work/v1
```

vLLM 的 xgrammar 拒绝 `minItems`、`maxItems`、`minLength`。发给它的 schema 已按现网 `ROUTER_GUIDED_JSON_SCHEMA` 的方式去掉这三项。2～8 条、非空、不重复仍在 Python 里检查，失败不改写成 direct、不截断、不去重。探测时带这三项的请求是 HTTP 400；去掉之后诊断里的拆分和改写请求是 HTTP 200。

## 3. 暴露 128 条的完整结果诊断

脚本：`code/retrieval/eval/run_b1_jev_router_diag.py`

数据：`b1_holdout_v1.jsonl` + `b1_blind_acceptance_v1.jsonl`。已暴露，不是 unseen。`clarify` 与 `mixed` 继续 skip，评分脚本没有把 clarify 补成 `knowledge`。

这次跑在拆分 few-shot **之前**。控制项 4/4。`unavailable=0`，`protocol=0`。`rewrite_error=0` 只表示改写调用没有抛错，不表示 `candidate_query` 与 gold 逐字一致。

```text
holdout  control=4/4  result=44/49  skip=11  splitter_error=2
blind    control=4/4  result=43/49  skip=11  splitter_error=2
b1_closed=no
main_path=not_connected
```

Jev 的 `primary_route` 在计入分母的题上是对的。单条 knowledge、system、unsupported、task_limit 的 `RouterResult` 与现有 gold 一致。

当时还没过的题：

```text
拆分调用被 Python 拒绝（Qwen HTTP 200，但只留下第一句）：
  h-039 h-040 ba-039 ba-040
  都是「A，以及 B」

拆分条数对，文本与实验 gold 不一致：
  h-044  补了「的要求 / 的规定」
  ba-046 留下了 A–H

改写生成了 candidate_query，但不是 gold：
  h-051 h-054  把历史原句和当前句拼在一起，没有替换对象
  ba-055        把绝缘监测和接地电阻合成一句

Jev 把 confirm_rewrite 判成 clarify，改写器没有被调用：
  ba-051 ba-053
```

这 128 条在拆分 few-shot 之后重跑过一次，口径仍是 multi 逐字比对：holdout `result=47/49`，blind `result=45/49`，两边 `splitter_error=0`、`rewrite_error=0`、`control=4/4`、`skip=11`。剩下的失败是 `h-051`、`h-054`、`ba-053`、`ba-055` 的改写文本，`ba-051` 被判成 clarify，以及 `ba-044` 每条多了「说明」。诊断脚本随后把 `ba-051` 标为 ambiguous、把 multi 改成 anchors 通过即可。新口径的活跑结果写在第 8 节，以那一节为准。

## 4. 拆分对照（新 24 条）

文件：`code/retrieval/eval/b1_jev_split_compare_v1.jsonl`  
脚本：`code/retrieval/eval/run_b1_jev_split_compare.py`  
`closed=false`。问法包括「以及」两句和三句、另外、先…再…、分别说明、分号、编号、字母、英文。参考拆分只用于对照，不是冻结 gold。

逐字 `exact` 不作为拆分是否成功的标准。两边模型都会改措辞。采用的是：

```text
parse    能否通过 2～8 条、非空、不重复
count    条数是否与参考拆分一致
anchors  同一位置的标准号或对象是否还在
```

Qwen 使用与候选拆分器相同的提示和 xgrammar。`gpt-4o-mini` 只有聊天补全，没有 xgrammar。

| | Qwen few-shot 前 | Qwen few-shot 后 | gpt-4o-mini |
|---|---|---|---|
| parse | 20/24 | 24/24 | 24/24 |
| count | 20/24 | 24/24 | 24/24 |
| anchors | 20/24 | 24/24 | 23/24 |
| exact | 6/24 | 19/24 | 2/24 |

few-shot 前，Qwen 的 4 次失败都是「以及」两句只输出第一句：`sc-001`、`sc-005`、`sc-006`、`sc-024`。这种问法共 7 条。

few-shot 后，这 7 条全部拆成 2 条，标准号留在原位。24 条的 parse、count、anchors 都是 24/24。没过 exact 的 5 条是句末问号、把「说明」挪到句尾，以及 3 条英文被译成中文或中英夹杂。例子全是中文。没有再出现丢后半句，也没有把前一个标准号套到后一个任务上。

`gpt-4o-mini` 的「以及」两句是 7/7，但 `sc-011` 把 `NASA-STD-6030` 套到了「兆瓦充电接口温升」上。Qwen 在 few-shot 之后这题 anchors 通过。

因此拆分器继续用本机 Qwen2.5-7B，不把拆分换成远端 `gpt-4o-mini`。few-shot 在 `code/retrieval/b1_jev/multi_splitter.py` 的 `splitter_messages()`。两个例子的标准号不在评测集里：`GB/T 5094.1` 与 `IEC 60204-1` 的「以及」；`ISO 2768-1`、`ASME Y14.5`、`GB/T 1184` 的「分别说明」。规则 11 写明「A，以及 B」是两个任务，不能合成一条。

暴露 128 条里的「以及」句式，以第 8 节为准：blind 的多任务拆分通过；holdout 同日有一批拆分调用失败，那是调用中断，不是又把后半句丢掉。

## 5. 改写还没有同样的证据

`rewrite_generator.py` 仍是规则提示，没有 few-shot。暴露诊断里的失败是内容错误和 Jev 误判，不是 HTTP 失败：

```text
该替换对象时，模型把历史原句和当前句拼接
该判 confirm_rewrite 时，Jev 给出 clarify，生成器不被调用
```

`code/retrieval/eval/b1_jev_rewrite_compare_v1.jsonl` 20 条在短 Prompt 下 `exact=20/20`、`replaced=20/20`，本轮补全后重跑未退化。短提示之后，暴露集上完整返回的改写里，只剩 `ba-055` 把「绝缘监测报警阈值」留在「接地电阻」那一句。`h-051`、`h-054`、`ba-053` 在短提示后的逐字比对里已经通过；第 8 节里 `h-051` 是改写调用失败，没有产出错误文本。`ba-051` 有两种合理补法，标为 ambiguous，不改 Jev Prompt，也不进准确率分母。

2026-09-29 决定不为 `ba-055` 加 few-shot。暴露集上的改写因此留下这一条已知内容残留。

## 6. 关闭前的待定项

2026-09-29 已按 `b1-contract-v2` 关闭。下面是关闭前的问题，定案写在新合同里，不再开放。

1. 候选顺序是否仍按已冻结的链路执行：先 `primary_route`，只有 knowledge 再问 `knowledge_mode`；system / unsupported 原出口；旧 Router 不回退；Qwen 只做拆分和改写。现状代码是按这个做的。
2. 拆分验收是否改为 parse + count + anchors，不要求 task 文本逐字等于某一条参考句。这组数据支持这样定。英文问句是否必须保持英文，要单独立：当前中文 few-shot 会把英文拆分结果中文化，标准号还在。
3. 拆分模型是否就定为现网这台 Qwen2.5-7B，而不是 `gpt-4o-mini`。对照结果支持继续用 Qwen。
4. `confirm_rewrite` 的已知内容残留只有 `ba-055`。新 20 条唯一槽位替换已通过。2026-09-29 决定不加 few-shot，改写提示随候选一起冻结。
5. `mixed` 继续排除。代码没有扩展。exposed clarify 没有可靠 `primary_route` gold，诊断继续 skip。
6. 转正前必须新做 unseen blind。不得再用 holdout、blind、24 条拆分集、20 条改写集宣称 CLOSED。
7. 2026-09-29 已把主路径接到 `JevQueryRouter`，再进现有 Dispatcher / B2。旧 `QueryRouter` 不在项目里。

```text
b1-contract-v2 = CLOSED
B1 CLOSED = yes
main_path = JevQueryRouter → route_for_b2 → B2
old QueryRouter = local_backup/b1-router-v3, gitignored
```

## 7. 本轮链路补全

本轮只把已有组件接到诊断和单测上。Splitter Prompt、Rewriter Prompt、Jev 问句都没有改。

```text
修改：
  code/retrieval/eval/run_b1_jev_router_diag.py
  code/retrieval/tests/test_b1_jev_router.py
  docs/proceeding/2026-09-28-b1-jev-candidate-status.md

未改现网：
  code/retrieval/b1/router.py
  code/retrieval/b1/client.py
  code/retrieval/b1/dispatcher.py
  code/retrieval/b1/confirmation.py
  code/retrieval/b1/types.py
  code/retrieval/b2/
  retrieval/__init__.py
  retrieval/b1/__init__.py
```

诊断现在在 `route()` 成功后调用 `knowledge_processing_requests()`，并记录 `splitter_called`、`rewriter_called`、`dispatcher_knowledge_request_count`。`ba-051` 的 lane 是 `ambiguous`。`multi_2_to_8` 用条数、全部 knowledge、同一位置的标准号或对象判定通过；逐字差异记 `split_exact=false`，例如多一个「说明」仍算 semantic pass。

```text
离线单测                         24 passed
拆分 24 条（Qwen，few-shot）    parse=24/24  count=24/24  anchors=24/24  exact=19/24
改写 20 条（Qwen，短 Prompt）   exact=20/20  replaced=20/20
```

仍未覆盖，且本轮不做：

```text
mixed
exposed clarify 的 primary_route gold
ba-051 的 Jev 边界
英文拆分被中文化（sc-021～sc-023，anchors 仍通过）
新的 unseen blind
```

## 8. 冻结与接入（2026-09-29）

不再改 Jev 问句、拆分 few-shot、短改写提示。主路径是 `build_jev_router()` 和 `route_for_b2()`：Jev 产出 `RouterResult`，`dispatch_knowledge_tasks()` 把 knowledge 任务交给 B2。旧判断核在 `local_backup/b1-router-v3/`，该目录被 gitignore，不进入仓库。诊断脚本是 `code/retrieval/eval/run_b1_jev_router_diag.py`。口径：`mixed` 与 exposed `clarify` 跳过；`ba-051` 为 ambiguous，不进分母；multi 按条数、全部 knowledge、同一位置的标准号或对象通过，逐字差异另记 `split_exact`。

```text
holdout  control=4/4  result_pass=38/46  skip=11  ambiguous=0
         unavailable=3  protocol=0  splitter_error=7  rewrite_error=1
blind    control=4/4  result_pass=47/48  skip=11  ambiguous=1
         unavailable=0  protocol=0  splitter_error=0  rewrite_error=0
b1_closed=no
main_path=not_connected
```

上面这一行 `main_path` 是接入前那次诊断的打印。接入之后，`diagnostic_status()` 改为 `main_path=connected`。

holdout 的 3 次 `unavailable` 是 `h-002`、`h-016`、`h-044`：第一问已经是 knowledge，第二问失败，没有 `knowledge_mode`。7 次 `splitter_error` 是 `h-039`–`h-043`、`h-045`、`h-046`：`knowledge_mode=multi`，拆分器被调用，没有 `RouterResult`。1 次 `rewrite_error` 是 `h-051`：改写器被调用，没有 `candidate_query`。这 11 条是调用中断。blind 同日拆分 `splitter_error=0`，`ba-039`–`ba-046` 通过，其中 `ba-044` 的原因是 `split_semantic_pass`，`split_exact=false`。

完整返回 `RouterResult` 之后，唯一的业务内容错误是 `ba-055`：改写仍含「绝缘监测报警阈值」。`h-052`–`h-055` 改写逐字通过。`ba-051` 被 Jev 判成 clarify，按歧义不计错。

```text
离线单测                         24 passed
拆分 24 条（Qwen，few-shot）    parse=24/24  count=24/24  anchors=24/24  exact=19/24
改写 20 条（Qwen，短 Prompt）   exact=20/20  replaced=20/20
已知内容残留                    ba-055
已知歧义                        ba-051
prompt tuning                   stopped
```
