# B1-Jev v0 实验合同（2026-09-23）

> 状态：`HISTORICAL`  
> 2026-09-29 B1 已关闭。权威合同是 `docs/Preprocessing/2026-09-29-B1_Jev_Router_Frozen_Requirements_v1.md`。下文是关闭前的实验记录。

## 现状（2026-09-27）

给上游的核实结果。实验合同的禁止项不变；本节只记录这一天核对过的事实。

```text
判断核仍未接入 QueryRouter。现网仍是 b1-contract-v1 + b1-router-v3 + Qwen2.5-7B-Instruct。
本文件仍不是 b1-contract-v2，不申请 CLOSED。
离线单测：同日用 .venv（Python 3.13.15）跑 test_b1_jev_experimental.py，32 passed。
暴露 128 条实跑：兔子 jev-1.13，记录在
docs/proceeding/artifacts/b1-jev-v0-diag-2026-09-27.txt。
holdout 与 blind 相同：control 4/4，route 49/49，knowledge_mode 39/39，
skip 11（mixed 6 + clarify 5），unavailable 0，protocol 0。
b1_closed=no。这 128 条仍是已暴露诊断，不得当作 unseen CLOSED set。
39/39 只比 knowledge_mode 标签，不比切分后的 task 文本。
不得与 v3 mode exact 51/60、45/60 并成同一次提高。
分数不产生 RouterResult。multi 没有切出的问句，confirm_rewrite 没有 candidate_query。
现有 Dispatcher 因此接不上 B2。B2 合同不改。
本机无独显（AMD Ryzen Z1 Extreme + 核显，约 16 GB 内存）。
已装 torch 2.11.0+cu128，cuda_available=False，device_count=0。
本地 GLiClass 权重还在 D:\Download\gliclass-multilang-mini。
冒烟脚本只跑 route，本机未跑。主诊断仍是远程 Jev，问句和近期对话会出域。
解释器：.venv 已改接到本机 Python 3.13.15。原 Miniconda 3.13.12 不在这台机器上。
跑实验用 ..\.venv\Scripts\python.exe，不要用裸 python。
```

2026-09-28 起，候选 Router（能输出 `RouterResult`、含拆分与改写）的现状不再写在本节。2026-09-29 该候选已冻结，停止调提示。上游续订整套替换时读：

`docs/proceeding/2026-09-28-b1-jev-candidate-status.md`

上面 2026-09-27 的数字只描述当时的标签实验：没有 `RouterResult`，没有 task 文本，没有 `candidate_query`。

## 0. 绑定

```text
不修改 QueryRouter / VLLMRouterClient / b1-router-v3
不从 retrieval.__init__ 或 retrieval.b1 导出本实验包
不宣称 B1 CLOSED
不宣称 Jev 为 production-accepted Router
不得用当前 regression 64 + blind 64 作为 unseen CLOSED set
换冻结模型 / 改 mode·route·接口 → 将来若转正必须升 B1 contract version
只改 Qwen prompt → 仍只升 router_prompt_version（本实验不是这条路）
```

代码落点：`code/retrieval/b1_jev_experimental/`  
单测：`code/retrieval/tests/test_b1_jev_experimental.py`  
现网路径：`code/retrieval/b1/router.py`（本轮未改）

---

## 1. 实验只做什么

第一问是 **三选一路由**：

```text
knowledge     走文档 RAG。不是兜底
system        问助手自身：能力、用法、已接入的数据或文档类型
unsupported   直接拒绝：要执行文档 RAG 之外的能力。不是兜底
```

仅当第一问得到 `knowledge` 时，再发第二问 **处理形态**（不是现网 `mode` 的 1:1）：

```text
direct            一条已完整的知识问题
multi             2～8 条独立知识问题（现网仍是 mode=direct）
task_limit        超过 8 条
confirm_rewrite   缺对象，历史能唯一补全
clarify           补不全或多种解释都合理
```

两问都是 Jev choice。不做 noul、不用 0.5 阈值、不生成 `candidate_query`、不切分 task 文本。

现网 `QueryRouter` 仍一次生成 `mode + route + tasks + candidate_query`。本包不替代它。

---

## 2. 总流程（实验）

```text
当前问句 + Active Memory
        ↓
build_jev_state()
        ↓
Jev 第一问：primary_route ∈ {knowledge, system, unsupported}
        ↓
parse_jev_answers()
        ↓
仅当得到的 primary_route=knowledge
        ↓
Jev 第二问：knowledge_mode ∈ {direct, multi, task_limit, confirm_rewrite, clarify}
        ↓
parse_jev_mode_answer()   ← 无阈值、不拼生产 mode
```

`decide_with_jev()` 和两条冒烟脚本都不做 B1.0。诊断集里的 control 题单独走现网 `validate_query_text`，不调 Jev。

禁止：

```text
Jev 输出自由 JSON / 自由句子
用 noul / task_count / 0.5 再拼 mode
低 confidence → knowledge / unsupported / clarify
Jev 失败后回退现网 Qwen（实验包不做 fallback）
把本包接进 QueryRouter.route()
```

---

## 3. Jev state

```text
state = {
  "current_query": <当前问句>,
  "history": [
    {
      "raw_user_query": ...,
      "canonical_user_query": ...,
      "assistant_response": ...
    },
    ...
  ]
}
```

只含 Active Memory，不含 A0/A6/库内是否有答案。  
问句和近期对话会发到上游 API。研究实验可接受；企业内网 / 数据不出域则本方案不得转正。

---

## 4. Jev questions（两次请求，各一题）

权威定义在 `code/retrieval/b1_jev_experimental/questions.py`。  
`JEV_QUESTION_KEYS = ("primary_route",)`  
`JEV_MODE_QUESTION_KEYS = ("knowledge_mode",)`

一次请求只带一题。第二问只在第一问为 `knowledge` 时发出。

**说明书**：请求里的 instructions 先嵌入现网 `ROUTER_SYSTEM_PROMPT`（`b1-router-v3`），再追加本题 guard。整段说明不等于说明书原文。  
**输出格式**仍是 System One 的 choice，不是 Qwen JSON。

### 4.0 请求字段（符合 `/v1/systemone`）

```text
POST /v1/systemone
body 只能有三个键：model、state、questions
state = 对象（当前问句 + 历史），不是 chat messages
questions = 只含 primary_route，或只含 knowledge_mode
choice.criteria = 对象（从键里选一个）
请求体 ≤ 32 KiB；不要 stream
```

| 字段 | 填什么 |
|---|---|
| `model` | 兔子：`jev-1.13`；Vercel：`typesafe-ai/jev`；TypeSafe：`jev-latest` |
| `state.current_query` | 当前 Query |
| `state.history[]` | `raw_user_query` / `canonical_user_query` / `assistant_response` |
| `questions.primary_route` | `type=choice`，键 `knowledge` / `system` / `unsupported` |
| `questions.knowledge_mode` | `type=choice`，键 `direct` / `multi` / `task_limit` / `confirm_rewrite` / `clarify` |

```text
knowledge     要从制造/工程技术文档检索或回答
system        问助手能力、用法、已接入数据/文档类型
unsupported   要执行文档 RAG 之外的能力
```

`knowledge` / `unsupported` 都不是兜底。问「标准对 ERP 如何规定」是 knowledge；「你能不能查库存」是 system；「导出 ERP 库存」是 unsupported。

---

## 5. 解析，没有 policy

`parse_jev_answers()` 只检查 `primary_route`。  
`parse_jev_mode_answer()` 只检查 `knowledge_mode`。  
confidence 缺省 1.0，只作记录，不改标签。

已删除：用 noul / `task_count` / 默认 0.5 拼生产 `mode`。

---

## 6. 阶段与明确不做

| 阶段 | 做 | 不做 |
|---|---|---|
| 当前 | 三选一路由 + knowledge 五选一形态 | 接主链路、切多任务文本、写 candidate_query |
| 转正 | 新 unseen blind + `b1-contract-v2` | 用现 128 条宣称 CLOSED |

---

## 7. 模块改动表

| 模块 | 本轮 | 以后若转正 |
|---|---|---|
| `docs/Preprocessing/2026-09-01-B1_*_v1.md` | 不改 | 另开 v2 |
| `code/retrieval/b1/router.py` 等现网 B1 | **不改** | 判断核可换，确定性外壳保留 |
| `code/retrieval/b1_jev_experimental/` | **新增（route + knowledge_mode）** | 可迁入 `b1/` 并升合同 |
| `retrieval/__init__.py`、`retrieval.b1` | **不导出** | 导出前须合同变更 |
| `b1_holdout_v1.jsonl` / `b1_blind_acceptance_v1.jsonl` | 不改题/gold；只作 exposed 诊断 | 新 unseen 集 |
| B2–B5 | 不改 | 不等待 B1 CLOSED |

---

## 8. 实验门禁

现网 CLOSED 门禁仍对 `b1-router-v3` 有效，**不对 Jev v0 生效**。

暴露 128 条（holdout 64 + blind 64）先评 **route**，再对 Jev 判成 knowledge 的题评 **knowledge_mode**：

```text
route 可评：同前（唯一 route / confirm_rewrite / task_limit / 同 route 多任务）

knowledge_mode 可评（须 gold_route=knowledge）：
  knowledge_single / long_tail / zh_en → direct
  multi_2_to_8 → multi
  task_limit_gt8 → task_limit
  confirm_rewrite → confirm_rewrite

skip route：mixed、clarify
skip mode：system / unsupported / mixed / clarify / control
  以及第一问不是 knowledge 的题（不发第二问）

control：复用现网 B1.0 validate_query_text，不调 Jev
unavailable / protocol 单独记账
不宣称 CLOSED
不得与 v3 的 mode exact 51/60、45/60 直接比分母
```

---

## 9. 本地 GLiClass（适配器在，本机未跑通）

主诊断走远程 Jev。本地适配器可以分别做 route 与 knowledge_mode，但冒烟脚本只调用 route。

```text
默认权重：D:\Download\gliclass-multilang-mini（2026-09-27 文件还在）
覆盖：B1_GLICLASS_MODEL_PATH
适配：code/retrieval/b1_jev_experimental/gliclass_adapter.py
route：decide_with_gliclass
knowledge_mode：decide_knowledge_mode_with_gliclass
冒烟：retrieval/eval/run_b1_gliclass_smoke.py 只跑 route，打印 task=route_only
```

两个 decide 函数都使用 `local_files_only=True`。有 CUDA 时用 `cuda:0`，否则 CPU。本地路径的问句不出域。

2026-09-27 本机没有独显。`torch 2.11.0+cu128` 的 `cuda_available=False`。因此即使用本地路径，也只会落 CPU。这次没有跑 GLiClass 冒烟，不能写成已验证。

---

## 10. 怎么跑（不碰现网）

逻辑（无模型）。2026-09-27：32 passed。

```powershell
cd "D:\code backup\MyMethod\all-in-rag-main\code"
..\.venv\Scripts\python.exe -m pytest retrieval/tests/test_b1_jev_experimental.py -q
```

GLiClass 冒烟只覆盖 route，本机尚未跑：

```powershell
cd "D:\code backup\MyMethod\all-in-rag-main\code"
..\.venv\Scripts\python.exe retrieval/eval/run_b1_gliclass_smoke.py
```

真实 Jev 冒烟（优先 兔子 `jev-1.13` `/v1/systemone`。不要把 key 写进仓库）：

```powershell
$env:TUZI_API_KEY = "<your-tuzi-key>"
cd "D:\code backup\MyMethod\all-in-rag-main\code"
..\.venv\Scripts\python.exe retrieval\eval\run_b1_jev_smoke.py
```

默认（有 `TUZI_API_KEY` 时）：

```text
POST https://api.tu-zi.com/v1/systemone
Authorization: Bearer $TUZI_API_KEY
model = jev-1.13
body = native JEV JSON（第一问 primary_route；knowledge 再问 knowledge_mode）
```

同时设置多个 key 时：先 `TUZI_API_KEY`（`jev-1.13`），再 `AI_GATEWAY_API_KEY`（`typesafe-ai/jev`，`https://ai-gateway.vercel.sh/typesafe/v1/systemone`），再 `TYPESAFE_API_KEY`（`jev-latest`）。现网 B1 仍不调用。

---

## 11. 兔子 API 上的 Jev

[`jev-1.13`](https://api.tu-zi.com/docs/models/jev-1-13) **就是** Jev，**只**接受原生 `POST /v1/systemone`，不能走 Chat Completions。见 [JevAI 指南](https://api.tu-zi.com/docs/providers/jevai)。

```text
公开名：jev-1.13
请求：model + state + questions（choice 的 criteria 为对象）
响应：顶层 answers，或 Community 的 data.answers
限制：同步非流式；拒绝 stream；请求体 ≤ 32 KiB
```

`run_b1_jev_smoke.py` 在设置 `TUZI_API_KEY` 后走这条路。`run_b1_tuzi_smoke.py` 仍是聊天 JSON，**不是** Jev。

不要把 key 写进仓库或聊天。禁止从 `QueryRouter` 调用。不宣称 CLOSED。

---

## 12. 暴露 128 条诊断（route + knowledge_mode）

`b1_holdout_v1.jsonl` + `b1_blind_acceptance_v1.jsonl` 只作 exposed 诊断，**不是** unseen CLOSED 集。gold 文件不改。

```powershell
$env:TUZI_API_KEY = "<your-tuzi-key>"
cd "D:\code backup\MyMethod\all-in-rag-main\code"
..\.venv\Scripts\python.exe retrieval\eval\run_b1_jev_diag.py
```

输出比 `gold_route`/`got_route` 和 `gold_mode`/`got_mode`。  
`system` / `unsupported` 不发第二问。mixed / clarify 的 route 记 `SKIP`。control 不调 Jev。

2026-09-27 已跑完，原文在 `docs/proceeding/artifacts/b1-jev-v0-diag-2026-09-27.txt`。兔子 `jev-1.13`。holdout 与 blind 相同：

```text
control=4/4
route=49/49
mode=39/39
skip=11
unavailable=0
protocol=0
b1_closed=no
main_path=not_connected
```

skip 的 11 条是 mixed 6 条和 clarify 5 条，不进 route 分母。`mode=39/39` 只表示 knowledge 形态标签一致。这次没有生成 `tasks` 或 `candidate_query`，所以不能把该分数当成 B1 已能把问句交给 B2。
