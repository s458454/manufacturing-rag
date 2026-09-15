# B1 / B2 实现交接（2026-09-11）

> 本文是实现验收记录，不是需求主文档。  
> B1 权威需求：`docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`  
> B2 权威需求：`docs/Preprocessing/2026_09_02_B2_Query_Processing_Frozen_Requirements_v1.md`  
> `docs/retrieval.md` 只是高层设计摘要；与冻结合同冲突时以 Preprocessing 合同为准。

## 当前验收状态

```text
B2 DESIGN
FROZEN / PASS
- Identity Processor（NO TRANSFORMATION）
- 只接收 B1 direct 分派出的单个 knowledge RoutedTask
- task_id = {request_id}:{task_index}，B2 不重编号
- 不读 history，不 strip，不改任何一个字符

B2 IMPLEMENTATION
PASS
- code/retrieval/b2/
- 未进入 B3，未 import knowledge_base / Milvus / C8 / C9

B2 UNIT TEST
PASS
- Windows: 42 collected retrieval tests, 42 passed（含 5 条 B2）
- 无 skip

B2 CLOSED
CLOSED
- 上游验收：Identity Processor 通过关闭

B1 DETERMINISTIC CORE
ACCEPTED
- Memory 写入方案 A：route() 不 commit；上层调用 commit_turn
- Context Budget 五个常量全部实现
- 每次 inference 前动态校验 actual_input_tokens + 2048 <= 32768
- observability 字段按合同齐全，不落 Query/History/Response 正文
- SystemHandler 只留 Protocol，不编能力说明
- Dispatcher 串行保序：当前实现选择，不是新合同不变量

B1 MODEL INTEGRATION PATH
ACCEPTED WITH V1 ADAPTER APPLIED（2026-09-14；2026-09-15 删除 guided oneOf）
- extra_body: guided_json（xgrammar 兼容投影，无 oneOf）+ guided_decoding_backend=xgrammar
- 与 vLLM 0.8.5 V1 `--guided-decoding-backend xgrammar` 逐字一致
- `auto` 与请求级 `xgrammar:no-fallback` 均拒绝；禁止换约束库

B1 IMPLEMENTATION
COMPLETE
- 合同 b1-contract-v1 / Schema b1-schema-v1
- prompt **b1-router-v3** = 7B reference baseline；**不是** production-accepted Router
- 7B v3 regression 51/60；7B v3 blind 45/60（该 blind 已 exposed）
- 14B+v3 regression 42/60：本轮否定切换；非永久否决 14B
- 停止 v4 及针对当前测试集的 prompt patch
- 详见 docs/proceeding/2026-09-15-b1-experiment-closeout.md

B1 CLOSED
NOT CLOSED
- 上游 ver-0.7 接受实验结论，仍不关闭 B1
- 只改 prompt → 升 router_prompt_version；换模型/route/mode/接口 → 升 contract version
- B1→B5 production E2E 不得宣称 PASS/CLOSED
```

---

## 1. 完成结论

| 项 | 结果 |
|---|---|
| 范围 | B2 Identity Processor；B1 确定性内核 + vLLM 客户端。停在 `ProcessedQuery` |
| 代码位置 | 新建 `code/retrieval/`；**未修改** A0–A6、`code/C8`、`code/C9` |
| 本机 pytest | **42 passed / 0 skipped / 0 failed**（`python -m pytest retrieval/tests -q`，cwd=`code/`） |
| 真实 vLLM smoke | xuanwu：`empty_query=ok` / `direct_mode=task_limit` / `b1_protocol_smoke=ok`（2026-09-14）。不是准确率验收。 |
| B2 CLOSED | **CLOSED** |
| B1 CLOSED | **不申请** |

---

## 2. 修改 / 新增文件

新增：

```text
code/retrieval/__init__.py
code/retrieval/b2/__init__.py
code/retrieval/b2/errors.py
code/retrieval/b2/types.py
code/retrieval/b2/processor.py
code/retrieval/b1/__init__.py
code/retrieval/b1/constants.py
code/retrieval/b1/errors.py
code/retrieval/b1/types.py
code/retrieval/b1/prompt.py
code/retrieval/b1/schema.py
code/retrieval/b1/validation.py
code/retrieval/b1/confirmation.py
code/retrieval/b1/business_validation.py
code/retrieval/b1/tokenizer.py
code/retrieval/b1/memory.py
code/retrieval/b1/observability.py
code/retrieval/b1/config.py
code/retrieval/b1/client.py
code/retrieval/b1/dispatcher.py
code/retrieval/b1/router.py
code/retrieval/tests/__init__.py
code/retrieval/tests/helpers.py
code/retrieval/tests/test_b2_identity.py
code/retrieval/tests/test_b1_validation.py
code/retrieval/tests/test_b1_confirmation.py
code/retrieval/tests/test_b1_business_validation.py
code/retrieval/tests/test_b1_memory.py
code/retrieval/tests/test_b1_dispatcher.py
code/retrieval/tests/test_b1_router.py
code/retrieval/tests/test_b1_config_observability.py
code/retrieval/tests/test_b1_client.py
code/retrieval/requirements-server.txt
code/retrieval/verify_b1_router_server.py
models/retrieval-models.manifest.json
docs/proceeding/2026-09-11-b1-b2-implementation.md
```

改过（状态/索引）：

```text
agent.md
docs/README.md
```

明确未改：

```text
code/knowledge_base/**
code/preprocessing/**
code/C8/**
code/C9/**
docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md
docs/Preprocessing/2026_09_02_B2_Query_Processing_Frozen_Requirements_v1.md
docs/retrieval.md          # 高层设计已同步：RRF k=60、UNIFIED_ICU、B1/B2/B3/B5 合同路径
```

---

## 3. 生产边界

公开合同类型只有：

```text
QueryRequest
ConversationTurn
RouterResult
RoutedTask
QueryProcessingRequest
ProcessedQuery
```

没有把聚合 Turn 对象冻成新的跨模块合同。

本轮输出：

```text
B1 RouterResult
  ↓
Dispatcher 只对 knowledge RoutedTask 调 B2
  ↓
ProcessedQuery
```

禁止从本包发起 Dense/BM25/`hybrid_search`。

`request_id` / `conversation_id` 由上层创建。B1/B2 不生成。

---

## 4. B2 行为

```python
retrieval_query = request.query
ProcessedQuery(
    request_id=request.request_id,
    task_id=request.task_id,
    original_query=request.query,
    retrieval_queries=(request.query,),
)
```

`task_id` 例：`req-123` 的 tasks = knowledge, system, knowledge → 进入 B2 的是 `req-123:0` 与 `req-123:2`。

合同失败 → `QueryProcessingContractError`（编排错误，无用户文案）。

Observability：`request_id` / `task_id` / `b2-contract-v1` / `identity-v1` / `retrieval_query_count=1` / `processing_latency_ms`。不落 Query 正文。

---

## 5. B1 行为要点

### 5.1 Context Budget

```text
ROUTER_CONTEXT_LIMIT = 32768
MAX_ROUTER_MEMORY_TOKENS = 16384
ROUTER_NON_HISTORY_RESERVE = 8192
ROUTER_OUTPUT_AND_SAFETY_RESERVE = 8192
ROUTER_MAX_NEW_TOKENS = 2048
```

启动：Prompt + Schema + 最大合法 Query + chat template ≤ 8192，否则服务不得 ready。  
每次真实推理前：`actual_input_tokens + 2048 <= 32768`。超限只 FIFO 完整 Turn；Active Memory 已空仍超限 → `RouterConfigurationError`。禁止截断 Current Query / Prompt / Schema。

### 5.2 Memory 方案 A

`QueryRouter.route()` **不写入** ConversationTurn。  
`commit_turn(store, conversation_id, turn, tokenizer)` 供上层在整轮业务完成并拿到最终 `assistant_response` 后调用。

不得对以下路径调用 `commit_turn`：

```text
clarify
task_limit
Router 错误
等待 rewrite 确认
```

超限单 Turn 进 Conversation Log，不进 Active Memory。FIFO 只淘汰 Active Memory。

### 5.3 Confirmation

确认词 / 拒绝词按合同 exact match（允许 strip、ASCII lower、去掉末尾单个 `。！!?？`）。  
确认后用 `pending_rewrite` 重新走完整 B1，不假设 `route=knowledge`。  
拒绝不打模型。

### 5.4 Dispatcher

v1 **串行**、按 `tasks` 原始下标生成 `task_id`。这是实现选择。  
`system` 只通过 `SystemHandler` 接口露出，本轮无 FAQ 正文。  
`unsupported` 使用合同固定句。

### 5.5 Observability

每次至少：

```text
request_id
conversation_id
router_contract_version
router_prompt_version
router_schema_version
model_alias
history_turn_count
history_tokens
current_query_weighted_length
router_mode
task_count
router_latency_ms
error_type
```

标准 `error_type`：`invalid_query` / `empty_query` / `query_too_long` / `router_unavailable` / `router_protocol_error` / `router_configuration_error`。

---

## 6. 真实模型接入前置

仓库已固定：

| 项 | 值 |
|---|---|
| Serving | `vllm==0.8.5`（`code/retrieval/requirements-server.txt`） |
| Backend | 引擎级 `xgrammar`（禁止 `auto` 与请求级 `xgrammar:no-fallback`） |
| 模型逻辑 ID | `Qwen/Qwen2.5-7B-Instruct` |
| Artifact revision | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Manifest | `models/retrieval-models.manifest.json` |

`B1_REQUEST_TIMEOUT_MS` 必须显式配置，代码不提供“最优默认值”。

服务器 protocol smoke：

```text
export B1_VLLM_BASE_URL=...
export B1_MODEL_ALIAS=Qwen/Qwen2.5-7B-Instruct
export B1_TOKENIZER_NAME_OR_PATH=...
export B1_REQUEST_TIMEOUT_MS=...   # 部署值，不是算法最优

python retrieval/verify_b1_router_server.py
```

该脚本只验证协议（空 Query 不打模型、direct 后 B2 恒等）。**不是路由准确率验收。**

若集群 vLLM 版本不是 0.8.5，必须先改仓库 pin 再跑真实 smoke，禁止运行时改 `backend=auto`、请求级 `xgrammar:no-fallback` 或正则抠 JSON。启动参数必须是 `--guided-decoding-backend xgrammar`。

---

## 7. 测试矩阵（Windows 本机）

```text
python -m pytest retrieval/tests -q
# 42 passed
```

| 文件 | 覆盖 |
|---|---|
| `test_b2_identity.py` | 恒等、不规范化、task_id、合同错误、log 无 Query |
| `test_b1_validation.py` | strip-only、空/非法/过长、CJK 权重 4 |
| `test_b1_confirmation.py` | 确认/拒绝/新 Query |
| `test_b1_business_validation.py` | 单任务相等、duplicate、过长生成 Query 算协议错误 |
| `test_b1_memory.py` | commit_turn、超限单 Turn、FIFO |
| `test_b1_dispatcher.py` | 原始下标、system 接口、unsupported 文案 |
| `test_b1_router.py` | 不 commit、一次推理、确认重入、动态 FIFO、启动超限 |
| `test_b1_config_observability.py` | 观测字段、timeout 必填、禁止 auto / 请求级 no-fallback |
| `test_b1_client.py` | 引擎级 xgrammar + guided 子集、传输失败 → unavailable |

---

## 8. 明确未做

```text
B1 最终测试集 / 准确率 / P50/P95
SystemHandler 能力说明正文
Safety Gate
跨进程 Conversation 持久化（本轮 InMemoryConversationStore）
```

---

## 9. 上游验收结论与补丁（2026-09-11）

```text
B2：CLOSED
B1 deterministic core：ACCEPTED
B1 model integration path：REQUIRED FIX APPLIED（xgrammar:no-fallback）
B1 overall：NOT CLOSED
```

本轮补丁：

1. 生产客户端 `guided_decoding_backend` 改为 `xgrammar:no-fallback`，拒绝明文 `xgrammar` 与 `auto`。
2. 修正本文此前误写 `docs/retrieval.md` 仍为 `k=55`。工作区核实时该摘要确实仍残留 `k=55`，已同步为 `k=60`、UNIFIED_ICU 与正确合同路径。
3. B2 正式关闭。

真实 vLLM protocol smoke 仍须在服务器环境执行。smoke 通过后记 `B1 IMPLEMENTATION COMPLETE`，仍不 CLOSED。

---

## 10. vLLM 0.8.5 V1 适配（2026-09-14）

xuanwu 上真实 serving 证明：vLLM **0.8.5 V1** 的 CLI 只接受 `--guided-decoding-backend {auto,guidance,xgrammar}`。请求里再带 `xgrammar:no-fallback` 会 400（请求值必须与启动值逐字相同）。引擎级 `xgrammar`（不是 `auto`）已经 fail-closed，不会换成 outlines / guidance。

同时，0.8.5 的 xgrammar 校验会拒绝冻结合同 Schema 里的 `minItems` / `maxItems` / `minLength`。

本轮实现调整（不改冻结合同输出形状，不升级 vLLM，不退回 V0）：

1. 生产客户端 `guided_decoding_backend` 改为 `xgrammar`，拒绝 `auto` 与 `xgrammar:no-fallback`。
2. `guided_json` 发送 `ROUTER_GUIDED_JSON_SCHEMA`（去掉上述三个关键字的兼容投影）。2026-09-15 已按上游要求删除 `oneOf`：业务组合仍只由冻结合同 Schema + business validation 执行，非法组合 → `RouterProtocolError`，不自动修复。
3. Manifest / `requirements-server.txt` / smoke 脚本 pin 同步为引擎级 `xgrammar`。
4. Serving 保持：`--guided-decoding-backend xgrammar`。

此前第 9 节的 `xgrammar:no-fallback` 是针对旧请求级 API 的 fail-closed 编码；在 0.8.5 V1 上由引擎级 `xgrammar` + 请求同名 backend 完成同一目标。

xuanwu 协议 smoke 曾得到 `mode=task_limit` 且 `tasks` 非空。二次校验正确拒绝。中间一度用 guided `oneOf` 锁字段组合；7B 被压成几乎总是 `task_limit` + 空 tasks。2026-09-15 已删除 `oneOf`。

重跑结果（2026-09-14，当时仍带 oneOf）：

```text
empty_query=ok
direct_mode=task_limit
b1_protocol_smoke=ok
```

该句被标成 `task_limit` 属于路由准确率失败，不是协议失败。B1 IMPLEMENTATION COMPLETE，SEMANTIC ACCURACY FAIL，仍不 CLOSED。
