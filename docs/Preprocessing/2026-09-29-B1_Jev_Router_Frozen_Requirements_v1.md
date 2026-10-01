# B1 Jev Router — Frozen Requirements v1

> Status: **FROZEN / CLOSED**
>
> Contract version: `b1-contract-v2`  
> Router version: `b1-jev-v1`  
> Result version: `b1-result-v1`
>
> 本文档是 B1 的权威需求，并关闭 B1。  
> `docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`（`b1-contract-v1`）不再约束现网。
>
> 关闭的是当前实现：`JevQueryRouter` 产出 `RouterResult`，`route_for_b2()` 把 knowledge 任务交给现有 B2。不关闭 B3–B6，也不把 B1→B5 整链标成 CLOSED。

---

# 1. 链路

```text
B1.0 校验
→ Jev primary_route ∈ {knowledge, system, unsupported}
   system / unsupported → 一个 direct 任务，不再问第二问
   knowledge → Jev knowledge_mode
      direct / task_limit / clarify → Python 组装，不调 Qwen
      multi → Qwen 拆分，2～8 条 knowledge，mode=direct
      confirm_rewrite → Qwen 生成 candidate_query，用户确认后整段重入
→ RouterResult
→ dispatch_knowledge_tasks()
→ B2 process_query()
```

旧 `QueryRouter` 不在这条链路上，也不做 fallback。它的源码在 gitignore 的 `local_backup/b1-router-v3/`。

入口：

```text
build_jev_router()
route_for_b2()
```

# 2. 服务

```text
判断：兔子 jev-1.13，POST /v1/systemone，TUZI_API_KEY
      两次独立 choice。一次调用，无重试。confidence 只解析，不改标签。
拆分和改写：本机 Qwen/Qwen2.5-7B-Instruct，vLLM，xgrammar
      只在 multi 与 confirm_rewrite 时调用。发给 xgrammar 的 schema 不含
      minItems / maxItems / minLength。2～8 条、非空、不重复由 Python 检查。
      失败不改写成 direct、不截断、不去重。
```

Jev 超时、网络失败、HTTP 5xx → `RouterUnavailableError`。  
拆分失败 → `splitter_error`，固定回复，不自动再拆。  
改写失败 → `rewrite_error`，固定回复。  
非法 choice 或非法 JSON → `RouterProtocolError`。

# 3. 仍有效的行为

```text
业务 route 只有 knowledge / system / unsupported
当前 Session 的 Active Memory，最大 16384 token，超限只 FIFO 丢掉最老完整 Turn
Current Query 上限沿用 B1.0：约 500 个中文字符 / 2000 个英文字符
direct 单任务不改写 Current Query
依赖历史的改写必须先给用户确认；拒绝则 RewriteRejected
显式多任务自动拆分，单请求最多 8 个任务
knowledge 与 unsupported 都不是兜底
基础设施失败与语义不清分开
B1 不访问语料，不判断库里有没有答案，不做安全策略门
B2 对 knowledge 任务做恒等处理，不改写检索句
```

`mixed` 不在本版。整句只有一个 `primary_route`。  
拆分验收是 parse、条数、同一位置的标准号或对象。不要求拆分文本与某一条参考句逐字相同。  
中文 few-shot 可以把英文问句拆成中文，标准号必须留在原任务。

# 4. 关闭时接受的残留

```text
ba-055    改写可能留下旧焦点「绝缘监测报警阈值」。关闭时不再加 few-shot。
ba-051    两种补法都成立，标为歧义，不进准确率，不改 Jev 问句。
mixed     无实现。
exposed clarify    诊断继续跳过，没有可靠的 primary_route gold。
```

关闭所用的实测是已经暴露的 holdout、blind、24 条拆分和 20 条改写。不再另做 unseen 集。这些集不能再拿来改提示。

```text
拆分 24 条     parse=24/24  count=24/24  anchors=24/24  exact=19/24
改写 20 条     exact=20/20  replaced=20/20
blind         result_pass=47/48  唯一内容错误 ba-055  ambiguous=1（ba-051）
holdout       同日有 Jev 第二问和 Qwen 调用中断；完整返回的题目没有新的判断错误
```

# 5. 版本

```text
router_contract_version = b1-contract-v2
router_prompt_version   = b1-jev-v1
router_schema_version   = b1-result-v1
B1 CLOSED               = yes
main_path               = connected
```

`b1-jev-v1` 的判断问句、拆分 few-shot、短改写提示已冻结。再改这些文本或更换判断模型，必须升版本，不能在本关闭上直接改。
