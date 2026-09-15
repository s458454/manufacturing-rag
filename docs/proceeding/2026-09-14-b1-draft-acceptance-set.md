# B1 草稿验收题集（2026-09-14）

> 本文和 `code/retrieval/eval/b1_draft_acceptance_set.jsonl` 是 **DEV/DIAGNOSTIC 草稿**，不是冻结合同，也 **不是** CLOSED holdout。  
> 64 条 CLOSED 门禁集见 `docs/proceeding/2026-09-15-b1-holdout-and-router-v2.md` 与 `code/retrieval/eval/b1_holdout_v1.jsonl`。  
> 权威需求仍是 `docs/Preprocessing/2026-09-01-B1_Query_Router_Frozen_Requirements_v1.md`。  
> **跑完本套题也不能把 B1 标成 CLOSED。**

## 1. 用途

给当前 xuanwu 上的 Qwen2.5-7B-Instruct Router 做一次可复现的 mode/route 摸底。协议 smoke（`b1_protocol_smoke=ok`）已经通过；本套题测的是 **它有没有填对 mode 和 route**。

不评检索命中、不评 D1、不设准确率/P50/P95 通过线。

## 2. 语料范围

只出 ver1 已入库的 17 篇（14 英文 + 3 中文，`a6_leaf_v1` = 2034 Leaf）。不出 ver2 扫描件（1970s Materials Data Handbook 等）。

| 文档 | 出题用途 |
|---|---|
| NASA-STD-5006A 焊接 | 普通 knowledge、合同混合题、follow-up |
| NASA-STD-8739.4A 压接 | 普通 knowledge |
| NASA-STD-4003A 电搭接 | 普通 knowledge |
| NASA-STD-5009C NDE | 普通 knowledge |
| NASA-STD-6030 / 6033 增材制造 | 普通 knowledge、比较题、follow-up |
| NASA-STD-6016C 材料工艺 | 普通 knowledge |
| NASA-STD-5017B 机构 | 普通 knowledge |
| NASA-STD-8739.1B 聚合物工艺 | 普通 knowledge |
| NASA-STD-8739.12A 计量校准 | 普通 knowledge |
| NASA-HDBK-8739.19-2 / 19-3 | 长尾 knowledge |
| NISTIR 7218 / 7734 | 长尾 knowledge |
| YH5WR-17/45 避雷器 | 中文 knowledge、中英跨语言 |
| 水贝变电站仿真实训室 | 中文 knowledge |
| 电动汽车兆瓦级充电 | 中文 knowledge |

## 3. 覆盖的合同样本类型

| bucket | 条数 | 对应合同 |
|---|---|---|
| ordinary | 20 | 普通：knowledge / system / unsupported |
| long_tail | 4 | 长尾：手册/实验室报告 |
| zh_en | 2 | 中英混合 |
| multi_intent | 3 | 多意图：拆任务、knowledge+unsupported |
| follow_up | 4 | 多轮：confirm_rewrite / clarify |
| boundary | 4 | 边界：表述不清；ERP 文档知识 vs 实时查询；8 任务 vs 9 任务 |
| control | 1 | 空查询，确定性内核，不打模型 |
| **合计** | **38** | |

Gold 规则按冻结合同：

- 单任务 `direct`：`tasks[0].query` 必须等于当前 Query，禁止改写
- `你能不能查询当前库存？` → **system**（问能力）
- `查询 ERP 中当前库存。` → **unsupported**（要执行）
- `标准中对 ERP 数据保存有什么要求？` → **knowledge**
- `那钢呢？`（上一轮已问 5006A 铝合金焊接）→ **confirm_rewrite**
- `那个呢？` → **clarify**
- 9 个彼此独立任务 → **task_limit**

`b1-001` 就是协议 smoke 那句。当前模型曾给出合法但错误的 `task_limit`，本套题会记 FAIL。

## 4. 怎么跑

vLLM 保持 `--guided-decoding-backend xgrammar`。XFTP 至少同步：

```text
code/retrieval/eval/b1_draft_acceptance_set.jsonl
code/retrieval/eval/run_b1_draft_acceptance.py
```

以及已在用的 `code/retrieval/b1/`。

```bash
cd /public/zhangkairan/MyMethod/all-in-rag-main/code
conda activate mfg-rag-preprocess
export B1_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export B1_MODEL_ALIAS=Qwen/Qwen2.5-7B-Instruct
export B1_TOKENIZER_NAME_OR_PATH=/public/zhangkairan/Qwen2.5-7B-Instruct
export B1_REQUEST_TIMEOUT_MS=60000
python retrieval/eval/run_b1_draft_acceptance.py
```

看每条 `PASS/FAIL` 和最后的 `mode_route_score=a/b`。`b1_closed=no` 是写死的。

38 条里除空查询外都会打模型，按约 5–10 秒/条估计大约数分钟。

## 5. 和关闭 B1 的关系

跑分只是摸底。CLOSED 门禁集已另冻为 64 条 `b1_holdout_v1.jsonl`，见 `docs/proceeding/2026-09-15-b1-holdout-and-router-v2.md`。本草稿不能自行升级为合同或 holdout。
