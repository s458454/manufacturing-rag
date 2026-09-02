# MinerU A0 接入评估

## 1. 状态

本文是 MinerU 接入 A0 的 `[PROVISIONAL]` 设计记录，不修改当前 `[FROZEN]` 的 A0 输出合同，
也不表示现有 Docling/RapidOCR 运行链已经被替换。

项目已保留 MinerU 官方稳定标签的源码快照：

```text
source: https://github.com/opendatalab/MinerU
tag: mineru-3.4.5-released
git ref: fbb1257a555a3fde78ae5aaaa931e3b3f8fb2883
path: mineru/
```

完整来源身份见 `mineru/UPSTREAM_REVISION`。上游该标签中的
`mineru/version.py` 仍报告 `3.4.4`；验收时必须同时校验标签 Git ref，不能只检查运行时版本字符串。

当前只部署源码，没有下载 MinerU 模型，也没有把 MinerU 重依赖安装到现有 A0 Python 环境。

## 2. 已核实的公开能力

MinerU 3.4.5 的公开 CLI 与 HTTP API 均支持：

```text
parse method: auto | txt | ocr
backend: pipeline | vlm-engine | hybrid-engine | vlm-http-client | hybrid-http-client
hybrid effort: medium | high
```

`--method ocr` 不是文档说明层面的别名：真实代码会在 pipeline 和 hybrid 后端中将
`ocr_enable` 强制设为 `True`。因此统一 OCR 路径可以直接提交原 PDF，不需要由项目先把每页
另存为图片后再调用 MinerU。

MinerU 会生成 Markdown、`middle.json`、`content_list.json`、
`content_list_v2.json`、模型输出、图片和可视化文件。上游明确标注
`content_list_v2.json` 仍在开发、格式可能调整，因此第一版适配器不应以 V2 格式作为稳定合同。

## 3. 推荐的系统边界

### 3.1 结论

A0 不应直接 import MinerU 内部的 `backend.*` Python 函数。推荐把 MinerU 作为独立本地服务，
A0 只依赖其公开 HTTP 协议：

```text
Raw PDF
  -> A0 MinerUAdapter
  -> mineru-api /tasks
  -> MinerU method=ocr
  -> result ZIP (Markdown + content_list + middle_json + images)
  -> 项目自有 semantic projection / quality gate
  -> transactional publish
  -> document.md + document.json + quality_report.json
  -> A1（保持不变）
```

选择进程外服务边界的原因：

1. MinerU 的公开 API 协议比内部 Python 模块稳定；内部结构在不同后端之间存在差异。
2. MinerU 要求 `transformers>=4.57.3`，并带有 pipeline、VLM、vLLM/LMDeploy 等重依赖；
   独立环境可以避免与现有 Docling/RapidOCR 验收环境互相污染。
3. `mineru-api` 已提供 `/health`、异步 `/tasks`、状态查询和结果下载，适合长文档、GPU 队列、
   超时与重试。
4. 后续在单机、远程 GPU 服务或多 GPU router 之间切换时，A0 投影合同不需要变化。

开发烟测可以使用同步 `/file_parse` 或官方 CLI；批量正式运行应使用异步 `/tasks`，不要让
一个长 PDF 长时间占用同步 HTTP 请求。

### 3.2 第一轮候选配置

面向“数字 PDF 与扫描 PDF 全部走 OCR”的第一轮配置为：

```text
backend = hybrid-engine
parse_method = ocr
effort = medium
formula_enable = true
table_enable = true
image_analysis = false
lang = ch
response_format_zip = true
return_md = true
return_middle_json = true
return_content_list = true
return_images = true
```

说明：

- `hybrid-engine` 符合优先验证 MinerU 黑盒完整能力的目标，但尚未冻结为最终后端。
- `pipeline + method=ocr` 必须作为无生成式幻觉、可在 CPU/GPU 运行的对照组。
- Hybrid `medium` 本身会关闭图片/图表语义分析；显式 `image_analysis=false` 与当前 A0
  “图片主体不进入正式正文”的边界一致。
- MinerU 3.4 将中文、英文、日文、繁体中文和 Latin 场景统一路由到 `ch` OCR 模型；
  第一轮中英混合语料先固定 `ch`，再由真实语料评测决定是否需要其他语言配置。

## 4. 输出如何进入现有链路

MinerU 自带 Markdown 适合人工对照，但不能直接复制成项目的正式 `document.md`，因为项目还要求
稳定的 PDF 页码标记、准入规则、质量报告和事务发布。

第一版适配器应以 `content_list.json` 为语义投影输入，以 `middle.json` 为审计证据：

| MinerU 字段/产物 | 项目用途 |
|---|---|
| `content_list[].page_idx` | 生成 `<!-- PDF page N -->`，由 0-based 转为 1-based |
| `type=text` + `text_level>0` | 映射为 Markdown heading |
| `type=text` + 无标题级别 | 映射为正文段落 |
| `type=list` | 映射为列表语义 |
| `type=table` | 保存表体、题注、脚注和坐标；是否进入正式正文属于待冻结决策 |
| `type=image/chart` | 保存审计资产与可信题注，不让图片主体文本自动进入正文 |
| `type=equation` | 保存 LaTeX 与页码/坐标，按 A0 投影规则决定正文表达 |
| `bbox` | 写入 `document.json`，不写入 Markdown |
| `middle.json` | 完整审计、问题定位、后续重投影；不得作为第二份正文入库 |
| MinerU Markdown | 人工 diff / 黑盒输出审计，不作为项目正式合同的唯一来源 |

`document.md` 继续作为 A1 的唯一正式正文输入，所以 MinerU 接入可以先只改 A0，A1-A5 不需要
同步迁移到 MinerU 私有 JSON Schema。

## 5. 运行环境

MinerU 应使用独立环境。源码快照的典型安装方式是：

```bash
python -m venv .venv-mineru
.venv-mineru/bin/python -m pip install -U pip
.venv-mineru/bin/python -m pip install -e './mineru[all]'
```

Windows 对应执行 `.venv-mineru/Scripts/python.exe`。生产服务器可以按实际后端只安装
`mineru[pipeline]`、`mineru[vlm]` 和需要的推理框架，避免无关依赖。

安装 Python 包并不等于可复现部署完成。正式替换 A0 前，还必须：

1. 固定所有 MinerU 模型文件的来源、大小和 SHA-256；
2. 禁止正式运行时静默切换模型版本；
3. 记录 GPU/CPU 后端、MinerU tag、模型 identity 和解析参数；
4. 为服务 `/health`、任务超时、失败重试和结果 ZIP 完整性建立质量门。

## 6. 必须通过的迁移验收

同一批文档至少比较以下两组：

```text
MinerU hybrid-engine + method=ocr + effort=medium
MinerU pipeline      + method=ocr
```

验收维度：

- OCR 字符准确率，中英文分开统计；
- 标题识别和标题层级；
- 多栏阅读顺序；
- 表格结构、跨页表格、空单元格和 rowspan/colspan；
- 页码 provenance 完整率；
- 空白页与低质量页的幻觉率；
- 单页/整文档耗时、显存峰值和失败率；
- 同一输入重复运行的一致性；
- 投影后的 `document.md` 是否继续通过 A1-A5 现有合同测试。

## 7. 尚未冻结的决策

以下事项必须由真实语料结果决定，不能因为源码已进入仓库就视为确定：

1. 最终后端采用 hybrid 还是 pipeline；
2. OCR 表格正文是否改变当前“不进入正式 Markdown”的冻结规则；
3. 是否保留现有独立整页方向标准化，或完全交给 MinerU；
4. MinerU 输出的哪些置信度/结构异常触发页面拒收；
5. 是否长期保留 Docling/RapidOCR 作为降级后端；
6. MinerU 模型资产的固定版本、许可证与本地 manifest。
