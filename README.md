# Manufacturing RAG

面向制造业公开技术文档的检索增强生成（RAG）实验项目。

当前版本为 `ver-0`：完成了项目方向定义、首批公开工程文档收集和原始
All-in-RAG 参考代码整理，作为后续制造业 RAG 重构的可追溯起点。它不是已完成的
生产系统。

## ver-0 范围

- 建立制造业公开技术文档 RAG 的 V0.1 设计规范。
- 收录 24 份来自 NASA、NIST 官方来源的材料、加工、连接制造、制造规范与质量检测文档。
- 保留原始 All-in-RAG 的 C1-C9 示例代码，供后续审计和重构时对照。
- 内置 Docling 源码快照，供文档解析方案研究。

当前 C1-C9 中仍存在教程、菜谱和 Graph RAG 示例逻辑；部分脚本依赖已从本仓库移除的
原教程数据，因此不保证能够直接运行。这些文件是重构参考，不代表最终业务实现。

## 目录结构

```text
.
├── code/                         # C1-C9 原始参考代码与依赖配置
├── data/engineering_docs/        # 制造业公开文档及可校验清单
├── docling/                      # Docling 2.115.0 源码快照
├── docs/
│   └── manufacturing-rag-v0.1-spec.md
└── models/                       # 模型目录占位符，不提交模型权重
```

## 数据集

数据清单位于
[`data/engineering_docs/manifest.csv`](data/engineering_docs/manifest.csv)，记录每份文档的
官方来源、相对路径、文件大小和 SHA-256。当前共 24 份 PDF，约 75 MiB。

这些文档来自公开的 NASA 和 NIST 官方站点。仓库中的副本仅用于研究与可复现实验；
各文档的权利和使用条件仍以其发布机构及原始页面为准。

## 开发状态

详细目标、数据方案、Metadata Schema、分块策略、检索链路和评测规划见
[`docs/manufacturing-rag-v0.1-spec.md`](docs/manufacturing-rag-v0.1-spec.md)。

后续工作的重点包括：

1. 审计 C8/C9 参考实现，识别保留、删除和改造模块。
2. 将菜谱领域的数据结构、路由和生成提示改造成制造业文档领域。
3. 增加 PDF 标准化解析、工业 Metadata、Reranking、引用式回答与离线评测。
4. 逐步补齐可复现的环境安装、测试和运行入口。

## 环境提示

- 建议使用 Python 3.10-3.12。
- 通用参考依赖位于 `code/requirements.txt`。
- C8、C9 分别提供独立的 `requirements.txt`。
- C9 的环境变量模板位于 `code/C9/.env.example`；请勿提交真实 API Key 或密码。
- Milvus 的本地容器配置位于 `code/docker-compose.yml`。

由于 `ver-0` 保留了不同阶段的参考实现，目前不建议一次性安装并运行所有模块。
应根据正在重构的模块选择对应依赖。

## 来源与许可证

本项目基于 Datawhale 的
[all-in-rag](https://github.com/datawhalechina/all-in-rag) 修改，原项目采用
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans)。
本仓库的衍生内容继续遵循该许可；第三方组件按其各自许可证使用。

完整说明见 [`LICENSE.md`](LICENSE.md) 和
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

