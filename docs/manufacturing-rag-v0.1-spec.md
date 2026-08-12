# 制造业公开技术文档 RAG：V0.1 项目规范

> 文档用途：V0.1 项目方案与下游交接基线。下一阶段在本规范基础上开展 C8 代码审计、数据源落实与模块映射。

## 项目定位

项目暂定为：

> **面向制造业公开技术文档的工程知识 RAG 系统**

V0.1 控制实现范围，不追求复刻完整的“图纸工艺智能体平台”。原案例包含三维图纸识别、几何与加工特征理解、通用大模型、工艺智能体，以及与 CAD、PDM/PLM、CAPP、ERP、SRM、MES 等系统的联动。当前项目只复刻其中与 RAG 直接相关的知识侧能力：**工艺规范、工艺文档和历史加工经验的知识检索与工艺匹配**。

V0.1 主链路：

```text
公开工程技术文档
PDF / Markdown / Word
        ↓
文档解析
        ↓
统一 Markdown / Document
        ↓
结构化 Chunk
        ↓
Parent-Child
        ↓
Embedding
        ↓
Dense + BM25
        ↓
RRF
        ↓
Cross-Encoder Reranker
        ↓
Top-K Evidence
        ↓
LLM Answer
        ↓
Retrieval + Answer Evaluation
```

多模态和 CAD 在 V0.1 只保留输入接口，不实现具体解析能力：

```text
CAD / STEP / STL / 图纸
        ↓
[外部解析模型 / VLM / CAD Parser 黑盒]
        ↓
标准化文本 + 结构化 Metadata
        ↓
进入同一套 RAG
```

**GraphRAG 暂不纳入 V0.1。**

### V0.1 文档解析边界

V0.1 聚焦 PDF 正文文本主链路，Docling 作为 PDF、Markdown 和 Word 的主解析基底。扫描页正文必须通过 OCR 进入文本链路；当常规解析和 OCR 仍不能可靠恢复正文时，允许在远端服务器调用不超过 14B 的文档解析模型作为兜底。V0.1 不要求完全离线运行。

V0.1 必须支持：

```text
正文、标题和章节层级提取
页码及来源定位
扫描 PDF 正文 OCR
表格、图片及图表区域识别与引用定位
解析质量检测
```

表格的单元格内容完全不进入 Dense、BM25 或 LLM Context，只索引表注、所在章节、页码、来源和区域引用。图片、曲线图、流程图及结构示意图采用相同策略，不进行视觉语义理解。没有表注或图注时，只生成“第 N 页未命名表格 M”或“第 N 页未命名图片 M”等定位性名称，不生成语义描述。

表格和图片按区域隔离，不因这些区域暂未解析而隔离整页或整份文档；同页合格正文仍正常入库。正文解析质量不达标的页面不得自动进入正式知识库，必须进入隔离区并记录失败原因、解析器及版本、质量指标和原始来源，供后续复核或重新处理。

当查询命中表注、图注或其邻近正文时，系统只返回原文来源、所在页、章节、表注或图注以及区域引用，并明确提示 V0.1 尚未解析其内部内容，禁止根据邻近正文猜测表格数值或图片语义。

后续表格解析器和视觉解析器通过统一扩展接口输出标准化文本、结构化 Metadata、来源页码和区域坐标，再进入同一套 RAG 链路。

## 一、功能范围

| 模块 | V0.1 是否实现 | 说明 |
| --- | ---: | --- |
| PDF 文档加载 | 必须 | 第一版主要输入 |
| Markdown 输入 | 必须 | C8 原生适配较好 |
| Word 输入 | 可选 | 调库实现即可 |
| CAD/图纸解析 | 只留接口 | 不自行训练模型 |
| 文档结构解析 | 必须 | 标题、章节、页码等 |
| 扫描页 OCR | 必须 | 对缺失或低质量文字层进行识别 |
| 表格内部解析 | 不做 | 单元格内容不进入检索和生成链路 |
| 表格定位引用 | 必须 | 保留表注、章节、页码、来源、区域引用 |
| 图片/图表语义理解 | 不做 | 不生成语义描述，不进入检索和生成链路 |
| 图片/图表定位引用 | 必须 | 保留图注、章节、页码、来源、区域引用 |
| 解析质量检测与隔离 | 必须 | 正文按页门禁；表格和图片按区域隔离 |
| Metadata 增强 | 必须 | 文档类型、材料、来源等 |
| Parent-Child Chunk | 必须 | 小块检索、大块生成 |
| Dense Retrieval | 必须 | C8 原有 |
| BM25 | 必须 | C8 原有 |
| RRF | 必须 | C8 原有 |
| Cross-Encoder | 建议新增 | C8 的主要增强点 |
| Query Rewrite | 保留 | C8 原有，后续评估收益 |
| Query Routing | 简化 | 删除菜谱场景的 list/detail/general 路由 |
| LLM Generation | 必须 | 基于证据回答 |
| 引用来源 | 必须 | `source/page/section` |
| Recall@K / MRR | 必须 | 检索评估 |
| Faithfulness | 必须 | LLM-as-a-Judge |
| Answer Relevance | 必须 | LLM-as-a-Judge |
| GraphRAG | 不做 | 当前非必要 |
| ERP/MES 等真实系统 | 不做 | 使用公开文档模拟知识侧 |

## 二、数据集总体方案

第一版数据集控制在：

> **20～30 份 PDF，覆盖 3～6 类知识。**

V0.1 不追求一次收集数百份文档。数据集重点满足以下条件：

1. 不同 PDF 之间存在较高语义重叠，能够真实测试 Retrieval。
2. 每类文档在业务上能够对应原案例中的工艺规范、工艺文档、制造知识或材料知识。

公开来源优先级：

```text
NASA NTRS
NIST
其他政府或科研机构公开技术报告
```

厂商手册在第二版补充。

## 三、文档分类

下游按照业务知识类型收集和整理数据：

| 一级分类 | 二级分类或示例 | 计划数量 | 对应原案例能力 | 主要测试能力 | V0.1 优先级 |
| --- | --- | ---: | --- | --- | ---: |
| **材料技术手册** | Aluminum 6061 / 7075 / 2219 / 2014 等 | 6～8 | 图纸材料属性、工艺匹配 | 型号、材料牌号、性能、制造方式 | ★★★★★ |
| **特种合金手册** | Inconel / Nickel Alloy 等 | 2～3 | 材料选型、加工建议 | 相似材料文档区分 | ★★★★☆ |
| **机械加工知识** | Machining、Shop Techniques、切削/成形 | 3～5 | 历史加工经验、工艺知识 | 加工方法、工具、参数知识 | ★★★★★ |
| **连接与制造工艺** | Welding、Brazing、Joining、Forming | 3～4 | 工艺匹配 | 工艺条件与材料关系 | ★★★★☆ |
| **制造规范/指南** | Manufacturing guideline / process guideline | 3～5 | 工艺规范、设计规范 | 长文档规范检索 | ★★★★★ |
| **质量/检测/数据规范** | Inspection、traceability、manufacturing data | 2～4 | 质量、制造知识沉淀 | 条件查询、规范类 QA | ★★★☆☆ |
| 设备操作手册 | CNC / PLC / machine manual | V0.2 | 设备知识、故障知识 | 型号/故障码精确检索 | 暂缓 |
| CAD/STEP/STL | 模型及配套 PDF | V0.2 | 图纸工艺 | 多模态输入 | 暂缓 |
| 采购/成本资料 | Cost estimation、supplier data | V0.2 | 采购成本优化 | Structured RAG / SQL | 暂缓 |

## 四、第一批数据配比

以最终收集 24 份 PDF 为例：

```text
材料技术手册            8
机械加工知识            4
连接/制造工艺           4
制造规范                4
质量/检测/数据规范       4
--------------------------
总计                   24
```

材料手册之间应具有较高相似性，例如：

```text
6061
7075
2014
2219
```

这些文档的章节结构和术语大量重合，适合测试 Embedding 是只能找到“铝合金相关内容”，还是能够精确找到“7075 对应内容”。BM25、Dense、Hybrid 和 Reranker 的差距也更容易在这种语料中表现出来。

## 五、知识库目录

数据按业务知识类型组织：

```text
data/
├── material/
│   ├── aluminum/
│   ├── nickel/
│   └── other_alloys/
│
├── machining/
│   ├── conventional/
│   ├── forming/
│   └── shop_practice/
│
├── joining/
│   ├── welding/
│   ├── brazing/
│   └── other/
│
├── manufacturing_guidelines/
│
└── quality_and_inspection/
```

不按照以下机构来源组织目录：

```text
NASA/
NIST/
Other/
```

**机构属于 Metadata，不属于知识分类。**

## 六、统一 Metadata Schema

C8 当前菜谱场景中的以下字段需要重新设计：

```text
dish_name
category
difficulty
parent_id
```

V0.1 建议 Schema：

```python
{
    "document_id": "...",
    "document_title": "...",
    "document_type": "material_handbook",
    "domain": "material",
    "material": "Aluminum 7075",
    "organization": "NASA",
    "year": 1972,

    "source": "...",
    "page": 35,

    "h1": "...",
    "h2": "...",
    "h3": "...",

    "parent_id": "...",
    "chunk_id": "...",

    "file_type": "pdf"
}
```

不是所有文档都具有以下字段，允许为空：

```text
material
year
h3
```

必须保证的字段和结构是：

```text
document_id
document_title
document_type
source
section hierarchy
parent_id
chunk_id
```

## 七、Chunk 策略

V0.1 使用“结构切分优先，长度切分兜底”的简单策略：

```text
PDF
 ↓
Parser
 ↓
Markdown / Structured Text
 ↓
标题结构切分
 ↓
如果 Section 过长
 ↓
Recursive Split
 ↓
Child Chunk
```

Parent-Child 定义：

```text
Parent：完整章节或较大的上下文
Child：较小的检索单元
```

检索对象：

```text
Child
```

最终发送给 LLM 的内容：

```text
Parent，或者 Child + 邻近上下文
```

保留 C8 中“小块检索，大块生成”的思想。

## 八、检索链路

V0.1 将 C8 检索主链路升级为：

```text
Query
  ↓
可选 Rewrite
  ↓
┌───────────────┐
│ Dense Top-20  │
│ BM25  Top-20  │
└───────┬───────┘
        ↓
       RRF
        ↓
     Top-20
        ↓
Cross-Encoder
        ↓
      Top-5
        ↓
Parent Context Recovery
        ↓
       LLM
```

相对 C8 原项目，主要新增点为 **Cross-Encoder Reranking**。

后续代码审计需要确认：

```text
当前 C8 的 Top-K 在哪里截断
RRF 前候选数是多少
RRF 后候选数是多少
Cross-Encoder 应插入在哪里
Parent 回填发生在 Rerank 前还是后
```

规定的执行顺序为：

```text
Child Retrieval
→ RRF
→ Cross-Encoder
→ Top-K Child
→ Parent Recovery
```

不得先把所有 Child 转换为 Parent 再进行 Rerank，避免长文档重新稀释相关性。

## 九、生成链路

V0.1 Prompt 只要求：

```text
1. 仅根据 Context 回答
2. Context 不足时明确说明
3. 返回来源信息
```

期望输出示例：

```text
Answer:
7075 铝合金在……

Evidence:
[1] xxx Handbook, Section 5.3, Page 42
[2] xxx Manufacturing Guide, Section 2.1, Page 18
```

V0.1 不引入复杂 Agent。

## 十、评测数据集

从 20～30 份 PDF 中建立：

> **50～100 条人工 Golden Queries。**

每条记录至少包含：

```python
{
    "query": "...",
    "relevant_document_ids": [...],
    "relevant_chunk_ids": [...],
    "reference_answer": "..."  # 可在第二阶段补充
}
```

Retriever 评估指标：

```text
Recall@5
Precision@5
MRR
```

当 Relevant Chunk 较多时，可增加：

```text
MAP
```

Generation 评估指标：

```text
Faithfulness
Answer Relevance
```

Generation 评估使用 LLM-as-a-Judge。

## 十一、下游 C8 代码审计任务

下一阶段不得直接开始重写代码。必须先完成以下审计：

| 审计对象 | 需要确定的问题 |
| --- | --- |
| `data_preparation.py` | 当前加载哪些格式；Metadata 如何生成 |
| Chunk | 是否仅使用 `MarkdownHeaderTextSplitter` |
| Parent-Child | `parent_id` / `chunk_id` 如何建立 |
| `index_construction.py` | 当前 Embedding 模型、FAISS 类型、持久化方式 |
| BM25 | 使用什么 Tokenizer；中英文处理是否合理 |
| Dense | Top-K 在哪里控制 |
| RRF | 输入候选数、RRF `k`、去重键是什么 |
| Rerank | 当前是否完全没有 Cross-Encoder |
| Parent Recovery | 具体发生在哪里 |
| Rewrite | 哪些 Query 会被重写 |
| Router | 菜谱业务逻辑可以删除哪些 |
| Generation | Prompt 中哪些属于菜谱业务逻辑 |
| Evaluation | C8 是否已有评测模块；若无则需要新建 |
| Config | 哪些参数应该配置化 |

代码审计的最终输出必须采用以下映射形式：

```text
C8 原模块
      ↓
保留 / 删除 / 修改 / 新增
      ↓
工业文档 RAG 模块
```

代码审计阶段只输出模块的保留、删除、修改、新增清单，不直接大改代码。

## 十二、与原案例的对应关系

| 原案例 | V0.1 |
| --- | --- |
| 三维图纸识别模型 | 黑盒接口，不实现 |
| 几何相似性搜索 | 不实现 |
| 模型差异化比对 | 不实现 |
| 加工特征识别 | 不实现 |
| 工艺规范 | **公开制造规范 PDF** |
| 工艺文档 | **公开加工/制造 PDF** |
| 历史加工经验 | **公开 shop practice / machining handbook** |
| 图纸特征 + 工艺匹配 | V0.2 再接结构化 Feature |
| 工艺智能体 | **RAG + LLM** |
| CAD/PDM/PLM 等系统 | 不真实接入，预留 Adapter |
| 知识关联与智能检索 | **Metadata + Hybrid Retrieval + Rerank** |
| GraphRAG | 当前不采用 |

原案例强调“图纸理解 → 工艺知识 → 业务应用”的完整链条。V0.1 只复刻其中的**工艺知识检索与问答层**。

## 下一阶段任务定义

> **以 Datawhale all-in-rag C8 为 Baseline，对其代码进行完整审计，并设计将“菜谱 RAG”重构为“制造业公开技术文档 RAG”的修改方案。第一版数据集规划为约 20～30 份公开材料手册、机械加工文档、连接制造工艺、制造规范和质量检测 PDF。保留 Parent-Child、Dense、BM25、RRF 等 C8 主干，新增 PDF → 标准化文档解析、工业 Metadata Schema、Cross-Encoder Reranking、引用式回答，以及 Recall@K、MRR、Faithfulness、Answer Relevance 评测。CAD/三维图纸识别仅保留黑盒接口，暂不实现 GraphRAG。代码审计阶段先输出模块的保留、删除、修改、新增清单，不直接重写代码。**
