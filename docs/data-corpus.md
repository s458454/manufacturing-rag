# Data and Corpus

## 1. Role of This Document

本文件只定义当前数据的**角色、边界、语言与扩展原则**。  
具体 PDF 解析逻辑见 `preprocessing.md`；Chunk/Index 见 `knowledge-base.md`；Golden QA 构造见 `evaluation.md`。

## 2. Current Corpus Role

当前开发数据来自可获取的公开制造业/工程技术 PDF，用来验证：

```text
PDF preprocessing
Markdown structure recovery
Chunking
Hybrid Retrieval
Reranking
Context Recovery
Evidence-bounded QA
```

当前公开语料主要是英文，这是**数据可获取性**导致的开发条件，不代表最终生产只面向英文文档。

不要把当前文档数量写死进核心代码；应从实际数据目录/manifest 获取。

## 3. Expected Production Language

长期要求：

```text
中文文档为主
兼顾英文技术文档
支持中英跨语言检索
```

尤其要覆盖：

```text
Chinese Query → English technical document
```

因此 A5 Embedding 的选型不能只看英文 Retrieval。

## 4. Knowledge Types to Cover

当前文档知识可包含：

```text
标准 / 规范
材料手册
制造 / 加工要求
焊接 / joining 要求
检验 / 质量要求
设备 / 工艺说明
参数、限制、条件、步骤、例外
表格中的可信结构化技术信息
```

这不是业务 Metadata Schema；只是说明测试语料应覆盖不同类型的技术知识。

## 5. Data Boundary

知识库不直接接收：

```text
raw PDF
CAD/STEP/STL
ERP/MES database
inventory state
unvalidated OCR table body
```

这些需要在其各自上游先变成当前知识库认可的标准输入。

## 6. Current Input Contract

对 A1 来说，数据已经被 A0 标准化为：

```text
document.md
```

Markdown 内需要保留：

```text
Heading hierarchy
PDF page marker
正文顺序
通过 A0 准入的 table/caption
```

## 7. Representative-document Evaluation

D1 第一阶段只针对一份具有代表性的真实文档构建约 50 Query，用于快速调试 Retrieval。

这不等于最终全知识库 benchmark。

系统稳定后再扩展：

```text
1 representative document
→ multiple documents
→ corpus-wide evaluation
```

详细规则见 `evaluation.md`。
