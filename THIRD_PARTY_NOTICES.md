# Third-party notices

This file records the principal upstream material retained in `ver-0`. It is not a substitute for
the complete license text distributed with each component.

## Datawhale all-in-rag

- Source: <https://github.com/datawhalechina/all-in-rag>
- Use in this repository: original C1-C9 tutorial/example code used as a refactoring baseline
- Upstream license: CC BY-NC-SA 4.0
- Copyright and attribution remain with the upstream authors and contributors.

## Docling

- Source: <https://github.com/docling-project/docling>
- Snapshot in this repository: `docling/`, package metadata version 2.115.0
- Upstream license: MIT
- Copyright: IBM Corp. and Docling contributors
- Included license: `docling/LICENSE`

## Preprocessing model dependencies

The repository tracks model identity, configuration, source URLs, sizes, and SHA-256 checksums in
`models/preprocessing-models.manifest.json`; model weights and the RapidOCR rendering font are not
committed to Git.

- Docling Layout Heron: <https://huggingface.co/docling-project/docling-layout-heron>, Apache-2.0
- Docling TableFormer assets: <https://huggingface.co/docling-project/docling-models/tree/v2.3.0>,
  CDLA-Permissive-2.0 and Apache-2.0 notices apply as stated by the upstream model repository
- RapidOCR model assets: <https://github.com/RapidAI/RapidOCR>, Apache-2.0 project; upstream notes
  that OCR model copyright belongs to Baidu
- PP-LCNet document-orientation ONNX export:
  <https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx>

Copyright, attribution, and model-specific usage conditions remain with the respective upstream
authors. Provisioned local assets are not relicensed by this repository.

## FlagEmbedding / Visualized-BGE

- Source: <https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge>
- Snapshot in this repository: `code/C3/visual_bge/`
- Upstream license: MIT
- The embedded EVA-CLIP-derived files retain their own source notices, including OpenAI CLIP
  attribution where stated in file headers.

## NASA and NIST documents

- Location: `data/engineering_docs/raw/`
- Source list: `data/engineering_docs/manifest.csv`
- Use in this repository: research corpus for retrieval and evaluation
- Each document remains subject to the rights, notices, and usage conditions of its issuing
  organization and original source. Inclusion here does not relicense those documents.
