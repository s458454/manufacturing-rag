# PP-LCNet page-orientation model assets

This directory is consumed by `code/preprocessing/page_orientation.py` before
Docling and RapidOCR run. It must be provisioned locally; preprocessing never
downloads assets at runtime.

Required local files:

```text
model.onnx       # ignored by Git; verified against manifest.json
inference.yml    # official preprocessing configuration
labels.json      # ordered labels matching inference.yml
manifest.json    # model identity, provenance, and SHA-256
```

`model.onnx` remains ignored because it is a binary model asset. The three
metadata/configuration files are deliberately trackable so that the inference
contract remains auditable.

Before running preprocessing, verify that `manifest.json` matches the
provisioned model's real version, official source URL, and SHA-256. Placeholder
values and a mismatched SHA-256 are intentionally rejected at runtime. The
current real-model acceptance baseline is
`af9a0a4f317ff0709ce752067807f819cb15d883f8ecad89f28df1c6ee2d9c92`.

Use the official `PP-LCNet_x1_0_doc_ori` ONNX export. Its expected ordered
labels are `0`, `90`, `180`, `270`; do not use the RapidOCR PP-OCRv4 text-line
classifier as a replacement.
