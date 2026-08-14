# TableFormer V1 accurate model asset requirement

The project expects the approved TableFormer V1 `accurate` runtime assets here.
Binary weights remain ignored by Git. They are downloaded only during controlled
model provisioning, never by the preprocessing program at runtime. The approved
upstream revision is fixed at `docling-project/docling-models` `v2.3.0`.

At minimum the runtime requires:

```text
models/docling-project--docling-models/model_artifacts/tableformer/accurate/
├── tm_config.json
└── tableformer_accurate.safetensors
```

The exact approved assets are recorded in
`models/preprocessing-models.manifest.json` and are verified before Docling
conversion:

| File | Size | SHA-256 |
| --- | ---: | --- |
| `tm_config.json` | 7,060 | `984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd` |
| `tableformer_accurate.safetensors` | 212,758,388 | `2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9` |

The program fails before Docling conversion if either file is missing or does
not match its approved size and SHA-256. It never downloads a model at runtime.
