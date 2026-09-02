"""A5 Dense Embedding defaults. Semantic switches are not defined here.

``dimension``, pooling, and L2 normalization are baseline invariants, not
runtime options. ``batch_size`` and ``device`` are deployment parameters.
"""

from __future__ import annotations

DEFAULT_MODEL_NAME_OR_PATH = "Qwen/Qwen3-Embedding-4B"
DEFAULT_MAX_INPUT_TOKENS = 8192
DEFAULT_BATCH_SIZE = 4
DEFAULT_DEVICE = "cuda"
EMBEDDING_DIMENSION = 2560
NORM_ATOL = 1e-4
DETERMINISM_RTOL = 1e-4
DETERMINISM_ATOL = 1e-5
