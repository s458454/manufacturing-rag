"""Official Qwen3-Reranker token template and B5 frozen limits."""

RERANKER_CONTRACT_VERSION = "b5-contract-v1"
RERANKER_LOGICAL_ID = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RERANKER_MODEL = "Qwen3-Reranker-0.6B"

RERANK_INPUT_LIMIT = 20
RERANK_OUTPUT_LIMIT = 5
RERANKER_MAX_INPUT_TOKENS = 4096
RERANKER_DEFAULT_BATCH_SIZE = 8
RERANKER_DEVICE = "cuda"
RERANKER_FORWARD_DTYPE = "fp16"

RERANKER_INSTRUCTION = (
    "Given a manufacturing and engineering query, determine whether the "
    "document passage contains relevant information needed to answer the query."
)

# Official Qwen3-Reranker Transformers prefix/suffix. Do not simplify.
RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)
RERANKER_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n"
    "</think>\n\n"
)

YES_TOKEN = "yes"
NO_TOKEN = "no"
