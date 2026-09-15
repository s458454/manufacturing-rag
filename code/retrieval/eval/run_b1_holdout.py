"""Run the immutable 64-case B1 regression set against a live vLLM Router.

This set was the v2 CLOSED holdout. From b1-router-v3 onward it is only
regression/diagnostic. Do not change items or gold. A perfect score does not
close B1.

FAIL / INFRA cases dump current query, error_detail, preview, and raw JSON.

Requires:
  B1_VLLM_BASE_URL
  B1_MODEL_ALIAS
  B1_TOKENIZER_NAME_OR_PATH
  B1_REQUEST_TIMEOUT_MS
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from b1_eval_common import run_eval_set

SET_PATH = SCRIPT_DIR / "b1_holdout_v1.jsonl"


def main() -> None:
    run_eval_set(
        SET_PATH,
        status="IMMUTABLE_REGRESSION_V1",
        role="DIAGNOSTIC_NOT_CLOSED_GATE",
        gates_met_label="regression_gates_met",
        extra_headers=(
            "closed_candidate_set=no",
            "perfect_score_closes_b1=no",
        ),
    )


if __name__ == "__main__":
    main()
