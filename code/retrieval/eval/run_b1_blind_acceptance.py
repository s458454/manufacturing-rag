"""Run the frozen B1 blind acceptance set against a live vLLM Router.

This 64-case set was the v3 unseen CLOSED candidate. From ver-0.7 it is
exposed diagnostic: keep items and gold, but do not treat it as unseen.
A passing score still does not close B1.

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

SET_PATH = SCRIPT_DIR / "b1_blind_acceptance_v1.jsonl"


def main() -> None:
    run_eval_set(
        SET_PATH,
        status="FROZEN_BLIND_ACCEPTANCE_V1",
        role="EXPOSED_DIAGNOSTIC_NOT_CLOSED_GATE",
        gates_met_label="blind_gates_met",
        extra_headers=(
            "closed_candidate_set=no",
            "set_exposed=yes",
            "perfect_score_closes_b1=no",
        ),
    )


if __name__ == "__main__":
    main()
