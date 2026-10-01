"""Jev route + knowledge-mode smoke. Not QueryRouter."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1_jev_experimental.client import (
    decide_knowledge_mode_with_jev,
    decide_with_jev,
    resolve_jev_connection,
)
from retrieval.b1_jev_experimental.probes import SMOKE_CASES


def main() -> int:
    connection = resolve_jev_connection()
    print(f"jev_provider={connection.provider}")
    print(f"jev_endpoint={connection.endpoint}")
    print(f"jev_model={connection.model}")
    print("task=route_then_knowledge_mode")
    print("main_path=not_connected")
    print("b1_closed=yes")
    for case in SMOKE_CASES:
        answers = decide_with_jev(case["query"], case["history"])
        line = (
            f"{case['id']} route={answers.primary_route} "
            f"confidence={answers.primary_route_confidence:.3f}"
        )
        if answers.primary_route == "knowledge":
            mode = decide_knowledge_mode_with_jev(case["query"], case["history"])
            line += (
                f" mode={mode.knowledge_mode} "
                f"mode_confidence={mode.knowledge_mode_confidence:.3f}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
