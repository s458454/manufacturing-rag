"""Chat-completions route smoke. Not jev-1.13 /systemone and not QueryRouter."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1_jev_experimental.probes import SMOKE_CASES
from retrieval.b1_jev_experimental.tuzi_adapter import (
    decide_with_tuzi,
    resolve_tuzi_connection,
)


def main() -> int:
    connection = resolve_tuzi_connection()
    print(f"provider={connection.provider}")
    print(f"endpoint={connection.endpoint}")
    print(f"model={connection.model}")
    print("task=route_only")
    print("main_path=not_connected")
    print("b1_closed=yes")
    print("not_jev=yes")
    for case in SMOKE_CASES:
        answers = decide_with_tuzi(case["query"], case["history"])
        print(
            f"{case['id']} route={answers.primary_route} "
            f"confidence={answers.primary_route_confidence:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
