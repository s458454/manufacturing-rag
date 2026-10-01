"""Local GLiClass route smoke. Does not call QueryRouter or TypeSafe."""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1_jev_experimental.gliclass_adapter import (
    decide_with_gliclass,
    resolve_gliclass_model_path,
)
from retrieval.b1_jev_experimental.probes import SMOKE_CASES


def main() -> int:
    path = resolve_gliclass_model_path()
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"gliclass_model_path={path}")
    print(f"torch={torch.__version__} device={device}")
    print("task=route_only")
    print("main_path=not_connected")
    print("b1_closed=yes")
    for case in SMOKE_CASES:
        answers = decide_with_gliclass(case["query"], case["history"], device=device)
        print(
            f"{case['id']} route={answers.primary_route} "
            f"confidence={answers.primary_route_confidence:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
