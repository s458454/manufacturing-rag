"""Manifest check for the pinned B5 Qwen3-Reranker-0.6B revision.

Real-model server acceptance is out of scope for this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "models" / "retrieval-models.manifest.json"
PINNED_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
PINNED_LOGICAL_ID = "Qwen/Qwen3-Reranker-0.6B"


def load_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assets = payload["assets"]
    reranker = next(
        (
            asset
            for asset in assets
            if asset.get("logical_id") == PINNED_LOGICAL_ID
        ),
        None,
    )
    if reranker is None:
        raise SystemExit("retrieval-models.manifest.json must pin Qwen3-Reranker-0.6B")
    if reranker.get("revision") != PINNED_REVISION:
        raise SystemExit(
            "Qwen3-Reranker-0.6B revision must be "
            f"{PINNED_REVISION}, got {reranker.get('revision')!r}"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B5 reranker pin check")
    parser.add_argument(
        "--check-manifest-only",
        action="store_true",
        help="Only verify the pinned Qwen3-Reranker-0.6B revision.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_manifest()
    print("manifest_check=ok")
    if not args.check_manifest_only:
        print("real_model_smoke=skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
