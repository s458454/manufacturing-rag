"""Live smoke: Jev routes, then B2 processes knowledge tasks. Does not score accuracy.

Requires:
  TUZI_API_KEY
  B1_VLLM_BASE_URL
  B1_MODEL_ALIAS
  B1_TOKENIZER_NAME_OR_PATH
  B1_REQUEST_TIMEOUT_MS

Serving pin: models/retrieval-models.manifest.json and
code/retrieval/requirements-server.txt (vllm==0.8.5, engine-level xgrammar).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

MANIFEST_PATH = PROJECT_ROOT / "models" / "retrieval-models.manifest.json"
PINNED_VLLM = "0.8.5"


def load_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    serving = payload["serving"]
    if serving["engine"] != "vllm" or serving["version"] != PINNED_VLLM:
        raise SystemExit(
            f"retrieval-models.manifest.json must pin vllm {PINNED_VLLM}"
        )
    if serving["structured_decoding_backend"] != "xgrammar":
        raise SystemExit("structured_decoding_backend must be xgrammar")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 real-vLLM protocol smoke")
    parser.add_argument(
        "--check-manifest-only",
        action="store_true",
        help="Only verify the pinned serving/model manifest.",
    )
    return parser.parse_args()


def run_protocol_smoke() -> None:
    from retrieval.b1.config import load_router_config
    from retrieval.b1.errors import (
        EmptyQueryError,
        RouterProtocolError,
        RouterUnavailableError,
    )
    from retrieval.b1.memory import InMemoryConversationStore
    from retrieval.b1.pipeline import build_jev_router, route_for_b2
    from retrieval.b1.tokenizer import load_router_tokenizer
    from retrieval.b1.types import QueryRequest

    config = load_router_config()
    tokenizer = load_router_tokenizer(config.tokenizer_name_or_path)
    router = build_jev_router(
        config=config,
        tokenizer=tokenizer,
        store=InMemoryConversationStore(),
    )

    try:
        router.route(QueryRequest(request_id="smoke-empty", conversation_id="c", query="  "))
    except EmptyQueryError:
        print("empty_query=ok")
    else:
        raise SystemExit("empty query must not call the model")

    knowledge = "NASA-STD-5006A 对铝合金焊接有什么要求？"
    try:
        result, processed = route_for_b2(
            router,
            QueryRequest(request_id="smoke-direct", conversation_id="c", query=knowledge),
        )
    except RouterUnavailableError as exc:
        raise SystemExit(f"router unavailable: {exc}") from exc
    except RouterProtocolError as exc:
        print("direct_mode=protocol_error")
        raise SystemExit(f"router protocol error: {exc}") from exc
    print(f"direct_mode={result.mode}")
    print(f"processed_query_count={len(processed)}")
    for item in processed:
        if item.original_query != item.retrieval_queries[0]:
            raise SystemExit("B2 identity invariant failed")
    print("b1_protocol_smoke=ok")


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    print(f"manifest={MANIFEST_PATH}")
    print(f"vllm_pin={manifest['serving']['version']}")
    print(f"model_revision={manifest['assets'][0]['revision']}")
    if args.check_manifest_only:
        print("manifest_check=ok")
        return
    required = (
        "TUZI_API_KEY",
        "B1_VLLM_BASE_URL",
        "B1_MODEL_ALIAS",
        "B1_TOKENIZER_NAME_OR_PATH",
        "B1_REQUEST_TIMEOUT_MS",
    )
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    if missing:
        print("protocol_smoke=skipped missing " + ",".join(missing))
        raise SystemExit(2)
    run_protocol_smoke()


if __name__ == "__main__":
    main()
