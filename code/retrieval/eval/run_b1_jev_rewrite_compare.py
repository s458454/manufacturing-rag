"""Score the rewrite validation set. Not a CLOSED set. Splitter is not involved."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.types import ConversationTurn, RouterConfig
from retrieval.b1_jev.rewrite_generator import rewrite_candidate
from retrieval.b1_jev.router import qwen_text_complete

DATA_PATH = Path(__file__).resolve().parent / "b1_jev_rewrite_compare_v1.jsonl"


def main() -> int:
    rows = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_category: dict[str, int] = {}
    for row in rows:
        by_category[str(row["category"])] = by_category.get(str(row["category"]), 0) + 1
    short = [name for name, count in by_category.items() if count < 2]
    if short or len(rows) < 20:
        raise SystemExit(f"rewrite set is not balanced: n={len(rows)} short={short}")
    config = RouterConfig(
        vllm_base_url="http://10.50.2.6:8000/v1",
        model_alias="Qwen/Qwen2.5-7B-Instruct",
        tokenizer_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        request_timeout_ms=60000,
    )
    complete = qwen_text_complete(config)
    exact_n = 0
    replace_n = 0
    print(f"set={DATA_PATH}")
    print(f"case_count={len(rows)}")
    print("closed=false")
    for row in rows:
        history = tuple(
            ConversationTurn(
                raw_user_query=item["raw_user_query"],
                canonical_user_query=item["canonical_user_query"],
                assistant_response=item["assistant_response"],
            )
            for item in row["history"]
        )
        gold = str(row["candidate_query"])
        old_slot = str(row["old_slot"])
        new_slot = str(row["new_slot"])
        try:
            got = rewrite_candidate(str(row["query"]), history, complete)
        except Exception as exc:
            print(f"FAIL {row['id']} category={row['category']} error={type(exc).__name__}: {exc}")
            continue
        exact = got == gold
        replaced = new_slot in got and old_slot not in got
        exact_n += int(exact)
        replace_n += int(replaced)
        mark = "PASS" if exact else "FAIL"
        print(
            f"{mark} {row['id']} category={row['category']} "
            f"exact={exact} replaced={replaced}"
        )
        print(f"  gold={gold}")
        print(f"  got={got}")
    print("---")
    print(f"exact={exact_n}/{len(rows)}")
    print(f"replaced={replace_n}/{len(rows)}")
    for name, count in by_category.items():
        print(f"category_{name}={count}")
    print("b1_closed=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
