"""Probe the same splitter prompt on a LiaoBots chat model.

Not part of JevQueryRouter. No xgrammar. Compares parsed tasks with the
experimental multi gold so a capability difference is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1_jev.multi_splitter import SplitterFailed, parse_split_payload, splitter_messages

SCRIPT_DIR = Path(__file__).resolve().parent
CASE_IDS = ("h-039", "h-040", "h-044", "ba-039", "ba-040", "ba-046")
DEFAULT_BASE_URL = "https://ai.liaobots.work/v1"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _gold() -> dict[str, list[str]]:
    gold: dict[str, list[str]] = {}
    for row in _load_jsonl(SCRIPT_DIR / "b1_jev_multi_task_gold_v1.jsonl"):
        tasks = row.get("tasks")
        if isinstance(tasks, list):
            gold[str(row["id"])] = [str(item) for item in tasks]
    return gold


def _cases() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("b1_holdout_v1.jsonl", "b1_blind_acceptance_v1.jsonl"):
        for row in _load_jsonl(SCRIPT_DIR / name):
            case_id = str(row.get("id"))
            if case_id in CASE_IDS:
                found[case_id] = str(row["query"])
    missing = [case_id for case_id in CASE_IDS if case_id not in found]
    if missing:
        raise SystemExit(f"missing cases: {missing}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LIAOBOTS_MODEL", "").strip())
    parser.add_argument("--base-url", default=os.environ.get("LIAOBOTS_BASE_URL", DEFAULT_BASE_URL))
    args = parser.parse_args()
    api_key = os.environ.get("LIAOBOTS_API_KEY", "").strip()
    if not api_key:
        print("set LIAOBOTS_API_KEY to the LiaoBots login code")
        return 1
    if not args.model:
        print("pass --model or set LIAOBOTS_MODEL")
        return 1
    try:
        from openai import OpenAI
    except ImportError:
        print("openai is not installed in this interpreter")
        return 1

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120, max_retries=0)
    gold = _gold()
    print(f"base_url={args.base_url}")
    print(f"model={args.model}")
    print("guided_json=no")
    for case_id, query in _cases().items():
        messages = splitter_messages(query)
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
                max_tokens=1024,
                stream=False,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            print(f"FAIL {case_id} api_error={type(exc).__name__}: {exc}")
            continue
        print(f"--- {case_id}")
        print(raw)
        try:
            queries = parse_split_payload(raw)
        except SplitterFailed as exc:
            print(f"parse=reject reason={exc}")
            continue
        expected = gold[case_id]
        mark = "match" if list(queries) == expected else "mismatch"
        print(f"parse=ok count={len(queries)} gold={mark}")
        for index, text in enumerate(queries, start=1):
            print(f"  {index}. {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
