"""Compare splitter outputs on b1_jev_split_compare_v1.jsonl.

Qwen uses the same guided schema as the candidate router.
LiaoBots is plain chat, same as the gpt-4o-mini probe.
Not a CLOSED set and not part of JevQueryRouter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1.constants import STRUCTURED_DECODING_BACKEND
from retrieval.b1_jev.multi_splitter import (
    SPLITTER_GUIDED_SCHEMA,
    SplitterFailed,
    parse_split_payload,
    splitter_messages,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "b1_jev_split_compare_v1.jsonl"
_ANCHOR = re.compile(
    r"NASA-STD-\d+(?:\.\d+)?[A-Z]?"
    r"|NASA-HDBK-\d+(?:\.\d+)?(?:-\d+)?"
    r"|NISTIR\s*\d+"
    r"|YH5WR-\d+/\d+"
    r"|水贝|兆瓦"
)


def _load(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise SystemExit(f"{path} row is not an object")
        if payload.get("closed") is not False:
            raise SystemExit(f"{payload.get('id')} must stay closed=false")
        rows.append(payload)
    return rows


def _anchors(text: str) -> list[str]:
    return [item.replace(" ", "") for item in _ANCHOR.findall(text)]


def _score(reference: list[str], got: list[str] | None, error: str | None) -> dict[str, object]:
    if got is None:
        return {
            "parse": False,
            "count": False,
            "exact": False,
            "anchors": False,
            "error": error,
        }
    exact = got == reference
    count = len(got) == len(reference)
    anchors = count and all(
        _anchors(left) == _anchors(right) for left, right in zip(reference, got)
    )
    return {"parse": True, "count": count, "exact": exact, "anchors": anchors, "error": None}


def _complete_qwen(messages: list[dict[str, str]]) -> str:
    from openai import OpenAI

    base_url = os.environ.get("B1_VLLM_BASE_URL", "http://10.50.2.6:8000/v1").strip()
    model = os.environ.get("B1_MODEL_ALIAS", "Qwen/Qwen2.5-7B-Instruct").strip()
    api_key = os.environ.get("B1_VLLM_API_KEY", "EMPTY").strip() or "EMPTY"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        extra_body={
            "guided_json": SPLITTER_GUIDED_SCHEMA,
            "guided_decoding_backend": STRUCTURED_DECODING_BACKEND,
        },
    )
    return response.choices[0].message.content or ""


def _complete_liaobots(messages: list[dict[str, str]], model: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("LIAOBOTS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("set LIAOBOTS_API_KEY to the LiaoBots login code")
    base_url = os.environ.get("LIAOBOTS_BASE_URL", "https://ai.liaobots.work/v1").strip()
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        stream=False,
    )
    return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("qwen", "liaobots"), required=True)
    parser.add_argument("--model", default=os.environ.get("LIAOBOTS_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()
    rows = _load(DATA_PATH)
    print(f"set={DATA_PATH}")
    print(f"case_count={len(rows)}")
    print(f"backend={args.backend}")
    print("closed=false")
    if args.backend == "qwen":
        print("guided_json=yes")
        print(f"model={os.environ.get('B1_MODEL_ALIAS', 'Qwen/Qwen2.5-7B-Instruct')}")
    else:
        print("guided_json=no")
        print(f"model={args.model}")

    tallies = {"parse": 0, "count": 0, "exact": 0, "anchors": 0}
    by_pattern: dict[str, dict[str, int]] = {}
    for row in rows:
        case_id = str(row["id"])
        pattern = str(row["pattern"])
        reference = [str(item) for item in row["tasks"]]  # type: ignore[index]
        messages = splitter_messages(str(row["query"]))
        error: str | None = None
        got: list[str] | None = None
        raw = ""
        try:
            if args.backend == "qwen":
                raw = _complete_qwen(messages)
            else:
                raw = _complete_liaobots(messages, args.model)
            got = list(parse_split_payload(raw))
        except SplitterFailed as exc:
            error = str(exc)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        scored = _score(reference, got, error)
        bucket = by_pattern.setdefault(
            pattern, {"n": 0, "parse": 0, "count": 0, "exact": 0, "anchors": 0}
        )
        bucket["n"] += 1
        for key in ("parse", "count", "exact", "anchors"):
            if scored[key]:
                tallies[key] += 1
                bucket[key] += 1
        mark = "PASS" if scored["parse"] and scored["count"] and scored["anchors"] else "FAIL"
        print(
            f"{mark} {case_id} pattern={pattern} "
            f"parse={scored['parse']} count={scored['count']} "
            f"anchors={scored['anchors']} exact={scored['exact']} "
            f"n_ref={len(reference)} n_got={None if got is None else len(got)}"
        )
        if got is not None:
            for index, text in enumerate(got, start=1):
                print(f"  {index}. {text}")
        if error:
            print(f"  error={error}")
            if raw:
                print(f"  raw={raw}")

    total = len(rows)
    print("---")
    print(f"parse={tallies['parse']}/{total}")
    print(f"count={tallies['count']}/{total}")
    print(f"anchors={tallies['anchors']}/{total}")
    print(f"exact={tallies['exact']}/{total}")
    for pattern, bucket in by_pattern.items():
        n = bucket["n"]
        print(
            f"pattern_{pattern} n={n} parse={bucket['parse']}/{n} "
            f"count={bucket['count']}/{n} anchors={bucket['anchors']}/{n} "
            f"exact={bucket['exact']}/{n}"
        )
    print("b1_closed=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
