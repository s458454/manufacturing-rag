"""Exposed 128-case route + knowledge-mode diagnostic. Not QueryRouter, not CLOSED."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from retrieval.b1_jev_experimental.client import (
    decide_knowledge_mode_with_jev,
    decide_with_jev,
    resolve_jev_connection,
)
from retrieval.b1_jev_experimental.constants import EXPERIMENTAL_CONTRACT_ID
from retrieval.b1_jev_experimental.diag import (
    gold_knowledge_mode,
    gold_route,
    history_from_case,
    route_lane,
    score_control,
    score_mode,
    score_route,
)
from retrieval.b1_jev_experimental.errors import JevProtocolError, JevUnavailableError

SCRIPT_DIR = Path(__file__).resolve().parent
SETS = {
    "holdout": SCRIPT_DIR / "b1_holdout_v1.jsonl",
    "blind": SCRIPT_DIR / "b1_blind_acceptance_v1.jsonl",
}


def _load_cases(path: Path) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _retry(call, *, retries: int):
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return call()
        except JevUnavailableError as exc:
            last = exc
            if attempt >= retries:
                raise
            time.sleep(min(8.0, 2.0 ** attempt))
    raise last or JevUnavailableError("Jev request failed")


def _call_route(case: dict[str, object], *, retries: int, timeout_s: float):
    history = history_from_case(case)
    return _retry(
        lambda: decide_with_jev(str(case["query"]), history, timeout_s=timeout_s),
        retries=retries,
    )


def _call_mode(case: dict[str, object], *, retries: int, timeout_s: float):
    history = history_from_case(case)
    return _retry(
        lambda: decide_knowledge_mode_with_jev(
            str(case["query"]),
            history,
            timeout_s=timeout_s,
        ),
        retries=retries,
    )


def _line_mark(row: dict[str, object]) -> str:
    if row.get("mark") in {"INFRA", "PROTOCOL"}:
        return str(row["mark"])
    if row.get("mode_mark") in {"INFRA", "PROTOCOL"}:
        return str(row["mode_mark"])
    if row.get("lane") == "skip":
        return "SKIP"
    if row.get("mark") == "FAIL" or row.get("mode_mark") == "FAIL":
        return "FAIL"
    return str(row.get("mark") or "SKIP")


def _summarize(set_name: str, rows: list[dict[str, object]]) -> None:
    control = [row for row in rows if row["lane"] == "control"]
    skipped = [row for row in rows if row["lane"] == "skip"]
    evaluable = [row for row in rows if row["lane"] == "route"]
    infra = [
        row
        for row in rows
        if row.get("mark") == "INFRA" or row.get("mode_mark") == "INFRA"
    ]
    protocol = [
        row
        for row in rows
        if row.get("mark") == "PROTOCOL" or row.get("mode_mark") == "PROTOCOL"
    ]
    healthy = [row for row in evaluable if row["mark"] not in {"INFRA", "PROTOCOL"}]
    passed = sum(1 for row in healthy if row["pass"])
    mode_rows = [
        row
        for row in rows
        if row.get("mode_lane") == "mode"
        and row.get("mode_mark") not in {"INFRA", "PROTOCOL", "SKIP"}
    ]
    mode_pass = sum(1 for row in mode_rows if row.get("mode_pass"))
    control_pass = sum(1 for row in control if row["pass"])
    print("---")
    print(f"set_name={set_name}")
    print("task=route_then_knowledge_mode")
    print(f"control={control_pass}/{len(control)}")
    print(f"route={passed}/{len(healthy)}")
    print(f"mode={mode_pass}/{len(mode_rows)}")
    print(f"skip={len(skipped)}")
    print(f"unavailable={len(infra)}")
    print(f"protocol={len(protocol)}")
    by_bucket: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_bucket.setdefault(str(row["bucket"]), []).append(row)
    for bucket, bucket_rows in by_bucket.items():
        lane = bucket_rows[0]["lane"]
        if lane == "control":
            ok = sum(1 for row in bucket_rows if row["pass"])
            print(f"bucket_{bucket}={ok}/{len(bucket_rows)} lane=control")
            continue
        if lane == "skip":
            print(f"bucket_{bucket}={len(bucket_rows)} lane=skip")
            continue
        route_ok = sum(
            1
            for row in bucket_rows
            if row["mark"] not in {"INFRA", "PROTOCOL"} and row["pass"]
        )
        route_n = sum(1 for row in bucket_rows if row["mark"] not in {"INFRA", "PROTOCOL"})
        if bucket_rows[0].get("mode_lane") == "mode":
            mode_ok = sum(
                1
                for row in bucket_rows
                if row.get("mode_mark") not in {"INFRA", "PROTOCOL", "SKIP"}
                and row.get("mode_pass")
            )
            mode_n = sum(
                1
                for row in bucket_rows
                if row.get("mode_mark") not in {"INFRA", "PROTOCOL", "SKIP"}
            )
            print(
                f"bucket_{bucket} route={route_ok}/{route_n} "
                f"mode={mode_ok}/{mode_n} lane=route"
            )
        else:
            print(f"bucket_{bucket}={route_ok}/{route_n} lane=route")
    print("b1_closed=yes")
    print("main_path=not_connected")


def _empty_mode_score(reason: str) -> dict[str, object]:
    return {
        "mode_lane": "skip",
        "mode_mark": "SKIP",
        "mode_pass": None,
        "gold_mode": None,
        "got_mode": None,
        "mode_reason": reason,
    }


def run_set(
    set_name: str,
    *,
    sleep_s: float,
    retries: int,
    timeout_s: float,
) -> list[dict[str, object]]:
    path = SETS[set_name]
    cases = _load_cases(path)
    print(f"set={path}")
    print(f"case_count={len(cases)}")
    print("status=EXPOSED_ROUTE_THEN_MODE_DIAGNOSTIC")
    print(f"experimental_contract_id={EXPERIMENTAL_CONTRACT_ID}")
    print("role=ROUTE_THEN_KNOWLEDGE_MODE_NOT_CLOSED")
    print("closed_candidate_set=no")
    rows: list[dict[str, object]] = []
    for case in cases:
        lane = route_lane(case)
        expected_route = gold_route(case)
        if lane == "control":
            scored = score_control(case)
        else:
            try:
                answers = _call_route(case, retries=retries, timeout_s=timeout_s)
                scored = score_route(case, answers)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                if answers.primary_route == "knowledge":
                    try:
                        mode = _call_mode(case, retries=retries, timeout_s=timeout_s)
                        scored.update(score_mode(case, mode, asked=True))
                    except JevUnavailableError as exc:
                        scored.update(
                            {
                                "mode_lane": mode_lane_or_skip(case),
                                "mode_mark": "INFRA",
                                "mode_pass": None,
                                "gold_mode": gold_knowledge_mode(case),
                                "got_mode": None,
                                "mode_reason": "jev_unavailable",
                                "error_detail": str(exc)[:300],
                            }
                        )
                    except JevProtocolError as exc:
                        scored.update(
                            {
                                "mode_lane": mode_lane_or_skip(case),
                                "mode_mark": "PROTOCOL",
                                "mode_pass": False
                                if gold_knowledge_mode(case) is not None
                                else None,
                                "gold_mode": gold_knowledge_mode(case),
                                "got_mode": None,
                                "mode_reason": "jev_protocol_error",
                                "error_detail": str(exc)[:300],
                            }
                        )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                else:
                    scored.update(score_mode(case, None, asked=False))
            except JevUnavailableError as exc:
                scored = {
                    "lane": lane,
                    "mark": "INFRA",
                    "pass": None,
                    "gold_route": expected_route,
                    "got_route": None,
                    "reason": "jev_unavailable",
                    "got_error": "jev_unavailable",
                    "error_detail": str(exc)[:300],
                }
                scored.update(_empty_mode_score("jev_unavailable"))
            except JevProtocolError as exc:
                scored = {
                    "lane": lane,
                    "mark": "PROTOCOL",
                    "pass": False if lane == "route" else None,
                    "gold_route": expected_route,
                    "got_route": None,
                    "reason": "jev_protocol_error",
                    "got_error": "jev_protocol_error",
                    "error_detail": str(exc)[:300],
                }
                scored.update(_empty_mode_score("jev_protocol_error"))
        scored["id"] = case["id"]
        scored["bucket"] = case["bucket"]
        rows.append(scored)
        extra = ""
        shown = _line_mark(scored)
        if shown in {"FAIL", "INFRA", "PROTOCOL"}:
            extra = f" query={_one_line(case['query'])}"
        print(
            f"{shown} {case['id']} bucket={case['bucket']} "
            f"gold={scored.get('gold_route')} got={scored.get('got_route')} "
            f"gold_mode={scored.get('gold_mode')} got_mode={scored.get('got_mode')} "
            f"reason={scored['reason']} mode_reason={scored.get('mode_reason')}{extra}"
        )
    _summarize(set_name, rows)
    return rows


def mode_lane_or_skip(case: dict[str, object]) -> str:
    return "mode" if gold_knowledge_mode(case) is not None else "skip"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("holdout", "blind", "both"), default="both")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    connection = resolve_jev_connection()
    print(f"jev_provider={connection.provider}")
    print(f"jev_endpoint={connection.endpoint}")
    print(f"jev_model={connection.model}")
    names = ("holdout", "blind") if args.set == "both" else (args.set,)
    for name in names:
        run_set(
            name,
            sleep_s=args.sleep,
            retries=args.retries,
            timeout_s=args.timeout,
        )
        print("===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
