"""A3.1 terminal Section / table token profiler.

This round does not choose chunk_size or overlap and does not emit Leafs.
CLI reuses A1 ``load_markdown_documents``; it does not rediscover directories
or re-read PDFs / audit JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from knowledge_base.markdown_loader import (
    LoadedMarkdownDocument,
    MarkdownLoadingError,
    load_markdown_documents,
)
from knowledge_base.structure_parser import (
    MarkdownTableSpan,
    ParsedMarkdownDocument,
    TerminalSection,
    parse_markdown_document,
)
from knowledge_base.token_count import (
    SectionProfileError,
    count_tokens,
    load_tokenizer,
    transformers_version,
)

_BODY_THRESHOLDS = (256, 384, 512, 768, 1024, 1536, 2048)
_OVERSIZE_TABLE_CANDIDATE = 512
_LONGEST_SECTIONS = 15
_LARGEST_TABLES = 10


def _percentile(values: list[int], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percent / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_stat(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _token_stats(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50": _round_stat(_percentile(values, 50)),
        "p75": _round_stat(_percentile(values, 75)),
        "p90": _round_stat(_percentile(values, 90)),
        "p95": _round_stat(_percentile(values, 95)),
        "max": max(values) if values else None,
        "mean": _round_stat(_mean(values)),
    }


def _threshold_counts(values: list[int]) -> dict[str, int]:
    return {
        f"gt_{threshold}": sum(1 for value in values if value > threshold)
        for threshold in _BODY_THRESHOLDS
    }


def _tables_in_span(
    tables: tuple[MarkdownTableSpan, ...],
    section: TerminalSection,
) -> list[MarkdownTableSpan]:
    return [
        table
        for table in tables
        if section.body_start_line <= table.start_line < section.body_end_line
    ]


def _heading_level_distribution(
    parsed_docs: list[ParsedMarkdownDocument],
) -> dict[str, int]:
    counts = {str(level): 0 for level in range(1, 7)}
    for parsed in parsed_docs:
        for heading in parsed.headings:
            counts[str(heading.level)] += 1
    return counts


def profile_documents(
    documents: list[LoadedMarkdownDocument],
    tokenizer_id: str,
    *,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Profile terminal Sections and tables. Does not mutate *documents*."""

    if not documents:
        raise SectionProfileError("No documents provided")

    loaded_tokenizer = tokenizer if tokenizer is not None else load_tokenizer(
        tokenizer_id
    )
    parsed_docs: list[ParsedMarkdownDocument] = []
    for document in documents:
        parsed_docs.append(
            parse_markdown_document(
                document.content,
                document_id=document.document_id,
            )
        )

    body_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    unheaded: list[dict[str, str]] = []

    for document, parsed in zip(documents, parsed_docs):
        if parsed.unheaded_document_body:
            unheaded.append({"document_id": document.document_id})
        for table in parsed.tables:
            table_tokens = count_tokens(table.source, loaded_tokenizer)
            table_rows.append(
                {
                    "document_id": document.document_id,
                    "start_line": table.start_line + 1,
                    "end_line": table.end_line,
                    "table_tokens": table_tokens,
                    "preview": table.source.strip().split("\n", 1)[0][:120],
                }
            )
        for section in parsed.terminal_sections:
            body_tokens = count_tokens(section.body_text, loaded_tokenizer)
            nested = _tables_in_span(parsed.tables, section)
            nested_token_counts = [
                count_tokens(table.source, loaded_tokenizer) for table in nested
            ]
            body_rows.append(
                {
                    "document_id": document.document_id,
                    "section_ordinal": section.ordinal,
                    "heading_level": section.heading_level,
                    "heading_text": section.heading_text,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "body_tokens": body_tokens,
                    "table_count": len(nested),
                    "max_table_tokens": (
                        max(nested_token_counts) if nested_token_counts else 0
                    ),
                }
            )

    body_token_values = [row["body_tokens"] for row in body_rows]
    with_body = [value for value in body_token_values if value > 0]
    table_token_values = [row["table_tokens"] for row in table_rows]
    body_stats = _token_stats(body_token_values)
    table_stats = _token_stats(table_token_values)

    longest = sorted(
        body_rows,
        key=lambda row: (
            -int(row["body_tokens"]),
            str(row["document_id"]),
            int(row["section_ordinal"]),
        ),
    )[:_LONGEST_SECTIONS]
    largest_tables = sorted(
        table_rows,
        key=lambda row: (
            -int(row["table_tokens"]),
            str(row["document_id"]),
            int(row["start_line"]),
        ),
    )[:_LARGEST_TABLES]
    oversize = [
        {
            "document_id": row["document_id"],
            "start_line": row["start_line"],
            "table_tokens": row["table_tokens"],
            "preview": row["preview"],
            "candidate_threshold": _OVERSIZE_TABLE_CANDIDATE,
        }
        for row in sorted(
            table_rows,
            key=lambda item: (
                -int(item["table_tokens"]),
                str(item["document_id"]),
                int(item["start_line"]),
            ),
        )
        if int(row["table_tokens"]) > _OVERSIZE_TABLE_CANDIDATE
    ]

    return {
        "tokenizer": tokenizer_id,
        "transformers_version": transformers_version(),
        "document_count": len(documents),
        "heading_level_distribution": _heading_level_distribution(parsed_docs),
        "terminal_section_count": len(body_rows),
        "terminal_sections_with_body": sum(
            1 for value in body_token_values if value > 0
        ),
        "terminal_sections_without_body": sum(
            1 for value in body_token_values if value == 0
        ),
        "terminal_section_body_tokens": {
            **body_stats,
            "with_body_count": len(with_body),
            "thresholds": _threshold_counts(body_token_values),
        },
        "tables": {
            "table_count": len(table_rows),
            "table_token_p50": table_stats["p50"],
            "table_token_p95": table_stats["p95"],
            "max_table_tokens": table_stats["max"],
            "mean": table_stats["mean"],
            "largest_tables": largest_tables,
        },
        "longest_terminal_sections": longest,
        "anomalies": {
            "missing_page_marker": [],
            "malformed_page_marker": [],
            "page_order_error": [],
            "unheaded_document_body": unheaded,
            "oversize_table": oversize,
        },
    }


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def format_profile_summary(profile: dict[str, Any]) -> str:
    body = profile["terminal_section_body_tokens"]
    tables = profile["tables"]
    heading_bits = [
        f"{level}:{count}"
        for level, count in profile["heading_level_distribution"].items()
        if count
    ]
    lines = [
        f"tokenizer={profile['tokenizer']}",
        f"transformers_version={profile['transformers_version']}",
        f"document_count={profile['document_count']}",
        "heading_level_distribution="
        + (",".join(heading_bits) if heading_bits else ""),
        f"terminal_section_count={profile['terminal_section_count']}",
        f"terminal_sections_with_body={profile['terminal_sections_with_body']}",
        "terminal_sections_without_body="
        f"{profile['terminal_sections_without_body']}",
        f"body_tokens_p50={_format_optional(body['p50'])}",
        f"body_tokens_p75={_format_optional(body['p75'])}",
        f"body_tokens_p90={_format_optional(body['p90'])}",
        f"body_tokens_p95={_format_optional(body['p95'])}",
        f"body_tokens_max={_format_optional(body['max'])}",
        f"body_tokens_mean={_format_optional(body['mean'])}",
    ]
    thresholds = body["thresholds"]
    lines.append(
        "body_token_thresholds="
        + ",".join(
            f">{threshold}:{thresholds[f'gt_{threshold}']}"
            for threshold in _BODY_THRESHOLDS
        )
    )
    lines.extend(
        [
            f"table_count={tables['table_count']}",
            f"table_token_p50={_format_optional(tables['table_token_p50'])}",
            f"table_token_p95={_format_optional(tables['table_token_p95'])}",
            f"max_table_tokens={_format_optional(tables['max_table_tokens'])}",
            "longest_terminal_sections:",
        ]
    )
    for row in profile["longest_terminal_sections"]:
        lines.append(
            "  "
            f"document_id={row['document_id']} "
            f"ordinal={row['section_ordinal']} "
            f"level={_format_optional(row['heading_level'])} "
            f"heading={row['heading_text']!r} "
            f"page={row['page_start']}-{row['page_end']} "
            f"body_tokens={row['body_tokens']} "
            f"table_count={row['table_count']} "
            f"max_table_tokens={row['max_table_tokens']}"
        )
    anomalies = profile["anomalies"]
    lines.append("anomalies:")
    for name in (
        "missing_page_marker",
        "malformed_page_marker",
        "page_order_error",
        "unheaded_document_body",
        "oversize_table",
    ):
        items = anomalies[name]
        lines.append(f"  {name}={len(items)}")
        for item in items:
            lines.append(f"    {item}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A3.1 terminal Section profiler. Loads A1 documents from an "
            "explicit canonical root, parses Markdown structure, and counts "
            "tokens with a configurable Qwen3 embedding tokenizer. "
            "Does not choose chunk_size or overlap."
        ),
        epilog=(
            "From the repository root, put code/ on PYTHONPATH, for example:\n"
            '  PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" '
            "python -m knowledge_base.section_profile "
            "--canonical-root <ABS_PATH> "
            "--tokenizer Qwen/Qwen3-Embedding-4B "
            "--output /tmp/a3-section-profile.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Formal A0 output root containing <document_id>/document.md",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help=(
            "Hugging Face tokenizer id or local tokenizer directory. "
            "Must be the Qwen3-Embedding-4B tokenizer for official profiling."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON profile output path",
    )
    args = parser.parse_args(argv)

    try:
        documents = load_markdown_documents(args.canonical_root)
        profile = profile_documents(documents, args.tokenizer)
        args.output.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (MarkdownLoadingError, SectionProfileError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"canonical_root={args.canonical_root.resolve()}")
    print(f"output={args.output.resolve()}")
    print(format_profile_summary(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
