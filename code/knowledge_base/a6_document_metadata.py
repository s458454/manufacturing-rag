"""A6 production document metadata: deterministic extraction, dedup, version governance.

Reads only trusted ``document.md`` (A1 loader) plus the same ``quality_report.json``
A2 already reads for identity (``source_sha256``). Never re-OCRs, never re-reads
the raw PDF, never calls an LLM/VLM, and never modifies A0-A5 contracts.

This module is the production A6 counterpart to the A6.0 profiling tool
(``a6_analyzer_profile.py``). It is intentionally free of ``torch``/``transformers``
imports so it can run on a Milvus-only machine without embedding weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from knowledge_base.document_registry import (
    DocumentRegistryError,
    build_document_registry,
)
from knowledge_base.markdown_loader import (
    LoadedMarkdownDocument,
    MarkdownLoadingError,
    load_markdown_documents,
)

# ---------------------------------------------------------------------------
# Status vocabulary (frozen, contract SS24)
# ---------------------------------------------------------------------------

STATUS_ACTIVE = "ACTIVE"
STATUS_HISTORICAL = "HISTORICAL"
STATUS_DUPLICATE = "DUPLICATE"
STATUS_VERSION_CONFLICT = "VERSION_CONFLICT"
ALL_STATUSES = frozenset(
    {STATUS_ACTIVE, STATUS_HISTORICAL, STATUS_DUPLICATE, STATUS_VERSION_CONFLICT}
)

# Fields governed by the SS15 candidate-conflict rule.
FIELD_DOCUMENT_NUMBER = "document_number"
FIELD_DOCUMENT_VERSION = "document_version"
FIELD_FINALIZED_AT = "finalized_at"
FIELD_EFFECTIVE_FROM = "effective_from"
FIELD_EFFECTIVE_TO = "effective_to"

METHOD_EXPLICIT_LABEL = "explicit_label"
METHOD_NOT_FOUND = "not_found"
METHOD_DERIVED_FROM_FINALIZED_AT = "derived_from_finalized_at"
METHOD_DURATION_CALCULATION = "duration_calculation"
METHOD_CONFLICTING_CANDIDATES = "conflicting_candidates"

CONFLICT_REASON_SAME_NUMBER_MISSING_FINALIZED_AT = (
    "same_document_number_missing_finalized_at"
)
CONFLICT_REASON_SAME_NUMBER_FINALIZED_AT_TIE = "same_document_number_finalized_at_tie"
CONFLICT_REASON_TITLE_MATCH_NUMBER_INSUFFICIENT = (
    "title_match_document_number_insufficient"
)

# SS7 scan window: front + tail effective (non-blank) text lines.
SCAN_HEAD_LINES = 300
SCAN_TAIL_LINES = 300

_PDF_PAGE_MARKER_LINE = re.compile(r"^<!--\s*PDF page \d+\s*-->$")
_LABEL_SEP = r"[:：]"

_DOCUMENT_NUMBER_LABELS: tuple[str, ...] = (
    "文件编号",
    "文档编号",
    "标准号",
    "规范号",
    "编号",
    "Document No.",
    "Document Number",
    "Standard No.",
)

_DOCUMENT_VERSION_LABELS: tuple[str, ...] = (
    "版本号",
    "版本",
    "版次",
    "修订号",
    "修订版",
    "Version",
    "Revision",
    "Rev.",
)

# SS10: priority tiers, highest first. Only the first non-empty tier counts.
_FINALIZED_AT_LABEL_TIERS: tuple[tuple[str, ...], ...] = (
    ("批准日期", "批准时间", "Approved Date", "Approval Date"),
    ("签发日期", "Issue Date"),
    ("发布日期", "Release Date", "Published Date"),
)

_EFFECTIVE_FROM_LABELS: tuple[str, ...] = (
    "实施日期",
    "生效日期",
    "Effective Date",
    "Effective From",
    "Valid From",
)

_EFFECTIVE_TO_LABELS: tuple[str, ...] = (
    "有效期至",
    "失效日期",
    "截止日期",
    "Expiration Date",
    "Effective Until",
    "Valid Through",
)

# SS11: "自 YYYY年MM月DD日起实施/起生效" is recognized without a colon label.
_EFFECTIVE_FROM_PHRASE = re.compile(r"自\s*(?P<date>[^，,]+?)\s*起\s*(?:实施|生效)")
# SS11 special case: "自发布之日起实施/起生效" derives from finalized_at.
_ISSUANCE_PHRASE = re.compile(r"自\s*发布\s*之\s*日\s*起\s*(?:实施|生效)")
# SS12: "自 <date> 起实施/起生效，有效期 N 年" is a deterministic duration calc.
_DURATION_PATTERN = re.compile(
    r"自\s*(?P<date>[^，,起]+?)\s*起\s*(?:实施|生效)\s*[，,]\s*有效期\s*(?P<years>\d+)\s*年"
)

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_DATE_CN = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_DATE_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_DATE_SLASH = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")
_DATE_DOT = re.compile(r"(\d{4})\.(\d{1,2})\.(\d{1,2})")
_DATE_EN_MDY = re.compile(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})")
_DATE_EN_DMY = re.compile(r"(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})")

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


class A6MetadataError(Exception):
    """Fail-fast error for A6 document metadata governance."""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentMetadataDraft:
    """A single document's deterministically extracted fields, pre-governance."""

    document_id: str
    file_name: str
    document_title: str
    document_number: str | None
    document_version: str | None
    source_sha256: str
    document_content_hash: str
    finalized_at: int | None
    effective_from: int | None
    effective_to: int | None
    ingested_at: int


DOCUMENT_METADATA_FIELD_NAMES: tuple[str, ...] = (
    "document_id",
    "file_name",
    "document_title",
    "document_number",
    "document_version",
    "source_sha256",
    "document_content_hash",
    "finalized_at",
    "effective_from",
    "effective_to",
    "ingested_at",
    "status",
)


@dataclass(frozen=True)
class DocumentMetadata:
    document_id: str
    file_name: str
    document_title: str
    document_number: str | None
    document_version: str | None
    source_sha256: str
    document_content_hash: str
    finalized_at: int | None
    effective_from: int | None
    effective_to: int | None
    ingested_at: int
    status: str

    def to_json_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in DOCUMENT_METADATA_FIELD_NAMES}


AUDIT_FIELD_NAMES: tuple[str, ...] = (
    "document_id",
    "field",
    "value",
    "method",
    "evidence",
    "line_start",
    "line_end",
    "conflict",
)


@dataclass(frozen=True)
class MetadataAuditRecord:
    document_id: str
    field: str
    value: Any
    method: str
    evidence: str | None
    line_start: int | None
    line_end: int | None
    conflict: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in AUDIT_FIELD_NAMES}


CONFLICT_FIELD_NAMES: tuple[str, ...] = (
    "document_id_a",
    "document_id_b",
    "reason",
    "created_at",
)


@dataclass(frozen=True)
class DocumentConflict:
    document_id_a: str
    document_id_b: str
    reason: str
    created_at: int

    def to_json_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in CONFLICT_FIELD_NAMES}


@dataclass(frozen=True)
class _Candidate:
    value: Any
    evidence: str
    line_start: int
    line_end: int
    method: str


@dataclass(frozen=True)
class GovernanceOutcome:
    catalog: tuple[DocumentMetadata, ...]
    new_conflicts: tuple[DocumentConflict, ...]


# ---------------------------------------------------------------------------
# Time helpers (SS13)
# ---------------------------------------------------------------------------


def date_to_utc_ms(value: date) -> int:
    dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def now_utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def add_years(value: date, years: int) -> date:
    """Deterministic calendar add. Feb 29 + N years folds back to Feb 28."""

    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date_token(text: str) -> date | None:
    """Parse the first recognizable explicit date inside *text*. No guessing."""

    match = _DATE_CN.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DATE_ISO.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DATE_SLASH.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DATE_DOT.search(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _DATE_EN_MDY.search(text)
    if match and match.group(1).lower() in _MONTH_NAMES:
        return _safe_date(
            int(match.group(3)), _MONTH_NAMES[match.group(1).lower()], int(match.group(2))
        )
    match = _DATE_EN_DMY.search(text)
    if match and match.group(2).lower() in _MONTH_NAMES:
        return _safe_date(
            int(match.group(3)), _MONTH_NAMES[match.group(2).lower()], int(match.group(1))
        )
    return None


# ---------------------------------------------------------------------------
# SS7 scan window
# ---------------------------------------------------------------------------


def effective_line_indices(lines: Sequence[str]) -> list[int]:
    """0-based indices of non-blank lines, in document order."""

    return [index for index, line in enumerate(lines) if line.strip()]


def scan_window_line_indices(text: str) -> frozenset[int]:
    """Front 300 + tail 300 effective lines. Naturally covers the whole
    document when it has <= 600 effective lines (the two slices overlap)."""

    lines = text.splitlines()
    effective = effective_line_indices(lines)
    head = effective[:SCAN_HEAD_LINES]
    tail = effective[-SCAN_TAIL_LINES:] if effective else []
    return frozenset(head) | frozenset(tail)


def _table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if "|" not in stripped:
        return None
    parts = [part.strip() for part in stripped.strip("|").split("|")]
    parts = [part for part in parts if part != ""]
    return parts if len(parts) >= 2 else None


def _match_labels_in_line(line: str, labels: Sequence[str]) -> tuple[str, str] | None:
    """Return (label, value) for a plain 'label:value' line or a two-cell
    Markdown table row '| label | value |'. Longest label matched first."""

    stripped = line.strip()
    ordered_labels = sorted(labels, key=len, reverse=True)
    for label in ordered_labels:
        pattern = re.compile(rf"^{re.escape(label)}\s*{_LABEL_SEP}\s*(.+)$")
        match = pattern.match(stripped)
        if match:
            value = match.group(1).strip().rstrip("|").strip()
            if value:
                return label, value
    cells = _table_cells(line)
    if cells:
        lowered_labels = {label.lower(): label for label in labels}
        for index in range(len(cells) - 1):
            cell_key = cells[index].rstrip(":：").strip().lower()
            if cell_key in lowered_labels:
                value = cells[index + 1].strip()
                if value:
                    return lowered_labels[cell_key], value
    return None


def _label_candidates(
    lines: Sequence[str],
    window: frozenset[int],
    labels: Sequence[str],
    *,
    value_transform: Any = None,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for index in sorted(window):
        line = lines[index]
        found = _match_labels_in_line(line, labels)
        if found is None:
            continue
        _label, raw_value = found
        value = raw_value if value_transform is None else value_transform(raw_value)
        if value is None:
            continue
        candidates.append(
            _Candidate(
                value=value,
                evidence=line.strip(),
                line_start=index + 1,
                line_end=index + 1,
                method=METHOD_EXPLICIT_LABEL,
            )
        )
    return candidates


def _date_label_candidates(
    lines: Sequence[str], window: frozenset[int], labels: Sequence[str]
) -> list[_Candidate]:
    def _to_ms(raw_value: str) -> int | None:
        parsed = parse_date_token(raw_value)
        return None if parsed is None else date_to_utc_ms(parsed)

    return _label_candidates(lines, window, labels, value_transform=_to_ms)


def _normalize_document_number(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _normalize_document_version(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


# ---------------------------------------------------------------------------
# SS15 candidate resolution
# ---------------------------------------------------------------------------


def _resolve_tier(
    document_id: str, field: str, candidates: Sequence[_Candidate]
) -> tuple[Any, MetadataAuditRecord]:
    distinct = {candidate.value for candidate in candidates}
    if len(distinct) == 1:
        chosen = candidates[0]
        return chosen.value, MetadataAuditRecord(
            document_id=document_id,
            field=field,
            value=chosen.value,
            method=chosen.method,
            evidence=chosen.evidence,
            line_start=chosen.line_start,
            line_end=chosen.line_end,
            conflict=False,
        )
    return None, MetadataAuditRecord(
        document_id=document_id,
        field=field,
        value=None,
        method=METHOD_CONFLICTING_CANDIDATES,
        evidence=None,
        line_start=None,
        line_end=None,
        conflict=True,
    )


def resolve_field(
    document_id: str,
    field: str,
    tiered_candidates: Sequence[Sequence[_Candidate]],
) -> tuple[Any, MetadataAuditRecord]:
    """SS15: only the first non-empty tier (in priority order) counts."""

    for tier in tiered_candidates:
        if tier:
            return _resolve_tier(document_id, field, tier)
    return None, MetadataAuditRecord(
        document_id=document_id,
        field=field,
        value=None,
        method=METHOD_NOT_FOUND,
        evidence=None,
        line_start=None,
        line_end=None,
        conflict=False,
    )


# ---------------------------------------------------------------------------
# SS8-SS12 field extraction
# ---------------------------------------------------------------------------


def _extract_document_number(
    document_id: str, lines: Sequence[str], window: frozenset[int]
) -> tuple[str | None, MetadataAuditRecord]:
    candidates = _label_candidates(
        lines, window, _DOCUMENT_NUMBER_LABELS, value_transform=_normalize_document_number
    )
    return resolve_field(document_id, FIELD_DOCUMENT_NUMBER, [candidates])


def _extract_document_version(
    document_id: str, lines: Sequence[str], window: frozenset[int]
) -> tuple[str | None, MetadataAuditRecord]:
    candidates = _label_candidates(
        lines, window, _DOCUMENT_VERSION_LABELS, value_transform=_normalize_document_version
    )
    return resolve_field(document_id, FIELD_DOCUMENT_VERSION, [candidates])


def _extract_finalized_at(
    document_id: str, lines: Sequence[str], window: frozenset[int]
) -> tuple[int | None, MetadataAuditRecord]:
    tiers = [
        _date_label_candidates(lines, window, tier_labels)
        for tier_labels in _FINALIZED_AT_LABEL_TIERS
    ]
    return resolve_field(document_id, FIELD_FINALIZED_AT, tiers)


def _issuance_phrase_candidates(
    lines: Sequence[str], window: frozenset[int], finalized_at: int | None
) -> list[_Candidate]:
    if finalized_at is None:
        return []
    candidates: list[_Candidate] = []
    for index in sorted(window):
        line = lines[index]
        if _ISSUANCE_PHRASE.search(line):
            candidates.append(
                _Candidate(
                    value=finalized_at,
                    evidence=line.strip(),
                    line_start=index + 1,
                    line_end=index + 1,
                    method=METHOD_DERIVED_FROM_FINALIZED_AT,
                )
            )
    return candidates


def _effective_from_phrase_candidates(
    lines: Sequence[str], window: frozenset[int]
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for index in sorted(window):
        line = lines[index]
        match = _EFFECTIVE_FROM_PHRASE.search(line)
        if not match:
            continue
        parsed = parse_date_token(match.group("date"))
        if parsed is None:
            continue
        candidates.append(
            _Candidate(
                value=date_to_utc_ms(parsed),
                evidence=line.strip(),
                line_start=index + 1,
                line_end=index + 1,
                method=METHOD_EXPLICIT_LABEL,
            )
        )
    return candidates


def _extract_effective_from(
    document_id: str,
    lines: Sequence[str],
    window: frozenset[int],
    finalized_at: int | None,
) -> tuple[int | None, MetadataAuditRecord]:
    candidates = list(_date_label_candidates(lines, window, _EFFECTIVE_FROM_LABELS))
    candidates.extend(_effective_from_phrase_candidates(lines, window))
    candidates.extend(_issuance_phrase_candidates(lines, window, finalized_at))
    return resolve_field(document_id, FIELD_EFFECTIVE_FROM, [candidates])


def _duration_candidates(lines: Sequence[str], window: frozenset[int]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for index in sorted(window):
        line = lines[index]
        match = _DURATION_PATTERN.search(line)
        if not match:
            continue
        start = parse_date_token(match.group("date"))
        if start is None:
            continue
        years = int(match.group("years"))
        end = add_years(start, years)
        candidates.append(
            _Candidate(
                value=date_to_utc_ms(end),
                evidence=line.strip(),
                line_start=index + 1,
                line_end=index + 1,
                method=METHOD_DURATION_CALCULATION,
            )
        )
    return candidates


def _extract_effective_to(
    document_id: str, lines: Sequence[str], window: frozenset[int]
) -> tuple[int | None, MetadataAuditRecord]:
    candidates = list(_date_label_candidates(lines, window, _EFFECTIVE_TO_LABELS))
    candidates.extend(_duration_candidates(lines, window))
    return resolve_field(document_id, FIELD_EFFECTIVE_TO, [candidates])


def extract_field_audit(
    document_id: str, document_md_text: str
) -> tuple[
    str | None,
    str | None,
    int | None,
    int | None,
    int | None,
    tuple[MetadataAuditRecord, ...],
]:
    """SS8-SS12: extract the five governed fields with a full audit trail."""

    lines = document_md_text.splitlines()
    window = scan_window_line_indices(document_md_text)
    document_number, audit_number = _extract_document_number(document_id, lines, window)
    document_version, audit_version = _extract_document_version(document_id, lines, window)
    finalized_at, audit_finalized = _extract_finalized_at(document_id, lines, window)
    effective_from, audit_from = _extract_effective_from(
        document_id, lines, window, finalized_at
    )
    effective_to, audit_to = _extract_effective_to(document_id, lines, window)
    audits = (audit_number, audit_version, audit_finalized, audit_from, audit_to)
    return document_number, document_version, finalized_at, effective_from, effective_to, audits


# ---------------------------------------------------------------------------
# SS17/SS18 dedup fingerprints
# ---------------------------------------------------------------------------


def canonical_document_markdown(text: str) -> str:
    """SS18 canonicalization: drop PDF page marker lines, CRLF->LF, NFC,
    strip trailing whitespace per line. Never lowercases or reorders content."""

    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    nfc = unicodedata.normalize("NFC", normalized_newlines)
    kept_lines = []
    for line in nfc.split("\n"):
        if _PDF_PAGE_MARKER_LINE.fullmatch(line.strip()) is not None:
            continue
        kept_lines.append(line.rstrip())
    return "\n".join(kept_lines)


def compute_document_content_hash(text: str) -> str:
    canonical = canonical_document_markdown(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_source_identity(document_dir: Path) -> tuple[str, str]:
    """Read (document_id, source_sha256) from the trusted quality_report.json.

    This is the same file A2 reads for identity; A6 does not import A2 private
    helpers and does not modify A2's public contract.
    """

    report_path = document_dir / "quality_report.json"
    if not report_path.is_file():
        raise A6MetadataError(f"quality_report.json missing or not a file: {report_path}")
    try:
        raw = report_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise A6MetadataError(f"Failed to read {report_path}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise A6MetadataError(f"Invalid JSON in {report_path}") from exc
    if not isinstance(payload, dict):
        raise A6MetadataError(f"quality_report.json root must be an object: {report_path}")
    document_id = payload.get("document_id")
    if not isinstance(document_id, str) or not document_id.strip():
        raise A6MetadataError(f"document_id missing or empty in {report_path}")
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, str) or _SHA256_HEX.match(source_sha256.strip()) is None:
        raise A6MetadataError(f"source_sha256 missing or invalid in {report_path}")
    return document_id.strip(), source_sha256.strip().lower()


# ---------------------------------------------------------------------------
# Draft construction
# ---------------------------------------------------------------------------


def build_document_metadata_draft(
    *,
    document_id: str,
    document_md_text: str,
    file_name: str,
    document_title: str,
    source_sha256: str,
    ingested_at: int,
) -> tuple[DocumentMetadataDraft, tuple[MetadataAuditRecord, ...]]:
    (
        document_number,
        document_version,
        finalized_at,
        effective_from,
        effective_to,
        audits,
    ) = extract_field_audit(document_id, document_md_text)
    draft = DocumentMetadataDraft(
        document_id=document_id,
        file_name=file_name,
        document_title=document_title,
        document_number=document_number,
        document_version=document_version,
        source_sha256=source_sha256,
        document_content_hash=compute_document_content_hash(document_md_text),
        finalized_at=finalized_at,
        effective_from=effective_from,
        effective_to=effective_to,
        ingested_at=ingested_at,
    )
    return draft, audits


# ---------------------------------------------------------------------------
# SS20-SS24 dedup + version governance
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """SS20.2: NFC, strip, collapse internal whitespace. No other transform."""

    nfc = unicodedata.normalize("NFC", title)
    return re.sub(r"\s+", " ", nfc.strip())


def _finalize(draft: DocumentMetadataDraft, status: str) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=draft.document_id,
        file_name=draft.file_name,
        document_title=draft.document_title,
        document_number=draft.document_number,
        document_version=draft.document_version,
        source_sha256=draft.source_sha256,
        document_content_hash=draft.document_content_hash,
        finalized_at=draft.finalized_at,
        effective_from=draft.effective_from,
        effective_to=draft.effective_to,
        ingested_at=draft.ingested_at,
        status=status,
    )


def govern_documents(
    existing_catalog: Sequence[DocumentMetadata],
    new_drafts: Sequence[DocumentMetadataDraft],
    *,
    now_ms: int | None = None,
) -> GovernanceOutcome:
    """SS17-SS24: assign dedup/version status. Deterministic, order-sensitive.

    *new_drafts* order (caller convention: sorted by document_id, matching A1)
    determines pairwise comparison order within a batch. Existing catalog
    entries may transition ACTIVE -> HISTORICAL or ACTIVE -> VERSION_CONFLICT;
    HISTORICAL entries are never revisited (once superseded, stays resolved).
    Re-processing an already-known document_id is idempotent: ingested_at is
    preserved and the entry is excluded from its own comparison pool.
    """

    created_at = now_ms if now_ms is not None else now_utc_ms()
    pool: dict[str, DocumentMetadata] = {entry.document_id: entry for entry in existing_catalog}
    conflicts: list[DocumentConflict] = []

    def _add_conflict(document_id_a: str, document_id_b: str, reason: str) -> None:
        conflicts.append(
            DocumentConflict(
                document_id_a=document_id_a,
                document_id_b=document_id_b,
                reason=reason,
                created_at=created_at,
            )
        )

    for draft in new_drafts:
        previous_self = pool.get(draft.document_id)
        ingested_at = previous_self.ingested_at if previous_self is not None else draft.ingested_at
        working_draft = replace(draft, ingested_at=ingested_at)
        comparison_pool = [
            entry
            for entry in pool.values()
            if entry.document_id != draft.document_id and entry.status != STATUS_DUPLICATE
        ]

        # SS17: byte-identical original file.
        if any(entry.source_sha256 == working_draft.source_sha256 for entry in comparison_pool):
            pool[draft.document_id] = _finalize(working_draft, STATUS_DUPLICATE)
            continue

        # SS18: byte-different original, identical trusted content.
        if any(
            entry.document_content_hash == working_draft.document_content_hash
            for entry in comparison_pool
        ):
            pool[draft.document_id] = _finalize(working_draft, STATUS_DUPLICATE)
            continue

        same_number: list[DocumentMetadata] = []
        if working_draft.document_number is not None:
            same_number = [
                entry
                for entry in comparison_pool
                if entry.document_number == working_draft.document_number
            ]

        if same_number:
            differing = [
                entry
                for entry in same_number
                if entry.document_content_hash != working_draft.document_content_hash
            ]
            if not differing:
                # Same number, identical content already caught above; be safe.
                pool[draft.document_id] = _finalize(working_draft, STATUS_DUPLICATE)
                continue
            active_matches = [entry for entry in differing if entry.status == STATUS_ACTIVE]
            if len(active_matches) == 1:
                active = active_matches[0]
                both_dated = (
                    working_draft.finalized_at is not None and active.finalized_at is not None
                )
                if both_dated and working_draft.finalized_at != active.finalized_at:
                    if working_draft.finalized_at > active.finalized_at:
                        pool[draft.document_id] = _finalize(working_draft, STATUS_ACTIVE)
                        pool[active.document_id] = replace(active, status=STATUS_HISTORICAL)
                    else:
                        pool[draft.document_id] = _finalize(working_draft, STATUS_HISTORICAL)
                else:
                    reason = (
                        CONFLICT_REASON_SAME_NUMBER_FINALIZED_AT_TIE
                        if both_dated
                        else CONFLICT_REASON_SAME_NUMBER_MISSING_FINALIZED_AT
                    )
                    pool[draft.document_id] = _finalize(working_draft, STATUS_VERSION_CONFLICT)
                    pool[active.document_id] = replace(active, status=STATUS_VERSION_CONFLICT)
                    _add_conflict(draft.document_id, active.document_id, reason)
            else:
                pool[draft.document_id] = _finalize(working_draft, STATUS_VERSION_CONFLICT)
                for entry in differing:
                    if entry.status in (STATUS_ACTIVE, STATUS_VERSION_CONFLICT):
                        pool[entry.document_id] = replace(entry, status=STATUS_VERSION_CONFLICT)
                    reason = (
                        CONFLICT_REASON_SAME_NUMBER_FINALIZED_AT_TIE
                        if entry.finalized_at is not None
                        and working_draft.finalized_at is not None
                        else CONFLICT_REASON_SAME_NUMBER_MISSING_FINALIZED_AT
                    )
                    _add_conflict(draft.document_id, entry.document_id, reason)
            continue

        normalized_title = normalize_title(working_draft.document_title)
        title_matches = [
            entry
            for entry in comparison_pool
            if entry.document_number is None
            and normalize_title(entry.document_title) == normalized_title
            and entry.document_content_hash != working_draft.document_content_hash
        ]
        if title_matches:
            pool[draft.document_id] = _finalize(working_draft, STATUS_VERSION_CONFLICT)
            for entry in title_matches:
                if entry.status in (STATUS_ACTIVE, STATUS_VERSION_CONFLICT):
                    pool[entry.document_id] = replace(entry, status=STATUS_VERSION_CONFLICT)
                _add_conflict(
                    draft.document_id,
                    entry.document_id,
                    CONFLICT_REASON_TITLE_MATCH_NUMBER_INSUFFICIENT,
                )
            continue

        pool[draft.document_id] = _finalize(working_draft, STATUS_ACTIVE)

    ordered = tuple(pool[document_id] for document_id in sorted(pool))
    return GovernanceOutcome(catalog=ordered, new_conflicts=tuple(conflicts))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    text = "\n".join(lines)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8", newline="\n")


def _append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_document_metadata_jsonl(path: Path, catalog: Sequence[DocumentMetadata]) -> None:
    """Full rewrite keyed by document_id (deterministic, no duplicate rows)."""

    _write_jsonl(path, [entry.to_json_dict() for entry in catalog])


def append_metadata_audit_jsonl(
    path: Path, records: Sequence[MetadataAuditRecord]
) -> None:
    _append_jsonl(path, [record.to_json_dict() for record in records])


def append_document_conflicts_jsonl(
    path: Path, conflicts: Sequence[DocumentConflict]
) -> None:
    _append_jsonl(path, [conflict.to_json_dict() for conflict in conflicts])


def load_document_metadata_jsonl(path: Path) -> tuple[DocumentMetadata, ...]:
    if not path.is_file():
        return ()
    text = path.read_text(encoding="utf-8")
    entries: list[DocumentMetadata] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise A6MetadataError(f"{path}:{line_no}: invalid JSON object") from exc
        if not isinstance(payload, dict):
            raise A6MetadataError(f"{path}:{line_no}: each line must be a JSON object")
        missing = [name for name in DOCUMENT_METADATA_FIELD_NAMES if name not in payload]
        if missing:
            raise A6MetadataError(f"{path}:{line_no}: missing fields {missing}")
        entries.append(DocumentMetadata(**{name: payload[name] for name in DOCUMENT_METADATA_FIELD_NAMES}))
    return tuple(entries)


# ---------------------------------------------------------------------------
# End-to-end batch helper (no torch/transformers dependency)
# ---------------------------------------------------------------------------


def build_drafts_from_documents(
    documents: Sequence[LoadedMarkdownDocument],
    registry: dict[str, Any],
    *,
    ingested_at: int,
) -> tuple[DocumentMetadataDraft, ...]:
    """Build one draft per A1 document, joined to A2 title/source and the
    trusted quality_report.json source_sha256. *registry* maps
    document_id -> DocumentRegistryEntry (A2 output, read-only)."""

    drafts: list[DocumentMetadataDraft] = []
    for document in documents:
        entry = registry.get(document.document_id)
        if entry is None:
            raise A6MetadataError(
                f"No A2 registry entry for document_id={document.document_id}"
            )
        _report_document_id, source_sha256 = read_source_identity(document.path.parent)
        draft, _audits = build_document_metadata_draft(
            document_id=document.document_id,
            document_md_text=document.content,
            file_name=entry.source,
            document_title=entry.document_title,
            source_sha256=source_sha256,
            ingested_at=ingested_at,
        )
        drafts.append(draft)
    return tuple(drafts)


def build_audits_for_documents(
    documents: Sequence[LoadedMarkdownDocument],
) -> tuple[MetadataAuditRecord, ...]:
    all_audits: list[MetadataAuditRecord] = []
    for document in documents:
        _n, _v, _f, _ef, _et, audits = extract_field_audit(
            document.document_id, document.content
        )
        all_audits.extend(audits)
    return tuple(all_audits)


def run_governance(
    canonical_root: Path,
    *,
    manifest_path: Path | None,
    existing_catalog_path: Path | None,
    ingested_at: int | None = None,
) -> tuple[GovernanceOutcome, tuple[MetadataAuditRecord, ...]]:
    documents = load_markdown_documents(canonical_root)
    registry = build_document_registry(documents, manifest_path)
    stamp = ingested_at if ingested_at is not None else now_utc_ms()
    drafts = build_drafts_from_documents(documents, registry, ingested_at=stamp)
    audits = build_audits_for_documents(documents)
    existing = (
        load_document_metadata_jsonl(existing_catalog_path)
        if existing_catalog_path is not None
        else ()
    )
    outcome = govern_documents(existing, drafts)
    return outcome, audits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A6 production document metadata: deterministic extraction, "
            "dedup, and version governance. Reads trusted document.md only; "
            "no LLM, no raw PDF re-read, no A0-A5 modification."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--existing-catalog",
        type=Path,
        default=None,
        help="Prior outputs/index_metadata/document_metadata.jsonl for incremental runs",
    )
    parser.add_argument("--metadata-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--conflicts-out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        outcome, audits = run_governance(
            args.canonical_root,
            manifest_path=args.manifest,
            existing_catalog_path=args.existing_catalog,
        )
        write_document_metadata_jsonl(args.metadata_out, outcome.catalog)
        append_metadata_audit_jsonl(args.audit_out, audits)
        append_document_conflicts_jsonl(args.conflicts_out, outcome.new_conflicts)
    except (MarkdownLoadingError, DocumentRegistryError, A6MetadataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    counts: dict[str, int] = {}
    for entry in outcome.catalog:
        counts[entry.status] = counts.get(entry.status, 0) + 1
    print(f"canonical_root={args.canonical_root.resolve()}")
    print(f"document_count={len(outcome.catalog)}")
    for status in sorted(ALL_STATUSES):
        print(f"{status}={counts.get(status, 0)}")
    print(f"new_conflicts={len(outcome.new_conflicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
