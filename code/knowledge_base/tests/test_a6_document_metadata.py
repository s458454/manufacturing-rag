"""A6 production document metadata: extraction, dedup, and version governance."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.a6_document_metadata import (  # noqa: E402
    STATUS_ACTIVE,
    STATUS_DUPLICATE,
    STATUS_HISTORICAL,
    STATUS_VERSION_CONFLICT,
    A6MetadataError,
    DocumentConflict,
    DocumentMetadata,
    DocumentMetadataDraft,
    add_years,
    build_document_metadata_draft,
    canonical_document_markdown,
    compute_document_content_hash,
    date_to_utc_ms,
    effective_line_indices,
    extract_field_audit,
    govern_documents,
    load_document_metadata_jsonl,
    normalize_title,
    parse_date_token,
    read_source_identity,
    run_governance,
    scan_window_line_indices,
    write_document_metadata_jsonl,
    append_metadata_audit_jsonl,
    append_document_conflicts_jsonl,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _draft(
    document_id: str = "doc-1",
    *,
    document_md_text: str = "# Title\nbody\n",
    file_name: str = "doc.pdf",
    document_title: str = "Doc Title",
    source_sha256: str = SHA_A,
    ingested_at: int = 1000,
) -> tuple[DocumentMetadataDraft, tuple]:
    return build_document_metadata_draft(
        document_id=document_id,
        document_md_text=document_md_text,
        file_name=file_name,
        document_title=document_title,
        source_sha256=source_sha256,
        ingested_at=ingested_at,
    )


# ---------------------------------------------------------------------------
# SS7 scan window
# ---------------------------------------------------------------------------


def test_effective_line_indices_skips_blank_lines() -> None:
    lines = ["a", "", "  ", "b", "\t", "c"]
    assert effective_line_indices(lines) == [0, 3, 5]


def test_scan_window_covers_whole_short_document() -> None:
    text = "\n".join(f"line{i}" for i in range(50))
    window = scan_window_line_indices(text)
    assert window == frozenset(range(50))


def test_scan_window_skips_middle_of_long_document() -> None:
    # 900 effective lines: front 300 + tail 300 leaves a 300-line gap untouched.
    lines = [f"line{i}" for i in range(900)]
    text = "\n".join(lines)
    window = scan_window_line_indices(text)
    assert 0 in window and 299 in window
    assert 600 in window and 899 in window
    assert 450 not in window  # inside the untouched middle
    assert len(window) == 600


def test_scan_window_tail_reaches_metadata_near_document_end() -> None:
    body = ["filler"] * 700
    body.append("批准日期：2026年1月1日")
    text = "\n".join(body)
    document_number, version, finalized_at, effective_from, effective_to, audits = (
        extract_field_audit("doc-tail", text)
    )
    assert finalized_at == date_to_utc_ms(date(2026, 1, 1))


def test_scan_window_ignores_metadata_in_untouched_middle() -> None:
    head = ["filler"] * 350
    middle = ["批准日期：2026年1月1日"]
    tail = ["filler"] * 350
    text = "\n".join(head + middle + tail)
    _n, _v, finalized_at, _ef, _et, _audits = extract_field_audit("doc-mid", text)
    assert finalized_at is None


# ---------------------------------------------------------------------------
# Date parsing / arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2026年1月9日", date(2026, 1, 9)),
        ("2026-01-09", date(2026, 1, 9)),
        ("2026/01/09", date(2026, 1, 9)),
        ("2026.01.09", date(2026, 1, 9)),
        ("January 9, 2026", date(2026, 1, 9)),
        ("9 January 2026", date(2026, 1, 9)),
        ("not a date", None),
    ],
)
def test_parse_date_token(text: str, expected: date | None) -> None:
    assert parse_date_token(text) == expected


def test_add_years_normal() -> None:
    assert add_years(date(2025, 7, 1), 3) == date(2028, 7, 1)


def test_add_years_leap_day_folds_back() -> None:
    assert add_years(date(2024, 2, 29), 1) == date(2025, 2, 28)


# ---------------------------------------------------------------------------
# document_number (SS8)
# ---------------------------------------------------------------------------


def test_document_number_explicit_label_accepted() -> None:
    text = "# Doc\n文件编号：ABC-123\nbody\n"
    number, _v, _f, _ef, _et, audits = extract_field_audit("d1", text)
    assert number == "ABC-123"
    record = next(a for a in audits if a.field == "document_number")
    assert record.method == "explicit_label"
    assert record.conflict is False
    assert record.line_start == 2 and record.line_end == 2


def test_document_number_consistent_multiple_labels_accepted() -> None:
    text = "文件编号：ABC-123\n标准号：ABC-123\n"
    number, *_rest = extract_field_audit("d1", text)
    assert number == "ABC-123"


def test_document_number_conflicting_labels_is_null_with_conflict_audit() -> None:
    text = "文件编号：ABC-123\n标准号：XYZ-999\n"
    number, _v, _f, _ef, _et, audits = extract_field_audit("d1", text)
    assert number is None
    record = next(a for a in audits if a.field == "document_number")
    assert record.conflict is True
    assert record.value is None


def test_document_number_not_found_without_label() -> None:
    text = "NASA-STD-5009C\nbody without any explicit numbering label\n"
    number, _v, _f, _ef, _et, audits = extract_field_audit("d1", text)
    assert number is None
    record = next(a for a in audits if a.field == "document_number")
    assert record.method == "not_found"
    assert record.conflict is False


def test_document_number_english_label_document_no() -> None:
    text = "Document No.: NASA-STD-4003A\n"
    number, *_rest = extract_field_audit("d1", text)
    assert number == "NASA-STD-4003A"


def test_document_number_table_row_form() -> None:
    text = "| 标准号 | GB/T 1234-2020 |\n"
    number, *_rest = extract_field_audit("d1", text)
    assert number == "GB/T 1234-2020"


# ---------------------------------------------------------------------------
# document_version (SS9)
# ---------------------------------------------------------------------------


def test_document_version_explicit_label() -> None:
    text = "版本号：V2.1\n"
    _n, version, *_rest = extract_field_audit("d1", text)
    assert version == "V2.1"


def test_document_version_chinese_edition_label() -> None:
    text = "版本：2025版\n"
    _n, version, *_rest = extract_field_audit("d1", text)
    assert version == "2025版"


def test_document_version_revision_english_labels() -> None:
    for label, value in (("Version", "2.1"), ("Revision", "C"), ("Rev.", "B")):
        text = f"{label}: {value}\n"
        _n, version, *_rest = extract_field_audit("d1", text)
        assert version == value


def test_document_version_t6_filename_lookalike_is_null_and_ignores_filename() -> None:
    """T6: filename looks like a version string but body has no explicit tag."""

    text = "# NASA-STD-5009C\n技术规范2025版说明，正文未声明版本字段。\n"
    draft, audits = _draft(
        document_id="doc-t6",
        document_md_text=text,
        file_name="技术规范_V3.pdf",
    )
    assert draft.document_version is None
    record = next(a for a in audits if a.field == "document_version")
    assert record.method == "not_found"
    # file_name must never leak into extraction.
    assert draft.file_name == "技术规范_V3.pdf"


# ---------------------------------------------------------------------------
# finalized_at (SS10)
# ---------------------------------------------------------------------------


def test_finalized_at_priority_approval_over_release() -> None:
    text = "批准日期：2026年1月1日\n发布日期：2026年2月1日\n"
    _n, _v, finalized_at, *_rest = extract_field_audit("d1", text)
    assert finalized_at == date_to_utc_ms(date(2026, 1, 1))


def test_finalized_at_falls_back_to_lower_tier_when_top_tier_absent() -> None:
    text = "发布日期：2026年2月1日\n"
    _n, _v, finalized_at, *_rest = extract_field_audit("d1", text)
    assert finalized_at == date_to_utc_ms(date(2026, 2, 1))


def test_finalized_at_conflicting_same_tier_is_null() -> None:
    text = "批准日期：2026年1月1日\nApproval Date: 2026年1月2日\n"
    _n, _v, finalized_at, _ef, _et, audits = extract_field_audit("d1", text)
    assert finalized_at is None
    record = next(a for a in audits if a.field == "finalized_at")
    assert record.conflict is True


def test_finalized_at_not_found() -> None:
    text = "no dates here\n"
    _n, _v, finalized_at, _ef, _et, audits = extract_field_audit("d1", text)
    assert finalized_at is None
    record = next(a for a in audits if a.field == "finalized_at")
    assert record.method == "not_found"


# ---------------------------------------------------------------------------
# effective_from (SS11)
# ---------------------------------------------------------------------------


def test_effective_from_explicit_label() -> None:
    text = "实施日期：2026年3月1日\n"
    _n, _v, _f, effective_from, *_rest = extract_field_audit("d1", text)
    assert effective_from == date_to_utc_ms(date(2026, 3, 1))


def test_effective_from_phrase_without_label() -> None:
    text = "自2026年3月1日起实施。\n"
    _n, _v, _f, effective_from, *_rest = extract_field_audit("d1", text)
    assert effective_from == date_to_utc_ms(date(2026, 3, 1))


def test_effective_from_derived_from_finalized_at_when_issuance_phrase() -> None:
    text = "批准日期：2026年1月1日\n自发布之日起实施。\n"
    _n, _v, finalized_at, effective_from, _et, audits = extract_field_audit("d1", text)
    assert effective_from == finalized_at
    record = next(a for a in audits if a.field == "effective_from")
    assert record.method == "derived_from_finalized_at"


def test_effective_from_issuance_phrase_without_finalized_at_is_null() -> None:
    text = "自发布之日起实施。\n"
    _n, _v, finalized_at, effective_from, *_rest = extract_field_audit("d1", text)
    assert finalized_at is None
    assert effective_from is None


# ---------------------------------------------------------------------------
# effective_to (SS12)
# ---------------------------------------------------------------------------


def test_effective_to_explicit_label() -> None:
    text = "有效期至：2028年1月1日\n"
    *_rest, effective_to, _audits = extract_field_audit("d1", text)
    assert effective_to == date_to_utc_ms(date(2028, 1, 1))


def test_effective_to_duration_calculation() -> None:
    text = "自2025年7月1日起实施，有效期3年。\n"
    _n, _v, _f, effective_from, effective_to, audits = extract_field_audit("d1", text)
    assert effective_from == date_to_utc_ms(date(2025, 7, 1))
    assert effective_to == date_to_utc_ms(date(2028, 7, 1))
    record = next(a for a in audits if a.field == "effective_to")
    assert record.method == "duration_calculation"


def test_effective_to_duration_without_start_date_is_null() -> None:
    text = "有效期3年。\n"
    *_rest, effective_to, _audits = extract_field_audit("d1", text)
    assert effective_to is None


# ---------------------------------------------------------------------------
# document_content_hash (SS18)
# ---------------------------------------------------------------------------


def test_content_hash_ignores_pdf_page_marker() -> None:
    a = "# Title\n<!-- PDF page 1 -->\nbody\n"
    b = "# Title\nbody\n"
    assert compute_document_content_hash(a) == compute_document_content_hash(b)


def test_content_hash_normalizes_crlf() -> None:
    a = "# Title\r\nbody\r\n"
    b = "# Title\nbody\n"
    assert compute_document_content_hash(a) == compute_document_content_hash(b)


def test_content_hash_strips_trailing_whitespace_only() -> None:
    a = "# Title  \nbody\t\n"
    b = "# Title\nbody\n"
    assert compute_document_content_hash(a) == compute_document_content_hash(b)


def test_content_hash_preserves_leading_whitespace_and_case() -> None:
    a = "  indented\nBODY\n"
    b = "indented\nbody\n"
    assert compute_document_content_hash(a) != compute_document_content_hash(b)


def test_content_hash_nfc_normalizes_unicode() -> None:
    # "é" as a single codepoint vs "e" + combining acute accent.
    nfc = "caf\u00e9\n"
    decomposed = "cafe\u0301\n"
    assert compute_document_content_hash(nfc) == compute_document_content_hash(decomposed)


def test_content_hash_differs_for_different_content() -> None:
    assert compute_document_content_hash("a\n") != compute_document_content_hash("b\n")


def test_canonical_document_markdown_matches_hash_input() -> None:
    text = "# T\r\n<!-- PDF page 2 -->\nbody \n"
    canonical = canonical_document_markdown(text)
    assert "PDF page" not in canonical
    assert "\r" not in canonical


# ---------------------------------------------------------------------------
# source identity (quality_report.json)
# ---------------------------------------------------------------------------


def test_read_source_identity_missing_report_fails(tmp_path: Path) -> None:
    folder = tmp_path / "doc-1"
    folder.mkdir()
    with pytest.raises(A6MetadataError):
        read_source_identity(folder)


def test_read_source_identity_invalid_sha_fails(tmp_path: Path) -> None:
    folder = tmp_path / "doc-1"
    folder.mkdir()
    (folder / "quality_report.json").write_text(
        json.dumps({"document_id": "doc-1", "source_sha256": "not-a-sha"}),
        encoding="utf-8",
    )
    with pytest.raises(A6MetadataError):
        read_source_identity(folder)


def test_read_source_identity_valid(tmp_path: Path) -> None:
    folder = tmp_path / "doc-1"
    folder.mkdir()
    (folder / "quality_report.json").write_text(
        json.dumps({"document_id": "doc-1", "source_sha256": SHA_A.upper()}),
        encoding="utf-8",
    )
    document_id, sha = read_source_identity(folder)
    assert document_id == "doc-1"
    assert sha == SHA_A


# ---------------------------------------------------------------------------
# T1/T2: dedup
# ---------------------------------------------------------------------------


def test_t1_identical_source_sha256_is_duplicate() -> None:
    d1, _a1 = _draft("doc-a", document_md_text="# A\nbody\n", source_sha256=SHA_A)
    d2, _a2 = _draft("doc-b", document_md_text="# A different body\n", source_sha256=SHA_A)
    outcome = govern_documents([], [d1, d2])
    statuses = {entry.document_id: entry.status for entry in outcome.catalog}
    assert statuses["doc-a"] == STATUS_ACTIVE
    assert statuses["doc-b"] == STATUS_DUPLICATE


def test_t2_different_source_same_content_hash_is_duplicate() -> None:
    text = "# Same content\nbody\n"
    d1, _a1 = _draft("doc-a", document_md_text=text, source_sha256=SHA_A)
    d2, _a2 = _draft("doc-b", document_md_text=text, source_sha256=SHA_B)
    outcome = govern_documents([], [d1, d2])
    statuses = {entry.document_id: entry.status for entry in outcome.catalog}
    assert statuses["doc-a"] == STATUS_ACTIVE
    assert statuses["doc-b"] == STATUS_DUPLICATE


# ---------------------------------------------------------------------------
# T3/T4/T5: version governance
# ---------------------------------------------------------------------------


def test_t3_same_number_different_content_reliable_dates_orders_versions() -> None:
    old_text = "文件编号：DOC-1\n批准日期：2020年1月1日\nold body\n"
    new_text = "文件编号：DOC-1\n批准日期：2026年1月1日\nnew body\n"
    old, _a = _draft("doc-old", document_md_text=old_text, source_sha256=SHA_A)
    new, _b = _draft("doc-new", document_md_text=new_text, source_sha256=SHA_B)
    outcome = govern_documents([], [old, new])
    statuses = {entry.document_id: entry.status for entry in outcome.catalog}
    assert statuses["doc-old"] == STATUS_HISTORICAL
    assert statuses["doc-new"] == STATUS_ACTIVE
    assert outcome.new_conflicts == ()


def test_t3_incremental_run_transitions_existing_active_to_historical() -> None:
    old_text = "文件编号：DOC-1\n批准日期：2020年1月1日\nold body\n"
    old_draft, _a = _draft("doc-old", document_md_text=old_text, source_sha256=SHA_A)
    first = govern_documents([], [old_draft])
    assert first.catalog[0].status == STATUS_ACTIVE

    new_text = "文件编号：DOC-1\n批准日期：2026年1月1日\nnew body\n"
    new_draft, _b = _draft("doc-new", document_md_text=new_text, source_sha256=SHA_B)
    second = govern_documents(first.catalog, [new_draft])
    statuses = {entry.document_id: entry.status for entry in second.catalog}
    assert statuses["doc-old"] == STATUS_HISTORICAL
    assert statuses["doc-new"] == STATUS_ACTIVE


def test_t4_same_number_missing_dates_is_version_conflict_both_retrievable() -> None:
    a_text = "文件编号：DOC-2\nbody A\n"
    b_text = "文件编号：DOC-2\nbody B\n"
    a, _a = _draft("doc-a", document_md_text=a_text, source_sha256=SHA_A)
    b, _b = _draft("doc-b", document_md_text=b_text, source_sha256=SHA_B)
    outcome = govern_documents([], [a, b])
    statuses = {entry.document_id: entry.status for entry in outcome.catalog}
    assert statuses["doc-a"] == STATUS_VERSION_CONFLICT
    assert statuses["doc-b"] == STATUS_VERSION_CONFLICT
    assert len(outcome.new_conflicts) == 1
    conflict = outcome.new_conflicts[0]
    assert {conflict.document_id_a, conflict.document_id_b} == {"doc-a", "doc-b"}
    assert conflict.reason == "same_document_number_missing_finalized_at"


def test_t4_same_number_equal_finalized_at_is_version_conflict() -> None:
    a_text = "文件编号：DOC-3\n批准日期：2026年1月1日\nbody A\n"
    b_text = "文件编号：DOC-3\n批准日期：2026年1月1日\nbody B\n"
    a, _a = _draft("doc-a", document_md_text=a_text, source_sha256=SHA_A)
    b, _b = _draft("doc-b", document_md_text=b_text, source_sha256=SHA_B)
    outcome = govern_documents([], [a, b])
    statuses = {entry.document_id: entry.status for entry in outcome.catalog}
    assert statuses["doc-a"] == STATUS_VERSION_CONFLICT
    assert statuses["doc-b"] == STATUS_VERSION_CONFLICT
    assert outcome.new_conflicts[0].reason == "same_document_number_finalized_at_tie"


def test_t5_title_match_only_is_version_conflict_neither_deleted() -> None:
    a, _a = _draft(
        "doc-a",
        document_md_text="body A without a number label\n",
        document_title="Same Title",
        source_sha256=SHA_A,
    )
    b, _b = _draft(
        "doc-b",
        document_md_text="body B without a number label\n",
        document_title="Same Title",
        source_sha256=SHA_B,
    )
    outcome = govern_documents([], [a, b])
    statuses = {entry.document_id: entry.status for entry in outcome.catalog}
    assert statuses["doc-a"] == STATUS_VERSION_CONFLICT
    assert statuses["doc-b"] == STATUS_VERSION_CONFLICT
    assert outcome.new_conflicts[0].reason == "title_match_document_number_insufficient"


def test_unrelated_documents_are_both_active() -> None:
    a, _a = _draft(
        "doc-a",
        document_md_text="# A\nunrelated body A\n",
        document_title="Title A",
        source_sha256=SHA_A,
    )
    b, _b = _draft(
        "doc-b",
        document_md_text="# B\nunrelated body B\n",
        document_title="Title B",
        source_sha256=SHA_B,
    )
    outcome = govern_documents([], [a, b])
    statuses = {entry.document_id: entry.status for entry in outcome.catalog}
    assert statuses["doc-a"] == STATUS_ACTIVE
    assert statuses["doc-b"] == STATUS_ACTIVE
    assert outcome.new_conflicts == ()


def test_title_normalization_collapses_whitespace_only() -> None:
    assert normalize_title("  Foo   Bar  ") == "Foo Bar"
    assert normalize_title("Foo Bar") == "Foo Bar"
    assert normalize_title("Foo  Bar") == normalize_title("Foo Bar")


def test_reprocessing_same_document_id_is_idempotent_and_keeps_ingested_at() -> None:
    d1, _a1 = _draft("doc-a", ingested_at=1000)
    first = govern_documents([], [d1])
    assert first.catalog[0].ingested_at == 1000

    d1_again, _a2 = _draft("doc-a", ingested_at=9999)
    second = govern_documents(first.catalog, [d1_again])
    assert len(second.catalog) == 1
    assert second.catalog[0].ingested_at == 1000
    assert second.catalog[0].status == STATUS_ACTIVE


# ---------------------------------------------------------------------------
# Persistence roundtrip
# ---------------------------------------------------------------------------


def test_write_and_load_document_metadata_jsonl_roundtrip(tmp_path: Path) -> None:
    d1, _a1 = _draft("doc-a")
    d2, _a2 = _draft("doc-b", source_sha256=SHA_B)
    outcome = govern_documents([], [d1, d2])
    path = tmp_path / "document_metadata.jsonl"
    write_document_metadata_jsonl(path, outcome.catalog)
    loaded = load_document_metadata_jsonl(path)
    assert loaded == outcome.catalog


def test_write_document_metadata_jsonl_is_a_full_rewrite_not_append(tmp_path: Path) -> None:
    path = tmp_path / "document_metadata.jsonl"
    d1, _a1 = _draft("doc-a")
    outcome1 = govern_documents([], [d1])
    write_document_metadata_jsonl(path, outcome1.catalog)

    d2, _a2 = _draft("doc-b", source_sha256=SHA_B)
    outcome2 = govern_documents(outcome1.catalog, [d2])
    write_document_metadata_jsonl(path, outcome2.catalog)

    loaded = load_document_metadata_jsonl(path)
    assert len(loaded) == 2


def test_append_metadata_audit_and_conflicts_jsonl(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    conflicts_path = tmp_path / "conflicts.jsonl"
    _n, _v, _f, _ef, _et, audits = extract_field_audit("doc-a", "文件编号：X\n")
    append_metadata_audit_jsonl(audit_path, audits)
    conflict = DocumentConflict(
        document_id_a="doc-a", document_id_b="doc-b", reason="x", created_at=1
    )
    append_document_conflicts_jsonl(conflicts_path, [conflict])
    audit_lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(audit_lines) == len(audits)
    for line in audit_lines:
        payload = json.loads(line)
        assert set(payload) == {
            "document_id",
            "field",
            "value",
            "method",
            "evidence",
            "line_start",
            "line_end",
            "conflict",
        }
    conflict_lines = conflicts_path.read_text(encoding="utf-8").splitlines()
    assert len(conflict_lines) == 1
    payload = json.loads(conflict_lines[0])
    assert set(payload) == {"document_id_a", "document_id_b", "reason", "created_at"}


def test_load_document_metadata_jsonl_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_document_metadata_jsonl(tmp_path / "missing.jsonl") == ()


# ---------------------------------------------------------------------------
# End-to-end wiring: real A1 loader + A2 registry (no-manifest fallback)
# ---------------------------------------------------------------------------


def _write_corpus_document(
    root: Path,
    document_id: str,
    *,
    content: str,
    source: str,
    source_sha256: str,
) -> None:
    folder = root / document_id
    folder.mkdir(parents=True)
    (folder / "document.md").write_bytes(content.encode("utf-8"))
    (folder / "quality_report.json").write_text(
        json.dumps(
            {
                "document_id": document_id,
                "source": source,
                "source_sha256": source_sha256,
                "ocr_text": "MUST-NOT-ENTER-A6-METADATA",
            }
        ),
        encoding="utf-8",
    )


def test_run_governance_end_to_end_with_real_a1_a2(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    _write_corpus_document(
        root,
        "doc-a",
        content="# Spec A\n文件编号：SPEC-1\n批准日期：2026年1月1日\nbody A\n",
        source="/srv/raw/规范书A.pdf",
        source_sha256=SHA_A,
    )
    _write_corpus_document(
        root,
        "doc-b",
        content="# Spec B\nbody B, no explicit metadata fields\n",
        source="/srv/raw/规范书B.pdf",
        source_sha256=SHA_B,
    )

    outcome, audits = run_governance(
        root, manifest_path=None, existing_catalog_path=None, ingested_at=42
    )

    by_id = {entry.document_id: entry for entry in outcome.catalog}
    assert by_id["doc-a"].document_number == "SPEC-1"
    assert by_id["doc-a"].file_name == "规范书A.pdf"
    assert by_id["doc-a"].document_title == "规范书A"
    assert by_id["doc-a"].status == STATUS_ACTIVE
    assert by_id["doc-a"].ingested_at == 42
    assert by_id["doc-b"].document_number is None
    assert by_id["doc-b"].status == STATUS_ACTIVE
    assert len(audits) == 10  # 5 governed fields x 2 documents


def test_run_governance_incremental_uses_existing_catalog(tmp_path: Path) -> None:
    root = tmp_path / "canonical"
    root.mkdir()
    _write_corpus_document(
        root,
        "doc-a",
        content="# Spec A v1\n文件编号：SPEC-9\n批准日期：2020年1月1日\nold body\n",
        source="/srv/raw/spec.pdf",
        source_sha256=SHA_A,
    )
    metadata_path = tmp_path / "document_metadata.jsonl"
    first_outcome, _audits = run_governance(
        root, manifest_path=None, existing_catalog_path=None, ingested_at=1
    )
    write_document_metadata_jsonl(metadata_path, first_outcome.catalog)

    root2 = tmp_path / "canonical_v2"
    root2.mkdir()
    _write_corpus_document(
        root2,
        "doc-a-v2",
        content="# Spec A v2\n文件编号：SPEC-9\n批准日期：2026年1月1日\nnew body\n",
        source="/srv/raw/spec_v2.pdf",
        source_sha256=SHA_B,
    )
    second_outcome, _audits2 = run_governance(
        root2,
        manifest_path=None,
        existing_catalog_path=metadata_path,
        ingested_at=2,
    )
    statuses = {entry.document_id: entry.status for entry in second_outcome.catalog}
    assert statuses["doc-a"] == STATUS_HISTORICAL
    assert statuses["doc-a-v2"] == STATUS_ACTIVE
