"""A2 document registry: SHA join, title/source rules, and fail-fast errors."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from knowledge_base.document_registry import (  # noqa: E402
    DocumentRegistryError,
    build_document_registry,
    is_machine_bound_filesystem_source,
)
from knowledge_base.markdown_loader import (  # noqa: E402
    LoadedMarkdownDocument,
    load_markdown_documents,
)

SHA_1 = "11" * 32
SHA_2 = "22" * 32
SHA_3 = "33" * 32
SHA_4003A = "5c5271413bd05f611828e1d5c0eb78c986e32c8df13e864955694e363dc4b14b"
SHA_6033 = "7e4c13bcf106152043ec169025a1a81d671c388269c4ce3e144e01a9f0435050"
SHA_5009C = "9ecc604979b627f584fe0e04348d1ff3cd654e68d00b2326859f5ed1e9c862f2"
REAL_MANIFEST = REPO_ROOT / "data" / "engineering_docs" / "manifest.csv"

SERVER_PATH_MARKERS = ("/public/zhangkairan/",)


def _write_quality_report(folder: Path, payload: dict[str, object]) -> None:
    (folder / "quality_report.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _make_document(
    root: Path,
    document_id: str,
    *,
    content: str = "# NASA-STD-5006A W/CHANGE 2\nbody\n",
    source: str = "/public/zhangkairan/MyMethod/all-in-rag-main/data/raw/doc.pdf",
    source_sha256: str = SHA_1,
    report_document_id: str | None = None,
    extra_report: dict[str, object] | None = None,
    write_report: bool = True,
) -> LoadedMarkdownDocument:
    folder = root / document_id
    folder.mkdir(parents=True)
    path = folder / "document.md"
    path.write_bytes(content.encode("utf-8"))
    if write_report:
        payload: dict[str, object] = {
            "document_id": (
                document_id if report_document_id is None else report_document_id
            ),
            "source": source,
            "source_sha256": source_sha256,
            "ocr_text": "MUST-NOT-ENTER-REGISTRY",
        }
        if extra_report:
            payload.update(extra_report)
        _write_quality_report(folder, payload)
    return LoadedMarkdownDocument(
        document_id=document_id,
        content=content,
        path=path.resolve(),
    )


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha256", "title", "source_url", "local_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _default_manifest_row(
    sha256: str,
    *,
    title: str = "Official Title",
    source_url: str = "https://standards.nasa.gov/example.pdf",
    local_path: str = "raw/joining/example.pdf",
) -> dict[str, str]:
    return {
        "sha256": sha256,
        "title": title,
        "source_url": source_url,
        "local_path": local_path,
    }


def test_manifest_join_uses_title_and_source_url(tmp_path: Path) -> None:
    documents = [
        _make_document(tmp_path, "doc-a", source_sha256=SHA_1),
    ]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _default_manifest_row(
                SHA_1,
                title="NASA-STD-4003A - Electrical Bonding",
                source_url="https://standards.nasa.gov/4003a.pdf",
            )
        ],
    )

    registry = build_document_registry(documents, manifest)

    entry = registry["doc-a"]
    assert entry.document_id == "doc-a"
    assert entry.document_title == "NASA-STD-4003A - Electrical Bonding"
    assert entry.source == "https://standards.nasa.gov/4003a.pdf"
    assert set(registry) == {"doc-a"}


def test_sha256_join_is_case_insensitive(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1.upper(), title="Case Folded")],
    )

    registry = build_document_registry(documents, manifest)

    assert registry["doc-a"].document_title == "Case Folded"


def test_join_is_by_sha256_not_filename(tmp_path: Path) -> None:
    documents = [
        _make_document(
            tmp_path,
            "completely_different_name-5c5271413bd05f61",
            source_sha256=SHA_1,
        )
    ]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _default_manifest_row(
                SHA_1,
                title="Matched By SHA",
                source_url="https://example.invalid/doc.pdf",
                local_path="raw/joining/15_nasa_std_4003a_electrical_bonding.pdf",
            )
        ],
    )

    registry = build_document_registry(documents, manifest)

    assert registry["completely_different_name-5c5271413bd05f61"].document_title == (
        "Matched By SHA"
    )
    assert (
        registry["completely_different_name-5c5271413bd05f61"].source
        == "https://example.invalid/doc.pdf"
    )


def test_first_heading_is_not_used_as_title(tmp_path: Path) -> None:
    content = "# NASA-STD-5006A W/CHANGE 2\n\nIntroduction\n"
    documents = [
        _make_document(
            tmp_path,
            "weld-doc",
            content=content,
            source_sha256=SHA_1,
        )
    ]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _default_manifest_row(
                SHA_1,
                title="NASA-STD-5006A - General Welding Requirements for Aerospace Materials",
            )
        ],
    )

    registry = build_document_registry(documents, manifest)

    assert registry["weld-doc"].document_title == (
        "NASA-STD-5006A - General Welding Requirements for Aerospace Materials"
    )
    assert "W/CHANGE 2" not in registry["weld-doc"].document_title
    assert documents[0].content == content


def test_document_id_mismatch_fails_fast(tmp_path: Path) -> None:
    documents = [
        _make_document(
            tmp_path,
            "doc-a",
            report_document_id="doc-b",
            source_sha256=SHA_1,
        )
    ]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )

    with pytest.raises(DocumentRegistryError, match="does not match A1 document_id"):
        build_document_registry(documents, manifest)


def test_duplicate_manifest_sha_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _default_manifest_row(SHA_1, title="First"),
            _default_manifest_row(SHA_1.upper(), title="Second"),
        ],
    )

    with pytest.raises(DocumentRegistryError, match="Duplicate manifest sha256"):
        build_document_registry(documents, manifest)


def test_missing_manifest_sha_fails_fast_without_partial_result(
    tmp_path: Path,
) -> None:
    documents = [
        _make_document(tmp_path, "doc-a", source_sha256=SHA_1),
        _make_document(tmp_path, "doc-b", source_sha256=SHA_2),
    ]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )

    with pytest.raises(DocumentRegistryError, match="No manifest row"):
        build_document_registry(documents, manifest)


def test_empty_manifest_title_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1, title="   ")],
    )

    with pytest.raises(DocumentRegistryError, match="title is empty"):
        build_document_registry(documents, manifest)


def test_empty_manifest_source_fields_fail_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1, source_url="", local_path="")],
    )

    with pytest.raises(DocumentRegistryError, match="source_url and local_path"):
        build_document_registry(documents, manifest)


def test_relative_local_path_used_when_source_url_empty(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _default_manifest_row(
                SHA_1,
                source_url="",
                local_path="raw/joining/15_nasa_std_4003a_electrical_bonding.pdf",
            )
        ],
    )

    registry = build_document_registry(documents, manifest)

    assert registry["doc-a"].source == (
        "raw/joining/15_nasa_std_4003a_electrical_bonding.pdf"
    )


def test_absolute_local_path_is_ignored_when_source_url_is_logical(
    tmp_path: Path,
) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _default_manifest_row(
                SHA_1,
                source_url="https://standards.nasa.gov/doc.pdf",
                local_path="/public/company/manual.pdf",
            )
        ],
    )

    registry = build_document_registry(documents, manifest)

    assert registry["doc-a"].source == "https://standards.nasa.gov/doc.pdf"


@pytest.mark.parametrize(
    "bad_source",
    [
        "/public/company/manual.pdf",
        r"D:\company\manual.pdf",
        r"\\fileserver\docs\manual.pdf",
        r"\foo\bar.pdf",
        "file:///public/company/docs/manual.pdf",
        "file:///D:/docs/manual.pdf",
        "file://server/share/manual.pdf",
        "FILE:///public/company/docs/manual.pdf",
        "File:C:/docs/manual.pdf",
    ],
)
def test_machine_bound_candidate_source_fails_fast(
    tmp_path: Path,
    bad_source: str,
) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1, source_url=bad_source, local_path="")],
    )

    with pytest.raises(DocumentRegistryError, match="machine-bound filesystem path"):
        build_document_registry(documents, manifest)


@pytest.mark.parametrize(
    "bad_local_path",
    [
        "/public/company/manual.pdf",
        r"D:\company\manual.pdf",
        r"\\fileserver\docs\manual.pdf",
    ],
)
def test_machine_bound_local_path_fails_when_source_url_empty(
    tmp_path: Path,
    bad_local_path: str,
) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1, source_url="", local_path=bad_local_path)],
    )

    with pytest.raises(DocumentRegistryError, match="machine-bound filesystem path"):
        build_document_registry(documents, manifest)


@pytest.mark.parametrize(
    "logical_source",
    [
        "raw/joining/foo.pdf",
        "joining/foo.pdf",
        "https://standards.nasa.gov/foo.pdf",
        "http://example.com/foo.pdf",
        "s3://bucket/key",
        "dms://document/123",
        "oss://bucket/key",
    ],
)
def test_logical_sources_are_allowed(tmp_path: Path, logical_source: str) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1, source_url=logical_source, local_path="")],
    )

    registry = build_document_registry(documents, manifest)

    assert registry["doc-a"].source == logical_source
    assert not is_machine_bound_filesystem_source(registry["doc-a"].source)


def test_no_manifest_fallback_uses_posix_basename(tmp_path: Path) -> None:
    documents = [
        _make_document(
            tmp_path,
            "doc-a",
            source="/foo/bar/my_doc.pdf",
            source_sha256=SHA_1,
        )
    ]

    registry = build_document_registry(documents, None)

    assert registry["doc-a"].document_title == "my_doc"
    assert registry["doc-a"].source == "my_doc.pdf"


def test_no_manifest_fallback_uses_windows_basename(tmp_path: Path) -> None:
    documents = [
        _make_document(
            tmp_path,
            "doc-a",
            source=r"D:\foo\bar\manual.pdf",
            source_sha256=SHA_1,
        )
    ]

    registry = build_document_registry(documents, None)

    assert registry["doc-a"].document_title == "manual"
    assert registry["doc-a"].source == "manual.pdf"


def test_no_manifest_does_not_store_absolute_server_path(tmp_path: Path) -> None:
    documents = [
        _make_document(
            tmp_path,
            "doc-a",
            source="/public/zhangkairan/MyMethod/all-in-rag-main/data/raw/x.pdf",
            source_sha256=SHA_1,
        )
    ]

    registry = build_document_registry(documents, None)

    assert registry["doc-a"].source == "x.pdf"
    for marker in SERVER_PATH_MARKERS:
        assert marker not in registry["doc-a"].source
        assert marker not in registry["doc-a"].document_title


def test_missing_quality_report_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", write_report=False)]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )

    with pytest.raises(DocumentRegistryError, match="quality_report.json missing"):
        build_document_registry(documents, manifest)


def test_invalid_utf8_quality_report_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", write_report=False)]
    (tmp_path / "doc-a" / "quality_report.json").write_bytes(b"\xff\xfe not utf-8")
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )

    with pytest.raises(DocumentRegistryError, match="Invalid UTF-8") as caught:
        build_document_registry(documents, manifest)

    assert isinstance(caught.value.__cause__, UnicodeError)


def test_invalid_utf8_manifest_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256=SHA_1)]
    (tmp_path / "manifest.csv").write_bytes(b"\xff\xfe not utf-8,csv")

    with pytest.raises(DocumentRegistryError, match="Failed to read manifest") as caught:
        build_document_registry(documents, tmp_path / "manifest.csv")

    assert isinstance(caught.value.__cause__, (OSError, UnicodeError, csv.Error))


def test_invalid_quality_report_json_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", write_report=False)]
    (tmp_path / "doc-a" / "quality_report.json").write_text(
        "{not json",
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )

    with pytest.raises(DocumentRegistryError, match="Invalid JSON"):
        build_document_registry(documents, manifest)


def test_invalid_source_sha256_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source_sha256="abc")]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )

    with pytest.raises(DocumentRegistryError, match="Invalid source_sha256"):
        build_document_registry(documents, manifest)


def test_no_manifest_missing_source_fails_fast(tmp_path: Path) -> None:
    documents = [_make_document(tmp_path, "doc-a", source="")]

    with pytest.raises(DocumentRegistryError, match="source is missing or empty"):
        build_document_registry(documents, None)


def test_a1_content_is_unchanged(tmp_path: Path) -> None:
    content = "# heading\n\ntrusted body\n"
    documents = [
        _make_document(
            tmp_path,
            "doc-a",
            content=content,
            source_sha256=SHA_1,
        )
    ]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )
    before = documents[0].content

    build_document_registry(documents, manifest)

    assert documents[0].content == before
    assert documents[0].content is before


def test_audit_json_is_not_persisted_in_registry(tmp_path: Path) -> None:
    documents = [
        _make_document(
            tmp_path,
            "doc-a",
            source_sha256=SHA_1,
            extra_report={"ocr_text": "secret-ocr", "pages": 12},
        )
    ]
    (tmp_path / "doc-a" / "document.json").write_text(
        '{"raw": "not body"}',
        encoding="utf-8",
    )
    (tmp_path / "doc-a" / "regions.json").write_text("[]", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1, title="Kept Title")],
    )

    registry = build_document_registry(documents, manifest)
    entry = registry["doc-a"]

    assert entry.__dict__ == {
        "document_id": "doc-a",
        "document_title": "Kept Title",
        "source": "https://standards.nasa.gov/example.pdf",
    }
    assert "ocr_text" not in entry.__dict__
    assert "secret-ocr" not in entry.document_title
    assert "secret-ocr" not in entry.source


def test_unused_manifest_rows_do_not_fail(tmp_path: Path) -> None:
    documents = [
        _make_document(tmp_path, "doc-a", source_sha256=SHA_1),
        _make_document(tmp_path, "doc-b", source_sha256=SHA_2),
        _make_document(tmp_path, "doc-c", source_sha256=SHA_3),
    ]
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _default_manifest_row(SHA_1, title="One"),
            _default_manifest_row(SHA_2, title="Two"),
            _default_manifest_row(SHA_3, title="Three"),
            _default_manifest_row("44" * 32, title="Unused"),
        ],
    )

    registry = build_document_registry(documents, manifest)

    assert list(registry) == ["doc-a", "doc-b", "doc-c"]
    assert registry["doc-a"].document_title == "One"
    assert registry["doc-c"].document_title == "Three"


def test_empty_document_list_fails_fast(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1)],
    )

    with pytest.raises(DocumentRegistryError, match="No documents provided"):
        build_document_registry([], manifest)


def test_real_nasa_manifest_spotcheck_titles_and_urls(tmp_path: Path) -> None:
    assert REAL_MANIFEST.is_file()
    specs = [
        (
            "completely_other_4003a_dir",
            SHA_4003A,
            "NASA-STD-4003A - Electrical Bonding for NASA Launch Vehicles Spacecraft Payloads and Flight Equipment",
            "https://standards.nasa.gov/system/files/tmp/NASA-STD-4003A_w-Change%201%20-%20Revalidated%2003-13-2026.pdf",
        ),
        (
            "19_nasa_std_6033_am_equipment_facility-7e4c13bcf1061520",
            SHA_6033,
            "NASA-STD-6033 - Additive Manufacturing Equipment and Facility Control",
            "https://standards.nasa.gov/system/files/tmp/2026-01-071%20NASA-STD-6033%20-%20Final%20revalidated.pdf",
        ),
        (
            "21_nasa_std_5009c_nde-9ecc604979b627f5",
            SHA_5009C,
            "NASA-STD-5009C - Nondestructive Evaluation Requirements for Fracture-Critical Metallic Components",
            "https://standards.nasa.gov/sites/default/files/standards/NASA/C/0/2023-08-03-NASA-STD-5009C-Approved.pdf",
        ),
    ]
    documents = [
        _make_document(
            tmp_path,
            document_id,
            source=f"/public/zhangkairan/unused/{document_id}.pdf",
            source_sha256=sha256,
        )
        for document_id, sha256, _title, _source in specs
    ]

    registry = build_document_registry(documents, REAL_MANIFEST)

    assert len(registry) == 3
    for document_id, _sha256, title, source in specs:
        assert registry[document_id].document_title == title
        assert registry[document_id].source == source
        for marker in SERVER_PATH_MARKERS:
            assert marker not in registry[document_id].source


def test_registry_composes_with_a1_loader(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    _make_document(canonical, "doc-a", source_sha256=SHA_1)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [_default_manifest_row(SHA_1, title="From A1")],
    )

    loaded = load_markdown_documents(canonical)
    registry = build_document_registry(loaded, manifest)

    assert loaded[0].document_id == "doc-a"
    assert registry["doc-a"].document_title == "From A1"
    assert loaded[0].content == "# NASA-STD-5006A W/CHANGE 2\nbody\n"
