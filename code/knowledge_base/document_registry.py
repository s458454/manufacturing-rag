"""A2: document-level provenance registry from A1 documents and A0 metadata.

This module assigns ``document_title`` and citation ``source`` to each
``LoadedMarkdownDocument``. It does not parse Markdown, chunk, embed, or index.

``quality_report.json`` is read only for identity fields
(``document_id``, ``source``, ``source_sha256``). Audit JSON is never treated
as knowledge-base body text.

Run the smoke CLI from the repository root with ``code/`` on ``PYTHONPATH``:

    PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" \\
    python -m knowledge_base.document_registry \\
      --canonical-root <ABS_PATH> \\
      [--manifest <ABS_OR_REL_PATH>]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from knowledge_base.markdown_loader import (
    LoadedMarkdownDocument,
    MarkdownLoadingError,
    load_markdown_documents,
)

_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
_WINDOWS_DRIVE_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_MANIFEST_COLUMNS = ("sha256", "title", "source_url", "local_path")


class DocumentRegistryError(Exception):
    """Fail-fast error for A2 document registry construction."""


@dataclass(frozen=True)
class DocumentRegistryEntry:
    document_id: str
    document_title: str
    source: str


@dataclass(frozen=True)
class _ManifestRow:
    sha256: str
    title: str
    source_url: str
    local_path: str


def is_machine_bound_filesystem_source(value: str) -> bool:
    """Return True if *value* is a machine-bound filesystem location.

    The check is OS-independent and applies to the final candidate ``source``
    in manifest mode. No-manifest fallback does not use this predicate.
    """

    if value.startswith("/") or value.startswith("\\"):
        return True
    if _WINDOWS_DRIVE_ABSOLUTE.match(value) is not None:
        return True
    if value.lower().startswith("file:"):
        return True
    return False


def _cross_platform_basename_and_stem(source: str) -> tuple[str, str]:
    """Extract filename and stem from a POSIX or Windows path string."""

    normalized = source.replace("\\", "/")
    path = PurePosixPath(normalized)
    name = path.name
    if not name:
        raise DocumentRegistryError(
            f"Could not extract a filename from quality_report source: {source!r}"
        )
    stem = path.stem
    if not stem:
        raise DocumentRegistryError(
            f"Could not extract a filename stem from quality_report source: {source!r}"
        )
    return name, stem


def _normalize_sha256(value: object, *, field_name: str, location: str) -> str:
    if not isinstance(value, str):
        raise DocumentRegistryError(
            f"{field_name} must be a string in {location}"
        )
    digest = value.strip()
    if _SHA256_HEX.match(digest) is None:
        raise DocumentRegistryError(
            f"Invalid {field_name} in {location}: {value!r}"
        )
    return digest.lower()


def _require_non_empty_string(value: object, *, field_name: str, location: str) -> str:
    if not isinstance(value, str):
        raise DocumentRegistryError(
            f"{field_name} must be a string in {location}"
        )
    text = value.strip()
    if not text:
        raise DocumentRegistryError(f"{field_name} is empty in {location}")
    return text


def _read_quality_report(document: LoadedMarkdownDocument) -> dict[str, object]:
    report_path = document.path.parent / "quality_report.json"
    if not report_path.is_file():
        raise DocumentRegistryError(
            f"quality_report.json missing or not a file: {report_path}"
        )
    try:
        raw = report_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DocumentRegistryError(
            f"Failed to read {report_path}"
        ) from exc
    except UnicodeError as exc:
        raise DocumentRegistryError(
            f"Invalid UTF-8 in {report_path}"
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DocumentRegistryError(
            f"Invalid JSON in {report_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise DocumentRegistryError(
            f"quality_report.json root must be an object: {report_path}"
        )
    return payload


def _identity_from_quality_report(
    document: LoadedMarkdownDocument,
) -> tuple[str, str, str]:
    report_path = document.path.parent / "quality_report.json"
    payload = _read_quality_report(document)
    location = str(report_path)
    report_document_id = _require_non_empty_string(
        payload.get("document_id"),
        field_name="document_id",
        location=location,
    )
    if report_document_id != document.document_id:
        raise DocumentRegistryError(
            "quality_report.document_id does not match A1 document_id: "
            f"a1={document.document_id!r} report={report_document_id!r} "
            f"path={report_path}"
        )
    source_sha256 = _normalize_sha256(
        payload.get("source_sha256"),
        field_name="source_sha256",
        location=location,
    )
    source = payload.get("source")
    source_text = source.strip() if isinstance(source, str) else ""
    return report_document_id, source_sha256, source_text


def _csv_row(row: dict[str | None, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized[key.strip()] = "" if value is None else value.strip()
    return normalized


def _load_manifest(manifest_path: Path) -> dict[str, _ManifestRow]:
    path = Path(manifest_path)
    if not path.exists():
        raise DocumentRegistryError(f"Manifest does not exist: {path}")
    if not path.is_file():
        raise DocumentRegistryError(f"Manifest is not a file: {path}")
    rows: dict[str, _ManifestRow] = {}
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
        with handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DocumentRegistryError(f"Manifest has no header: {path}")
            columns = {
                name.strip()
                for name in reader.fieldnames
                if name is not None and name.strip()
            }
            missing = [name for name in _MANIFEST_COLUMNS if name not in columns]
            if missing:
                raise DocumentRegistryError(
                    "Manifest is missing required columns "
                    f"{missing}: {path}"
                )
            rows = {}
            for line_number, raw_row in enumerate(reader, start=2):
                row = _csv_row(raw_row)
                location = f"{path}:{line_number}"
                digest = _normalize_sha256(
                    row.get("sha256", ""),
                    field_name="sha256",
                    location=location,
                )
                if digest in rows:
                    raise DocumentRegistryError(
                        f"Duplicate manifest sha256 {digest} in {path}"
                    )
                rows[digest] = _ManifestRow(
                    sha256=digest,
                    title=row.get("title", ""),
                    source_url=row.get("source_url", ""),
                    local_path=row.get("local_path", ""),
                )
    except DocumentRegistryError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DocumentRegistryError(f"Failed to read manifest: {path}") from exc
    if not rows:
        raise DocumentRegistryError(f"Manifest has no data rows: {path}")
    return rows


def _candidate_source(row: _ManifestRow, *, document_id: str) -> str:
    if row.source_url:
        candidate = row.source_url
    elif row.local_path:
        candidate = row.local_path
    else:
        raise DocumentRegistryError(
            "Manifest source_url and local_path are both empty for "
            f"document_id={document_id}"
        )
    if is_machine_bound_filesystem_source(candidate):
        raise DocumentRegistryError(
            "Manifest source is a machine-bound filesystem path for "
            f"document_id={document_id}: {candidate!r}"
        )
    return candidate


def _entry_from_manifest(
    document: LoadedMarkdownDocument,
    source_sha256: str,
    rows: dict[str, _ManifestRow],
) -> DocumentRegistryEntry:
    row = rows.get(source_sha256)
    if row is None:
        raise DocumentRegistryError(
            "No manifest row for source_sha256 "
            f"{source_sha256} (document_id={document.document_id})"
        )
    title = row.title.strip()
    if not title:
        raise DocumentRegistryError(
            f"Manifest title is empty for document_id={document.document_id}"
        )
    return DocumentRegistryEntry(
        document_id=document.document_id,
        document_title=title,
        source=_candidate_source(row, document_id=document.document_id),
    )


def _entry_from_source_filename(
    document: LoadedMarkdownDocument,
    quality_source: str,
    report_path: Path,
) -> DocumentRegistryEntry:
    if not quality_source:
        raise DocumentRegistryError(
            f"quality_report.source is missing or empty: {report_path}"
        )
    filename, stem = _cross_platform_basename_and_stem(quality_source)
    return DocumentRegistryEntry(
        document_id=document.document_id,
        document_title=stem,
        source=filename,
    )


def build_document_registry(
    documents: list[LoadedMarkdownDocument],
    manifest_path: Path | None = None,
) -> dict[str, DocumentRegistryEntry]:
    """Build a document-level title/source registry.

    If *manifest_path* is provided, it is the metadata authority for this build:
    every document must join by full ``source_sha256``, and the final ``source``
    value must not be a machine-bound filesystem path. Unused extra manifest
    rows are allowed. Duplicate manifest SHA-256 values fail fast.

    If *manifest_path* is omitted, every document uses a deterministic filename
    fallback derived from ``quality_report.source``. The two modes are never
    mixed.
    """

    if not documents:
        raise DocumentRegistryError("No documents provided")

    seen_ids: set[str] = set()
    for document in documents:
        if document.document_id in seen_ids:
            raise DocumentRegistryError(
                f"Duplicate document_id: {document.document_id}"
            )
        seen_ids.add(document.document_id)

    manifest_rows = (
        _load_manifest(Path(manifest_path)) if manifest_path is not None else None
    )

    registry: dict[str, DocumentRegistryEntry] = {}
    for document in documents:
        _document_id, source_sha256, quality_source = _identity_from_quality_report(
            document
        )
        if manifest_rows is not None:
            entry = _entry_from_manifest(document, source_sha256, manifest_rows)
        else:
            entry = _entry_from_source_filename(
                document,
                quality_source,
                document.path.parent / "quality_report.json",
            )
        registry[document.document_id] = entry
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A2 document registry. Loads A1 documents from an explicit canonical "
            "root, then joins optional corpus manifest metadata by source SHA-256."
        ),
        epilog=(
            "From the repository root, put code/ on PYTHONPATH, for example:\n"
            '  PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" '
            "python -m knowledge_base.document_registry "
            "--canonical-root <ABS_PATH> "
            "[--manifest data/engineering_docs/manifest.csv]"
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
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional corpus manifest CSV. When provided it is the metadata "
            "authority; do not omit it if titles/URLs must come from the manifest."
        ),
    )
    args = parser.parse_args(argv)

    try:
        documents = load_markdown_documents(args.canonical_root)
        registry = build_document_registry(documents, args.manifest)
    except (MarkdownLoadingError, DocumentRegistryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"canonical_root={args.canonical_root.resolve()}")
    print(
        "manifest="
        + (str(args.manifest.resolve()) if args.manifest is not None else "")
    )
    print(f"registry_size={len(registry)}")
    for document_id, entry in registry.items():
        print(f"document_id={document_id}")
        print(f"document_title={entry.document_title}")
        print(f"source={entry.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
