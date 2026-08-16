"""A1: lossless Markdown loading from an explicit A0 canonical publish root.

This module only discovers and reads formal ``<canonical_root>/<document_id>/document.md``.
It does not parse Markdown, enrich metadata, chunk, embed, or index.

Run the smoke CLI from the repository root with ``code/`` on ``PYTHONPATH``:

    PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" \\
    python -m knowledge_base.markdown_loader \\
      --canonical-root <ABS_PATH>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


class MarkdownLoadingError(Exception):
    """Fail-fast error for A1 Markdown loading."""


@dataclass(frozen=True)
class LoadedMarkdownDocument:
    document_id: str
    content: str
    path: Path


def load_markdown_documents(canonical_root: Path) -> list[LoadedMarkdownDocument]:
    """Discover and losslessly read every first-level ``document.md``.

    Discovery is non-recursive: only ``<canonical_root>/<document_id>/document.md``
    is accepted. First-level document directories and ``document.md`` files must
    not be symlinks. Any discovered file that cannot be read as strict UTF-8
    fails the entire load.
    """

    root = Path(canonical_root)
    discovered: list[Path] = []
    try:
        if not root.exists():
            raise MarkdownLoadingError(f"Canonical root does not exist: {root}")
        if not root.is_dir():
            raise MarkdownLoadingError(f"Canonical root is not a directory: {root}")
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if child.is_symlink():
                raise MarkdownLoadingError(
                    f"Refusing to load a symlinked document directory: {child}"
                )
            candidate = child / "document.md"
            if candidate.is_symlink():
                raise MarkdownLoadingError(
                    f"Refusing to load a symlinked document.md: {candidate}"
                )
            if not candidate.is_file():
                continue
            discovered.append(candidate)
    except MarkdownLoadingError:
        raise
    except OSError as exc:
        raise MarkdownLoadingError(
            f"Failed to enumerate canonical root: {root}"
        ) from exc

    if not discovered:
        raise MarkdownLoadingError(
            "No document.md found under direct children of: "
            f"{root}"
        )

    discovered.sort(key=lambda path: path.parent.name)

    loaded: list[LoadedMarkdownDocument] = []
    for path in discovered:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise MarkdownLoadingError(f"Failed to read {path}") from exc
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MarkdownLoadingError(f"Invalid UTF-8 in {path}") from exc
        loaded.append(
            LoadedMarkdownDocument(
                document_id=path.parent.name,
                content=content,
                path=path.resolve(),
            )
        )
    return loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "A1 lossless Markdown loader. "
            "Requires an explicit A0 canonical publish root; "
            "does not guess outputs, smoke, or acceptance directories."
        ),
        epilog=(
            "From the repository root, put code/ on PYTHONPATH, for example:\n"
            '  PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}" '
            "python -m knowledge_base.markdown_loader "
            "--canonical-root <ABS_PATH>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        required=True,
        help="Formal A0 output root containing <document_id>/document.md",
    )
    args = parser.parse_args(argv)

    try:
        documents = load_markdown_documents(args.canonical_root)
    except MarkdownLoadingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    first = documents[0]
    lossless = first.content.encode("utf-8") == first.path.read_bytes()
    print(f"canonical_root={args.canonical_root.resolve()}")
    print(f"discovered={len(documents)}")
    print(f"loaded={len(documents)}")
    print(f"document_ids={','.join(doc.document_id for doc in documents)}")
    print(f"lossless_check={first.document_id}:{'ok' if lossless else 'fail'}")
    return 0 if lossless else 1


if __name__ == "__main__":
    raise SystemExit(main())
