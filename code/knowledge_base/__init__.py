from __future__ import annotations

from typing import Any

_EXPORTS = {
    "LoadedMarkdownDocument": "markdown_loader",
    "MarkdownLoadingError": "markdown_loader",
    "load_markdown_documents": "markdown_loader",
    "DocumentRegistryEntry": "document_registry",
    "DocumentRegistryError": "document_registry",
    "build_document_registry": "document_registry",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(f"{__name__}.{module_name}", fromlist=[name])
    return getattr(module, name)
