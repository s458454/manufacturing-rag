from __future__ import annotations

from typing import Any

_EXPORTS = {
    "LoadedMarkdownDocument": "markdown_loader",
    "MarkdownLoadingError": "markdown_loader",
    "load_markdown_documents": "markdown_loader",
    "DocumentRegistryEntry": "document_registry",
    "DocumentRegistryError": "document_registry",
    "build_document_registry": "document_registry",
    "SectionProfileError": "token_count",
    "count_tokens": "token_count",
    "load_tokenizer": "token_count",
    "parse_markdown_document": "structure_parser",
    "profile_documents": "section_profile",
    "A3ChunkingError": "leaf_chunker",
    "Leaf": "leaf_chunker",
    "SectionRef": "leaf_chunker",
    "chunk_documents": "leaf_chunker",
    "chunk_parsed_documents": "leaf_chunker",
    "DEFAULT_CHUNK_SIZE": "chunking_config",
    "DEFAULT_OVERLAP_TOKENS": "chunking_config",
    "A4HierarchyError": "section_hierarchy",
    "SectionNode": "section_hierarchy",
    "SectionHierarchy": "section_hierarchy",
    "build_section_hierarchy": "section_hierarchy",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = __import__(f"{__name__}.{module_name}", fromlist=[name])
    return getattr(module, name)
