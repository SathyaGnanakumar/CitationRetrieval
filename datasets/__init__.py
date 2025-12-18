"""
Datasets package for retrieval evaluation.

This package contains dataset loaders and preprocessing utilities
for various academic paper datasets used in retrieval evaluation.
"""

from datasets.scholarcopilot import (
    build_citation_corpus,
    clean_text,
    iter_citations,
    load_dataset,
)
from datasets.scholarcopilot.query_extraction import (
    extract_citation_queries,
    extract_query_from_paper_simple,
    find_citation_contexts,
    get_citation_keys_from_bib_info,
)

__all__ = [
    # From loader
    "load_dataset",
    "build_citation_corpus",
    "iter_citations",
    "clean_text",
    # From query_extraction
    "find_citation_contexts",
    "extract_query_from_paper_simple",
    "get_citation_keys_from_bib_info",
    "extract_citation_queries",
]
