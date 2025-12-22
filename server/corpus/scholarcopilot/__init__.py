"""
ScholarCopilot dataset loader and preprocessing utilities.
"""

from corpus.scholarcopilot.loader import (
    build_citation_corpus,
    clean_text,
    iter_citations,
    load_dataset,
)

__all__ = [
    "load_dataset",
    "build_citation_corpus",
    "iter_citations",
    "clean_text",
]
