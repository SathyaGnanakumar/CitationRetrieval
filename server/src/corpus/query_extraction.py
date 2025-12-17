"""Extract queries from citation context in papers."""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def find_citation_contexts(
    paper_text: str,
    citation_keys: List[str],
    context_chars: int = 400,
) -> List[Tuple[str, str]]:
    """
    Find contexts around citations in a paper.

    Args:
        paper_text: Full text of the paper
        citation_keys: List of citation keys to look for (e.g., ["smith2020", "jones2021"])
        context_chars: Number of characters before and after citation (default: 400)

    Returns:
        List of (citation_key, context) tuples
    """
    contexts = []

    # First try to find <|cite_N|> style markers (ScholarCopilot format)
    cite_pattern = r"<\|cite_(\d+)\|>"
    cite_matches = list(re.finditer(cite_pattern, paper_text))

    if cite_matches:
        # Map citation indices to keys
        for idx, cite_key in enumerate(citation_keys):
            # Find the corresponding cite marker
            for match in cite_matches:
                cite_idx = int(match.group(1))
                if cite_idx == idx:
                    start_pos = match.start()

                    # Extract context around the citation
                    context_start = max(0, start_pos - context_chars)
                    context_end = min(len(paper_text), start_pos + context_chars)

                    context = paper_text[context_start:context_end].strip()

                    # Remove the cite marker from context
                    context = re.sub(cite_pattern, "", context)

                    # Clean up the context
                    context = re.sub(r"\s+", " ", context)

                    contexts.append((cite_key, context))
                    break
    else:
        # Fall back to LaTeX citation format
        for cite_key in citation_keys:
            # Look for various citation formats
            # \cite{smith2020}, \citep{smith2020}, \citet{smith2020}, etc.
            patterns = [
                rf"\\cite\{{{re.escape(cite_key)}\}}",
                rf"\\citep\{{{re.escape(cite_key)}\}}",
                rf"\\citet\{{{re.escape(cite_key)}\}}",
                rf"\\cite\w*\{{{re.escape(cite_key)}\}}",
                rf"\[{re.escape(cite_key)}\]",
            ]

            for pattern in patterns:
                matches = list(re.finditer(pattern, paper_text, re.IGNORECASE))

                for match in matches:
                    start_pos = match.start()

                    # Extract context around the citation
                    context_start = max(0, start_pos - context_chars)
                    context_end = min(len(paper_text), start_pos + context_chars)

                    context = paper_text[context_start:context_end].strip()

                    # Clean up the context
                    context = re.sub(r"\s+", " ", context)

                    contexts.append((cite_key, context))
                    break  # Only take first occurrence of this citation

                if contexts and contexts[-1][0] == cite_key:
                    break  # Found this citation, move to next

    return contexts


def extract_query_from_paper_simple(paper_text: str, max_words: int = 50) -> str:
    """
    Simple extraction: take first N words from paper.

    Args:
        paper_text: Full text of the paper
        max_words: Maximum number of words to extract

    Returns:
        Query string
    """
    words = paper_text.split()[:max_words]
    return " ".join(words)


def get_citation_keys_from_bib_info(bib_info: Dict) -> List[str]:
    """
    Extract all citation keys from a paper's bib_info.

    Args:
        bib_info: The bib_info dict from a paper

    Returns:
        List of citation keys
    """
    cite_keys = []

    if not isinstance(bib_info, dict):
        return cite_keys

    for entries in bib_info.values():
        if not isinstance(entries, list):
            continue
        for citation in entries:
            if isinstance(citation, dict):
                key = citation.get("citation_key") or citation.get("paper_id")
                if key:
                    cite_keys.append(str(key))

    return cite_keys


def extract_citation_queries(
    paper: Dict,
    context_chars: int = 400,
    max_queries: int = None,
) -> List[Dict]:
    """
    Extract citation-based queries from a paper.

    For each citation in the paper, finds the context around where it's cited
    and creates a query from that context.

    Args:
        paper: Paper dict with 'paper' text and 'bib_info'
        context_chars: Number of chars before/after citation (default: 400)
        max_queries: Maximum number of queries to extract (None = all)

    Returns:
        List of dicts with 'query', 'target_citation_key', 'target_citation_id'
    """
    paper_text = paper.get("paper", "")
    bib_info = paper.get("bib_info", {}) or {}

    if not paper_text or not bib_info:
        return []

    # Get all citation keys
    cite_keys = get_citation_keys_from_bib_info(bib_info)

    if not cite_keys:
        return []

    # Find contexts for each citation
    contexts = find_citation_contexts(paper_text, cite_keys, context_chars)

    # Build query objects
    queries = []

    # Create mapping from citation_key to full citation info
    key_to_citation = {}
    for entries in bib_info.values():
        if isinstance(entries, list):
            for citation in entries:
                if isinstance(citation, dict):
                    key = citation.get("citation_key") or citation.get("paper_id")
                    if key:
                        key_to_citation[str(key)] = citation

    for cite_key, context in contexts:
        citation_info = key_to_citation.get(cite_key, {})
        citation_id = (
            citation_info.get("citation_key")
            or citation_info.get("paper_id")
            or citation_info.get("id")
        )

        queries.append(
            {
                "query": context,
                "target_citation_key": cite_key,
                "target_citation_id": str(citation_id) if citation_id else cite_key,
                "target_citation_title": citation_info.get("title", ""),
                "source_paper_id": paper.get("paper_id"),
                "source_paper_title": paper.get("title", ""),
            }
        )

        if max_queries and len(queries) >= max_queries:
            break

    return queries
